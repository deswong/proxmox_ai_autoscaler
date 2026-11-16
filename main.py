import datetime
import re
import time
import logging
import gc
from config import (
    POLL_INTERVAL_SECONDS,
    EXCLUDED_LXCS,
    EXCLUDED_VMS,
    setup_file_logging,
)
import storage
from proxmox_api import ProxmoxClient
from predictor import Predictor
from scaler import Scaler

# Configure root logger for the service
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

INITIAL_ALLOCATION_MARKER = "[Proxmox AI Autoscaler - Initial Allocation]"
_INITIAL_ALLOCATION_RE = re.compile(
    re.escape(INITIAL_ALLOCATION_MARKER)
    + r"\s*CPU Cores:\s*(\d+)\s*RAM:\s*(\d+)\s*MB"
)

# Entities booted less than this many seconds ago are skipped to avoid
# learning from (and reacting to) boot storms.
MIN_UPTIME_SECONDS = 900


def parse_initial_allocation(notes: str):
    """
    Extracts (cpus, ram_mb) from the initial-allocation stamp written into an
    entity's Notes on first detection. Returns None if no stamp is present.
    """
    if not notes:
        return None
    match = _INITIAL_ALLOCATION_RE.search(notes)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def ensure_initial_allocation(px_client, entity_id, entity_type, current_metrics):
    """
    Returns the entity's original (cpus, ram_mb) allocation, recording it into
    the Proxmox Notes on first detection. Falls back to the current allocation
    if the notes cannot be read (and never writes in that case, so existing
    user notes are not clobbered by a transient API failure).
    """
    current = (
        int(current_metrics["allocated_cpus"]),
        int(current_metrics["allocated_ram_mb"]),
    )

    notes = px_client.get_entity_notes(entity_id, entity_type)
    if notes is None:
        return current

    initial = parse_initial_allocation(notes)
    if initial is not None:
        return initial

    if INITIAL_ALLOCATION_MARKER in notes:
        # Marker present but unparseable (hand-edited) — don't append a second one.
        return current

    logger.info(
        f"[{entity_type} {entity_id}] First time detection. Recording initial resources to notes."
    )
    initial_allocation_stamp = (
        f"\n\n{INITIAL_ALLOCATION_MARKER}\n"
        f"CPU Cores: {current[0]}\n"
        f"RAM: {current[1]} MB\n"
    )
    px_client.set_entity_notes(entity_id, entity_type, notes + initial_allocation_stamp)
    return current


def build_dynamic_baseline(entity_type, initial, current_metrics):
    """
    Zero-config baseline anchored to the entity's ORIGINAL allocation.

    Previously this was derived from the *current* allocation every cycle, so
    min_ram_mb == current RAM (RAM could never shrink, and host-pressure
    reclaim was a no-op) and max_ram_mb == 2 × current RAM (the ceiling doubled
    after every scale-up). Anchoring to the recorded initial allocation stops
    that ratchet. The max bounds never drop below the current allocation so an
    entity that already grew past them is not forcibly shrunk below its live
    usage; it simply cannot grow further.
    """
    initial_cpus, initial_ram_mb = initial
    alloc_cpus = int(current_metrics["allocated_cpus"])
    alloc_ram_mb = int(current_metrics["allocated_ram_mb"])

    ram_floor = 128 if entity_type == "LXC" else 1024
    ram_ceiling_floor = 256 if entity_type == "LXC" else 1024

    return {
        "min_cpus": 1,
        "min_ram_mb": max(ram_floor, initial_ram_mb),
        "max_cpus": max(initial_cpus + 4, alloc_cpus),
        "max_ram_mb": max(ram_ceiling_floor, initial_ram_mb * 2, alloc_ram_mb),
    }


def log_prediction(entity_id, predicted_usage, current_metrics, baseline, host_metrics, now):
    """Records a prediction for the nightly trainer to review and learn from."""
    storage.log_prediction(
        entity_id,
        predicted_usage["cpu_percent"],
        predicted_usage["ram_usage_mb"],
        predicted_usage.get("predicted_swap_mb", 0.0),
        predicted_usage.get("recent_peak_disk_read", 0.0),
        predicted_usage.get("recent_peak_disk_write", 0.0),
        predicted_usage.get("recent_peak_net_in", 0.0),
        predicted_usage.get("recent_peak_net_out", 0.0),
        ctx_hour=now.hour,
        ctx_dow=now.weekday(),
        ctx_host_load_1m=host_metrics.get("load_avg_1m", 0.0),
        ctx_host_load_5m=host_metrics.get("load_avg_5m", 0.0),
        ctx_cpu_overcommit=host_metrics.get("cpu_overcommit_ratio", 0.0),
        ctx_ram_overcommit=host_metrics.get("ram_overcommit_ratio", 0.0),
        ctx_container_count=int(host_metrics.get("container_count", 0)),
        ctx_ksm_sharing_mb=host_metrics.get("ksm_sharing_mb", 0.0),
        ctx_actual_cpu=current_metrics.get("cpu_percent", 0.0),
        ctx_actual_ram=current_metrics.get("ram_usage_mb", 0.0),
        ctx_actual_swap=current_metrics.get("swap_mb", 0.0),
        ctx_min_cpus=baseline.get("min_cpus", 0),
        ctx_max_cpus=baseline.get("max_cpus", 0),
        ctx_min_ram=baseline.get("min_ram_mb", 0),
        ctx_max_ram=baseline.get("max_ram_mb", 0),
    )


def process_entity(
    px_client, predictor, scaler, entity_id, entity_type,
    current_metrics, explicit_baselines, host_metrics, now,
):
    """Runs one predict → log → scale pass for a single LXC or VM."""
    tag = f"[{entity_type} {entity_id}]"

    # Skip recently booted entities to prevent learning from boot storms
    uptime = current_metrics.get("uptime", 0)
    if uptime < MIN_UPTIME_SECONDS:
        logger.info(
            f"{tag} Skipping scaling. Entity just booted "
            f"(Uptime: {uptime}s < {MIN_UPTIME_SECONDS}s)."
        )
        return

    initial = ensure_initial_allocation(px_client, entity_id, entity_type, current_metrics)

    # Determine baseline: Use explicit if it exists, otherwise build a dynamic one
    if entity_id in explicit_baselines:
        baseline = explicit_baselines[entity_id]
    else:
        baseline = build_dynamic_baseline(entity_type, initial, current_metrics)
        logger.debug(f"{tag} Using dynamic fallback baseline: {baseline}")

    # Retrieve recent RRD time-series (last hour is sufficient; we only use 15 points)
    if entity_type == "LXC":
        historical_metrics = px_client.get_lxc_rrd_history(entity_id, timeframe="hour")
    else:
        historical_metrics = px_client.get_vm_rrd_history(entity_id, timeframe="hour")

    # Predict impending usage (pass live host metrics as ML context)
    predicted_usage = predictor.predict_next_usage(
        entity_id, historical_metrics, entity_type=entity_type,
        host_context=host_metrics,
    )
    if predicted_usage is None:
        return

    try:
        log_prediction(entity_id, predicted_usage, current_metrics, baseline, host_metrics, now)
    except Exception as db_err:
        logger.warning(
            f"{tag} Failed to log prediction for reinforcement learning: {db_err}"
        )

    if entity_type == "LXC":
        scaler.evaluate_and_scale(
            entity_id, "LXC", baseline, predicted_usage, current_metrics,
            host_metrics=host_metrics,
        )
    else:
        # Compute optimal next-reboot sizing from 14-day rolling peak.
        # No live hotplug is attempted for VMs — the config is written as
        # a pending Proxmox entry that takes effect on the next reboot,
        # ensuring the VM always has headroom without risking guest OS crashes.
        rolling_peaks = storage.get_vm_rolling_peaks(entity_id)
        scaler.apply_vm_pending_config(
            entity_id, baseline, predicted_usage, current_metrics, rolling_peaks,
            host_metrics=host_metrics,
        )


def build_host_metrics(px_client, all_lxc_metrics, all_vm_metrics):
    """
    Fetches host node metrics once per cycle — used as ML context AND safety caps —
    and enriches them with overcommit ratios and worst-case committed totals.
    """
    host_metrics = px_client.get_host_usage()

    # Cluster overcommit ratios from data already in RAM (zero extra API calls).
    all_containers = {**all_lxc_metrics, **all_vm_metrics}
    physical_cpus = max(host_metrics.get("physical_cpus", 1), 1)
    total_ram_mb = max(host_metrics.get("total_ram_mb", 1.0), 1.0)
    total_alloc_cpus = sum(m.get("allocated_cpus", 0) for m in all_containers.values())
    total_alloc_ram = sum(m.get("allocated_ram_mb", 0.0) for m in all_containers.values())
    host_metrics["cpu_overcommit_ratio"] = total_alloc_cpus / physical_cpus
    host_metrics["ram_overcommit_ratio"] = total_alloc_ram / total_ram_mb
    host_metrics["container_count"] = float(len(all_containers))

    # Worst-case committed capacity — includes ALL entities (running + stopped).
    # If every stopped VM/LXC were powered on simultaneously, the host must still
    # be able to service all their configured allocations. The scaler refuses
    # any scale-up that would push the total committed resources over the limit.
    committed = px_client.get_all_committed_resources()
    host_metrics["committed_cpus"] = committed["committed_cpus"]
    host_metrics["committed_ram_mb"] = committed["committed_ram_mb"]
    host_metrics["total_entity_count"] = committed["total_entity_count"]
    host_metrics["stopped_entity_count"] = committed["stopped_entity_count"]

    if committed["stopped_entity_count"] > 0:
        logger.debug(
            f"Worst-case capacity: {committed['committed_cpus']} CPUs / "
            f"{committed['committed_ram_mb']:.0f} MB RAM committed across "
            f"{committed['total_entity_count']} entities "
            f"({committed['stopped_entity_count']} currently stopped)."
        )
    return host_metrics


def run():
    setup_file_logging()
    logger.info("Starting Proxmox Universal AI Autoscaler Service...")

    # Initialize SQLite database and baseline configurations
    storage.init_db()

    # Initialize Proxmox API Client
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to initialize Proxmox Client. Fatal Error.")
        return

    predictor = Predictor(prediction_horizon=2)  # Predict 2 cycles (minutes) ahead
    scaler = Scaler(px_client)

    # Warn if poll interval is below Proxmox's RRD resolution (1-minute granularity).
    # Polling faster than 60s wastes API calls since no new RRD data will be available yet.
    if POLL_INTERVAL_SECONDS < 60:
        logger.warning(
            f"POLL_INTERVAL_SECONDS is set to {POLL_INTERVAL_SECONDS}s, which is below Proxmox's "
            f"RRD resolution of 60s. Predictions will repeat the same data until a new data point "
            f"arrives. Consider setting POLL_INTERVAL_SECONDS=60 or higher."
        )

    # Main run loop
    while True:
        cycle_start_time = time.time()

        try:
            # Re-fetch explicit baselines (from .env/DB)
            explicit_baselines = storage.get_baselines()

            # Bulk fetch telemetry for all running LXCs and VMs
            all_lxc_metrics = px_client.get_all_lxc_metrics()
            all_vm_metrics = px_client.get_all_vm_metrics()

            if not all_lxc_metrics and not all_vm_metrics:
                logger.warning(
                    "No running LXCs or VMs found on this node. Nothing to monitor."
                )

            host_metrics = build_host_metrics(px_client, all_lxc_metrics, all_vm_metrics)
            now = datetime.datetime.now()

            work = [
                ("LXC", all_lxc_metrics, EXCLUDED_LXCS),
                ("VM", all_vm_metrics, EXCLUDED_VMS),
            ]
            for entity_type, metrics_by_id, excluded in work:
                for entity_id, current_metrics in metrics_by_id.items():
                    if entity_id in excluded:
                        logger.debug(
                            f"[{entity_type} {entity_id}] Skipping "
                            f"(Listed in EXCLUDED_{entity_type}S blacklist)."
                        )
                        continue
                    try:
                        process_entity(
                            px_client, predictor, scaler, entity_id, entity_type,
                            current_metrics, explicit_baselines, host_metrics, now,
                        )
                    except Exception as e:
                        logger.error(
                            f"[{entity_type} {entity_id}] Exception during autoscaling cycle: {e}",
                            exc_info=True,
                        )
        except Exception as e:
            # A transient failure (e.g. SQLite locked by the nightly trainer) must not
            # kill the daemon — log it and retry on the next cycle.
            logger.error(f"Autoscaling cycle failed: {e}", exc_info=True)

        # Force garbage collection to keep daemon memory usage very low over time
        gc.collect()

        # Sleep until next poll interval
        elapsed = time.time() - cycle_start_time
        sleep_time = max(0.5, POLL_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
