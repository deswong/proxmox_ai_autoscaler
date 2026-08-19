import math
import logging
import storage
from config import (
    MAX_HOST_CPU_ALLOCATION_PERCENT,
    MAX_HOST_RAM_ALLOCATION_PERCENT,
    MAX_HOST_SWAP_USAGE_PERCENT,
    LXC_TARGET_SWAP_MB,
    LXC_MIN_SWAP_MB,
    HOST_RAM_RESERVE_PERCENT,
    SWAP_FLUSH_THRESHOLD_PERCENT,
)
from proxmox_api import ProxmoxClient

logger = logging.getLogger("scaler")

# Host RAM % above which we actively push idle containers down to reclaim headroom.
# Lowered from 90% → 80% so reclaiming starts well before the host is critically stressed.
HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = 80.0
# A container is "idle relative to allocation" when it uses less than this fraction.
CONTAINER_IDLE_ALLOCATION_RATIO = 0.5
# Target: scale idle containers to this multiple of current usage (+ 30% buffer).
ACTIVE_SCALEDOWN_HEADROOM_RATIO = 1.5


class Scaler:
    def __init__(self, proxmox_client: ProxmoxClient):
        self.px = proxmox_client

        # Buffer percentages applied to the prediction to ensure we don't scale
        # too tightly to the absolute exact predicted Mb/Cpu, leaving overhead.
        self.ram_buffer_percent = 30.0
        self.cpu_buffer_percent = 20.0

    def evaluate_and_scale(
        self,
        entity_id: str,
        entity_type: str,
        baseline: dict,
        predicted: dict,
        current_metrics: dict,
    ):
        """
        Evaluates predictions against max/min baselines and overall node health.
        Triggers a scaling action if requirements change.

        Host pressure is handled at three tiers:
          < threshold     — normal operation
          threshold–90%   — block scale-ups (existing safety cap)
          > 90% RAM       — block scale-ups AND actively scale down idle containers
        """

        if not current_metrics:
            logger.warning(
                f"[{entity_type} {entity_id}] No current metrics to base scaling on. Skipping."
            )
            return

        # Fetch live host node metrics early so they are available for all safety guards
        host_metrics = self.px.get_host_usage()
        host_ram_pct = host_metrics.get("ram_percent", 0.0)
        host_cpu_pct = host_metrics["cpu_percent"]
        host_swap_pct = host_metrics.get("swap_percent", 0.0)

        # Hardcoded emergency safeguards (caps user config at max 95%)
        safe_cpu_limit = min(MAX_HOST_CPU_ALLOCATION_PERCENT, 95.0)
        safe_ram_limit = min(MAX_HOST_RAM_ALLOCATION_PERCENT, 95.0)
        safe_swap_limit = min(MAX_HOST_SWAP_USAGE_PERCENT, 95.0)

        # 0. Pre-calculate swap status (Needed for RAM headroom planning)
        swap_used = 0.0
        swap_alloc = 0.0

        if entity_type == "LXC":
            swap_used = current_metrics.get("swap_mb", 0.0)
            swap_alloc = current_metrics.get("allocated_swap_mb", 0.0)

        # 1. Calculate the raw desired resources from the predictor.
        #    Use the higher of the ML forecast, the observed recent peak, or
        #    the current actual usage to ensure we never under-allocate.
        current_usage_mb = current_metrics.get("ram_usage_mb", 0.0)
        peak_ram_mb = max(
            predicted["ram_usage_mb"],
            predicted.get("recent_peak_ram", 0.0),
            current_usage_mb
        )

        # "Natural Reclaim" RAM Boost: If the container is actively using swap, we
        # MUST increase its RAM headroom to give the OS enough physical space
        # to naturally page those swap blocks back into RAM on its own schedule.
        if swap_used > 5 and entity_type == "LXC":
            needed_for_swap = (current_usage_mb + swap_used) * 1.15
            if peak_ram_mb < needed_for_swap:
                logger.debug(
                    f"[{entity_type} {entity_id}] Boosting target RAM to {needed_for_swap:.0f} MB "
                    f"to allow natural reclaim of {swap_used:.0f} MB swap."
                )
                peak_ram_mb = needed_for_swap

        # Tapered RAM buffer: the headroom shrinks proportionally as the host approaches its
        # safety threshold, so we never blindly over-allocate with a flat 30% when RAM is tight.
        # At 0% host usage → full ram_buffer_percent. At the threshold → min 5%.
        host_ram_pressure_ratio = min(host_ram_pct / max(safe_ram_limit, 1.0), 1.0)
        effective_ram_buffer = max(5.0, self.ram_buffer_percent * (1.0 - host_ram_pressure_ratio))

        if effective_ram_buffer < self.ram_buffer_percent:
            logger.debug(
                f"[{entity_type} {entity_id}] Host RAM at {host_ram_pct:.1f}% — tapering "
                f"RAM buffer from {self.ram_buffer_percent:.0f}% → {effective_ram_buffer:.1f}%."
            )
        desired_ram_mb = peak_ram_mb * (1 + effective_ram_buffer / 100.0)

        # Hard ceiling: we must never allocate more vCPUs than the host physically has.
        physical_cpus = max(int(host_metrics.get("physical_cpus", 1)), 1)

        # CPU scaling heuristic using a blended signal:
        #   60% weight to the EWMA-smoothed reading (spike-dampened)
        #   40% weight to the ML model prediction
        # This prevents a single-cycle burst from immediately adding cores, while
        # still responding meaningfully when load is genuinely and persistently high.
        smoothed_cpu = predicted.get("smoothed_cpu_percent", predicted["cpu_percent"])
        blended_cpu = 0.6 * smoothed_cpu + 0.4 * predicted["cpu_percent"]

        desired_cpus = current_metrics["allocated_cpus"]
        if blended_cpu > 85.0:
            overshoot = blended_cpu - 85.0
            cores_to_add = max(1, int(overshoot / 15))
            desired_cpus += cores_to_add
        elif (
            blended_cpu < 25.0
            and smoothed_cpu < 25.0  # EWMA must also agree — prevents scale-down on brief lulls
            and current_metrics["cpu_percent"] < 25.0  # live reading confirms it's genuinely idle
        ):
            undershoot = 25.0 - blended_cpu
            cores_to_remove = max(1, int(undershoot / 30))
            desired_cpus = max(1, desired_cpus - cores_to_remove)

        # Clamp desired_cpus to what the host can physically provide BEFORE applying baselines.
        desired_cpus = min(desired_cpus, physical_cpus)

        logger.info(
            f"[{entity_type} {entity_id}] Analyzing... Current State: "
            f"{current_metrics['allocated_cpus']} Cores, {current_metrics['allocated_ram_mb']} MB RAM. "
            f"Predicted Need: {desired_cpus} Cores "
            f"(blended CPU: {blended_cpu:.1f}%, smoothed: {smoothed_cpu:.1f}%, raw: {predicted['cpu_percent']:.1f}%), "
            f"{predicted['ram_usage_mb']:.0f} MB RAM."
        )

        # 2. Bound against configured baselines (min/max for this entity)
        # Apply a hard system floor (64MB LXC, 1024MB VM) to prevent OS crashes
        system_floor = 64 if entity_type == "LXC" else 1024
        target_ram = max(
            baseline["min_ram_mb"],
            system_floor,
            min(int(desired_ram_mb), baseline["max_ram_mb"]),
        )

        # Final Guard: Ensure we never shrink RAM if container is heavily swapped
        # Priority 1: Ensure enough headroom (usage + swap).
        if swap_used > 5 and entity_type == "LXC":
            # Ensure physical capacity for pages
            target_ram = max(target_ram, int(current_usage_mb + swap_used + 128))
            # Prevent scale-down unless host is in emergency state (>95%)
            if host_ram_pct < 95.0:
                target_ram = max(target_ram, int(current_metrics["allocated_ram_mb"]))
            target_ram = min(target_ram, baseline["max_ram_mb"])

        # Clamp target_cpus: respect baseline bounds AND never exceed physical host CPU count.
        target_cpus = max(baseline["min_cpus"], min(desired_cpus, baseline["max_cpus"], physical_cpus))
        if target_cpus < desired_cpus:
            logger.debug(
                f"[{entity_type} {entity_id}] CPU clamped to {target_cpus} "
                f"(physical host limit: {physical_cpus}, baseline max: {baseline['max_cpus']})."
            )

        # 3. Check physical node limits before scaling UP
        # Hardcoded emergency safeguard (caps user config at max 95%)

        # Apply Host Swap Safety Cap
        # If the host is heavily swapping, completely block all scale-ups to prevent
        # exacerbating an already memory-starved hypervisor.
        if (
            host_swap_pct > safe_swap_limit
            and (target_cpus > current_metrics["allocated_cpus"] or target_ram > current_metrics["allocated_ram_mb"])
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale up. Host Node Swap is over "
                f"threshold ({host_swap_pct:.1f}% > {safe_swap_limit}%)."
            )
            # Limit scale up to current allocation for both
            target_cpus = min(target_cpus, current_metrics["allocated_cpus"])
            target_ram = min(target_ram, current_metrics["allocated_ram_mb"])

        if (
            host_cpu_pct > safe_cpu_limit
            and target_cpus > current_metrics["allocated_cpus"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale CPU up. Host Node CPU is over "
                f"threshold ({host_cpu_pct:.1f}% > {safe_cpu_limit}%)."
            )
            # Limit scale up to current allocation
            target_cpus = current_metrics["allocated_cpus"]

        if (
            host_ram_pct > safe_ram_limit
            and target_ram > current_metrics["allocated_ram_mb"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale RAM up. Host Node RAM is over "
                f"threshold ({host_ram_pct:.1f}% > {safe_ram_limit}%)."
            )
            # But we can allow scaling down RAM, just not UP.
            target_ram = min(target_ram, current_metrics["allocated_ram_mb"])

        # Worst-case committed capacity check.
        # We treat every stopped VM/LXC as a latent demand that could materialise at
        # any moment. Before approving a scale-up we verify that the proposed new
        # allocation still leaves the host able to start ALL entities simultaneously.
        #
        # committed_cpus / committed_ram_mb come from get_all_committed_resources()
        # and include stopped entities — this is the key difference from the current
        # running-only overcommit ratios computed in main.py.
        total_ram_mb = host_metrics.get("total_ram_mb", 0.0)
        committed_cpus   = host_metrics.get("committed_cpus", 0)
        committed_ram_mb = host_metrics.get("committed_ram_mb", 0.0)

        if total_ram_mb > 0 and committed_cpus > 0:
            # --- CPU worst-case budget ---
            cpu_delta = target_cpus - current_metrics["allocated_cpus"]
            if cpu_delta > 0:
                # Max cores we are willing to commit across ALL entities
                max_committable_cpus = int(physical_cpus * (safe_cpu_limit / 100.0))
                projected_committed_cpus = committed_cpus + cpu_delta
                if projected_committed_cpus > max_committable_cpus:
                    # Calculate how many cores are actually available in the budget
                    available_cpu_budget = max(0, max_committable_cpus - committed_cpus)
                    clamped_cpus = current_metrics["allocated_cpus"] + available_cpu_budget
                    clamped_cpus = max(baseline["min_cpus"], min(clamped_cpus, target_cpus))
                    if clamped_cpus < target_cpus:
                        logger.warning(
                            f"[{entity_type} {entity_id}] WORST-CASE CPU CAP: "
                            f"Scaling to {target_cpus} cores would push total committed "
                            f"to {projected_committed_cpus} (budget: {max_committable_cpus} "
                            f"of {physical_cpus} physical). Clamping to {clamped_cpus}."
                        )
                        target_cpus = clamped_cpus

            # --- RAM worst-case budget ---
            ram_delta = target_ram - current_metrics["allocated_ram_mb"]
            if ram_delta > 0:
                # Max RAM we are willing to commit across ALL entities (reserve stays free)
                reserved_mb = total_ram_mb * (HOST_RAM_RESERVE_PERCENT / 100.0)
                max_committable_ram = total_ram_mb - reserved_mb
                projected_committed_ram = committed_ram_mb + ram_delta
                if projected_committed_ram > max_committable_ram:
                    available_ram_budget = max(0.0, max_committable_ram - committed_ram_mb)
                    ram_ceiling_mb = int(current_metrics["allocated_ram_mb"] + available_ram_budget)
                    ram_ceiling_mb = max(int(baseline["min_ram_mb"]), min(ram_ceiling_mb, target_ram))
                    if ram_ceiling_mb < target_ram:
                        logger.warning(
                            f"[{entity_type} {entity_id}] WORST-CASE RAM CAP: "
                            f"Scaling to {target_ram} MB would push total committed "
                            f"to {projected_committed_ram:.0f} MB "
                            f"(budget: {max_committable_ram:.0f} MB, "
                            f"reserve: {reserved_mb:.0f} MB of {total_ram_mb:.0f} MB total). "
                            f"Clamping to {ram_ceiling_mb} MB."
                        )
                        target_ram = ram_ceiling_mb
        elif total_ram_mb > 0:
            # Fallback: committed data not yet available — use live free-RAM ceiling
            # (original guard, kept as a safety net on first cycle before API data arrives)
            reserved_mb = total_ram_mb * (HOST_RAM_RESERVE_PERCENT / 100.0)
            host_free_mb = total_ram_mb * (1.0 - host_ram_pct / 100.0)
            available_for_scale = host_free_mb - reserved_mb
            ram_delta = target_ram - current_metrics["allocated_ram_mb"]
            if available_for_scale <= 0 < ram_delta:
                logger.warning(
                    f"[{entity_type} {entity_id}] AVAILABLE-RAM CAP: No headroom left after "
                    f"{HOST_RAM_RESERVE_PERCENT:.0f}% reserve ({reserved_mb:.0f} MB). "
                    "Blocking scale-up."
                )
                target_ram = current_metrics["allocated_ram_mb"]
            elif ram_delta > 0:
                ram_ceiling_mb = int(current_metrics["allocated_ram_mb"] + available_for_scale)
                target_ram = min(target_ram, ram_ceiling_mb)



        # 3b. Active scale-down: when the host is critically RAM-stressed AND this
        #     container is genuinely idle relative to its allocation, reclaim headroom.
        #     Guard: both conditions must be true simultaneously so a busy container
        #     during a host-wide load event is never aggressively shrunk.
        if (
            entity_type == "LXC"
            and host_ram_pct > HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
        ):
            ram_usage = current_metrics.get("ram_usage_mb", 0.0)
            alloc_ram = current_metrics["allocated_ram_mb"]
            if (
                alloc_ram > 0
                and ram_usage / alloc_ram < CONTAINER_IDLE_ALLOCATION_RATIO
            ):
                # Container is using < 50% of its allocation while the host is struggling.
                # Nudge it down to usage × 1.5, floored at min_ram_mb.
                reclaimed_target = max(
                    int(ram_usage * ACTIVE_SCALEDOWN_HEADROOM_RATIO),
                    baseline["min_ram_mb"],
                )
                if reclaimed_target < alloc_ram:
                    logger.warning(
                        f"[{entity_type} {entity_id}] HOST PRESSURE RECLAIM: Host RAM at "
                        f"{host_ram_pct:.1f}% (>{HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD:.0f}%). "
                        f"Container idle ({ram_usage:.0f}/{alloc_ram:.0f} MB used). "
                        f"Actively reducing RAM to {reclaimed_target} MB to ease host pressure."
                    )
                    target_ram = reclaimed_target

        if entity_type == "LXC":
            if swap_used > 5:
                logger.info(
                    f"[LXC {entity_id}] Natural Reclaim active ({swap_used:.0f}/{swap_alloc:.0f} MB used). "
                    "Waiting for OS to page back to RAM."
                )

        # 5. Compute the target swap cap for this LXC.
        #    Auto mode (-1): size swap like RAM — use observed peak + 30% buffer,
        #    floored at LXC_MIN_SWAP_MB so no container is ever left fully swapless
        #    during the model cold-start period.
        if entity_type == "LXC":
            if LXC_TARGET_SWAP_MB == -1:
                peak_swap = max(
                    predicted.get("predicted_swap_mb", 0.0),
                    predicted.get("recent_peak_swap", 0.0),
                )
                target_swap = max(
                    int(peak_swap * (1 + self.ram_buffer_percent / 100.0)),
                    LXC_MIN_SWAP_MB,
                )
            else:
                target_swap = max(LXC_TARGET_SWAP_MB, LXC_MIN_SWAP_MB)

            # NATURAL RECLAIM "DO NO HARM" FLOOR:
            # Never set the swap limit lower than the active swap usage plus a 32MB buffer.
            # Why? Because lowering the cgroup limit below usage forces the Linux kernel into
            # a synchronous reclaim (pausing the container, reading disk sequentially to RAM).
            # This causes catastrophic I/O stalls in Proxmox. We MUST let the OS page it back
            # gently on its own schedule.
            safe_floor = int(swap_used + 32)
            if target_swap < safe_floor and swap_used > 5:
                logger.debug(f"[LXC {entity_id}] Adjusting target swap from {target_swap} MB to safe floor {safe_floor} MB to prevent cgroup stall.")
                target_swap = safe_floor
        else:
            target_swap = 0  # VMs manage swap internally; we don't set this

        # 6. Apply changes if different from currently allocated.
        #    Triggers on: CPU change, RAM change (>=32 MB), swap cap change (>=32 MB).
        ram_diff = abs(target_ram - current_metrics["allocated_ram_mb"])
        current_swap_alloc = current_metrics.get("allocated_swap_mb", 0.0)
        swap_diff = abs(target_swap - current_swap_alloc) if entity_type == "LXC" else 0.0

        if (
            target_cpus != current_metrics["allocated_cpus"]
            or ram_diff >= 32
            or swap_diff >= 32  # match RAM threshold — avoids micro-updates on 1-MB rounding changes
        ):
            cpu_action = "UNCHANGED"
            if target_cpus > current_metrics["allocated_cpus"]:
                cpu_action = "UP"
            elif target_cpus < current_metrics["allocated_cpus"]:
                cpu_action = "DOWN"

            ram_action = "UNCHANGED"
            if target_ram > current_metrics["allocated_ram_mb"]:
                ram_action = "UP"
            elif target_ram < current_metrics["allocated_ram_mb"]:
                ram_action = "DOWN"

            logger.info(
                f"[{entity_type} {entity_id}] Scaling Required. "
                f"CPU: {cpu_action} to {target_cpus} (was {current_metrics['allocated_cpus']}), "
                f"RAM: {ram_action} to {target_ram} MB (was {current_metrics['allocated_ram_mb']} MB), "
                f"Swap: {target_swap} MB"
            )

            if entity_type == "LXC":
                self.px.update_lxc_resources(
                    entity_id, target_cpus, target_ram, swap_mb=target_swap
                )
                trigger = (
                    "host_pressure"
                    if host_metrics.get("ram_percent", 0) > HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
                    else "prediction"
                )
                try:
                    storage.log_scale_event(
                        entity_id=entity_id,
                        entity_type="LXC",
                        cpus_before=current_metrics["allocated_cpus"],
                        cpus_after=target_cpus,
                        ram_before_mb=current_metrics["allocated_ram_mb"],
                        ram_after_mb=target_ram,
                        trigger=trigger,
                        swap_before_mb=float(current_metrics.get("swap_mb", 0.0)),
                        swap_after_mb=float(target_swap),
                    )
                except Exception as log_err:
                    logger.debug(f"[LXC {entity_id}] Scale event log failed: {log_err}")
            elif entity_type == "VM":
                self.px.update_vm_resources(entity_id, target_cpus, target_ram)
        else:
            logger.debug(
                f"[{entity_type} {entity_id}] Resources adequate, no significant scaling required."
            )

        # 7. Swap flush — runs every cycle, independent of whether a scale event fired.
        #    A container can be swap-saturated while its RAM/CPU allocation is stable
        #    (no scale event triggers), leaving swap stuck indefinitely. This block
        #    fires whenever swap saturation exceeds the threshold AND the container has
        #    enough free RAM headroom to safely absorb the pages back.
        if entity_type == "LXC" and swap_used > 5:
            swap_alloc_check = current_metrics.get("allocated_swap_mb", 0.0)
            if (
                swap_alloc_check > 0
                and (swap_used / swap_alloc_check * 100.0) >= SWAP_FLUSH_THRESHOLD_PERCENT
            ):
                # Use target_ram (post-scale headroom) to check if flush is safe.
                # If a scale-up just fired, target_ram already accounts for the new size.
                ram_headroom = target_ram - current_usage_mb
                if ram_headroom >= swap_used * 1.1:
                    logger.info(
                        f"[LXC {entity_id}] Swap flush triggered: "
                        f"{swap_used:.0f}/{swap_alloc_check:.0f} MB used "
                        f"({swap_used/swap_alloc_check*100:.0f}% ≥ {SWAP_FLUSH_THRESHOLD_PERCENT:.0f}% threshold), "
                        f"RAM headroom {ram_headroom:.0f} MB ≥ {swap_used*1.1:.0f} MB required."
                    )
                    self.px.flush_lxc_swap(entity_id)
                else:
                    logger.debug(
                        f"[LXC {entity_id}] Swap flush deferred: headroom "
                        f"{ram_headroom:.0f} MB < {swap_used*1.1:.0f} MB required to safely absorb swap."
                    )

    def apply_vm_pending_config(
        self,
        vm_id: str,
        baseline: dict,
        predicted: dict,
        current_metrics: dict,
        rolling_peaks: dict,
    ):
        """
        Computes the optimal CPU / RAM sizing for a VM using a blended 14-day
        statistic from the telemetry log plus safety headroom, then writes that
        as a *pending* Proxmox config entry. The change takes effect on the next
        reboot — no live hotplug is ever attempted.

        CPU Sizing formula (spike-resistant)
        -------------------------------------
        Uses a weighted blend of three statistics from the 14-day window:
          cpu_basis = 0.50 × p95_cpu_pct + 0.30 × avg_cpu_pct + 0.20 × peak_cpu_pct

        Why blend, not peak?
          Using raw MAX meant a single spike from 13 days ago permanently over-sized
          the VM, potentially causing launch failures if the host has fewer physical
          CPUs available at boot time. The blend reflects sustained load while still
          providing headroom for real bursts.

        needed_cores = ceil(cpu_basis / 100 × base_cpus × cpu_buffer) + 1
        target_cpus  = clamp(needed_cores, min_cpus, min(max_cpus, physical_cpus))

        RAM Sizing formula
        ------------------
        peak_ram_mb  = MAX(14-day observed RAM peak, prediction recent_peak)
        target_ram   = clamp(int(peak_ram * headroom), max(min_ram_mb, 1024), max_ram_mb)

        Config is only written when recommendation differs from current allocation
        by > 5% RAM or >= 1 CPU core.
        """
        if not current_metrics:
            logger.warning(f"[VM {vm_id}] No current metrics. Skipping pending config.")
            return

        sample_count = rolling_peaks.get("sample_count", 0)
        alloc_cpus   = current_metrics["allocated_cpus"]
        alloc_ram_mb = current_metrics["allocated_ram_mb"]

        # Hard ceiling: never schedule more vCPUs than the host physically has.
        # This prevents a VM from failing to launch on the next boot because the
        # config demands cores the hypervisor cannot provide.
        host_metrics = self.px.get_host_usage()
        physical_cpus = max(int(host_metrics.get("physical_cpus", alloc_cpus)), 1)

        # Fetch actual configuration from API to avoid logging changes that are already pending
        if hasattr(self.px, "get_vm_config"):
            current_config = self.px.get_vm_config(vm_id) or {}
        else:
            current_config = {}
        config_ram_mb = current_config.get("ram_mb", alloc_ram_mb)

        if sample_count > 0:
            # Primary path: real observed statistics from the telemetry log.
            # Blend P95 (50%), average (30%), and peak (20%) to get a representative
            # CPU demand that is robust against single-cycle outliers.
            p95_cpu   = rolling_peaks.get("p95_cpu_pct", rolling_peaks["peak_cpu_pct"])
            avg_cpu   = rolling_peaks.get("avg_cpu_pct", rolling_peaks["peak_cpu_pct"])
            peak_cpu  = rolling_peaks["peak_cpu_pct"]
            cpu_basis = 0.50 * p95_cpu + 0.30 * avg_cpu + 0.20 * peak_cpu

            peak_ram_mb  = rolling_peaks["peak_ram_mb"]
            source_label = (
                f"{sample_count} telemetry samples "
                f"(P95={p95_cpu:.1f}%, avg={avg_cpu:.1f}%, peak={peak_cpu:.1f}%)"
            )
        else:
            # Bootstrap: no log data yet (day one). Use the ML prediction smoothed
            # values where available, falling back to raw predictions.
            smoothed = predicted.get("smoothed_cpu_percent", None)
            avg_pred  = predicted.get("recent_avg_cpu", None)
            raw_peak  = max(predicted["cpu_percent"], predicted.get("recent_peak_cpu", 0.0))

            if smoothed is not None and avg_pred is not None:
                cpu_basis = 0.50 * smoothed + 0.30 * avg_pred + 0.20 * raw_peak
            else:
                cpu_basis = raw_peak

            peak_ram_mb  = max(
                predicted["ram_usage_mb"],
                predicted.get("recent_peak_ram", 0.0),
            )
            source_label = "ML prediction (no log data yet)"

        logger.info(
            f"[VM {vm_id}] CPU basis for sizing: {cpu_basis:.1f}% "
            f"({source_label}). "
            f"Host physical CPUs: {physical_cpus}."
        )

        # Apply RAM headroom above peak
        headroom   = 1 + self.ram_buffer_percent / 100.0
        target_ram = int(peak_ram_mb * headroom)
        target_ram = max(target_ram, 1024)              # Proxmox VM floor
        target_ram = max(target_ram, baseline["min_ram_mb"])
        target_ram = min(target_ram, baseline["max_ram_mb"])

        # CPU: translate blended % demand into vCPU core count.
        # In Proxmox QEMU topology, current total vCPUs = sockets * cores.
        config_sockets = current_config.get("sockets", 1)
        config_cores   = current_config.get("cores", alloc_cpus)   # cores per socket
        # Use the config-defined vCPU count exclusively as the sizing base.
        # Do NOT mix in alloc_cpus (the live running vcpu count from the status API) —
        # alloc_cpus can lag behind a pending config change or reflect the hotplugged
        # vcpu count rather than the full cores×sockets allocation. Including it as a
        # max() caused needed_cores to escalate on every cycle after a reboot because
        # the newly active vCPU count became the new multiplier base.
        base_vcpus     = max(config_cores * config_sockets, 1)

        # cpu_basis is percentage demand for the VM (0-100% per VM allocation).
        # Clamp cpu_basis to 100.0% max so historical multi-core RRD metric artifacts
        # or spikes do not artificially multiply core requirements.
        effective_cpu_pct = min(cpu_basis, 100.0)

        # Calculate actual vCPUs needed with headroom buffer:
        # e.g., at 50% load on 1 vCPU, needed_cores = ceil(0.50 * 1 * 1.20) = 1 vCPU.
        needed_cores = max(
            1,
            math.ceil((effective_cpu_pct / 100.0) * base_vcpus * (1 + self.cpu_buffer_percent / 100.0))
        )

        # Step cap: prevent single-cycle overprovisioning jumps greater than +2 vCPUs.
        # Cap is relative to base_vcpus (the config-defined count), not alloc_cpus, so
        # it cannot escalate by more than 2 cores per reboot cycle even under sustained load.
        step_capped = min(needed_cores, base_vcpus + 2)
        if step_capped < needed_cores:
            logger.debug(
                f"[VM {vm_id}] Step cap applied: needed {needed_cores} cores "
                f"but limiting to base_vcpus ({base_vcpus}) + 2 = {step_capped}."
            )
        needed_cores = step_capped

        # Clamp: baseline bounds AND physical host limit so VM can always launch.
        target_cpus = max(
            baseline["min_cpus"],
            min(needed_cores, baseline["max_cpus"], physical_cpus),
        )
        if needed_cores > physical_cpus:
            logger.warning(
                f"[VM {vm_id}] CPU ceiling enforced: needed {needed_cores} cores "
                f"but host only has {physical_cpus} physical CPUs. "
                f"Clamping to {target_cpus}."
            )

        # Always normalise to 1 socket; all vCPUs are assigned as cores on socket 0.
        # This keeps the QEMU socket-id valid (range 0:0) and avoids NUMA topology
        # mismatches that occur when sockets > 1 is left in a pending config.
        target_sockets = 1

        # Only write when change is significant compared to existing CONFIG
        ram_delta_pct = abs(target_ram - config_ram_mb) / max(config_ram_mb, 1) * 100
        cpu_changed   = (target_cpus != config_cores) or (target_sockets != config_sockets)

        if ram_delta_pct < 5.0 and not cpu_changed:
            logger.debug(
                f"[VM {vm_id}] Pending config unchanged "
                f"(delta {ram_delta_pct:.1f}% RAM, CPU same). Skipping write."
            )
            return

        logger.info(
            f"[VM {vm_id}] PENDING CONFIG (applies on next reboot): "
            f"{target_cpus} CPUs/cores (was {config_cores}), "
            f"{target_sockets} sockets (was {config_sockets}), "
            f"{target_ram} MB RAM (was {config_ram_mb:.0f} MB). "
            f"Basis: {source_label} — "
            f"blended cpu_basis {cpu_basis:.1f}% + {self.cpu_buffer_percent:.0f}% buffer, "
            f"14-day peak RAM {peak_ram_mb:.0f} MB + {self.ram_buffer_percent:.0f}% headroom."
        )
        # Only pass sockets when it actually needs correcting.
        # Always writing sockets=1 even when the active config is already sockets=1
        # creates unnecessary pending churn. More importantly: if a previous cycle already
        # wrote sockets=1 as *pending* but the VM hasn't rebooted yet (active=2, pending=1),
        # re-writing it every cycle extends the window where the mismatch can cause issues.
        needs_sockets_fix = (config_sockets != target_sockets)
        self.px.update_vm_resources(
            vm_id, target_cpus, target_ram,
            sockets=target_sockets if needs_sockets_fix else None,
        )
        try:
            storage.log_scale_event(
                entity_id=vm_id,
                entity_type="VM",
                cpus_before=float(alloc_cpus),
                cpus_after=float(target_cpus),
                ram_before_mb=float(alloc_ram_mb),
                ram_after_mb=float(target_ram),
                trigger="vm_pending_config",
            )
        except Exception as log_err:
            logger.debug(f"[VM {vm_id}] Scale event log failed: {log_err}")
