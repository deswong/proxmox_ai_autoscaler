import datetime
import numpy as np
import xgboost as xgb
import logging
import os
from typing import List, Optional

logger = logging.getLogger("predictor")


class Predictor:
    def __init__(self, prediction_horizon=2, models_dir="./models"):
        """
        prediction_horizon: Number of future intervals (minutes) to predict.
        models_dir: Location where the nightly training cron task will save the XGBoost .json weights.
        """
        self.prediction_horizon = prediction_horizon
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self._model_cache = {}  # Store loaded models in RAM
        self._model_mtimes = {}  # File modification times to detect fresh nightly trains

    def _get_model(self, model_path: str):
        """
        Retrieves a cached XGBoost Booster from RAM, or loads it from disk if new/updated.
        """
        if not os.path.exists(model_path):
            if model_path in self._model_cache:
                del self._model_cache[model_path]
                del self._model_mtimes[model_path]
            return None

        mtime = os.path.getmtime(model_path)
        if (
            model_path in self._model_cache
            and self._model_mtimes.get(model_path) == mtime
        ):
            return self._model_cache[model_path]

        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
            self._model_cache[model_path] = booster
            self._model_mtimes[model_path] = mtime
            return booster
        except Exception as e:
            logger.error(f"Failed to load XGBoost model from {model_path}: {e}")
            return None

    @staticmethod
    def _build_context_features(metrics: list, host_context: dict) -> list:
        """
        Constructs the 17-element global context vector appended after the
        90 per-interval history features. Same layout used by training.

        Slots [0-9]  — node health (10 values):
          host_cpu%, host_ram%, host_swap%,
          load_avg_1m, load_avg_5m, ksm_sharing_mb,
          cpu_overcommit_ratio, ram_overcommit_ratio, container_count, (reserved=0)

        Slots [10-11] — temporal context (2 values):
          hour_of_day (0-23), day_of_week (0=Mon...6=Sun)

        Slots [12-17] — rate-of-change deltas (6 values):
          delta_cpu%, delta_mem_mb, delta_diskread, delta_diskwrite,
          delta_netin, delta_netout
          (last interval − first interval in the 15-point window)
        """
        now = datetime.datetime.now()  # pylint: disable=disallowed-name

        # Rate-of-change deltas across the window
        first, last = metrics[0], metrics[-1]
        delta_cpu = (last.get("cpu", 0.0) - first.get("cpu", 0.0)) * 100
        delta_mem = (last.get("mem", 0.0) - first.get("mem", 0.0)) / (1024 * 1024)
        delta_dr = last.get("diskread", 0.0) - first.get("diskread", 0.0)
        delta_dw = last.get("diskwrite", 0.0) - first.get("diskwrite", 0.0)
        delta_ni = last.get("netin", 0.0) - first.get("netin", 0.0)
        delta_no = last.get("netout", 0.0) - first.get("netout", 0.0)

        return [
            # Node health (10)
            float(host_context.get("cpu_percent", 0.0)),
            float(host_context.get("ram_percent", 0.0)),
            float(host_context.get("swap_percent", 0.0)),
            float(host_context.get("load_avg_1m", 0.0)),
            float(host_context.get("load_avg_5m", 0.0)),
            float(host_context.get("ksm_sharing_mb", 0.0)),
            float(host_context.get("cpu_overcommit_ratio", 0.0)),
            float(host_context.get("ram_overcommit_ratio", 0.0)),
            float(host_context.get("container_count", 0.0)),
            0.0,  # reserved for future use
            # Temporal (2)
            float(now.hour),
            float(now.weekday()),
            # Rate-of-change deltas (6)
            delta_cpu,
            delta_mem,
            delta_dr,
            delta_dw,
            delta_ni,
            delta_no,
        ]

    def predict_next_usage(
        self,
        entity_id: str,
        rrd_data: List[dict],
        entity_type: str = "LXC",
        host_context: Optional[dict] = None,
    ) -> dict:
        """
        Takes chronological data from Proxmox RRD API.
        Only performs fast inference using pre-trained XGBoost weights.
        If no weights exist yet (first day), falls back to the latest telemetry reading.

        Feature vector layout (107 total):
          [0–89]   Per-container history:  6 × 15 intervals
                   cpu%, mem_mb, diskread, diskwrite, netin, netout
          [90–99]  Node health context:    10 scalars
                   host_cpu%, host_ram%, host_swap%, load_1m, load_5m,
                   ksm_mb, cpu_overcommit, ram_overcommit, container_count, reserved
          [100–101] Temporal:             2 scalars (hour_of_day, day_of_week)
          [101–106] Deltas:               6 scalars (last − first of window)

        host_context keys (all optional, default 0.0):
          cpu_percent, ram_percent, swap_percent, load_avg_1m, load_avg_5m,
          ksm_sharing_mb, cpu_overcommit_ratio, ram_overcommit_ratio, container_count
        """
        if host_context is None:
            host_context = {}

        valid_metrics = [
            m for m in rrd_data if m.get("cpu") is not None and m.get("mem") is not None
        ]

        if not valid_metrics:
            logger.warning(
                f"No valid telemetry data received for {entity_type} {entity_id}. "
                "Aborting prediction to prevent dangerous scale-down."
            )
            return None

        metrics = valid_metrics[-15:]

        # Capture peaks
        highest_recent_cpu = float(max(m.get("cpu", 0.0) * 100 for m in metrics))
        highest_recent_ram = float(max(m.get("mem", 0.0) / (1024 * 1024) for m in metrics))
        highest_recent_swap = float(max(m.get("swap", 0.0) / (1024 * 1024) for m in metrics))
        highest_recent_disk_read = float(max(m.get("diskread", 0.0) for m in metrics))
        highest_recent_disk_write = float(max(m.get("diskwrite", 0.0) for m in metrics))
        highest_recent_net_in = float(max(m.get("netin", 0.0) for m in metrics))
        highest_recent_net_out = float(max(m.get("netout", 0.0) for m in metrics))

        del rrd_data
        del valid_metrics

        latest = metrics[-1]
        fallback_cpu = latest.get("cpu", 0.0) * 100
        fallback_ram = latest.get("mem", 0.0) / (1024 * 1024)

        if len(metrics) < 15:
            fallback_swap = latest.get("swap", 0.0) / (1024 * 1024)
            del metrics
            return {
                "cpu_percent": fallback_cpu,
                "ram_usage_mb": fallback_ram,
                "recent_peak_cpu": highest_recent_cpu,
                "recent_peak_ram": highest_recent_ram,
                "predicted_swap_mb": fallback_swap,
                "recent_peak_swap": highest_recent_swap,
                "recent_peak_disk_read": highest_recent_disk_read,
                "recent_peak_disk_write": highest_recent_disk_write,
                "recent_peak_net_in": highest_recent_net_in,
                "recent_peak_net_out": highest_recent_net_out,
            }

        prefix = entity_type.lower()
        cpu_model_path = os.path.join(self.models_dir, f"{prefix}_{entity_id}_cpu.json")
        ram_model_path = os.path.join(self.models_dir, f"{prefix}_{entity_id}_ram.json")

        pred_cpu = fallback_cpu
        pred_ram = fallback_ram
        pred_swap = latest.get("swap", 0.0) / (1024 * 1024)

        if os.path.exists(cpu_model_path) and os.path.exists(ram_model_path):
            try:
                # 90 per-interval features (6 × 15)
                X_features = []
                for m in metrics:
                    X_features.append(m.get("cpu", 0.0) * 100)
                    X_features.append(m.get("mem", 0.0) / (1024 * 1024))
                    X_features.append(m.get("diskread", 0.0))
                    X_features.append(m.get("diskwrite", 0.0))
                    X_features.append(m.get("netin", 0.0))
                    X_features.append(m.get("netout", 0.0))

                # 17 global context features (node health + temporal + deltas)
                X_features.extend(self._build_context_features(metrics, host_context))

                model_cpu = self._get_model(cpu_model_path)
                model_ram = self._get_model(ram_model_path)

                if model_cpu and model_ram:
                    dmatrix = xgb.DMatrix(np.array([X_features]))
                    pred_cpu = max(0.0, float(model_cpu.predict(dmatrix)[0]))
                    pred_ram = max(0.0, float(model_ram.predict(dmatrix)[0]))

                    swap_model_path = os.path.join(
                        self.models_dir, f"{prefix}_{entity_id}_swap.json"
                    )
                    model_swap = self._get_model(swap_model_path)
                    if model_swap:
                        pred_swap = max(0.0, float(model_swap.predict(dmatrix)[0]))

            except Exception as e:
                logger.error(
                    f"Failed to run XGBoost inference for {entity_type} {entity_id}: {e}"
                )
        else:
            logger.debug(
                f"No XGBoost models found yet for {entity_type} {entity_id}. Falling back to live metrics."
            )

        del metrics

        return {
            "cpu_percent": pred_cpu,
            "ram_usage_mb": pred_ram,
            "recent_peak_cpu": highest_recent_cpu,
            "recent_peak_ram": highest_recent_ram,
            "predicted_swap_mb": pred_swap,
            "recent_peak_swap": highest_recent_swap,
            "recent_peak_disk_read": highest_recent_disk_read,
            "recent_peak_disk_write": highest_recent_disk_write,
            "recent_peak_net_in": highest_recent_net_in,
            "recent_peak_net_out": highest_recent_net_out,
        }
