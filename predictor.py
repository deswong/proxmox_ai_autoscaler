import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from typing import List, Tuple

logger = logging.getLogger("predictor")

class Predictor:
    def __init__(self, prediction_horizon=3):
        """
        prediction_horizon: Number of future intervals (minutes) to predict.
        We predict 3 minutes into the future by default to proactively scale before spikes.
        """
        self.prediction_horizon = prediction_horizon
        
    def predict_next_usage(self, rrd_data: List[dict]) -> dict:
        """
        Takes a chronological list of dicts from Proxmox RRD API:
        [{'time': UNIX_EPOCH, 'cpu': 0.05, 'maxcpu': 2, 'mem': bytes, 'maxmem': bytes}]
        Returns predicted CPU percent and RAM usage (MB) using Linear Regression forecasting.
        """
        # Filter out invalid or purely null/0 data points that might appear in RRD
        valid_metrics = [m for m in rrd_data if m.get('cpu') is not None and m.get('mem') is not None]
        
        if not valid_metrics:
            return {"cpu_percent": 0.0, "ram_usage_mb": 0.0}
            
        # We want the most recent 15 valid data points for a smooth, fast trend
        metrics = valid_metrics[-15:]
        
        # Explicit memory optimization: We no longer need the heavy original RRD json array
        # or the large filtered array. Free them before we spin up Scikit-Learn matrices.
        del rrd_data
        del valid_metrics
            
        if len(metrics) < 3:
            # Not enough data for a reliable trend, return the most recent reading
            latest = metrics[-1]
            return {
                "cpu_percent": (latest.get('cpu', 0.0) * 100),
                "ram_usage_mb": (latest.get('mem', 0.0) / (1024 * 1024))
            }
            
        # Prepare data for Scikit-Learn
        # We'll normalize X (time) to relative minute intervals 0, 1, 2, ...
        # We just use index position as time interval to keep regression simple
        X = np.arange(len(metrics)).reshape(-1, 1)
        
        # Proxmox CPU graph data is a ratio 0.0 -> 1.0 per core usually, 
        # or overall ratio. Multiply by 100 for percent.
        y_cpu = np.array([(m.get('cpu', 0.0) * 100) for m in metrics])
        y_ram = np.array([(m.get('mem', 0.0) / (1024 * 1024)) for m in metrics])
        
        # Explicit memory optimization: Free the intermediate array list
        del metrics
        
        # Fit Linear Regression for CPU
        model_cpu = LinearRegression()
        model_cpu.fit(X, y_cpu)
        
        # Fit Linear Regression for RAM
        model_ram = LinearRegression()
        model_ram.fit(X, y_ram)
        
        # Predict at X = current_index + prediction_horizon
        future_X = np.array([[len(metrics) - 1 + self.prediction_horizon]])
        
        pred_cpu = model_cpu.predict(future_X)[0]
        pred_ram = model_ram.predict(future_X)[0]
        
        # Ensure predictions don't drop below 0
        pred_cpu = max(0.0, float(pred_cpu))
        pred_ram = max(0.0, float(pred_ram))
        
        # Also check against recent actual peaks to ensure we don't scale down immediately 
        # during transient drops when the trend isn't steeply downwards yet.
        highest_recent_cpu = max(y_cpu)
        highest_recent_ram = max(y_ram)
        
        # If the prediction is lower than highest recent limit, but the trend isn't severely downward, 
        # we might want to blend the prediction with peak memory.
        # But for autoscaler safety, returning the pure regression prediction is fine as long as
        # the scaler adds an overhead buffer (e.g. +20%).
        
        return {
            "cpu_percent": pred_cpu,
            "ram_usage_mb": pred_ram,
            "recent_peak_cpu": float(highest_recent_cpu),
            "recent_peak_ram": float(highest_recent_ram)
        }
