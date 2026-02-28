import time
from predictor import Predictor

def test_predictor():
    predictor = Predictor(prediction_horizon=2)
    
    # Simulate a rising trend
    # format: [{'time': timestamp, 'cpu': ratio, 'mem': bytes}]
    base_time = time.time() - 300
    metrics = [
        {'time': base_time,       'cpu': 0.10, 'mem': 512 * 1024 * 1024},
        {'time': base_time + 60,  'cpu': 0.20, 'mem': 600 * 1024 * 1024},
        {'time': base_time + 120, 'cpu': 0.30, 'mem': 750 * 1024 * 1024},
        {'time': base_time + 180, 'cpu': 0.50, 'mem': 900 * 1024 * 1024},
        {'time': base_time + 240, 'cpu': 0.75, 'mem': 1024 * 1024 * 1024},
    ]
    
    predictions = predictor.predict_next_usage(metrics)
    print("Testing Rising Trend Prediction:")
    print(f"Metrics: CPU rising 10 -> 75, RAM rising 512 -> 1024")
    print(f"Predicted Output: {predictions}")
    
    # Assertions
    assert predictions['cpu_percent'] > 75.0, "CPU Prediction should extrapolate upwards"
    assert predictions['ram_usage_mb'] > 1024.0, "RAM Prediction should extrapolate upwards"

    # Simulate a falling trend
    metrics_falling = [
        {'time': base_time,       'cpu': 0.90, 'mem': 2048 * 1024 * 1024},
        {'time': base_time + 60,  'cpu': 0.80, 'mem': 1900 * 1024 * 1024},
        {'time': base_time + 120, 'cpu': 0.50, 'mem': 1500 * 1024 * 1024},
        {'time': base_time + 180, 'cpu': 0.30, 'mem': 1024 * 1024 * 1024},
        {'time': base_time + 240, 'cpu': 0.15, 'mem': 512 * 1024 * 1024},
    ]
    
    predictions_fallback = predictor.predict_next_usage(metrics_falling)
    print("\nTesting Falling Trend Prediction:")
    print(f"Metrics: CPU falling 90 -> 15, RAM falling 2048 -> 512")
    print(f"Predicted Output: {predictions_fallback}")
    
    assert predictions_fallback['cpu_percent'] < 15.0, "CPU Prediction should extrapolate downwards"
    assert predictions_fallback['ram_usage_mb'] < 512.0, "RAM Prediction should extrapolate downwards"
    
if __name__ == "__main__":
    print("Running Mock AI Predictor Tests...")
    test_predictor()
    print("All mock tests passed!")
