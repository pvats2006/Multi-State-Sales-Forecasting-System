"""
monitor.py
Monitors forecast accuracy against real observed sales.
Triggers automatic retraining if MAPE exceeds a threshold.
"""

import json
import os
import pandas as pd
from datetime import datetime


REGISTRY_PATH  = 'model_registry.json'
FORECASTS_PATH = 'data/forecasts.json'
MONITOR_LOG    = 'data/monitor_log.json'


def compute_mape(actuals, predictions):
    """Returns MAPE as a percentage."""
    if len(actuals) != len(predictions):
        min_len  = min(len(actuals), len(predictions))
        actuals  = actuals[:min_len]
        predictions = predictions[:min_len]
    errors = [
        abs(a - p) / a * 100
        for a, p in zip(actuals, predictions) if a != 0
    ]
    return round(sum(errors) / len(errors), 2) if errors else 0.0


def compute_rmse(actuals, predictions):
    """Returns RMSE."""
    min_len  = min(len(actuals), len(predictions))
    actuals  = actuals[:min_len]
    predictions = predictions[:min_len]
    mse = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    return round((mse / len(actuals)) ** 0.5, 2)


def load_stored_forecast(state):
    """
    Loads the stored predicted_sales values for a state
    from data/forecasts.json.
    """
    if not os.path.exists(FORECASTS_PATH):
        raise FileNotFoundError(
            f"{FORECASTS_PATH} not found. "
            "Run src/forecast.py first.")

    with open(FORECASTS_PATH, 'r') as f:
        forecasts = json.load(f)
        
    if state not in forecasts:
        raise ValueError(
            f"State '{state}' not found in {FORECASTS_PATH}.")

    return [row['predicted_sales'] for row in forecasts[state]]


def trigger_retrain(state):
    """
    Retrains all models for one state and updates the registry
    and forecasts.json.
    """
    print(f"\n  [Monitor] Triggering retrain for {state}...")

    # Lazy imports to avoid circular dependency
    # Note: Using direct imports as they are in the same directory (src)
    try:
        from evaluate import select_best_model
        from forecast import generate_forecast
    except ImportError:
        from src.evaluate import select_best_model
        from src.forecast import generate_forecast

    csv_name = state.lower().replace(' ', '_') + ".csv"
    csv_path = f"data/processed/{csv_name}"
    # Try alternate if exact name was used (some versions used Title Case in filename)
    if not os.path.exists(csv_path):
        csv_path = f"data/processed/{state}.csv"

    if not os.path.exists(csv_path):
        print(f"  [Monitor] ERROR: {csv_path} not found. "
              "Cannot retrain.")
        return False

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Call select_best_model with correct signature (state_name, state_df)
    select_best_model(state, df)

    # Re-generate production forecasts
    new_forecast = generate_forecast(state, n_weeks=8)

    forecasts = {}
    if os.path.exists(FORECASTS_PATH):
        with open(FORECASTS_PATH, 'r') as f:
            forecasts = json.load(f)
            
    forecasts[state] = new_forecast
    with open(FORECASTS_PATH, 'w') as f:
        json.dump(forecasts, f, indent=2)

    print(f"  [Monitor] Retrain complete for {state}. "
          "Forecast updated.")
    return True


def check_accuracy_drift(state, actual_sales_list,
                          threshold_mape=25.0):
    """
    Compares stored forecast against observed actuals.

    Returns:
        dict with keys: state, mape, rmse, retrain_triggered, timestamp
    """
    print(f"\n[Monitor] Checking accuracy for {state}...")

    try:
        predictions = load_stored_forecast(state)
    except Exception as e:
        print(f"  [Monitor] Error loading forecast: {e}")
        return None

    mape = compute_mape(actual_sales_list, predictions)
    rmse = compute_rmse(actual_sales_list, predictions)

    retrain_triggered = False

    if mape > threshold_mape:
        print(f"  ALERT: {state} accuracy degraded. "
              f"MAPE={mape}% (threshold={threshold_mape}%).")
        print(f"  RMSE={rmse}. Triggering automatic retrain...")
        retrain_triggered = trigger_retrain(state)
    else:
        print(f"  [OK] {state} accuracy OK. "
              f"MAPE={mape}% | RMSE={rmse}")

    result = {
        "state":              state,
        "mape":               mape,
        "rmse":               rmse,
        "threshold":          threshold_mape,
        "retrain_triggered":  retrain_triggered,
        "timestamp":          datetime.now().strftime(
                                  '%Y-%m-%dT%H:%M:%S')
    }

    # Append to monitor log
    log = []
    if os.path.exists(MONITOR_LOG):
        try:
            with open(MONITOR_LOG, 'r') as f:
                log = json.load(f)
        except:
            log = []
            
    log.append(result)
    os.makedirs(os.path.dirname(MONITOR_LOG), exist_ok=True)
    with open(MONITOR_LOG, 'w') as f:
        json.dump(log, f, indent=2)

    return result


def run_monitoring(actuals_dict, threshold_mape=25.0):
    """
    Runs accuracy checks for all states in actuals_dict.
    """
    print("\n" + "="*60)
    print("  ACCURACY MONITORING REPORT")
    print("="*60)

    results = []
    for state, actuals in actuals_dict.items():
        try:
            result = check_accuracy_drift(
                state, actuals, threshold_mape)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [Monitor] ERROR for {state}: {e}")

    # -- Summary table --
    print(f"\n{'='*60}")
    print(f"  {'State':<20} {'MAPE%':>8} {'RMSE':>10} "
          f"{'Retrained':>12}")
    print(f"  {'-'*50}")
    for r in results:
        flag = "YES !!!" if r['retrain_triggered'] else "No"
        print(f"  {r['state']:<20} {r['mape']:>7.1f}% "
              f"{r['rmse']:>10.1f} {flag:>12}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    # Example usage - replace with real observed values
    sample_actuals = {
        "California": [3500, 3400, 3600, 3450,
                       3700, 3550, 3480, 3620],
        "New York":   [2700, 2800, 2650, 2900,
                       2750, 2820, 2700, 2780],
        "Texas":      [2600, 2700, 2500, 2800,
                       2650, 2720, 2600, 2750],
    }
    run_monitoring(sample_actuals, threshold_mape=25.0)
