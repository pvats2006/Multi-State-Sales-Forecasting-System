import pandas as pd
import json
import joblib
import os
import numpy as np
from pathlib import Path

# Import prediction functions
from train_arima import predict_arima
from train_prophet import predict_prophet
from train_xgboost import predict_xgboost
try:
    from train_lstm import predict_lstm
except ImportError:
    pass

# ── ARIMA confidence intervals ──────────────────────────────
def get_arima_intervals(model, n_periods=8, alpha=0.10):
    """
    Returns (predictions, lower_bounds, upper_bounds).
    alpha=0.10 -> 90% prediction interval.
    """
    try:
        preds, conf_int = model.predict(
            n_periods=n_periods, return_conf_int=True, alpha=alpha)
        lower = conf_int[:, 0].tolist()
        upper = conf_int[:, 1].tolist()
    except Exception:
        # Fallback if prediction intervals fail
        preds = model.predict(n_periods=n_periods)
        margin = [abs(p) * 0.15 for p in preds]
        lower  = [p - m for p, m in zip(preds, margin)]
        upper  = [p + m for p, m in zip(preds, margin)]
    return list(preds), lower, upper


# ── Prophet confidence intervals ────────────────────────────
def get_prophet_intervals(model, n_periods=8):
    """
    Prophet natively returns yhat_lower and yhat_upper.
    """
    future   = model.make_future_dataframe(periods=n_periods, freq='W')
    forecast = model.predict(future)
    last_n   = forecast.tail(n_periods)
    preds    = last_n['yhat'].tolist()
    lower    = last_n['yhat_lower'].tolist()
    upper    = last_n['yhat_upper'].tolist()
    return preds, lower, upper


# ── XGBoost bootstrap confidence intervals ──────────────────
def get_xgboost_intervals(train_df, state_name, n_periods=8,
                           n_bootstrap=10):
    """
    Trains n_bootstrap XGBoost models with different random seeds.
    Confidence band = 10th / 90th percentile across bootstrap predictions.
    """
    from train_xgboost import train_xgboost, predict_xgboost

    all_preds = []
    for seed in range(n_bootstrap):
        try:
            m = train_xgboost(
                train_df,
                state_name=f"{state_name}_boot{seed}",
                random_state=seed,
                silent=True
            )
            p = predict_xgboost(m, train_df, n_periods=n_periods)
            all_preds.append(p)
        except Exception as e:
            print(f"Bootstrap iteration {seed} failed: {e}")
            continue

    if not all_preds:
        return None, None, None

    arr   = np.array(all_preds)           # shape: (n_bootstrap, n_periods)
    preds = arr.mean(axis=0).tolist()
    lower = np.percentile(arr, 10, axis=0).tolist()
    upper = np.percentile(arr, 90, axis=0).tolist()
    return preds, lower, upper


# ── LSTM Monte Carlo Dropout intervals ──────────────────────
def get_lstm_intervals(model, scaler, train_series,
                        n_periods=8, mc_runs=20):
    """
    Runs LSTM prediction mc_runs times with dropout active (MC Dropout).
    Confidence band = mean +/- 1.5 * std across runs.
    """
    import tensorflow as tf

    all_preds = []
    for _ in range(mc_runs):
        try:
            seq_len = model.input_shape[1]
            scaled  = scaler.transform(
                train_series.values.reshape(-1, 1)).flatten()
            seq     = list(scaled[-seq_len:])
            run_preds = []
            for _ in range(n_periods):
                x    = np.array(seq[-seq_len:]).reshape(1, seq_len, 1)
                # training=True keeps Dropout active for MC Dropout
                pred = model(x, training=True).numpy()[0][0]
                run_preds.append(pred)
                seq.append(pred)
            inv = scaler.inverse_transform(
                np.array(run_preds).reshape(-1, 1)).flatten()
            all_preds.append(inv.tolist())
        except Exception as e:
            print(f"MC Dropout run failed: {e}")
            continue

    if not all_preds:
        return None, None, None

    arr   = np.array(all_preds)
    mean  = arr.mean(axis=0)
    std   = arr.std(axis=0)
    preds = mean.tolist()
    lower = (mean - 1.5 * std).tolist()
    upper = (mean + 1.5 * std).tolist()
    return preds, lower, upper


def generate_forecast(state_name, n_weeks=8):
    """
    Loads the best model for a state, generates n_weeks predictions
    with upper/lower confidence bounds.
    """
    print(f"Generating forecast for {state_name}...")
    
    # 1. Load Registry
    registry_path = Path('model_registry.json')
    if not registry_path.exists():
        raise FileNotFoundError("model_registry.json not found.")
        
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        
    if state_name not in registry:
        raise ValueError(f"State '{state_name}' not in model registry.")

    best_model_type = registry[state_name]['best_model']
    
    # 2. Load Processed Data
    csv_name = state_name.lower().replace(' ', '_') + ".csv"
    data_path = Path('data/processed') / csv_name
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    series = df['sales']

    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks, freq='W')

    preds = lower = upper = None

    if best_model_type == 'arima':
        model = joblib.load(f"models/{state_name}_arima.pkl")
        preds, lower, upper = get_arima_intervals(model, n_weeks)

    elif best_model_type == 'prophet':
        model = joblib.load(f"models/{state_name}_prophet.pkl")
        preds, lower, upper = get_prophet_intervals(model, n_weeks)

    elif best_model_type == 'xgboost':
        from features import create_features
        feat_df = create_features(df, state_name=state_name)
        # We use bootstrap for intervals
        preds, lower, upper = get_xgboost_intervals(feat_df, state_name, n_weeks)

    elif best_model_type == 'lstm':
        import tensorflow as tf
        model_path = Path('models') / f"{state_name}_lstm.keras"
        model = tf.keras.models.load_model(model_path)
        scaler_path = Path('models') / f"{state_name}_lstm_scaler.pkl"
        scaler = joblib.load(scaler_path)
        preds, lower, upper = get_lstm_intervals(model, scaler, series, n_weeks)

    # Fallback: flat +/-15% if intervals failed
    if preds is not None and lower is None:
        lower = [p * 0.85 for p in preds]
        upper = [p * 1.15 for p in preds]

    if preds is None:
        raise RuntimeError(f"Failed to generate forecast for {state_name}")

    results = []
    for date, p, lo, hi in zip(future_dates, preds, lower, upper):
        results.append({
            "date": date.strftime('%Y-%m-%d'),
            "predicted_sales": round(float(p),  2),
            "lower_bound":     round(float(lo), 2),
            "upper_bound":     round(float(hi), 2)
        })

    return results

def generate_all_forecasts(n_weeks=8):
    """
    Generates forecasts for all states and saves to data/forecasts.json.
    """
    registry_path = Path('model_registry.json')
    if not registry_path.exists():
        return
        
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        
    all_forecasts = {}
    for state_name in registry.keys():
        if state_name.startswith('_'):
            continue
        try:
            forecast = generate_forecast(state_name, n_weeks=n_weeks)
            all_forecasts[state_name] = forecast
        except Exception as e:
            print(f"Failed to generate forecast for {state_name}: {e}")
            
    # Save results
    output_path = Path('data/forecasts.json')
    with open(output_path, 'w') as f:
        json.dump(all_forecasts, f, indent=4)
        
    print(f"\nAll forecasts saved to {output_path}")
    return all_forecasts

if __name__ == "__main__":
    generate_all_forecasts(n_weeks=8)
