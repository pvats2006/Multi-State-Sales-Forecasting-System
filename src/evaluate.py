import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import custom modules
from features import create_features, train_val_split
from train_arima import train_arima, predict_arima
from train_prophet import train_prophet, predict_prophet
from train_xgboost import train_xgboost, predict_xgboost

def evaluate_model(actual, predicted):
    """
    Computes RMSE, MAE, and MAPE.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero for MAPE
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    return {
        'rmse': round(float(rmse), 4),
        'mae': round(float(mae), 4),
        'mape': round(float(mape), 4)
    }

def select_best_model(state_name, state_df):
    """
    Trains all models, evaluates on validation set, and records the winner.
    """
    print(f"\n--- Selecting Best Model for {state_name} ---")
    
    # 1. Prepare data splits
    # For ARIMA/LSTM (Series-based)
    series = state_df.set_index('date')['sales']
    train_series = series.iloc[:-8]
    val_series = series.iloc[-8:]
    
    # For Prophet/XGBoost (DF-based)
    df_featured = create_features(state_df, state_name=state_name)
    train_df, val_df = train_val_split(df_featured, val_weeks=8)
    
    # history for recursive XGBoost prediction
    history_end_idx = state_df[state_df['date'] < val_df['date'].min()].index[-1]
    history_df = state_df.loc[:history_end_idx]

    # history for LSTM
    SL = 12
    last_seq_lstm = train_series.iloc[-SL:].values
    
    results = {}
    all_preds = {"arima": None, "prophet": None, "xgboost": None, "lstm": None}
    
    # --- 1. ARIMA ---
    try:
        m_arima = train_arima(train_series, state_name)
        p_arima = predict_arima(m_arima, n_periods=8)
        all_preds["arima"] = p_arima
        results['arima'] = evaluate_model(val_series.values, p_arima)
    except Exception as e:
        print(f"ARIMA failed for {state_name}: {e}")

    # --- 2. Prophet ---
    try:
        # Prophet wants columns [date, sales]
        m_prophet = train_prophet(state_df.iloc[:-8], state_name)
        p_prophet = predict_prophet(m_prophet, n_periods=8)
        all_preds["prophet"] = p_prophet
        results['prophet'] = evaluate_model(val_series.values, p_prophet)
    except Exception as e:
        print(f"Prophet failed for {state_name}: {e}")

    # --- 3. XGBoost ---
    try:
        m_xgb = train_xgboost(train_df, state_name)
        p_xgb = predict_xgboost(m_xgb, history_df, n_periods=8)
        all_preds["xgboost"] = p_xgb
        results['xgboost'] = evaluate_model(val_series.values, p_xgb)
    except Exception as e:
        print(f"XGBoost failed for {state_name}: {e}")

    # --- 4. LSTM ---
    import time
    lstm_preds = None
    try:
        from train_lstm import train_lstm, predict_lstm, LSTM_AVAILABLE
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow not installed.")
        t0 = time.time()
        lstm_model, lstm_scaler = train_lstm(train_series, state_name)
        if lstm_model is not None and time.time() - t0 < 120:
            lstm_preds = predict_lstm(lstm_model, lstm_scaler,
                                      train_series, n_periods=8)
            all_preds["lstm"] = lstm_preds
            results['lstm'] = evaluate_model(val_series.values, lstm_preds)
            print(f"  [LSTM] Predictions: {[round(p,2) for p in lstm_preds]}")
        else:
            print(f"  [LSTM] Skipped for {state_name} (timeout or no model).")
    except Exception as e:
        print(f"  [LSTM] Skipped for {state_name}: {e}")
    
    if not results:
        return None, None

    # --- Run Diagnostics ---
    run_full_diagnostics(state_name, train_series, val_series, all_preds)
    for model_name, preds in all_preds.items():
        if preds:
            print_forecast_vs_actual(state_name, model_name, val_series, preds)

    # Determine winner based on RMSE
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_metrics = results[best_model_name]
    
    # Update model_registry.json
    registry_path = Path('model_registry.json')
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
        
    registry[state_name] = {
        "best_model": best_model_name,
        "rmse": best_metrics['rmse'],
        "mae": best_metrics['mae'],
        "mape": best_metrics['mape'],
        "all_models": results
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
        
    return best_model_name, best_metrics

def run_all_states():
    """
    Loops through all processed state files and selects the best model for each.
    """
    processed_dir = Path('data/processed/')
    if not processed_dir.exists():
        print("No processed data found. Run preprocess.py first.")
        return
        
    summary = []
    for csv_file in processed_dir.glob('*.csv'):
        state_name = csv_file.stem.replace('_', ' ').title()
        state_df = pd.read_csv(csv_file)
        state_df['date'] = pd.to_datetime(state_df['date'])
        
        best_name, metrics = select_best_model(state_name, state_df)
        if best_name:
            summary.append({
                'State': state_name,
                'Best Model': best_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE %': metrics['mape']
            })
            
    # Print summary table
    if summary:
        print("\n" + "="*50)
        print(f"{'State':<15} | {'Best Model':<10} | {'RMSE':<10} | {'MAPE %':<8}")
        print("-" * 50)
        for row in summary:
            print(f"{row['State']:<15} | {row['Best Model']:<10} | {row['RMSE']:<10.2f} | {row['MAPE %']:<8.2f}")
        print("="*50)

    # -- Update model_registry.json metadata --
    from datetime import datetime
    registry_path = Path('model_registry.json')
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        registry['_metadata'] = {
            "last_run": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "total_states": len(summary),
            "pipeline_version": "1.1",
            "models_available": ["arima", "prophet", "xgboost", "lstm"]
        }
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        print("[Registry] Metadata updated.")

def print_forecast_vs_actual(state, model_name, val_series, predictions):
    """
    Prints a side-by-side comparison table of actual vs predicted values.
    """
    import os

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print(f"  Forecast vs Actual — {state} | Model: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Week':<6} {'Actual':>10} {'Predicted':>12} {'Error%':>10}")
    print(f"  {'-'*40}")

    actual = list(val_series)
    pred   = list(predictions)

    for i, (a, p) in enumerate(zip(actual, pred)):
        err = abs(a - p) / a * 100 if a != 0 else 0
        print(f"  {i+1:<6} {a:>10.2f} {p:>12.2f} {err:>9.1f}%")

    rmse = (sum((a - p)**2 for a, p in zip(actual, pred)) / len(actual)) ** 0.5
    mape = sum(abs(a - p) / a * 100 for a, p in zip(actual, pred)
               if a != 0) / len(actual)
    print(f"\n  RMSE: {rmse:.2f}  |  MAPE: {mape:.2f}%")
    print(f"{'='*60}\n")


def run_full_diagnostics(state, train_series, val_series, all_predictions):
    """
    all_predictions: dict like
      {"arima": [...], "prophet": [...], "xgboost": [...], "lstm": [...]}

    Saves a multi-model comparison chart to data/diagnostics/{state}_model_comparison.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs('data/diagnostics', exist_ok=True)

    # --- Data summary ---
    print(f"\n{'='*60}")
    print(f"  Data Summary — {state}")
    print(f"{'='*60}")
    print(f"  Training rows  : {len(train_series)}")
    print(f"  Validation rows: {len(val_series)}")
    print(f"  Train mean     : {train_series.mean():.2f}")
    print(f"  Train std      : {train_series.std():.2f}")
    print(f"  Train min      : {train_series.min():.2f}")
    print(f"  Train max      : {train_series.max():.2f}")
    print(f"{'='*60}\n")

    # --- Matplotlib chart ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot last 12 weeks of training as historical context
    history = train_series.iloc[-12:]
    ax.plot(range(len(history)), history.values,
            color='black', linewidth=2, label='Historical (last 12w)')

    # X axis for validation period
    val_x = range(len(history) - 1, len(history) + len(val_series))
    actual_plot = [history.values[-1]] + list(val_series.values)
    ax.plot(val_x, actual_plot,
            color='black', linewidth=2, linestyle='--', label='Actual')

    # Plot each model
    colors = {'arima': '#E24B4A', 'prophet': '#378ADD',
              'xgboost': '#1D9E75', 'lstm': '#BA7517'}

    for model_name, preds in all_predictions.items():
        if preds is None:
            continue
        rmse = (sum((a - p)**2 for a, p in
                    zip(val_series.values, preds)) / len(preds)) ** 0.5
        mape = sum(abs(a - p) / a * 100 for a, p in
                   zip(val_series.values, preds)
                   if a != 0) / len(preds)
        pred_plot = [history.values[-1]] + list(preds)
        ax.plot(val_x, pred_plot,
                color=colors.get(model_name, 'gray'),
                linewidth=1.5, linestyle='-.',
                label=f"{model_name.upper()} "
                      f"(RMSE={rmse:.0f}, MAPE={mape:.1f}%)")

    ax.set_title(f"{state} — Validation: Actual vs All Models",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"data/diagnostics/{state}_model_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Diagnostic chart saved] -> {save_path}\n")

if __name__ == "__main__":
    run_all_states()
