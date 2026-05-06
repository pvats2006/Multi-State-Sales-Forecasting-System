import pandas as pd
import pmdarima as pm
import joblib
import os
from pathlib import Path

def train_arima(train_series, state_name):
    """
    Trains an ARIMA model using auto_arima.
    If auto_arima fails, falls back through simpler non-seasonal ARIMA models.
    """
    print(f"Training ARIMA for {state_name} (Samples: {len(train_series)})...")
    
    try:
        # 1. Primary: auto_arima with quarterly seasonality (m=4)
        model = pm.auto_arima(
            train_series,
            seasonal=True,
            m=4,  # Use quarterly instead of m=52 to accommodate shorter datasets
            max_p=3, max_q=3, max_d=2,
            max_P=2, max_Q=2, max_D=1,
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic',
            trace=False
        )
    except Exception as e:
        print(f"Auto-ARIMA failed for {state_name}: {e}. Trying fallback 1 (2,1,2)...")
        try:
            # 2. First Fallback: non-seasonal ARIMA(2,1,2)
            model = pm.ARIMA(order=(2, 1, 2), suppress_warnings=True).fit(train_series)
            print(f"Fallback 1 (2,1,2) used for {state_name}.")
        except Exception as e2:
            print(f"Fallback 1 failed: {e2}. Trying fallback 2 (1,1,1)...")
            # 3. Second Fallback: non-seasonal ARIMA(1,1,1)
            model = pm.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(train_series)
            print(f"Fallback 2 (1,1,1) used for {state_name}.")
    
    # Save the model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{state_name}_arima.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def predict_arima(model, n_periods=8):
    """
    Forecasts next n_periods steps.
    """
    forecast = model.predict(n_periods=n_periods)
    return forecast.tolist()

if __name__ == "__main__":
    # Test with sample data
    from preprocess import load_data, preprocess_states
    
    data_path = 'data/sales_data.xlsx'
    if not os.path.exists(data_path):
        print("Run src/preprocess.py first to generate data.")
    else:
        df_raw = load_data(data_path)
        processed_dict = preprocess_states(df_raw)
        
        # Pick one state
        state = list(processed_dict.keys())[0]
        state_df = processed_dict[state]
        
        # Set date as index and take 'sales' series
        state_df = state_df.set_index('date')
        series = state_df['sales']
        
        # Split (manual simple split for testing)
        train_series = series.iloc[:-8]
        val_series = series.iloc[-8:]
        
        # Train
        model = train_arima(train_series, state)
        
        # Predict
        predictions = predict_arima(model, n_periods=8)
        print(f"Predictions for {state}: {predictions}")
        print(f"Actual values: {val_series.tolist()}")
