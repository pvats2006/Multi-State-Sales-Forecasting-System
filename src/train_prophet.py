import pandas as pd
from prophet import Prophet
import joblib
import os
from pathlib import Path

def train_prophet(train_df, state_name):
    """
    Trains a Prophet model with Indian holidays and multiplicative seasonality.
    """
    print(f"Training Prophet for {state_name}...")
    
    # Prepare dataframe for Prophet
    df_p = train_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Initialize Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add Indian public holidays
    model.add_country_holidays(country_name='IN')
    
    # Fit the model
    model.fit(df_p)
    
    # Save the model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{state_name}_prophet.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def predict_prophet(model, n_periods=8):
    """
    Forecasts next n_periods steps using Prophet.
    """
    # Note: make_future_dataframe freq 'W' defaults to Sunday, matching our data
    future = model.make_future_dataframe(periods=n_periods, freq='W')
    forecast = model.predict(future)
    
    # Return yhat for the last n_periods
    predictions = forecast['yhat'].iloc[-n_periods:].tolist()
    return predictions

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
        
        # Split
        train_df = state_df.iloc[:-8]
        val_df = state_df.iloc[-8:]
        
        # Train
        model = train_prophet(train_df, state)
        
        # Predict
        predictions = predict_prophet(model, n_periods=8)
        print(f"Prophet Predictions for {state}: {predictions}")
        print(f"Actual values: {val_df['sales'].tolist()}")
