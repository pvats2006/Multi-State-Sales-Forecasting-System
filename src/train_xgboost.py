import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import holidays
import os
from pathlib import Path

# Feature list as specified
FEATURES = [
    'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
    'rolling_mean_4', 'rolling_std_4',
    'week_of_year', 'month', 'day_of_week', 'is_holiday'
]

def train_xgboost(train_df, state_name, random_state=42, silent=False):
    """
    Trains an XGBoost model on feature-engineered data.
    """
    # 1. Identify which features are present (some lags might have been dropped)
    actual_features = [f for f in FEATURES if f in train_df.columns]
    X = train_df[actual_features]
    y = train_df['sales']
    
    if not silent:
        print(f"[XGBoost] Training on {len(train_df)} rows for {state_name}")
        print(f"[XGBoost] Features: {actual_features}")
    
    if len(train_df) < 10:
        raise ValueError(f"Insufficient training data for {state_name}: {len(train_df)} rows. Need at least 10.")

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    model.fit(X, y)
    
    # Save the model (only if not silent/bootstrap)
    if not silent:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"{state_name}_xgboost.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model

def predict_xgboost(model, last_known_df, n_periods=8):
    """
    Recursive forecasting: predicts 1 step at a time and uses predictions as future lags.
    """
    # Ensure date is datetime
    history = last_known_df.copy()
    history['date'] = pd.to_datetime(history['date'])
    
    predictions = []
    in_holidays = holidays.country_holidays('IN')
    
    # Identify features used during training
    actual_features = model.feature_names_in_
    
    for _ in range(n_periods):
        # 1. Get next date
        next_date = history.iloc[-1]['date'] + pd.Timedelta(weeks=1)
        
        # 2. Build feature row from history
        # Check history length based on active lags
        max_lag = 1
        if 'sales_lag_30' in actual_features: max_lag = 30
        elif 'sales_lag_7' in actual_features: max_lag = 7
        
        if len(history) < max_lag:
            raise ValueError(f"History too short for recursive prediction. Need {max_lag}, got {len(history)}.")
            
        # Extract features for current step
        feat_dict = {}
        if 'sales_lag_1' in actual_features: feat_dict['sales_lag_1'] = history.iloc[-1]['sales']
        if 'sales_lag_7' in actual_features: feat_dict['sales_lag_7'] = history.iloc[-7]['sales']
        if 'sales_lag_30' in actual_features: feat_dict['sales_lag_30'] = history.iloc[-30]['sales']
        
        if 'rolling_mean_4' in actual_features or 'rolling_std_4' in actual_features:
            rolling_window = history.iloc[-4:]['sales']
            if 'rolling_mean_4' in actual_features: feat_dict['rolling_mean_4'] = rolling_window.mean()
            if 'rolling_std_4' in actual_features: 
                std = rolling_window.std()
                feat_dict['rolling_std_4'] = 0 if pd.isna(std) else std
        
        if 'week_of_year' in actual_features: feat_dict['week_of_year'] = int(next_date.isocalendar().week)
        if 'month' in actual_features: feat_dict['month'] = next_date.month
        if 'day_of_week' in actual_features: feat_dict['day_of_week'] = next_date.dayofweek
        if 'is_holiday' in actual_features: feat_dict['is_holiday'] = 1 if next_date in in_holidays else 0
        
        # Construct input row
        input_data = pd.DataFrame([feat_dict])[actual_features]
        
        # 3. Predict
        pred = model.predict(input_data)[0]
        predictions.append(float(pred))
        
        # 4. Update history with prediction for next step
        new_row = pd.DataFrame({'date': [next_date], 'sales': [pred]})
        history = pd.concat([history, new_row], ignore_index=True)
        
    return predictions

if __name__ == "__main__":
    # Test with sample data
    from preprocess import load_data, preprocess_states
    from features import create_features, train_val_split
    
    data_path = 'data/sales_data.xlsx'
    if not os.path.exists(data_path):
        print("Run src/preprocess.py first to generate data.")
    else:
        df_raw = load_data(data_path)
        processed_dict = preprocess_states(df_raw)
        
        # Pick one state
        state = list(processed_dict.keys())[0]
        state_df = processed_dict[state]
        
        # Create features
        df_featured = create_features(state_df)
        
        # Split
        train_df, val_df = train_val_split(df_featured, val_weeks=8)
        
        # Train
        model = train_xgboost(train_df, state)
        
        # For prediction, we need the history immediately preceding the validation set
        # Since train_df is already clipped of NaNs and last 8 weeks, 
        # we need the rows from df_featured just before val_df
        # Or more simply, the last 30+ rows of the training set (including the period shifted)
        
        # The history should be the state_df (original processed) up to the start of validation
        history_end_idx = state_df[state_df['date'] < val_df['date'].min()].index[-1]
        history_df = state_df.loc[:history_end_idx]
        
        # Predict
        predictions = predict_xgboost(model, history_df, n_periods=8)
        print(f"XGBoost Predictions for {state}: {predictions}")
        print(f"Actual values: {val_df['sales'].tolist()}")
