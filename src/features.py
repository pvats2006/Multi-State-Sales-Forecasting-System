import pandas as pd
import numpy as np
import holidays
from pathlib import Path

def create_features(df, min_rows=30, state_name="Unknown"):
    """
    Creates lag, rolling, calendar, and holiday features for a single state DataFrame.
    Dynamically drops high-period lags if they reduce the dataset below min_rows.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Base lag features
    df['sales_lag_1'] = df['sales'].shift(1)
    df['sales_lag_7'] = df['sales'].shift(7)
    df['sales_lag_30'] = df['sales'].shift(30)
    
    # 2. Rolling features: 4-week window
    df['rolling_mean_4'] = df['sales'].rolling(window=4, min_periods=1).mean()
    df['rolling_std_4'] = df['sales'].rolling(window=4, min_periods=1).std()
    
    # 3. Calendar features
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # 4. Holiday flag
    in_holidays = holidays.country_holidays('IN')
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in in_holidays else 0)
    
    # --- Dynamic Lag Management ---
    # Attempt to keep all lags
    df_clean = df.dropna()
    
    if len(df_clean) < min_rows:
        # Drop lag 30 and try again
        df_no_30 = df.drop(columns=['sales_lag_30']).dropna()
        if len(df_no_30) < min_rows:
            # Drop lag 7 too
            df_final = df.drop(columns=['sales_lag_30', 'sales_lag_7']).dropna()
            print(f"Warning [{state_name}]: Dropped lag_30 and lag_7 to maintain {len(df_final)} rows.")
            return df_final.reset_index(drop=True)
        else:
            print(f"Warning [{state_name}]: Dropped lag_30 to maintain {len(df_no_30)} rows.")
            return df_no_30.reset_index(drop=True)
            
    return df_clean.reset_index(drop=True)

def train_val_split(df, val_weeks=8):
    """
    Splits the data into training and validation sets based on time.
    Last val_weeks rows go to validation.
    """
    if len(df) <= val_weeks:
        print(f"Warning: Dataframe length ({len(df)}) is less than or equal to val_weeks ({val_weeks}).")
        return df, df.iloc[0:0] # Return empty validation
        
    train_df = df.iloc[:-val_weeks]
    val_df = df.iloc[-val_weeks:]
    
    return train_df, val_df

if __name__ == "__main__":
    # Test with one of the processed files
    processed_dir = Path('data/processed/')
    if not processed_dir.exists():
        print("Processed data not found. Please run src/preprocess.py first.")
    else:
        # Pick the first CSV found
        csv_files = list(processed_dir.glob('*.csv'))
        if not csv_files:
            print("No CSV files found in data/processed/.")
        else:
            test_file = csv_files[0]
            print(f"Testing features with {test_file}...")
            
            df = pd.read_csv(test_file)
            df_featured = create_features(df)
            
            print(f"Features created. New shape: {df_featured.shape}")
            print("Columns:", df_featured.columns.tolist())
            
            train, val = train_val_split(df_featured, val_weeks=8)
            print(f"Split complete: Train={len(train)}, Val={len(val)}")
            
            print("\nFirst 5 rows of features:")
            print(df_featured.head())
