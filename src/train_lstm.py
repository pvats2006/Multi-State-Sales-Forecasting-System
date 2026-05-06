"""
train_lstm.py
LSTM model training and prediction with adaptive complexity
based on dataset size, and graceful TensorFlow import handling.
"""

LSTM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
    print(f"[LSTM] TensorFlow {tf.__version__} loaded successfully.")
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        LSTM_AVAILABLE = True
        print("[LSTM] Keras loaded successfully.")
    except ImportError:
        LSTM_AVAILABLE = False
        print("[LSTM] WARNING: TensorFlow/Keras not found. "
              "Install with: pip install tensorflow-cpu")

import numpy as np
import joblib
import os


def build_lstm_model(input_shape, units_1=64, units_2=32,
                     stacked=True):
    """Build and compile a Keras LSTM model."""
    if not LSTM_AVAILABLE:
        raise ImportError(
            "TensorFlow/Keras not installed. "
            "Run: pip install tensorflow-cpu")

    model = Sequential()

    if stacked:
        model.add(LSTM(units_1, return_sequences=True,
                       input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units_2))
        model.add(Dropout(0.2))
    else:
        model.add(LSTM(units_1, input_shape=input_shape))
        model.add(Dropout(0.1))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def create_sequences(data, seq_len):
    """Convert a 1D array into (X, y) sequence pairs."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def train_lstm(train_series, state_name):
    """
    Train an LSTM model on a weekly sales series.
    Automatically adjusts complexity based on dataset size.

    Parameters:
        train_series : pd.Series - weekly sales with datetime index
        state_name   : str - used for saving model artifacts

    Returns:
        (model, scaler) tuple, or (None, None) on failure
    """
    if not LSTM_AVAILABLE:
        raise ImportError(
            "TensorFlow/Keras not installed. "
            "Run: pip install tensorflow-cpu")

    from sklearn.preprocessing import MinMaxScaler

    n = len(train_series)
    print(f"  [LSTM] Training on {n} samples for {state_name}.")

    # --- Adaptive config based on dataset size ---
    if n < 40:
        seq_len  = 4
        epochs   = 30
        stacked  = False
        units_1  = 32
        units_2  = 16
        print(f"  [LSTM] Small dataset - using "
              f"seq_len={seq_len}, single LSTM({units_1}).")
    elif n < 60:
        seq_len  = 8
        epochs   = 40
        stacked  = False
        units_1  = 48
        units_2  = 24
        print(f"  [LSTM] Medium dataset - using "
              f"seq_len={seq_len}, single LSTM({units_1}).")
    else:
        seq_len  = 12
        epochs   = 50
        stacked  = True
        units_1  = 64
        units_2  = 32
        print(f"  [LSTM] Large dataset - using "
              f"seq_len={seq_len}, stacked LSTM({units_1},{units_2}).")

    # --- Scale data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(
        train_series.values.reshape(-1, 1)).flatten()

    # --- Create sequences ---
    X, y = create_sequences(scaled, seq_len)

    if len(X) < 5:
        print(f"  [LSTM] Not enough sequences after windowing "
              f"(got {len(X)}). Skipping.")
        return None, None

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # --- Build model ---
    model = build_lstm_model(
        input_shape=(seq_len, 1),
        units_1=units_1,
        units_2=units_2,
        stacked=stacked
    )

    # --- Train with early stopping ---
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )

    val_split = 0.15 if len(X) >= 20 else 0.0

    model.fit(
        X, y,
        epochs=epochs,
        batch_size=8,
        validation_split=val_split,
        callbacks=[early_stop] if val_split > 0 else [],
        verbose=0
    )

    # --- Save artifacts ---
    os.makedirs('models', exist_ok=True)
    model_path  = f"models/{state_name}_lstm.keras"
    scaler_path = f"models/{state_name}_lstm_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"  [LSTM] Model saved -> {model_path}")
    return model, scaler


def predict_lstm(model, scaler, train_series, n_periods=8,
                 seq_len=None):
    """
    Recursively forecast n_periods steps ahead using the trained LSTM.

    Parameters:
        model        : trained Keras model
        scaler       : fitted MinMaxScaler
        train_series : full training series (pd.Series)
        n_periods    : number of future steps to forecast
        seq_len      : sequence length (auto-detected if None)

    Returns:
        list of float - forecasted values in original scale
    """
    if not LSTM_AVAILABLE:
        raise ImportError("TensorFlow/Keras not installed.")

    # Auto-detect seq_len from model input shape
    if seq_len is None:
        seq_len = model.input_shape[1]

    scaled = scaler.transform(
        train_series.values.reshape(-1, 1)).flatten()

    # Seed with the last seq_len values
    current_seq = list(scaled[-seq_len:])
    predictions_scaled = []

    for _ in range(n_periods):
        x = np.array(current_seq[-seq_len:]).reshape(1, seq_len, 1)
        pred = model.predict(x, verbose=0)[0][0]
        predictions_scaled.append(pred)
        current_seq.append(pred)

    # Inverse transform
    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten().tolist()

    return predictions
