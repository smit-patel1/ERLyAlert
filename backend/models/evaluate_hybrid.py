import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def load_data():
    df = pd.read_csv("data/processed_er_data.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    return df

def prepare_lstm_training_data(data, sequence_length=7):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y, scaler

def evaluate_hybrid():
    df = load_data()
    prophet_model = joblib.load("trained_models/prophet_model.pkl")
    lstm_model = tf.keras.models.load_model("trained_models/lstm_model.keras")

    forecast = prophet_model.predict(df[["ds"]])
    df["yhat"] = forecast["yhat"]
    residuals = df["y"] - df["yhat"]

    X, y_true, scaler = prepare_lstm_training_data(residuals)
    y_pred_scaled = lstm_model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    hybrid_yhat = df["yhat"].values[-len(y_pred):] + y_pred
    actual_y = df["y"].values[-len(y_pred):]
    dates = df["ds"].values[-len(y_pred):]

    df_result = pd.DataFrame({
        "date": dates,
        "y": actual_y,
        "yhat_prophet": df["yhat"].values[-len(y_pred):],
        "yhat_hybrid": hybrid_yhat
    })

    mae = mean_absolute_error(actual_y, hybrid_yhat)
    return df_result, mae

if __name__ == "__main__":
    df_hybrid, mae_hybrid = evaluate_hybrid()
    print(f"Hybrid MAE: {mae_hybrid:.2f}")
