import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def load_er_data(filepath="data/processed_er_data.csv"):
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    return df

def prepare_lstm_input(data, sequence_length=7):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    input_seq = scaled[-sequence_length:].reshape(1, sequence_length, 1)
    return input_seq, scaler

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

def predict_hybrid(days_ahead=7):
    df = load_er_data()
    prophet_model = joblib.load("trained_models/prophet_model.pkl")
    lstm_model = tf.keras.models.load_model("trained_models/lstm_model.keras")

    future = prophet_model.make_future_dataframe(periods=days_ahead)
    forecast = prophet_model.predict(future)

    merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    residuals = merged["y"] - merged["yhat"]

    input_seq, scaler = prepare_lstm_input(residuals)
    residual_pred_scaled = lstm_model.predict(input_seq, verbose=0)
    residual_pred = scaler.inverse_transform(residual_pred_scaled)[0][0]

    forecast_tail = forecast.tail(days_ahead).copy()
    forecast_tail["yhat_hybrid"] = forecast_tail["yhat"] + residual_pred
    forecast_tail["high_risk"] = forecast_tail["yhat_hybrid"] > (df["y"].mean() + df["y"].std())

    output = forecast_tail[["ds", "yhat_hybrid", "high_risk"]].rename(
        columns={"ds": "date", "yhat_hybrid": "forecast"}
    )

    return output.to_dict(orient="records")

def evaluate_hybrid_model():
    df = load_er_data()
    prophet_model = joblib.load("trained_models/prophet_model.pkl")
    lstm_model = tf.keras.models.load_model("trained_models/lstm_model.keras")

    forecast = prophet_model.predict(df[["ds"]])
    merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    residuals = merged["y"] - merged["yhat"]

    X, y_true, scaler = prepare_lstm_training_data(residuals)
    y_pred_scaled = lstm_model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    prophet_yhat = merged["yhat"].values[-len(y_pred):]
    hybrid_yhat = prophet_yhat + y_pred
    actual_y = merged["y"].values[-len(y_pred):]

    mae = mean_absolute_error(actual_y, hybrid_yhat)
    return mae

if __name__ == "__main__":
    result = predict_hybrid()
    for row in result:
        print(row)

    mae_score = evaluate_hybrid_model()
    print(f"\nHybrid Model MAE on historical data: {mae_score:.2f}")
