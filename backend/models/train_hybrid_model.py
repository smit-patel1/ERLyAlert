import os
import pandas as pd
import numpy as np
import tensorflow as tf
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime
import os
import pandas as pd

def load_er_data(county=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "../../datasets/processed_er_data.csv")
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])
    if county:
        df = df[df["county"] == county].copy()
    return df[["ds", "y"]]


def prepare_lstm_data(data, sequence_length=7):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y, scaler

def train_hybrid_model(county, days_ahead=7):
    df = load_er_data(county=county)

    if df.empty or len(df) < 30:
        return {
            "region": county,
            "forecast_days": days_ahead,
            "mae": 0,
            "forecast": []
        }

    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    prophet_path = f"{model_dir}/prophet_model_{county}.pkl"
    lstm_path = f"{model_dir}/lstm_model_{county}.keras"
    scaler_path = f"{model_dir}/scaler_{county}.pkl"

    if not (os.path.exists(prophet_path) and os.path.exists(lstm_path) and os.path.exists(scaler_path)):
        prophet_model = Prophet()
        prophet_model.fit(df)

        forecast = prophet_model.predict(df[["ds"]])
        merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
        residuals = merged["y"] - merged["yhat"]

        X, y, scaler = prepare_lstm_data(residuals)

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X, y, epochs=75, batch_size=16, verbose=0)

        joblib.dump(prophet_model, prophet_path)
        joblib.dump(scaler, scaler_path)
        lstm_model.save(lstm_path)
    else:
        prophet_model = joblib.load(prophet_path)
        lstm_model = tf.keras.models.load_model(lstm_path)
        scaler = joblib.load(scaler_path)

    future = prophet_model.make_future_dataframe(periods=60)
    forecast_df = prophet_model.predict(future)

    today = pd.to_datetime(datetime.today().date())
    forecast_df = forecast_df[forecast_df["ds"] >= today]
    if forecast_df.empty:
        return {
            "region": county,
            "forecast_days": days_ahead,
            "mae": 0,
            "forecast": []
        }

    forecast_tail = forecast_df.head(days_ahead).copy()
    recent_residuals = df.merge(forecast_df[["ds", "yhat"]], on="ds", how="inner")
    residuals = recent_residuals["y"] - recent_residuals["yhat"]

    if len(residuals) >= 7:
        input_seq = residuals.values[-7:].reshape(-1, 1)
        input_scaled = scaler.transform(input_seq)
        input_seq_reshaped = input_scaled.reshape(1, 7, 1)
        residual_pred_scaled = lstm_model.predict(input_seq_reshaped, verbose=0)
        residual_pred = scaler.inverse_transform(residual_pred_scaled)[0][0]
    else:
        residual_pred = 0.0

    forecast_tail["yhat_hybrid"] = forecast_tail["yhat"] + residual_pred
    forecast_tail["high_risk"] = forecast_tail["yhat_hybrid"] > (df["y"].mean() + df["y"].std())

    final_output = forecast_tail[["ds", "yhat_hybrid", "high_risk"]].rename(
        columns={"ds": "date", "yhat_hybrid": "forecast"}
    )

    merged = df.merge(forecast_df[["ds", "yhat"]], on="ds", how="inner")
    if len(merged) > 0:
        mae = mean_absolute_error(merged["y"], merged["yhat"])
    else:
        mae = 0

    return {
        "region": county,
        "forecast_days": days_ahead,
        "mae": mae,
        "forecast": final_output.to_dict(orient="records")
    }

if __name__ == "__main__":
    result = train_hybrid_model("Cabarrus", days_ahead=7)
    for row in result["forecast"]:
        print(row)
