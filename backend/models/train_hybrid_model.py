import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
# from backend.external_data import load_external_data
# from backend.risk_logic import combine_factors

def load_er_data(filepath="data\processed_er_data.csv", county=None):
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])
    df["county"] = df["county"].astype(str).str.strip()
    if county:
        df = df[df["county"] == county].copy()
        if df.empty:
            print(f"[WARN] No data found for {county}")
            return pd.DataFrame()
        df = df[["ds", "y"]]
    return df

def prepare_lstm_data(data, sequence_length=7):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_hybrid_model(county, days_ahead=7):
    df = load_er_data(county=county)
    if df.empty or len(df) < 10:
        return {"forecast": [], "mae": 0.0, "region": county}

    prophet_model_path = f"trained_models/prophet_model_{county}.pkl"
    lstm_model_path = f"trained_models/lstm_model_{county}.keras"

    if os.path.exists(prophet_model_path) and os.path.exists(lstm_model_path):
        prophet_model = joblib.load(prophet_model_path)
        lstm_model = tf.keras.models.load_model(lstm_model_path)
    else:
        prophet_model = Prophet()
        prophet_model.fit(df)
        forecast_all = prophet_model.predict(df[["ds"]])
        merged = df.merge(forecast_all[["ds", "yhat"]], on="ds", how="inner")
        residuals = merged['y'] - merged['yhat']
        if len(residuals) < 7:
            print(f"[WARN] Not enough residuals to train LSTM for {county}")
            return {"forecast": [], "mae": 0.0, "region": county}
        X, y, scaler = prepare_lstm_data(residuals, sequence_length=7)
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X, y, epochs=75, batch_size=16, verbose=0)
        os.makedirs("trained_models", exist_ok=True)
        joblib.dump(prophet_model, prophet_model_path)
        lstm_model.save(lstm_model_path)
        joblib.dump(scaler, f"trained_models/scaler_{county}.pkl")
        joblib.dump(residuals, f"trained_models/residuals_{county}.pkl")

    forecast = prophet_model.make_future_dataframe(periods=days_ahead)
    forecast = prophet_model.predict(forecast)
    merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    residuals = merged["y"] - merged["yhat"]

    if len(residuals) < 7:
        print(f"[WARN] Not enough residuals to make prediction for {county}")
        return {"forecast": [], "mae": 0.0, "region": county}

    input_seq = residuals.values[-7:].reshape(-1, 1)
    scaler = MinMaxScaler()
    input_seq_scaled = scaler.fit_transform(input_seq).reshape(1, 7, 1)
    residual_pred_scaled = lstm_model.predict(input_seq_scaled, verbose=0)
    residual_pred = scaler.inverse_transform(residual_pred_scaled)[0][0]

    forecast_tail = forecast.tail(days_ahead).copy()
    forecast_tail["yhat_hybrid"] = forecast_tail["yhat"] + residual_pred

    threshold = df["y"].mean() + df["y"].std()
    forecast_tail["high_risk"] = forecast_tail["yhat_hybrid"] > threshold

    # Optional: hook in external data later
    # external_data = load_external_data(county)
    # forecast_tail = combine_factors(forecast_tail, external_data)

    forecast_tail = forecast_tail.rename(columns={"yhat_hybrid": "forecast"})
    output = forecast_tail[["ds", "forecast", "high_risk"]]
    output = output.rename(columns={"ds": "date"})
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")

    mae = float(mean_absolute_error(merged["y"].tail(days_ahead), forecast_tail["forecast"]))

    return {
        "forecast": output.to_dict(orient="records"),
        "mae": mae,
        "region": county
    }

if __name__ == "__main__":
    out = train_hybrid_model("Durham")
    print(out)
