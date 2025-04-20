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

COUNTIES = ["Cabarrus", "Caldwell", "Mecklenburg", "Pitt", "Davidson", "Durham"]

def load_er_data(filepath="datasets/processed_er_data.csv", county=None):
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
        print(f"[WARN] Not enough data for {county}")
        return

    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    prophet_path = f"{model_dir}/prophet_model_{county}.pkl"
    lstm_path = f"{model_dir}/lstm_model_{county}.keras"
    scaler_path = f"{model_dir}/scaler_{county}.pkl"

    should_train = not (
        os.path.exists(prophet_path)
        and os.path.exists(lstm_path)
        and os.path.exists(scaler_path)
    )

    if should_train:
        print(f"[INFO] Training hybrid model for {county}")
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
        print(f"[INFO] Model for {county} already exists. Skipping training.")

if __name__ == "__main__":
    for county in COUNTIES:
        train_hybrid_model(county, days_ahead=7)
