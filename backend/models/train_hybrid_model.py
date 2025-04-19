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

def load_er_data(filepath="data/processed_er_data.csv", county=None):
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])
    if county:
        df = df[df["county"] == county].copy()
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
    prophet_model_path = f"trained_models/prophet_model_{county}.pkl"
    lstm_model_path = f"trained_models/lstm_model_{county}.keras"

    if os.path.exists(prophet_model_path) and os.path.exists(lstm_model_path):
        prophet_model = joblib.load(prophet_model_path)
        lstm_model = tf.keras.models.load_model(lstm_model_path)
    else:
        prophet_model = Prophet()
        prophet_model.fit(df)
        merged = df.merge(prophet_model.predict(df[["ds"]])[["ds", "yhat"]], on="ds", how="inner")
        residuals = merged['y'] - merged['yhat']
        sequence_length = 7
        X, y, scaler = prepare_lstm_data(residuals, sequence_length)
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X, y, epochs=75, batch_size=16, verbose=0)
        os.makedirs("trained_models", exist_ok=True)
        joblib.dump(prophet_model, prophet_model_path)
        lstm_model.save(lstm_model_path)
        scaler_path = f"trained_models/scaler_{county}.pkl"
        joblib.dump(scaler, scaler_path)
        residuals_path = f"trained_models/residuals_{county}.pkl"
        joblib.dump(residuals, residuals_path)

    # TODO: Future forecast using Prophet
    # forecast = prophet_model.make_future_dataframe(periods=days_ahead)
    # forecast = prophet_model.predict(forecast)
    # merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    # residuals = merged["y"] - merged["yhat"]

    # TODO: Predict residual using LSTM on recent window
    # input_seq, scaler = prepare_lstm_data(residuals[-7:])
    # residual_pred = lstm_model.predict(input_seq)

    # TODO: Combine Prophet forecast + residual to get hybrid forecast
    # forecast_tail = forecast.tail(days_ahead).copy()
    # forecast_tail["yhat_hybrid"] = forecast_tail["yhat"] + residual_pred

    # TODO: Risk Flag (basic)
    # forecast_tail["high_risk"] = forecast_tail["yhat_hybrid"] > (df["y"].mean() + df["y"].std())

    # external_data = load_external_data(county)
    # forecast_tail = combine_factors(forecast_tail, external_data)

    # TODO: Final formatting before return
    # forecast_tail = forecast_tail.rename(columns={"yhat_hybrid": "forecast"})
    # return {
    #     "forecast": forecast_tail.to_dict(orient="records"),
    #     "mae": calculated_mae,
    #     "region": county
    # }

if __name__ == "__main__":
    out = train_hybrid_model("Durham")
