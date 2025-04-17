import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os

DATA_PATH = os.path.join("data", "processed_er_data.csv")

def load_er_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "ds" not in df.columns:
        df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def forecast_er_visits(days_ahead: int = 7):
    df = load_er_data()
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    recent = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_ahead).copy()
    recent["high_risk"] = recent["yhat"] > (df["y"].mean() + df["y"].std())
    recent = recent.rename(columns={"ds": "date", "yhat": "forecast"})

    merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    mae = mean_absolute_error(merged["y"], merged["yhat"])

    return {
        "region": "Charlotte",
        "forecast_days": days_ahead,
        "mae": mae,
        "forecast": recent.to_dict(orient="records")
    }
