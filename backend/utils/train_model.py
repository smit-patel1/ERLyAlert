import pandas as pd
from prophet import Prophet
import joblib

def train_model():
    df = pd.read_csv("data/processed_er_data.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    model = Prophet()
    model.fit(df)

    joblib.dump(model, "models/prophet_model.pkl")
