from fastapi import FastAPI
from typing import List
from backend.models.prophet_model import forecast_er_visits

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ERLyAlert API is running."}

@app.get("/forecast")
def get_forecast(days: int = 7):
    forecast = forecast_er_visits(days)
    return {
        "region": "Charlotte",
        "forecast_days": days,
        "forecast": forecast
    }
