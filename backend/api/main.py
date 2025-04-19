import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.models.train_hybrid_model import train_hybrid_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ERLyAlert Hybrid API running."}

@app.get("/forecast")
def get_forecast(county: str = Query(...), days: int = 7):
    result = train_hybrid_model(county=county, days_ahead=days)
    return {
        "forecast": result["forecast"],
        "mae": float(result["mae"]),
        "region": county
    }
