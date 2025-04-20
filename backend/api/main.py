import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.models.train_hybrid_model import train_hybrid_model
from backend.data_ingestion.external_data import build_external_context
from backend.utils.risk_scoring import combine_factors

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

    print("[DEBUG] Raw forecast output:")
    for f in result.get("forecast", []):
        print(f)

    forecast = result.get("forecast", [])
    if not forecast:
        return {
            "forecast": [],
            "mae": result.get("mae", 0),
            "region": county
        }

    external_ctx = build_external_context("North Carolina")
    enriched_forecast = combine_factors(forecast, external_ctx)

    return {
        "forecast": enriched_forecast,
        "mae": float(result["mae"]),
        "region": county
    }
