import sys
import os
import json
from pathlib import Path
from isoweek import Week
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.data_ingestion.process_event_counts import fetch_and_save_events
from backend.data_ingestion.process_weather_anomalies import fetch_and_save_weather_anomalies
from backend.data_ingestion.fetch_flu import fetch_flu

def _load_json(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def epiweek_to_date(epiweek_str):
    year = int(epiweek_str[:4])
    week = int(epiweek_str[4:])
    return Week(year, week).monday()

def build_external_context(region: str) -> dict:
    events_path = fetch_and_save_events()
    weather_path = fetch_and_save_weather_anomalies()
    flu_data = fetch_flu("nc")

    events_ctx = _load_json(events_path)
    weather_ctx = _load_json(weather_path)

    flu_raw = {}
    for item in flu_data:
        if "date" in item and str(item.get("flu_activity_level", "")).replace(".", "").isdigit():
            epi_date = epiweek_to_date(item["date"])
            iso_date = epi_date.strftime("%Y-%m-%d")
            flu_raw[iso_date] = float(item["flu_activity_level"])

    if flu_raw:
        latest_date = max(flu_raw.keys())
        latest_value = flu_raw[latest_date]
        for i in range(1, 15):
            future_date = datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=i)
            iso = future_date.strftime("%Y-%m-%d")
            flu_raw[iso] = latest_value

    flu_ctx = {date: {"flu_index": value} for date, value in flu_raw.items()}

    external_ctx = {}
    for ctx in (events_ctx, weather_ctx, flu_ctx):
        for date, values in ctx.items():
            external_ctx.setdefault(date, {}).update(values)

    return external_ctx

if __name__ == "__main__":
    context = build_external_context("North Carolina")
    print(json.dumps(context, indent=2))
