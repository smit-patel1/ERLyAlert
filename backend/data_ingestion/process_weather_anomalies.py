import json
import os
from collections import defaultdict
from datetime import datetime

INPUT_PATH = "backend/data/external_factors/weather_Charlotte.json"
OUTPUT_PATH = "backend/data/external_factors/weather_daily_charlotte.json"

MONTHLY_CLIMATES = {
    1: 4.5, 2: 6.0, 3: 11.2, 4: 16.8,
    5: 21.5, 6: 25.6, 7: 28.2, 8: 27.5,
    9: 23.8, 10: 17.2, 11: 11.1, 12: 6.0
}

def fetch_and_save_weather_anomalies():
    if not os.path.exists(INPUT_PATH):
        print("Missing weather forecast input.")
        return

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    daily_temps = defaultdict(list)

    for entry in data:
        dt = datetime.strptime(entry["datetime"], "%Y-%m-%d %H:%M:%S")
        day = dt.strftime("%Y-%m-%d")
        temp = entry.get("temperature")
        if temp is not None:
            daily_temps[day].append(temp)

    result = {}

    for date, temps in daily_temps.items():
        avg = round(sum(temps) / len(temps), 2)
        month = int(date.split("-")[1])
        baseline = MONTHLY_CLIMATES.get(month, 20.0)
        anomaly = round(avg - baseline, 2)
        result[date] = {"temp_anomaly": anomaly}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved temperature anomalies to {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    fetch_and_save_weather_anomalies()