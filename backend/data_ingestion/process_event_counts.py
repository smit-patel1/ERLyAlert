import json
import os
from collections import defaultdict
from datetime import datetime

INPUT_PATH = "backend/data/external_factors/nc_events_details.json"
OUTPUT_PATH = "backend/data/external_factors/events_daily.json"

def fetch_and_save_events():
    if not os.path.exists(INPUT_PATH):
        print("Raw event file not found.")
        return

    with open(INPUT_PATH, "r") as f:
        raw_events = json.load(f)

    event_count_by_date = defaultdict(int)
    for event in raw_events:
        date = event.get("date")
        if date:
            try:
                normalized_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
                event_count_by_date[normalized_date] += 1
            except ValueError:
                continue

    daily_event_data = {date: {"event_count": count} for date, count in event_count_by_date.items()}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(daily_event_data, f, indent=2)

    print(f"Saved daily event counts to {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    fetch_and_save_events()
