"""
Fetch public event schedules from APIs like Eventbrite or Ticketmaster (mock version).
"""
import json
import os
from datetime import datetime, timedelta

def fetch_events(city: str) -> dict:
    # Generate mock events for the next 5 days
    today = datetime.today()
    events = []

    sample_events = [
        "Concert in the Park",
        "Food Truck Festival",
        "Health Awareness Fair",
        "Downtown Parade",
        "Local Football Game"
    ]

    for i in range(5):
        date = today + timedelta(days=i)
        events.append({
            "date": date.strftime("%Y-%m-%d"),
            "event_name": sample_events[i % len(sample_events)],
            "region": city,
            "attendees_estimate": 500 + i * 100
        })

    output_path = f"backend/data/external_factors/events_{city}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    return events

#  call
if __name__ == "__main__":
    fetch_events("Charlotte")
