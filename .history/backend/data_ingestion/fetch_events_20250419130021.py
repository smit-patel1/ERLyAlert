"""
Fetch public event schedules from Eventbrite API.
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EVENTBRITE_API_KEY")

def fetch_events(city: str) -> dict:
    url = f"https://www.eventbriteapi.com/v3/events/search/?location.address={city}&location.within=10km&start_date.keyword=this_week"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Error fetching event data:", response.status_code, response.text)
        return {}

    data = response.json()
    events = []

    for event in data.get("events", []):
        events.append({
            "event_name": event.get("name", {}).get("text"),
            "start_time": event.get("start", {}).get("local"),
            "end_time": event.get("end", {}).get("local"),
            "region": city,
            "url": event.get("url")
        })

    output_path = f"backend/data/external_factors/events_{city}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    return events

#  call
if __name__ == "__main__":
    fetch_events("Charlotte")
