import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TICKETMASTER_API_KEY")

BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
STATE_CODE = "NC"
COUNTRY_CODE = "US"
DETAILS_OUTPUT_FILE = "backend/data/external_factors/nc_events_details.json"

def fetch_nc_events():
    if not API_KEY:
        print("API key not found. Set TICKETMASTER_API_KEY in your .env file.")
        return

    os.makedirs(os.path.dirname(DETAILS_OUTPUT_FILE), exist_ok=True)

    params = {
        "apikey": API_KEY,
        "stateCode": STATE_CODE,
        "countryCode": COUNTRY_CODE,
        "size": 200,
        "sort": "date,asc"
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch events: {response.status_code} - {response.text}")
        return

    data = response.json()
    events = data.get('_embedded', {}).get('events', [])
    if not events:
        print("No events found.")
        return

    event_details = []
    for event in events:
        event_details.append({
            "name": event.get("name", "N/A"),
            "venue": event.get("_embedded", {}).get("venues", [{}])[0].get("name", "N/A"),
            "date": event.get("dates", {}).get("start", {}).get("localDate", "N/A"),
            "time": event.get("dates", {}).get("start", {}).get("localTime", "N/A"),
            "sales_status": event.get("dates", {}).get("status", {}).get("code", "N/A")
        })

    event_details.sort(key=lambda x: x.get("date", "9999-99-99"))
    with open(DETAILS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(event_details, f, indent=2)

    print(f"Saved {len(event_details)} events to {DETAILS_OUTPUT_FILE}")
    return DETAILS_OUTPUT_FILE

if __name__ == "__main__":
    fetch_nc_events()
