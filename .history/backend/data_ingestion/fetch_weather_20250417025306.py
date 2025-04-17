"""
Fetch weather forecast data from external APIs
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("a3b755a0f418212f476b0af7805a3c9d")

def fetch_weather(city: str) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching data")
        return {}

    data = response.json()
    structured_data = []

    for entry in data["list"]:
        structured_data.append({
            "datetime": entry["dt_txt"],
            "temperature": entry["main"]["temp"],
            "weather": entry["weather"][0]["description"]
        })

    # Save it
    output_path = f"backend/data/external_factors/weather_{city}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_data, f, indent=2)

    return structured_data

# Example call
if __name__ == "__main__":
    fetch_weather("Charlotte")
