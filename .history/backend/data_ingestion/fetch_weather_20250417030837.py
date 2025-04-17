"""
Fetch weather forecast data from external APIs
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")

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

    output_path = f"backend/data/weather_{city}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_data, f, indent=2)

    return structured_data
