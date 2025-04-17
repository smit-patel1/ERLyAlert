"""
Fetch mock flu activity trends
"""
import json
import os
from datetime import datetime, timedelta

def fetch_flu(region: str) -> dict:
    today = datetime.today()
    structured_data = []

    for i in range(7):
        date = today + timedelta(days=i)
        structured_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "flu_activity_level": round(2 + 1.5 * (i % 3), 2),  # mock values
            "region": region
        })

    output_path = f"backend/data/external_factors/flu_{region}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_data, f, indent=2)
    return structured_data

# call
if __name__ == "__main__":
    fetch_flu("Charlotte")
