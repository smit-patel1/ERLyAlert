"""
Fetch real flu activity trends from CDC FluView API.
"""
import json
import os
from sodapy import Socrata
from dotenv import load_dotenv

load_dotenv()

def fetch_flu(region: str = "North Carolina") -> dict:
    client = Socrata("data.cdc.gov", None)

    # Pull latest 7 entries for the given state
    results = client.get(
        "qx9z-2fkw",
        limit=7,
        order="week_end DESC",
        where=f"region_description='{region}'"
    )

    structured_data = []
    for entry in results:
        structured_data.append({
            "date": entry["week_end"],
            "flu_activity_level": entry.get("activity_level", "N/A"),
            "region": region
        })

    output_path = f"backend/data/external_factors/flu_{region.replace(' ', '_')}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_data, f, indent=2)

    return structured_data

# Example call
if __name__ == "__main__":
    fetch_flu("North Carolina")
