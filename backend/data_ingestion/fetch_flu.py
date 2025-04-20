import json
import os
from delphi_epidata import Epidata
from datetime import date, timedelta
from isoweek import Week
from dotenv import load_dotenv

load_dotenv()

def get_epiweek(d: date):
    year, week, _ = d.isocalendar()
    return int(f"{year}{week:02d}")

def fetch_flu(region: str = "nc") -> dict:
    today = date.today()
    recent_epiweeks = [get_epiweek(today - timedelta(weeks=i)) for i in range(7)]

    response = Epidata.fluview(
        regions=[region],
        epiweeks=recent_epiweeks
    )

    if response["result"] != 1 or not response.get("epidata"):
        raise Exception("API error: no results")

    structured_data = []
    for entry in response["epidata"]:
        structured_data.append({
            "date": str(entry["epiweek"]),
            "flu_activity_level": entry.get("wili", "N/A"),
            "region": region
        })

    output_path = f"backend/data/external_factors/flu_{region.upper()}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_data, f, indent=2)

    return structured_data

if __name__ == "__main__":
    fetch_flu("nc")
