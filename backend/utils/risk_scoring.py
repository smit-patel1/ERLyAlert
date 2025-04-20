from datetime import datetime

def _normalise(date_val: str) -> str:
    """Return YYYY‑MM‑DD no matter what comes in."""
    if isinstance(date_val, str):
        if "T" in date_val:          
            date_val = date_val.split("T")[0]
        try:
            return datetime.fromisoformat(date_val).strftime("%Y-%m-%d")
        except ValueError:
            return date_val[:10]
    return date_val.strftime("%Y-%m-%d")

def combine_factors(forecast_rows: list[dict], external_ctx: dict) -> list[dict]:
    enriched = []

    for row in forecast_rows:
        date_key = _normalise(row["date"])
        ctx      = external_ctx.get(date_key, {})

        score, factors = 0, []

        # flu
        flu = ctx.get("flu_index", 0)
        if flu > 0.6:
            score += 1
            factors.append({"type": "flu", "value": f"{flu:.2f}", "impact": "+1"})

        # events
        ev = ctx.get("event_count", 0)
        if ev >= 10:
            score += 2
            factors.append({"type": "events", "value": str(ev), "impact": "+2"})

        # temperature
        anom = ctx.get("temp_anomaly", 0)
        if anom > 8:
            score += 1
            factors.append({"type": "temperature", "value": f"{anom:.1f}°C", "impact": "+1"})

        row["contributing_factors"] = factors
        row["risk_level"] = ("High"   if score >= 3 else
                             "Medium" if score == 2 else
                             "Low")
        row["explanation"] = (
            "ER visits likely to surge due to " + " and ".join(f["type"] for f in factors) + "."
            if factors else
            "No significant risk factors."
        )

        enriched.append(row)

    return enriched
