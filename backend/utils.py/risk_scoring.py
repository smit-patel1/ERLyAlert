def calculate_risk_score(row):
    """
    Calculates the risk level (Low, Medium, High) based on external factors.

    Parameters:
        row (dict): A dictionary containing contextual data for a single day.
                    Expected keys: 'temp_anomaly', 'flu_index', 'event_count'

    Returns:
        str: One of "Low", "Medium", or "High"
    """
    score = 0

    if row.get("temp_anomaly", 0) > 8:
        score += 1
    if row.get("flu_index", 0) > 0.6:
        score += 1
    if row.get("event_count", 0) >= 2:
        score += 2

    if score >= 3:
        return "High"
    elif score == 2:
        return "Medium"
    else:
        return "Low"


def generate_risk_explanation(row):
    """
    Generates a human-readable explanation for the risk level.

    Parameters:
        row (dict): A dictionary containing contextual data for a single day.

    Returns:
        str: Explanation string summarizing the cause of risk.
    """
    reasons = []

    if row.get("flu_index", 0) > 0.6:
        reasons.append("high flu activity")
    if row.get("event_count", 0) >= 2:
        reasons.append("multiple large public events")
    if row.get("temp_anomaly", 0) > 8:
        reasons.append("unusual temperature anomaly")

    if not reasons:
        return "No significant risk factors."
    else:
        return "ER visits likely to surge due to " + " and ".join(reasons) + "."


def combine_factors(forecast_df, external_data_dict):
    """
    Combines model forecast data with contextual risk factors.

    Parameters:
        forecast_df (list): List of forecast dictionaries for each date.
        external_data_dict (dict): Dictionary with external data keyed by date.

    Returns:
        list: Updated forecast list with risk_level and explanation added.
    """
    enriched_forecast = []

    for row in forecast_df:
        date = row.get("date")
        context = external_data_dict.get(date, {})

        risk_level = calculate_risk_score(context)
        explanation = generate_risk_explanation(context)

        row["risk_level"] = risk_level
        row["explanation"] = explanation

        enriched_forecast.append(row)

    return enriched_forecast
