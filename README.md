# ERLyAlert

ERLyAlert is a predictive analytics platform that forecasts emergency room (ER) visit surges using a hybrid machine learning model and real-time contextual data. Our goal is to help hospitals, emergency planners, and public health agencies proactively allocate staff and resources to prevent ER overload.

## Project Overview

ERLyAlert provides:

- **7-Day Forecasts** for ER admissions across selected counties in NC.
- **Risk Classification** using flu activity, local events, and weather.
- **Factor Breakdown** explaining what causes surge risks (e.g. events, flu, heat spikes).
- **Interactive Streamlit Dashboard** with region selector, filters, confidence intervals, and insights.
- **Natural Language Forecast Assistant** that answers user queries in English.
- **Exportable Reports** in CSV or JSON formats.

---

## Tech Stack

### Backend

- **FastAPI**: for real-time API endpoints (`/forecast`)
- **Hybrid Forecasting Model**:
  - **Prophet** captures seasonality, trends, and holidays.
  - **LSTM** models recent residuals (non-linear error patterns) to increase accuracy.
  - Both models are trained **per-county** and combined for a hybrid prediction.
- **Risk Scoring**:
  - Calculates risk levels based on flu index, event count, and temperature anomaly.
  - Generates human-readable explanations and factor impact metadata.
- **External Data Sources**:
  - Flu: CDC FluView API
  - Events: Ticketmaster (via API)
  - Weather: OpenWeatherMap
- **Libraries**: pandas, scikit-learn, tensorflow/keras, joblib, Prophet

### Frontend

- **Streamlit**: Interactive Dashboard
- **Plotly**: For charting predicted visit volumes and confidence
- **Interactive Visualization:** Line and bar charts with dynamic filters

---

## Project Structure and File Overview

###

- `.env` - API keys and environment config
- `trained_models/` - saved Prophet, LSTM, and scaler files for each county

### `frontend/`

- `streamlit_app.py` - the main dashboard application

### `backend/api/`

- `main.py` - FastAPI app with `/forecast` route, returns hybrid forecast improved with contextual risk factors

### `backend/data_ingestion/`

- `fetch_events.py` - fetches NC event listings from Ticketmaster
- `fetch_weather.py` - pulls 5-day forecast data for a specified city
- `fetch_flu.py` - queries CDC FluView for weekly flu activity
- `process_event_counts.py` - converts raw event list to daily event count JSON
- `process_weather_anomalies.py` - calculates temperature anomalies vs monthly baselines
- `external_data.py` - merges flu, weather, and event data into a unified context dictionary

### `backend/models/`

- `train_hybrid_model.py` - trains or loads per-county Prophet+LSTM models and returns 7-day forecasts
- `predict_hybrid_models.py` - runs inference using saved hybrid models
- `retrain_all.py` - retrains hybrid models for all counties

### `backend/utils/`

- `risk_scoring.py` - calculates risk scores from contextual data and generates explanations
- `preprocessing.py` - handles ER data cleaning

### `backend/data/external_factors/`

- `events_daily.json`, `flu_NC.json`, `weather_daily_charlotte.json` - contextual inputs used for risk scoring

---

## Model Summary

We are using a **hybrid time series model**:

- **Facebook Prophet** is trained on historical ER visits for each county
- Forecast residuals (errors) are fed into an **LSTM neural network**
- The LSTM learns non-linear deviations that Prophet misses
- Final forecast = `Prophet prediction + LSTM residual adjustment`
- High-risk days are flagged when predictions exceed `mean + 1 standard deviation`

The contextual risk scoring further analyzes each day by matching the date with real-world external factors and produces:

- `risk_level`: Low, Medium, High
- `explanation`: why it’s risky (e.g., "flu and events")
- `contributing_factors`: flu index, event counts, temperature anomalies
