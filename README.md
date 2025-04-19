# ERLyAlert

ERLyAlert predicts surges in emergency room visits, giving hospitals the data they need to prepare. Our platform uses a hybrid model of Facebook Prophet & LSTM with real-time external data to help healthcare facilities manage resources better when the need arises.

## Project Overview

ERLyAlert helps emergency departments prepare by providing:

- **7-Day ER Visit Forecasts:** 7-day ER admission trends before they happen
- **Automated Risk Alerts:** Get warnings when our system predicts high-traffic days
- **Factor Analysis:** Understand what's driving increased visits - weather events, local gatherings, disease outbreaks, and air quality issues
- **Interactive Dashboard:** Select your region, adjust timeframes, and dig into the metrics that matter
- **Natural Language Querying ChatBot:** Ask questions naturally and get straightforward answers

## Tech Stack

### Backend

- **API Framework:** FastAPI
- **Forecasting Models:** We're using a hybrid approach that combines Facebook Prophet with LSTM neural networks. Prophet handles the seasonal patterns and trends really well, while the LSTM catches the non-linear relationships that happen over time. By combining these two approaches our model is more accurate especially when the ER gets unexpectedly slammed or when visit patterns suddenly change.
- **Data Handling:** Pandas, NumPy, and scikit-learn
- **External Data Integration:** Securely integrated using Python requests and python-dotenv

### Frontend

- **Dashboard Framework:** Streamlit
- **Visualizations:** Plotly for interactive visualizations of forecast data
- **Map Integration:** Pydeck shows geographic risk levels through intuitive maps

## Implementation

- **Data Preprocessing:** We wrote scripts to convert historical ER data into daily forecasts
- **Forecasting Model:** Our Prophet model delivers usable predictions
- **Dashboard:** The prototype shows forecasts, risk levels, and factor analysis
- **Optimization and Validation:** Rigorous testing and validation to increase accuracy
- **Expanded External Data Sources:** Expanding our real-time data integration (detailed epidemic tracking, event impacts, etc.)

## License

This project is open-source under the MIT License.
