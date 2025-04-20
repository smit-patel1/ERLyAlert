import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json
import numpy as np
import pydeck as pdk
from collections import Counter

st.set_page_config(
    page_title="ERLyAlert - ER Visit Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }

    /* Sidebar height fix for fullscreen */
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow-y: auto;
        padding-bottom: 2rem;
    }

    /* Reduce main top spacing */
    .block-container {
        padding-top: 1rem;
    }

    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }

    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }

    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }

    .risk-low {
        background-color: #00cc96;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }

    .factor-box {
        background-color: #f1f3f6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }

    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .header-text {
        margin-left: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_regions():
    """Fetch available regions from API or use fallback"""
    try:
        response = requests.get("http://localhost:8000/regions")
        if response.status_code == 200:
            return response.json()
    except:
        pass

    return [
        {"id": "ct_yale", "name": "Yale New Haven Health System (CT)"}
    ]

@st.cache_data
def get_forecast_data(county, days=7):
    try:
        response = requests.get(f"http://localhost:8000/forecast?county={county}&days={days}")
        if response.status_code == 200:
            data = response.json()
            forecast = data.get("forecast", [])
            baseline = np.mean([day.get("forecast", 0) for day in forecast]) if forecast else 100

            transformed = []
            for day in forecast:
                transformed.append({
                    "date": pd.to_datetime(day["date"]).strftime("%Y-%m-%d"),
                    "predicted_visits": int(day.get("forecast", 0)),
                    "confidence_lower": int(day.get("yhat_lower", day.get("forecast", 0) * 0.9)),
                    "confidence_upper": int(day.get("yhat_upper", day.get("forecast", 0) * 1.1)),
                    "risk_level":  day.get("risk_level", "High" if day.get("high_risk") else "Low"),
                    "contributing_factors": day.get("contributing_factors", [])
                })

            return {
                "forecast": transformed,
                "metrics": {
                    "mae": round(data.get("mae", 0), 2),
                    "accuracy_7day": "N/A"
                },
                "region_info": {
                    "name": data.get("region", county),
                    "baseline_avg": int(baseline)
                }
            }
    except Exception as e:
        st.error(f"Error fetching forecast: {e}")

    return {"forecast": [], "metrics": {}, "region_info": {}}


def display_metric_card(title, value, delta=None, delta_good="increase"):
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin-top:0; color:#555;">{title}</h4>
            <h2 style="margin-bottom:5px;">{value}</h2>
            {f'<p style="margin:0; color:{"green" if (delta_good=="increase" and delta>0) or (delta_good=="decrease" and delta<0) else "red"};">{"+" if delta>0 else ""}{delta}%</p>' if delta is not None else ''}
        </div>
        """, unsafe_allow_html=True)

def risk_badge(risk_level):
    colors = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}
    return f'<span class="{colors.get(risk_level.lower(), "risk-low")}">{risk_level.upper()}</span>'

# App Header
st.markdown("""
    <div class="header-container">
        <h1>🏥 ERLyAlert</h1>
        <div class="header-text">
            <h3>Emergency Room Visit Prediction & Early Warning System</h3>
        </div>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Dashboard Settings")

    counties = ["Cabarrus", "Caldwell", "Mecklenburg", "Pitt", "Davidson", "Durham"]
    selected_county = st.selectbox("Select County", counties)

    forecast_days = st.slider("Forecast Days", min_value=3, max_value=14, value=7, key="forecast_slider", help="Select how many future days to forecast.")

    st.markdown("---")
    st.subheader("Display Settings")
    show_confidence_interval = st.checkbox("Show Confidence Interval", value=True, key="confidence_checkbox", help="Enable to show confidence bands on forecast chart.")
    visualization_type = st.radio("Visualization Type", ["Line Chart", "Bar Chart"], horizontal=True, key="visualization_type_radio")

    st.markdown("---")
    st.subheader("Advanced Filters")
    filter_by_risk = st.multiselect(
        "Filter by Risk Level",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
        key="risk_level_multiselect",
        help="Filter forecast output by predicted risk levels."
    )

forecast_data = get_forecast_data(selected_county, forecast_days)
region_info = forecast_data.get("region_info", {})
forecast = forecast_data.get("forecast", [])
metrics = forecast_data.get("metrics", {})

if filter_by_risk and len(filter_by_risk) < 3:
    forecast = [day for day in forecast if day["risk_level"].lower() in [r.lower() for r in filter_by_risk]]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Risk Summary", "Ask Forecast", "Export & Notify", "Help & Info"])

with tab1:
    st.header(f"ER Visit Forecast for {region_info.get('name', selected_county)}")

    df = pd.DataFrame(forecast)

    if df.empty:
        st.warning("No forecast data available for the selected filters.")
        st.stop()   
    if "date" not in df.columns:
        st.warning("Forecast data is missing the 'date' column.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%b %d (%a)")
    df["risk_color"] = df["risk_level"].map({
        "High": "#ff4b4b",
        "Medium": "#ffa500",
        "Low": "#00cc96"
    }).fillna("#ccc")

    if visualization_type == "Line Chart":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["date_str"], 
            y=df["predicted_visits"],
            mode='lines+markers',
            name='Predicted Visits',
            line=dict(color='#4361ee', width=3),
            marker=dict(size=10, color=df["risk_color"])
        ))
        
        if show_confidence_interval:
            fig.add_trace(go.Scatter(
                x=df["date_str"],
                y=df["confidence_upper"],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df["date_str"],
                y=df["confidence_lower"],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(67, 97, 238, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
        
        baseline = region_info.get("baseline_avg", 100)
        fig.add_trace(go.Scatter(
            x=[df["date_str"].iloc[0], df["date_str"].iloc[-1]],
            y=[baseline, baseline],
            mode='lines',
            name='Baseline Average',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Predicted ER Visits by Day",
            xaxis_title="Date",
            yaxis_title="Predicted Visits",
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 1600])
        )

        
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df["date_str"],
            y=df["predicted_visits"],
            marker_color=df["risk_color"],
            name='Predicted Visits',
            text=df["predicted_visits"],
            textposition='auto',
        ))
        
        baseline = region_info.get("baseline_avg", 100)
        fig.add_trace(go.Scatter(
            x=df["date_str"],
            y=[baseline] * len(df),
            mode='lines',
            name='Baseline Average',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        if show_confidence_interval:
            fig.add_trace(go.Scatter(
                x=df["date_str"],
                y=df["predicted_visits"],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df["confidence_upper"] - df["predicted_visits"],
                    arrayminus=df["predicted_visits"] - df["confidence_lower"],
                    color='rgba(0,0,0,0.3)'
                ),
                mode='markers',
                marker=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Predicted ER Visits by Day",
            xaxis_title="Date",
            yaxis_title="Predicted Visits",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    risk_legend_cols = st.columns(3)
    with risk_legend_cols[0]:
        st.markdown(f'<span class="risk-low">LOW RISK</span> - Normal ER volume expected', unsafe_allow_html=True)
    with risk_legend_cols[1]:
        st.markdown(f'<span class="risk-medium">MEDIUM RISK</span> - Higher than normal volume', unsafe_allow_html=True)
    with risk_legend_cols[2]:
        st.markdown(f'<span class="risk-high">HIGH RISK</span> - Surge conditions likely', unsafe_allow_html=True)
        
    st.subheader("Daily Forecast Details")
    
    for index, row in df.iterrows():
        risk_html = risk_badge(row["risk_level"])
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <h3 style="margin: 0;">{row['date_str']}</h3>
            <div>{risk_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        detail_cols = st.columns([1, 2])
        with detail_cols[0]:
            st.metric("Predicted Visits", int(row["predicted_visits"]), 
                      int(row["predicted_visits"] - region_info.get("baseline_avg", 100)))
            
            if show_confidence_interval:
                st.caption(f"Range: {int(row['confidence_lower'])} - {int(row['confidence_upper'])}")
        
        with detail_cols[1]:
            if "contributing_factors" in row and row["contributing_factors"]:
                st.markdown("##### Contributing Factors")
                
                for factor in row["contributing_factors"]:
                    factor_type = factor["type"].capitalize()
                    factor_value = factor["value"]
                    factor_impact = factor["impact"]
                    
                    st.markdown(f"""
                    <div class="factor-box">
                        <strong>{factor_type}:</strong> {factor_value} <span style="float:right; color:#ff4b4b;">{factor_impact}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("No significant contributing factors")
        
        st.markdown("<hr>", unsafe_allow_html=True)

with tab2:
    st.header("Forecast Summary")

    max_day = max(forecast, key=lambda x: x["predicted_visits"])
    min_day = min(forecast, key=lambda x: x["predicted_visits"])
    high_risk_days = sum(1 for day in forecast if day["risk_level"].lower() == "high")
    baseline = region_info.get("baseline_avg", 100)

    # Optional accuracy calc
    accuracy_val = f"{round(100 - metrics.get('mae', 0) / baseline * 100, 1)}%" if baseline > 0 else "N/A"
    metrics["accuracy_7day"] = accuracy_val

    st.markdown("#### Peak ER Day")
    max_date = datetime.strptime(max_day["date"], "%Y-%m-%d").strftime("%b %d (%a)")

    st.markdown(f"""
    <div style="background:white; border-radius:8px; padding:1rem 1.25rem; box-shadow:0 2px 4px rgba(0,0,0,0.08); margin-bottom:1rem;">
        <div style="font-size:1rem; font-weight:600; color:#444;">{max_date}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top:0.5rem;">
            <div>
                <div style="font-size:2rem; font-weight:700;">{int(max_day["predicted_visits"])}</div>
                <div style="font-size:0.85rem; color:#777;">expected visits</div>
            </div>
            <div style="align-self: flex-end;">{risk_badge(max_day["risk_level"])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Forecast Insights")
    insight_cols = st.columns(2)

    with insight_cols[0]:
        if show_confidence_interval:
            avg_range = int((df["confidence_upper"] - df["confidence_lower"]).mean())
            display_metric_card("Avg CI Range", f"± {avg_range} visits")
        else:
            display_metric_card("Avg CI Range", "Hidden")

    with insight_cols[1]:
        all_factors = []
        for row in df["contributing_factors"]:
            all_factors.extend([f["type"] for f in row])
        top_factor = Counter(all_factors).most_common(1)
        if top_factor:
            display_metric_card("Top Risk Factor", top_factor[0][0].title())
        else:
            display_metric_card("Top Risk Factor", "—")

    st.markdown("#### Risk Level Summary")
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for day in forecast:
        risk_counts[day["risk_level"].lower()] += 1

    risk_summary_cols = st.columns(3)
    risk_colors = {"high": "#ff4b4b", "medium": "#ffa500", "low": "#00cc96"}
    for i, level in enumerate(["high", "medium", "low"]):
        with risk_summary_cols[i]:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size:1.8rem; font-weight:700;">{risk_counts[level]}</div>
                <div style="color: {risk_colors[level]}; font-weight: bold;">{level.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Top Contributing Factors")

    all_factors = []
    for day in forecast:
        for factor in day.get("contributing_factors", []):
            all_factors.append({
                "type": factor["type"],
                "value": factor["value"],
                "impact": int(factor["impact"].replace("+", ""))
            })

    factor_summary = {}
    for factor in all_factors:
        key = f"{factor['type']}:{factor['value']}"
        if key not in factor_summary:
            factor_summary[key] = {
                "type": factor["type"],
                "value": factor["value"],
                "total_impact": 0,
                "count": 0
            }
        factor_summary[key]["total_impact"] += factor["impact"]
        factor_summary[key]["count"] += 1

    top_factors = sorted(factor_summary.values(), key=lambda x: x["total_impact"], reverse=True)[:5]

    for factor in top_factors:
        factor_type = factor["type"].capitalize()
        factor_value = factor["value"]
        impact = factor["total_impact"]
        days = factor["count"]

        st.markdown(f"""
        <div class="factor-box">
            <strong>{factor_type}: {factor_value}</strong>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <div>Impact: <span style="color:#ff4b4b;">+{impact}</span></div>
                <div>Present in {days} days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

with tab3:
    st.subheader("🤖 Ask about the Forecast")
    st.caption("Ask questions about the forecast such as:")

    st.markdown("""
    <ul style='padding-left:1.2rem;'>
        <li>When is the highest risk day?</li>
        <li>Which day has the lowest predicted visits?</li>
        <li>How many visits are expected on Friday?</li>
    </ul>
    """, unsafe_allow_html=True)

    user_question = st.text_input("Enter your question here")

    if user_question and forecast:
        df = pd.DataFrame(forecast)
        df["date_obj"] = pd.to_datetime(df["date"])

        response = ""

        if "highest" in user_question.lower():
            row = df.loc[df["predicted_visits"].idxmax()]
            response = f"The highest predicted ER visits are on **{row['date_obj'].strftime('%A, %B %d')}** with **{row['predicted_visits']} visits**."
        
        elif "lowest" in user_question.lower():
            row = df.loc[df["predicted_visits"].idxmin()]
            response = f"The lowest predicted ER visits are on **{row['date_obj'].strftime('%A, %B %d')}** with **{row['predicted_visits']} visits**."

        elif "friday" in user_question.lower():
            friday = df[df["date_obj"].dt.day_name() == "Friday"]
            if not friday.empty:
                row = friday.iloc[0]
                response = f"**{row['predicted_visits']} ER visits** are predicted on **Friday, {row['date']}**."
            else:
                response = "I couldn’t find a forecast for Friday."

        else:
            response = "Sorry, I couldn't interpret that. Try asking about the highest or lowest forecast day."

        st.markdown(f"<div style='padding: 1rem; background-color: #f1f3f6; border-radius: 8px;'>{response}</div>", unsafe_allow_html=True)

    elif user_question:
        st.warning("No forecast data available.")

st.markdown("---")

with tab4:
    st.subheader("Export Forecast Data")
    export_format = st.selectbox("Export Format", ["CSV Data", "JSON Data"])

    if forecast:
        if export_format == "CSV Data":
            df = pd.DataFrame([
                {
                    "date": d["date"],
                    "predicted_visits": d["predicted_visits"],
                    "yhat_lower": d["confidence_lower"],
                    "yhat_upper": d["confidence_upper"],
                    "risk_level": d["risk_level"],
                    "contributing_factors": json.dumps(d.get("contributing_factors", []))
                }
                for d in forecast
            ])
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"erlyalert_forecast_{selected_county}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        elif export_format == "JSON Data":
            json_data = json.dumps(forecast, indent=2)
            st.download_button(
                label="🧾 Download JSON",
                data=json_data,
                file_name=f"erlyalert_forecast_{selected_county}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    else:
        st.info("No forecast data available to export.")

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>© 2025 ERLyAlert. All rights reserved. Emergency Room Load Prediction System.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
with tab5:
    st.header("Help & Information")

    st.markdown("""
    ### About ERLyAlert
    
    ERLyAlert is an advanced forecasting system that predicts Emergency Room visit volumes to help hospitals prepare for surges and optimize staffing.

    ### How to use this dashboard:
    1. **Select a region**
    2. Adjust the **forecast days**
    3. Toggle **visualization options**
    4. Use **filters** to focus on specific risk levels
    5. Ask questions about the forecast

    ### Understanding Risk Levels:
    - **Low Risk** (Green): Normal ER conditions expected
    - **Medium Risk** (Orange): Higher than average volume
    - **High Risk** (Red): Surge conditions likely


    """)
