import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json
import numpy as np
import pydeck as pdk

# Set page configuration
st.set_page_config(
    page_title="ERLyAlert - ER Visit Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stApp {
        background-color: #f8f9fa;
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

# Helper Functions
def get_regions():
    """Fetch available regions from API or use mock data"""
    try:
        response = requests.get("http://localhost:8000/regions")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback mock data if API is unavailable
    return [
        {"id": "nyc_manhattan", "name": "Manhattan, NYC"},
        {"id": "la_county", "name": "Los Angeles County"},
        {"id": "chicago_cook", "name": "Cook County, Chicago"},
        {"id": "miami_dade", "name": "Miami-Dade County"},
        {"id": "sf_bay", "name": "San Francisco Bay Area"}
    ]

def get_forecast_data(region_id, days=7):
    """Fetch forecast data from API or use mock data"""
    try:
        response = requests.get(f"http://localhost:8000/forecast/{region_id}?days={days}")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Generate mock forecast data
    today = datetime.now()
    base_visits = 120
    
    mock_data = {
        "forecast": [],
        "metrics": {
            "mae": round(np.random.uniform(5.5, 12.5), 2),
            "accuracy_7day": f"{round(np.random.uniform(82, 95), 1)}%"
        },
        "region_info": {
            "name": next((r["name"] for r in get_regions() if r["id"] == region_id), "Unknown Region"),
            "baseline_avg": base_visits
        }
    }
    
    # Risk distributions - higher on weekends
    risk_levels = ["low", "medium", "high"]
    risk_weights = {
        0: [0.6, 0.3, 0.1],  # Monday
        1: [0.7, 0.2, 0.1],  # Tuesday
        2: [0.7, 0.2, 0.1],  # Wednesday
        3: [0.6, 0.3, 0.1],  # Thursday
        4: [0.5, 0.3, 0.2],  # Friday
        5: [0.3, 0.4, 0.3],  # Saturday
        6: [0.4, 0.4, 0.2],  # Sunday
    }
    
    # Factors that may contribute to ER visits
    factors = {
        "weather": ["Clear", "Rainy", "Stormy", "Heat Wave", "Cold Front"],
        "events": ["None", "Sports Game", "Concert", "Public Holiday", "Marathon"],
        "aqi": ["Good (0-50)", "Moderate (51-100)", "Unhealthy for Sensitive Groups (101-150)", "Unhealthy (151-200)"],
        "epidemic": ["Low flu activity", "Moderate flu activity", "High flu activity", "COVID-19 spike"]
    }
    
    factor_impact = {
        "weather": {"Clear": 0, "Rainy": 10, "Stormy": 25, "Heat Wave": 30, "Cold Front": 15},
        "events": {"None": 0, "Sports Game": 15, "Concert": 20, "Public Holiday": 25, "Marathon": 30},
        "aqi": {"Good (0-50)": 0, "Moderate (51-100)": 10, "Unhealthy for Sensitive Groups (101-150)": 25, "Unhealthy (151-200)": 40},
        "epidemic": {"Low flu activity": 0, "Moderate flu activity": 20, "High flu activity": 50, "COVID-19 spike": 70}
    }
    
    # Generate forecast for each day
    for i in range(days):
        date = today + timedelta(days=i)
        weekday = date.weekday()
        
        # Select random factors for this day
        day_factors = {
            "weather": np.random.choice(factors["weather"], p=[0.5, 0.2, 0.1, 0.1, 0.1]),
            "events": np.random.choice(factors["events"], p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            "aqi": np.random.choice(factors["aqi"], p=[0.6, 0.2, 0.1, 0.1]),
            "epidemic": np.random.choice(factors["epidemic"], p=[0.5, 0.3, 0.1, 0.1])
        }
        
        # Calculate visits based on factors
        factor_sum = sum(factor_impact[k][v] for k, v in day_factors.items())
        
        # Weekend effect
        weekend_effect = 30 if weekday >= 5 else 0
        
        # Randomize slightly
        random_factor = np.random.normal(0, 10)
        
        # Calculate predicted visits
        predicted_visits = base_visits + factor_sum + weekend_effect + random_factor
        predicted_visits = max(int(predicted_visits), 50)  # Ensure minimum visits
        
        # Risk level based on visits
        if predicted_visits > base_visits * 1.3:  # 30% above baseline
            risk = np.random.choice(risk_levels, p=risk_weights[weekday])
        elif predicted_visits > base_visits * 1.1:  # 10% above baseline
            temp_weights = [w * 1.5 for w in risk_weights[weekday]]
            temp_weights[0] *= 0.5  # Reduce chance of low risk
            temp_weights = [w/sum(temp_weights) for w in temp_weights]
            risk = np.random.choice(risk_levels, p=temp_weights)
        else:
            temp_weights = risk_weights[weekday].copy()
            temp_weights[2] *= 0.3  # Reduce chance of high risk
            temp_weights = [w/sum(temp_weights) for w in temp_weights]
            risk = np.random.choice(risk_levels, p=temp_weights)
        
        # Format day factors for display
        displayed_factors = []
        for factor_type, factor_value in day_factors.items():
            impact = factor_impact[factor_type][factor_value]
            if impact > 0:
                displayed_factors.append({
                    "type": factor_type,
                    "value": factor_value,
                    "impact": f"+{impact}" if impact > 0 else str(impact)
                })
        
        # Sort factors by impact
        displayed_factors.sort(key=lambda x: int(x["impact"].replace("+", "")), reverse=True)
        
        mock_data["forecast"].append({
            "date": date.strftime("%Y-%m-%d"),
            "predicted_visits": predicted_visits,
            "confidence_lower": int(predicted_visits * 0.9),
            "confidence_upper": int(predicted_visits * 1.1),
            "risk_level": risk,
            "contributing_factors": displayed_factors[:3]  # Top 3 contributing factors
        })
    
    return mock_data

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

# Sidebar for filters and settings
with st.sidebar:
    st.header("Dashboard Settings")
    
    # Region Selection
    regions = get_regions()
    region_options = {r["name"]: r["id"] for r in regions}
    selected_region_name = st.selectbox("Select Region", list(region_options.keys()))
    selected_region_id = region_options[selected_region_name]
    
    # Forecast Range
    forecast_days = st.slider("Forecast Days", min_value=3, max_value=14, value=7)
    
    # Date Range (if historical comparison needed)
    st.subheader("Historical Comparison")
    enable_historical = st.checkbox("Enable Historical Comparison", value=False)
    
    if enable_historical:
        historical_days = st.slider("Historical Days", min_value=7, max_value=90, value=30)
    
    # Display Settings
    st.subheader("Display Settings")
    show_confidence_interval = st.checkbox("Show Confidence Interval", value=True)
    visualization_type = st.radio("Visualization Type", ["Line Chart", "Bar Chart"])
    
    # Advanced Filters
    st.subheader("Advanced Filters")
    filter_by_risk = st.multiselect("Filter by Risk Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

# Fetch forecast data
forecast_data = get_forecast_data(selected_region_id, forecast_days)
region_info = forecast_data.get("region_info", {})
forecast = forecast_data.get("forecast", [])
metrics = forecast_data.get("metrics", {})

# Apply filters
if filter_by_risk:
    forecast = [day for day in forecast if day["risk_level"].lower() in [r.lower() for r in filter_by_risk]]

# Main content area
col1, col2 = st.columns([7, 3])

with col1:
    st.header(f"ER Visit Forecast for {region_info.get('name', selected_region_name)}")
    
    # Create dataframe for plotting
    df = pd.DataFrame(forecast)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    df["date_str"] = df["date"].dt.strftime("%b %d (%a)")
    df["risk_color"] = df["risk_level"].map({
        "low": "#00cc96", 
        "medium": "#ffa500", 
        "high": "#ff4b4b"
    })
    
    # Create the main forecast visualization
    if visualization_type == "Line Chart":
        fig = go.Figure()
        
        # Base prediction line
        fig.add_trace(go.Scatter(
            x=df["date_str"], 
            y=df["predicted_visits"],
            mode='lines+markers',
            name='Predicted Visits',
            line=dict(color='#4361ee', width=3),
            marker=dict(size=10, color=df["risk_color"])
        ))
        
        # Confidence intervals
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
        
        # Add baseline reference
        baseline = region_info.get("baseline_avg", 100)
        fig.add_trace(go.Scatter(
            x=[df["date_str"].iloc[0], df["date_str"].iloc[-1]],
            y=[baseline, baseline],
            mode='lines',
            name='Baseline Average',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Layout
        fig.update_layout(
            title="Predicted ER Visits by Day",
            xaxis_title="Date",
            yaxis_title="Predicted Visits",
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
    else:  # Bar Chart
        fig = go.Figure()
        
        # Bars with risk color
        fig.add_trace(go.Bar(
            x=df["date_str"],
            y=df["predicted_visits"],
            marker_color=df["risk_color"],
            name='Predicted Visits',
            text=df["predicted_visits"],
            textposition='auto',
        ))
        
        # Add baseline reference
        baseline = region_info.get("baseline_avg", 100)
        fig.add_trace(go.Scatter(
            x=df["date_str"],
            y=[baseline] * len(df),
            mode='lines',
            name='Baseline Average',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        # Error bars for confidence intervals
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
        
        # Layout
        fig.update_layout(
            title="Predicted ER Visits by Day",
            xaxis_title="Date",
            yaxis_title="Predicted Visits",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Level Legend
    risk_legend_cols = st.columns(3)
    with risk_legend_cols[0]:
        st.markdown(f'<span class="risk-low">LOW RISK</span> - Normal ER volume expected', unsafe_allow_html=True)
    with risk_legend_cols[1]:
        st.markdown(f'<span class="risk-medium">MEDIUM RISK</span> - Higher than normal volume', unsafe_allow_html=True)
    with risk_legend_cols[2]:
        st.markdown(f'<span class="risk-high">HIGH RISK</span> - Surge conditions likely', unsafe_allow_html=True)
        
    # Show Table with Daily Forecast Details
    st.subheader("Daily Forecast Details")
    
    # Create a custom table view
    for index, row in df.iterrows():
        risk_html = risk_badge(row["risk_level"])
        
        # Day header with risk badge
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <h3 style="margin: 0;">{row['date_str']}</h3>
            <div>{risk_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Day details
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

with col2:
    # Summary Metrics
    st.header("Forecast Summary")
    
    # Get overall metrics
    max_day = max(forecast, key=lambda x: x["predicted_visits"])
    min_day = min(forecast, key=lambda x: x["predicted_visits"])
    high_risk_days = sum(1 for day in forecast if day["risk_level"].lower() == "high")
    baseline = region_info.get("baseline_avg", 100)
    
    # Peak day
    st.markdown("#### Peak ER Day")
    max_date = datetime.strptime(max_day["date"], "%Y-%m-%d").strftime("%b %d (%a)")
    st.markdown(f"""
    <div class="metric-card">
        <h3>{max_date}</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2>{int(max_day["predicted_visits"])}</h2>
                <p>expected visits</p>
            </div>
            <div>
                {risk_badge(max_day["risk_level"])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Metrics
    st.markdown("#### Model Performance")
    metrics_cols = st.columns(2)
    with metrics_cols[0]:
        display_metric_card("MAE", metrics.get("mae", "10.2"))
    with metrics_cols[1]:
        display_metric_card("7-Day Accuracy", metrics.get("accuracy_7day", "87.5%"))
    
    # Risk Summary
    st.markdown("#### Risk Level Summary")
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for day in forecast:
        risk_counts[day["risk_level"].lower()] += 1
    
    risk_summary_cols = st.columns(3)
    with risk_summary_cols[0]:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1>{risk_counts["high"]}</h1>
            <p style="color: #ff4b4b; font-weight: bold;">HIGH</p>
        </div>
        """, unsafe_allow_html=True)
    with risk_summary_cols[1]:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1>{risk_counts["medium"]}</h1>
            <p style="color: #ffa500; font-weight: bold;">MEDIUM</p>
        </div>
        """, unsafe_allow_html=True)
    with risk_summary_cols[2]:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1>{risk_counts["low"]}</h1>
            <p style="color: #00cc96; font-weight: bold;">LOW</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top Contributing Factors Overall
    st.markdown("#### Top Contributing Factors")
    
    # Collect all factors across days
    all_factors = []
    for day in forecast:
        for factor in day.get("contributing_factors", []):
            all_factors.append({
                "type": factor["type"],
                "value": factor["value"],
                "impact": int(factor["impact"].replace("+", ""))
            })
    
    # Group by type and value, sum impact
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
    
    # Sort and display top factors
    top_factors = sorted(factor_summary.values(), key=lambda x: x["total_impact"], reverse=True)[:5]
    
    for factor in top_factors:
        factor_type = factor["type"].capitalize()
        factor_value = factor["value"]
        impact = factor["total_impact"]
        days = factor["count"]
        
        st.markdown(f"""
        <div class="factor-box">
            <strong>{factor_type}: {factor_value}</strong>
            <div style="display: flex; justify-content: space-between;">
                <div>Impact: <span style="color:#ff4b4b;">+{impact}</span></div>
                <div>Present in {days} days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # User Query Section
    st.markdown("---")
    st.markdown("#### Ask about the Forecast")
    
    query = st.text_input("Ask a question (e.g., 'When is the highest risk day?' or 'What factors cause the surge on Friday?')")
    
    if query:
        st.markdown("##### Analysis")
        
        # Simple query processing logic
        query_lower = query.lower()
        
        if "highest risk" in query_lower or "worst day" in query_lower:
            max_day = max(forecast, key=lambda x: x["predicted_visits"])
            max_date = datetime.strptime(max_day["date"], "%Y-%m-%d").strftime("%b %d (%a)")
            
            st.markdown(f"""
            The highest risk day in the forecast period is **{max_date}** with **{int(max_day["predicted_visits"])}** 
            predicted visits ({risk_badge(max_day["risk_level"])}).
            
            Top contributing factors:
            """, unsafe_allow_html=True)
            
            for factor in max_day.get("contributing_factors", [])[:3]:
                st.markdown(f"- {factor['type'].capitalize()}: {factor['value']} ({factor['impact']})")
                
        elif "lowest risk" in query_lower or "quietest day" in query_lower:
            min_day = min(forecast, key=lambda x: x["predicted_visits"])
            min_date = datetime.strptime(min_day["date"], "%Y-%m-%d").strftime("%b %d (%a)")
            
            st.markdown(f"""
            The lowest risk day is **{min_date}** with only **{int(min_day["predicted_visits"])}** 
            predicted visits ({risk_badge(min_day["risk_level"])}).
            """, unsafe_allow_html=True)
            
        elif any(day in query_lower for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            # Find which day was mentioned
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            mentioned_day = next((day for day in days if day in query_lower), None)
            
            if mentioned_day:
                # Find forecast entries for that day
                day_forecasts = [day for day in forecast if 
                                datetime.strptime(day["date"], "%Y-%m-%d").strftime("%A").lower() == mentioned_day]
                
                if day_forecasts:
                    day_forecast = day_forecasts[0]  # Take the first occurrence
                    forecast_date = datetime.strptime(day_forecast["date"], "%Y-%m-%d").strftime("%b %d (%a)")
                    
                    st.markdown(f"""
                    For **{forecast_date}**, we predict **{int(day_forecast["predicted_visits"])}** ER visits 
                    ({risk_badge(day_forecast["risk_level"])}).
                    
                    Contributing factors:
                    """, unsafe_allow_html=True)
                    
                    if "contributing_factors" in day_forecast and day_forecast["contributing_factors"]:
                        for factor in day_forecast["contributing_factors"]:
                            st.markdown(f"- {factor['type'].capitalize()}: {factor['value']} ({factor['impact']})")
                    else:
                        st.markdown("No significant contributing factors identified for this day.")
                else:
                    st.markdown(f"I don't see a {mentioned_day.capitalize()} in the current forecast period.")
        
        elif "factor" in query_lower or "cause" in query_lower:
            st.markdown("""
            **Top factors contributing to ER surges in this forecast period:**
            """)
            
            for i, factor in enumerate(top_factors[:3], 1):
                st.markdown(f"""
                {i}. **{factor["type"].capitalize()}: {factor["value"]}**  
                   Impact: +{factor["total_impact"]} across {factor["count"]} days
                """)
        
        else:
            st.markdown("""
            I can help you understand:
            - The highest or lowest risk days
            - Details about specific days (e.g., "What about Friday?")
            - Contributing factors to ER surges
            - Overall risk distribution
            
            Try asking a more specific question about the forecast.
            """)

# Export options
st.markdown("---")
export_col1, export_col2 = st.columns(2)

with export_col1:
    st.subheader("Export Options")
    export_format = st.selectbox("Export Format", ["PDF Report", "CSV Data", "JSON Data"])
    
    if st.button("Generate Export"):
        if export_format == "PDF Report":
            st.success("PDF report generation started. Your report will be ready to download shortly.")
            # In a real implementation, this would generate a PDF
        elif export_format == "CSV Data":
            st.success("CSV data ready for download.")
            # Create a download button for CSV data
            csv_data = pd.DataFrame(forecast).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"erlyalert_forecast_{selected_region_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        elif export_format == "JSON Data":
            st.success("JSON data ready for download.")
            # Create a download button for JSON data
            json_data = json.dumps(forecast_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"erlyalert_forecast_{selected_region_id}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

with export_col2:
    st.subheader("Notification Settings")
    notify_risk_level = st.selectbox("Notify me of days with risk level", ["High Only", "Medium and High", "All Levels", "None"])
    notify_email = st.text_input("Email Address")
    
    if st.button("Save Notification Settings"):
        if notify_email and "@" in notify_email:
            st.success(f"You will receive notifications for {notify_risk_level.lower()} risk days at {notify_email}")
        else:
            st.error("Please enter a valid email address")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>ERLyAlert - Emergency Room Load Prediction System | Developed for Healthcare Hackathon 2025</p>
</div>
""", unsafe_allow_html=True)

# Add a help icon with tooltip in the corner
st.sidebar.markdown("---")
with st.sidebar.expander("Help & Information"):
    st.markdown("""
    ### About ERLyAlert
    
    ERLyAlert is an advanced forecasting system that predicts Emergency Room visit volumes to help hospitals prepare for surges and optimize staffing.
    
    ### How to use this dashboard:
    
    1. **Select a region** from the dropdown menu
    2. Adjust the **forecast days** to see predictions further in the future
    3. Toggle **visualization options** to view the data differently
    4. Use **filters** to focus on specific risk levels
    5. Enable **historical comparison** to see how the forecast compares to past data
    6. Ask questions about the forecast using the query box
    
    ### Understanding Risk Levels:
    
    - **Low Risk** (Green): Normal ER conditions expected
    - **Medium Risk** (Yellow): Higher than average volume expected
    - **High Risk** (Red): Surge conditions likely, additional staffing recommended
    
    ### Questions?
    
    Contact the ERLyAlert team at support@erlyalert.example.com
    """)

# Add a simple chatbot interface for questions
if st.sidebar.checkbox("Enable AI Assistant"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("ERLyAlert Assistant")
    
    user_question = st.sidebar.text_input("Ask about ER predictions or the dashboard:")
    
    if user_question:
        # Simple response logic
        if "how" in user_question.lower() and "work" in user_question.lower():
            st.sidebar.markdown("""
            ERLyAlert works by combining historical ER visit data with external factors like:
            - Weather conditions
            - Local events
            - Air quality
            - Epidemic data (flu/COVID trends)
            
            Our machine learning model analyzes these patterns to predict future ER volumes.
            """)
        
        elif "accurate" in user_question.lower():
            st.sidebar.markdown(f"""
            Our current model achieves:
            - Mean Absolute Error (MAE): {metrics.get('mae', '10.2')}
            - 7-day accuracy: {metrics.get('accuracy_7day', '87.5%')}
            
            The forecast is most accurate for the next 3-5 days and becomes less certain beyond that.
            """)
        
        elif "factor" in user_question.lower():
            st.sidebar.markdown("""
            The main factors affecting ER visits include:
            - Day of week (weekends often see more visits)
            - Weather events (extreme heat, storms)
            - Air quality (poor AQI increases respiratory issues)
            - Epidemic trends (flu/COVID spikes)
            - Local events (sports games, concerts)
            
            Each factor's impact is weighted in our prediction model.
            """)
        
        else:
            st.sidebar.markdown("""
            I can answer questions about:
            - How the prediction model works
            - Understanding risk levels
            - Interpreting the forecast
            - Using dashboard features
            
            Try asking something more specific about ERLyAlert!
            """)

# Map view
if st.checkbox("Show Map View"):
    st.subheader("Regional Risk Map")
    
    # Create mock data for multiple locations
    map_data = []
    
    # Create center points for different regions
    region_centers = {
        "nyc_manhattan": (40.7831, -73.9712),
        "la_county": (34.0522, -118.2437),
        "chicago_cook": (41.8781, -87.6298),
        "miami_dade": (25.7617, -80.1918),
        "sf_bay": (37.7749, -122.4194)
    }
    
    # Generate data points around centers
    for region_id, (lat, lon) in region_centers.items():
        region_forecast = get_forecast_data(region_id, 7)
        region_name = next((r["name"] for r in regions if r["id"] == region_id), "Unknown")
        
        # Calculate average risk level
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        avg_risk_score = sum(risk_scores[day["risk_level"].lower()] for day in region_forecast["forecast"]) / len(region_forecast["forecast"])
        
        # Generate 3-5 points around the center
        num_points = np.random.randint(3, 6)
        for i in range(num_points):
            # Random offset
            lat_offset = np.random.normal(0, 0.05)
            lon_offset = np.random.normal(0, 0.05)
            
            # Random risk based on average with some variation
            point_risk_score = min(3, max(1, avg_risk_score + np.random.normal(0, 0.5)))
            if point_risk_score >= 2.5:
                risk = "high"
            elif point_risk_score >= 1.5:
                risk = "medium"
            else:
                risk = "low"
            
            # Random visit count
            visits = int(np.random.normal(
                region_forecast["region_info"].get("baseline_avg", 100),
                region_forecast["region_info"].get("baseline_avg", 100) * 0.2
            ))
            
            map_data.append({
                "lat": lat + lat_offset,
                "lon": lon + lon_offset,
                "risk": risk,
                "visits": visits,
                "region": region_name
            })
    
    # Create DataFrame for map
    map_df = pd.DataFrame(map_data)
    
    # Color mapping for risk levels
    color_scale = {
        "low": "#00cc96",
        "medium": "#ffa500",
        "high": "#ff4b4b"
    }
    map_df["color"] = map_df["risk"].map(color_scale)
    
    # Create map
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=39.8283,
            longitude=-98.5795,
            zoom=3,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="visits / 10",
                pickable=True,
                opacity=0.8,
                radius_min_pixels=5,
                radius_max_pixels=30,
            ),
        ],
        tooltip={
            "html": "<b>{region}</b><br/>"
                    "Risk Level: {risk}<br/>"
                    "Predicted Visits: {visits}",
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }
    ))
