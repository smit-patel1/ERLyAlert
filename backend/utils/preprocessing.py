import pandas as pd
from datetime import datetime

def expand_monthly_to_daily(df):
    df.columns = df.columns.str.strip().str.replace("\u00a0", " ").str.replace("\xa0", " ")  # Clean whitespace
    print("Cleaned Columns:", df.columns.tolist())

    for col in df.columns:
        if "Month" in col and "of" in col:
            df.rename(columns={col: "Month"}, inplace=True)
            break

    df['date'] = pd.to_datetime(df['Month'], format='%B %Y')

    expanded_rows = []

    for _, row in df.iterrows():
        start_date = row['date']
        days_in_month = (start_date + pd.offsets.MonthEnd(0)).day
        base_daily = row['Total ED Visits'] / days_in_month

        for i in range(days_in_month):
            current_date = start_date + pd.Timedelta(days=i)
            weekday = current_date.weekday()
            multiplier = {
                0: 1.15,
                1: 1.05,
                2: 0.95,
                3: 1.00,
                4: 1.00,
                5: 1.10,
                6: 1.20
            }[weekday]
            visits = round(base_daily * multiplier)
            expanded_rows.append({
                'ds': current_date,
                'county': row['County'],
                'y': visits
            })

    return pd.DataFrame(expanded_rows)

def preprocess_er_visits_from_new_data(input_path, output_path):
    df = pd.read_csv(input_path)
    daily_df = expand_monthly_to_daily(df)
    daily_df = daily_df.sort_values(by=["ds", "county"])
    daily_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_er_visits_from_new_data(
        "data/NC_ER_DATA.csv",
        "data/processed_er_data.csv"
    )
