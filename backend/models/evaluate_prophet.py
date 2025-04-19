import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

def evaluate_prophet():
    df = pd.read_csv("data/processed_er_data.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    model = Prophet()
    model.fit(df)

    forecast = model.predict(df[["ds"]])
    df["yhat"] = forecast["yhat"]
    mae = mean_absolute_error(df["y"], df["yhat"])

    df = df[["ds", "y", "yhat"]].rename(columns={"ds": "date"})
    return df, mae

if __name__ == "__main__":
    df_prophet, mae_prophet = evaluate_prophet()
    print(f"Prophet MAE: {mae_prophet:.2f}")
