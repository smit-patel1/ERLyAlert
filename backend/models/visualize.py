import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from prophet import Prophet

COUNTY = "Mecklenburg"
SEQLEN = 7

df = pd.read_csv("datasets/processed_er_data.csv")
df = df[df["county"] == COUNTY][["ds", "y"]].dropna()
df["ds"] = pd.to_datetime(df["ds"])

prophet = joblib.load(f"trained_models/prophet_model_{COUNTY}.pkl")
lstm    = tf.keras.models.load_model(f"trained_models/lstm_model_{COUNTY}.keras")
scaler  = joblib.load(f"trained_models/scaler_{COUNTY}.pkl")

forecast_df = prophet.predict(df[["ds"]])
merged = df.merge(forecast_df[["ds", "yhat"]], on="ds", how="inner")
residuals = merged["y"] - merged["yhat"]

if len(residuals) >= SEQLEN:
    input_seq = residuals.values[-SEQLEN:].reshape(-1, 1)
    input_scaled = scaler.transform(input_seq)
    input_reshaped = input_scaled.reshape(1, SEQLEN, 1)
    residual_pred_scaled = lstm.predict(input_reshaped, verbose=0)
    residual_pred = scaler.inverse_transform(residual_pred_scaled)[0][0]
else:
    residual_pred = 0.0

# Build tail with real y and predicted yhat
tail = merged.tail(SEQLEN).copy()
tail["yhat_hybrid"] = tail["yhat"] + residual_pred

# Print comparison
for _, row in tail.iterrows():
    print(f"{row['ds'].date()} A:{row['y']:.1f} P:{row['yhat']:.1f} H:{row['yhat_hybrid']:.1f}")

# MAE
mae_p = np.mean(np.abs(tail["y"] - tail["yhat"]))
mae_h = np.mean(np.abs(tail["y"] - tail["yhat_hybrid"]))
print(f"\nProphet MAE {mae_p:.2f} | Hybrid MAE {mae_h:.2f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(tail["ds"], tail["yhat"], "--",  label="Prophet")
plt.plot(tail["ds"], tail["yhat_hybrid"], "-", label="Hybrid")
plt.plot(tail["ds"], tail["y"], "o:", label="Actual")
plt.title(f"{COUNTY} – Last {SEQLEN} Days")
plt.ylabel("ER Visits")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
