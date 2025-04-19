import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib.pyplot as plt
from backend.models.evaluate_prophet import evaluate_prophet
from backend.models.evaluate_hybrid import evaluate_hybrid

df_prophet, mae_prophet = evaluate_prophet()
df_hybrid, mae_hybrid = evaluate_hybrid()

plt.figure(figsize=(12, 6))

plt.plot(df_hybrid["date"], df_hybrid["y"], label="Actual", color="black", linewidth=2)
plt.plot(df_hybrid["date"], df_hybrid["yhat_prophet"], label=f"Prophet (MAE: {mae_prophet:.2f})", linestyle="--")
plt.plot(df_hybrid["date"], df_hybrid["yhat_hybrid"], label=f"Hybrid (MAE: {mae_hybrid:.2f})", linestyle="-.")

plt.title("Actual vs Prophet vs Hybrid Forecast")
plt.xlabel("Date")
plt.ylabel("ER Visits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
