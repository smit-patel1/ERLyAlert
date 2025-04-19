import pandas as pd

df = pd.read_csv("data/processed_er_data.csv")

# Clean any whitespace or special characters
df["county"] = df["county"].astype(str).str.strip()

# Print unique counties
print("Available counties:", df["county"].unique())

# Check row count for each
for county in df["county"].unique():
    count = len(df[df["county"] == county])
    print(f"{county}: {count} rows")
