import pyreadr
import pandas as pd

result = pyreadr.read_r("data/5v_cleandf.rdata")

df = list(result.values())[0]

print("Shape:", df.shape)
print(df.head())

df.to_csv("data/converted_er_data.csv", index=False)
print("Saved to: data/converted_er_data.csv")
