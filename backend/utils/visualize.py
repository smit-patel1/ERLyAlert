import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\processed_er_visits.csv', parse_dates=['ds'])

plt.figure(figsize=(10, 5))
df['y'].plot()
plt.title('Daily ER Visit Counts')
plt.ylabel('Visits')
plt.xlabel('Date')
plt.grid(True)
plt.show()
