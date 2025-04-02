import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

np.random.seed(42)
time = pd.date_range(start="2024-01-01", periods=1000, freq='H')
energy_consumption = np.random.normal(loc=50, scale=10, size=len(time))

anomalies = np.random.choice(len(time), size=20, replace=False)
energy_consumption[anomalies] *= np.random.uniform(2, 4, size=len(anomalies))

df = pd.DataFrame({'Time': time, 'Energy': energy_consumption})

# Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['Iso_Anomaly'] = iso_forest.fit_predict(df[['Energy']])

# Z-Score
df['Z_Score'] = zscore(df['Energy'])
df['Z_Anomaly'] = (abs(df['Z_Score']) > 3).astype(int)

# IQR
Q1 = df['Energy'].quantile(0.25)
Q3 = df['Energy'].quantile(0.75)
IQR = Q3 - Q1
df['IQR_Anomaly'] = ((df['Energy'] < (Q1 - 1.5 * IQR)) | (df['Energy'] > (Q3 + 1.5 * IQR))).astype(int)

# кількість аномалій
print("Isolation Forest anomalies:", df['Iso_Anomaly'].value_counts().to_dict())
print("Z-Score anomalies:", df['Z_Anomaly'].sum())
print("IQR anomalies:", df['IQR_Anomaly'].sum())

# Візуалізація методів
plt.figure(figsize=(14, 6))
plt.plot(df['Time'], df['Energy'], label="Energy Consumption", color='blue', alpha=0.7)

# Isolation Forest
plt.scatter(df['Time'][df['Iso_Anomaly'] == -1], df['Energy'][df['Iso_Anomaly'] == -1],
            color='red', label="Isolation Forest", marker='o', s=300, edgecolors='black')
# IQR
plt.scatter(df['Time'][df['IQR_Anomaly'] == 1], df['Energy'][df['IQR_Anomaly'] == 1],
            color='purple', label="IQR", marker='s', s=80)
# Z-Score
plt.scatter(df['Time'][df['Z_Anomaly'] == 1], df['Energy'][df['Z_Anomaly'] == 1],
            color='green', label="Z-Score", marker='x', s=120)


plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.title("Comparison of Anomaly Detection Methods")
plt.legend()
plt.xticks(rotation=45)
plt.show()