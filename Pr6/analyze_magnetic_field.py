import pandas as pd
import matplotlib.pyplot as plt

file_paths = [
    ("Raw Data_10.csv", "10 см"),
    ("Raw Data_1.csv", "50 см"),
    ("Raw Data_50.csv", "1 м")
]

dfs = []
device_name = "Ноутбук"

for path, distance in file_paths:
    df = pd.read_csv(path)
    df['Відстань'] = distance
    df['Пристрій'] = device_name
    dfs.append(df)

# Об'єдную в один датафрейм
df_all = pd.concat(dfs, ignore_index=True)

# усереднення абсолютного магнітного поля по відстані
avg_by_distance = df_all.groupby('Відстань')['Absolute field (µT)'].mean().reset_index()

distance_order = ["10 см", "50 см", "1 м"]
avg_by_distance['Відстань'] = pd.Categorical(avg_by_distance['Відстань'], categories=distance_order, ordered=True)
avg_by_distance = avg_by_distance.sort_values('Відстань')

# вивід таблиці з усередненими значеннями
print(avg_by_distance)

# побудова графіка
plt.figure(figsize=(8, 5))
plt.plot(avg_by_distance['Відстань'], avg_by_distance['Absolute field (µT)'], marker='o')
plt.title(f"Середній рівень магнітного поля біля пристрою: {device_name}")
plt.xlabel("Відстань")
plt.ylabel("Середнє магнітне поле (µT)")
plt.grid(True)
plt.tight_layout()
plt.show()
