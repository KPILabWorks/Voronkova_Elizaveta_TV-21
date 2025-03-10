import pandas as pd
import numpy as np

chunksize = 100_000

output_file = "processed_data.csv"

# очистка файлу
with open(output_file, "w") as f:
    f.write("")

for chunk in pd.read_csv("data_2019-03.csv", chunksize=chunksize):
    # залишаю тільки ті, де passenger_count > 1
    chunk = chunk[chunk["passenger_count"] > 1]

    # групую
    chunk["passenger_category"] = np.select(
        [
            chunk["passenger_count"] == 2,
            chunk["passenger_count"] <= 4,
            chunk["passenger_count"] > 4
        ],
        ["Couple", "Small Group", "Large Group"],
    )

    # записую шматок у файл
    chunk.to_csv(output_file, mode="a", index=False, header=False)

print("Дані збережено у 'processed_data.csv'")
