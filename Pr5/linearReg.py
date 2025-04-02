import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Створення синтетичних даних
np.random.seed(42)
n_samples = 1_000_000  # 1 млн зразків
n_features = 50  # 50 ознак
X = np.random.rand(n_samples, n_features)
y = (2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] +
     np.sum(X[:, 3:], axis=1) * 0.1 +
     np.random.normal(0, 0.1, n_samples))

columns = [f'Feature{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df['Target'] = y

X = df[columns]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання моделі
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Прогнозування
y_pred = model.predict(X_test_scaled)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Коефіцієнти моделі:")
for feature, coef in zip(columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Вільний член (intercept): {model.intercept_:.4f}")
print(f"Середньоквадратична помилка (MSE): {mse:.4f}")
print(f"Коефіцієнт детермінації (R²): {r2:.4f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.title('Фактичні vs Прогнозовані значення')
plt.tight_layout()
plt.show()

# Прогноз для нових даних
new_data = np.random.rand(1, n_features)
new_data_df = pd.DataFrame(new_data, columns=columns)
new_data_scaled = scaler.transform(new_data_df)
prediction = model.predict(new_data_scaled)
print(f"\nПрогноз для нових даних: {prediction[0]:.4f}")