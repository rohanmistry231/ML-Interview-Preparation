import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Regression Evaluation Metrics
# This script demonstrates MSE, MAE, and R² for regression.

# Tasks:
# 1. Generate synthetic regression data.
# 2. Split data into training and testing sets.
# 3. Train a Linear Regression model.
# 4. Calculate MSE, MAE, and R².
# 5. Visualize actual vs predicted values.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + 1 + np.random.randn(100) * 2

# Convert to DataFrame
data = pd.DataFrame({'X': X.flatten(), 'y': y})

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions and calculate metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Step 5: Visualize actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y demolition_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Metrics: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.savefig('regression_metrics.png')
plt.close()