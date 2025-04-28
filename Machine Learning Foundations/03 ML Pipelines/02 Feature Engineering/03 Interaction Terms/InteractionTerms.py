import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Interaction Terms
# This script demonstrates adding interaction terms to capture feature interactions.

# Tasks:
# 1. Generate synthetic data with interacting features.
# 2. Apply PolynomialFeatures to include interaction terms.
# 3. Train Linear Regression models with and without interaction terms.
# 4. Evaluate performance (MSE).
# 5. Visualize model predictions.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 5 * X[:, 0] * X[:, 1] + np.random.randn(100) * 0.1

data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
data['Target'] = y

# Step 2: Apply PolynomialFeatures for interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = poly.fit_transform(X)

# Step 3: Train Linear Regression models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_inter, X_test_inter = train_test_split(X_interaction, test_size=0.2, random_state=42)

# Linear model (no interactions)
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Interaction model
model_inter = LinearRegression()
model_inter.fit(X_train_inter, y_train)
y_pred_inter = model_inter.predict(X_test_inter)
mse_inter = mean_squared_error(y_test, y_pred_inter)

print(f'MSE (Linear): {mse_linear:.2f}')
print(f'MSE (Interaction Terms): {mse_inter:.2f}')

# Step 4: Visualize actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='green', label='Linear Predictions')
plt.scatter(y_test, y_pred_inter, color='red', label='Interaction Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Interaction Terms: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.savefig('interaction_terms.png')
plt.close()