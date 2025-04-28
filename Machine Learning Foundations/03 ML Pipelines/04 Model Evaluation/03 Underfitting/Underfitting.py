import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Underfitting
# This script demonstrates underfitting using a low-degree polynomial regression.

# Tasks:
# 1. Generate synthetic non-linear data.
# 2. Train a low-degree polynomial regression model.
# 3. Compare with a better-fitting model.
# 4. Evaluate training and testing errors.
# 5. Visualize underfitting.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train models
# Underfit model (degree 1)
poly_underfit = PolynomialFeatures(degree=1)
X_train_underfit = poly_underfit.fit_transform(X_train)
X_test_underfit = poly_underfit.transform(X_test)
model_underfit = LinearRegression()
model_underfit.fit(X_train_underfit, y_train)
y_pred_underfit = model_underfit.predict(X_test_underfit)
mse_underfit = mean_squared_error(y_test, y_pred_underfit)

# Better model (degree 3)
poly_better = PolynomialFeatures(degree=3)
X_train_better = poly_better.fit_transform(X_train)
X_test_better = poly_better.transform(X_test)
model_better = LinearRegression()
model_better.fit(X_train_better, y_train)
y_pred_better = model_better.predict(X_test_better)
mse_better = mean_squared_error(y_test, y_pred_better)

print(f'MSE (Underfit, degree=1): {mse_underfit:.2f}')
print(f'MSE (Better, degree=3): {mse_better:.2f}')

# Step 4: Visualize underfitting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot_underfit = model_underfit.predict(poly_underfit.transform(X_plot))
y_plot_better = model_better.predict(poly_better.transform(X_plot))

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_plot, y_plot_underfit, color='red', label='Underfit Model (degree=1)')
plt.plot(X_plot, y_plot_better, color='purple', label='Better Model (degree=3)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Underfitting: Underfit vs Better Model')
plt.legend()
plt.grid(True)
plt.savefig('underfitting.png')
plt.close()