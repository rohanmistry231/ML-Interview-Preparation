import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Overfitting
# This script demonstrates overfitting using a high-degree polynomial regression.

# Tasks:
# 1. Generate synthetic non-linear data.
# 2. Train a high-degree polynomial regression model.
# 3. Compare with a simpler model.
# 4. Evaluate training and testing errors.
# 5. Visualize overfitting.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train models
# Simple model (degree 3)
poly_simple = PolynomialFeatures(degree=3)
X_train_simple = poly_simple.fit_transform(X_train)
X_test_simple = poly_simple.transform(X_test)
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)

# Overfit model (degree 15)
poly_overfit = PolynomialFeatures(degree=15)
X_train_overfit = poly_overfit.fit_transform(X_train)
X_test_overfit = poly_overfit.transform(X_test)
model_overfit = LinearRegression()
model_overfit.fit(X_train_overfit, y_train)
y_pred_overfit = model_overfit.predict(X_test_overfit)
mse_overfit = mean_squared_error(y_test, y_pred_overfit)

print(f'MSE (Simple, degree=3): {mse_simple:.2f}')
print(f'MSE (Overfit, degree=15): {mse_overfit:.2f}')

# Step 4: Visualize overfitting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot_simple = model_simple.predict(poly_simple.transform(X_plot))
y_plot_overfit = model_overfit.predict(poly_overfit.transform(X_plot))

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_plot, y_plot_simple, color='red', label='Simple Model (degree=3)')
plt.plot(X_plot, y_plot_overfit, color='purple', label='Overfit Model (degree=15)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Overfitting: Simple vs Overfit Model')
plt.legend()
plt.grid(True)
plt.savefig('overfitting.png')
plt.close()