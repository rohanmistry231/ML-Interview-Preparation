import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Polynomial Features
# This script demonstrates adding polynomial features to improve regression.

# Tasks:
# 1. Generate synthetic non-linear data.
# 2. Apply PolynomialFeatures to create polynomial terms.
# 3. Train Linear Regression models with and without polynomial features.
# 4. Evaluate performance (MSE).
# 5. Visualize regression fits.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Step 2: Apply PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Step 3: Train Linear Regression models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_poly, X_test_poly = train_test_split(X_poly, test_size=0.2, random_state=42)

# Linear model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Polynomial model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print(f'MSE (Linear): {mse_linear:.2f}')
print(f'MSE (Polynomial): {mse_poly:.2f}')

# Step 4: Visualize regression fits
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot_linear = model_linear.predict(X_plot)
y_plot_poly = model_poly.predict(poly.transform(X_plot))

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_plot_linear, color='green', label='Linear Fit')
plt.plot(X_plot, y_plot_poly, color='red', label='Polynomial Fit (degree=3)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Features: Linear vs Polynomial Regression')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_features.png')
plt.close()