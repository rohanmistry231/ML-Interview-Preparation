import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Bias-Variance Tradeoff
# This script demonstrates the bias-variance tradeoff using polynomial regression.

# Tasks:
# 1. Generate synthetic non-linear data.
# 2. Train polynomial regression models with varying degrees.
# 3. Evaluate training and testing errors.
# 4. Visualize bias-variance tradeoff.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train models with varying polynomial degrees
degrees = [1, 3, 10]
train_errors = []
test_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Step 4: Visualize bias-variance tradeoff
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker='o', label='Training Error', color='blue')
plt.plot(degrees, test_errors, marker='o', label='Testing Error', color='red')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.savefig('bias_variance_tradeoff.png')
plt.close()