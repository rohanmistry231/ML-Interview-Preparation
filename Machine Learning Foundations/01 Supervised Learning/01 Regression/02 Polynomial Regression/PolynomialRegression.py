import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Polynomial Regression
# This script demonstrates Polynomial Regression using synthetic non-linear data.

# Tasks:
# 1. Generate synthetic non-linear data (quadratic relationship).
# 2. Split data into training and testing sets.
# 3. Create and train a Polynomial Regression model (degree 2).
# 4. Make predictions and evaluate performance (MSE, R²).
# 5. Visualize the polynomial fit.

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.sort(6 * np.random.rand(100, 1) - 3, axis=0)  # Values between -3 and 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5  # Quadratic + noise

# Convert to DataFrame
data = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y.flatten(), test_size=0.2, random_state=42)

# Step 3: Create and train Polynomial Regression model
degree = 2
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_train, y_train)

# Step 4: Make predictions
X_test_sorted = np.sort(X_test, axis=0)
y_pred = polyreg.predict(X_test)
y_pred_plot = polyreg.predict(X_test_sorted)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Step 5: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test_sorted, y_pred_plot, color='red', label='Polynomial fit (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_regression.png')
plt.close()