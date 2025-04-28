import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Linear Regression
# This script demonstrates Linear Regression using a synthetic dataset to predict house prices based on size.

# Tasks:
# 1. Generate synthetic data for house sizes and prices.
# 2. Split data into training and testing sets.
# 3. Train a Linear Regression model.
# 4. Make predictions and evaluate performance (MSE, R²).
# 5. Visualize the regression line and predictions.

# Step 1: Generate synthetic data
np.random.seed(42)
house_sizes = np.random.rand(100, 1) * 200  # Size in square feet (0-200)
prices = 50 + 3 * house_sizes + np.random.randn(100, 1) * 10  # Price = 50 + 3*size + noise

# Convert to DataFrame for clarity
data = pd.DataFrame({'Size': house_sizes.flatten(), 'Price': prices.flatten()})

# Step 2: Split data
X = data[['Size']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print(f'Coefficients: {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')

# Step 5: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('House Size (sq ft)/100)')
plt.ylabel('Price ($1000)')
plt.title('Linear Regression: House Size vs Price')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression.png')
plt.close()

# Save the plot for reference
# The plot shows the regression line fitting the data, with training and testing points.