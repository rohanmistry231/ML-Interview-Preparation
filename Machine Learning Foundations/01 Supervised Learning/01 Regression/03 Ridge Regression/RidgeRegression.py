import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ridge Regression
# This script demonstrates Ridge Regression with synthetic data to handle multicollinearity.

# Tasks:
# 1. Generate synthetic data with correlated features.
# 2. Split data into training and testing sets.
# 3. Train a Ridge Regression model with regularization.
# 4. Make predictions and evaluate performance (MSE, R²).
# 5. Visualize feature coefficients.

# Step 1: Generate synthetic data
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
# Introduce multicollinearity
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.1  # Correlated feature
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.5

# Convert to DataFrame
data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(n_features)])
data['Target'] = y

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Ridge Regression model
ridge = Ridge(alpha=1.0)  # Regularization strength
ridge.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = ridge.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print('Feature Coefficients:', ridge.coef_)

# Step 5: Visualize feature coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(n_features), ridge.coef_, color='blue')
plt.xticks(range(n_features), [f'Feature_{i}' for i in range(n_features)])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression: Feature Coefficients')
plt.grid(True)
plt.savefig('ridge_regression.png')
plt.close()