import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Lasso Regression
# This script demonstrates Lasso Regression with synthetic data for feature selection.

# Tasks:
# 1. Generate synthetic data with some irrelevant features.
# 2. Split data into training and testing sets.
# 3. Train a Lasso Regression model with L1 regularization.
# 4. Make predictions and evaluate performance (MSE, R²).
# 5. Visualize feature coefficients to show sparsity.

# Step 1: Generate synthetic data
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
# Only first 3 features are relevant
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

# Convert to DataFrame
data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(n_features)])
data['Target'] = y

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Lasso Regression model
lasso = Lasso(alpha=0.1)  # Regularization strength
lasso.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = lasso.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print('Feature Coefficients:', lasso.coef_)

# Step 5: Visualize feature coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(n_features), lasso.coef_, color='blue')
plt.xticks(range(n_features), [f'Feature_{i}' for i in range(n_features)])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression: Feature Coefficients')
plt.grid(True)
plt.savefig('lasso_regression.png')
plt.close()