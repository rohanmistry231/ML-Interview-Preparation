import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Normalization Demonstration
# This script focuses on Normalization techniques using MinMaxScaler on the Iris dataset.

# Tasks:
# 1. Load and explore the Iris dataset.
# 2. Apply MinMaxScaler for normalization (scales data to [0, 1] or custom range).
# 3. Train a Logistic Regression model on normalized data.
# 4. Compare performance with raw data.
# 5. Visualize the effect of normalization.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)

# Explore raw data
print("Raw Data Statistics:")
print(data.describe())

# Step 2: Apply Normalization
# Default range [0, 1]
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)

# Custom range example [-1, 1]
minmax_scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_normalized_custom = minmax_scaler_custom.fit_transform(X)

# Step 3: Train Logistic Regression
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_norm, X_test_norm = train_test_split(X_normalized, test_size=0.2, random_state=42)

# Raw data model
model_raw = LogisticRegression(random_state=42, max_iter=200)
model_raw.fit(X_train_raw, y_train)
y_pred_raw = model_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test, y_pred_raw)

# Normalized data model
model_norm = LogisticRegression(random_state=42, max_iter=200)
model_norm.fit(X_train_norm, y_train)
y_pred_norm = model_norm.predict(X_test_norm)
acc_norm = accuracy_score(y_test, y_pred_norm)

# Step 4: Print results
print(f'\nAccuracy (Raw Data): {acc_raw:.2f}')
print(f'Accuracy (Normalized [0,1]): {acc_norm:.2f}')

# Step 5: Visualize
plt.figure(figsize=(10, 6))

# Raw data
plt.subplot(2, 1, 1)
sns.boxplot(data=data)
plt.title('Raw Features')
plt.xticks(rotation=45)

# Normalized data [0, 1]
plt.subplot(2, 1, 2)
sns.boxplot(data=pd.DataFrame(X_normalized, columns=iris.feature_names))
plt.title('Normalized Features (MinMaxScaler [0,1])')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('normalization_effect.png')
plt.close()

# Additional: Show custom range effect
plt.figure(figsize=(6, 4))
sns.boxplot(data=pd.DataFrame(X_normalized_custom, columns=iris.feature_names))
plt.title('Normalized Features (MinMaxScaler [-1,1])')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('normalization_custom_range.png')
plt.close()

print("\nNormalization complete. Check 'normalization_effect.png' and 'normalization_custom_range.png' for visualizations.")