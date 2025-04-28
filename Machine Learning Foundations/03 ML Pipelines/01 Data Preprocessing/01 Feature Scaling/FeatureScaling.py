import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Feature Scaling
# This script demonstrates Normalization and Standardization on the Iris dataset.

# Tasks:
# 1. Load the Iris dataset.
# 2. Apply Normalization (MinMaxScaler) and Standardization (StandardScaler).
# 3. Train a Logistic Regression model on scaled data.
# 4. Compare model performance (accuracy).
# 5. Visualize feature distributions before and after scaling.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)

# Step 2: Apply scaling
# Normalization (MinMaxScaler)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)

# Standardization (StandardScaler)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

# Step 3: Train Logistic Regression on each dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_norm, X_test_norm = train_test_split(X_normalized, test_size=0.2, random_state=42)
X_train_std, X_test_std = train_test_split(X_standardized, test_size=0.2, random_state=42)

# Raw data
model_raw = LogisticRegression(random_state=42)
model_raw.fit(X_train_raw, y_train)
y_pred_raw = model_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test, y_pred_raw)

# Normalized data
model_norm = LogisticRegression(random_state=42)
model_norm.fit(X_train_norm, y_train)
y_pred_norm = model_norm.predict(X_test_norm)
acc_norm = accuracy_score(y_test, y_pred_norm)

# Standardized data
model_std = LogisticRegression(random_state=42)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

print(f'Accuracy (Raw): {acc_raw:.2f}')
print(f'Accuracy (Normalized): {acc_norm:.2f}')
print(f'Accuracy (Standardized): {acc_std:.2f}')

# Step 4: Visualize feature distributions
plt.figure(figsize=(12, 8))

# Raw data
plt.subplot(3, 1, 1)
sns.boxplot(data=data)
plt.title('Raw Features')
plt.xticks(rotation=45)

# Normalized data
plt.subplot(3, 1, 2)
sns.boxplot(data=pd.DataFrame(X_normalized, columns=iris.feature_names))
plt.title('Normalized Features (MinMaxScaler)')
plt.xticks(rotation=45)

# Standardized data
plt.subplot(3, 1, 3)
sns.boxplot(data=pd.DataFrame(X_standardized, columns=iris.feature_names))
plt.title('Standardized Features (StandardScaler)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('feature_scaling.png')
plt.close()