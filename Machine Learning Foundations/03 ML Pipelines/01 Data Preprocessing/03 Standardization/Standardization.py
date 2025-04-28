import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Standardization Demonstration
# This script focuses on Standardization techniques using StandardScaler on the Iris dataset.

# Tasks:
# 1. Load and explore the Iris dataset.
# 2. Apply StandardScaler for standardization (zero mean, unit variance).
# 3. Train a Logistic Regression model on standardized data.
# 4. Compare performance with raw data.
# 5. Visualize the effect of standardization.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)

# Explore raw data
print("Raw Data Statistics:")
print(data.describe())

# Step 2: Apply Standardization
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

# Step 3: Train Logistic Regression
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_std, X_test_std = train_test_split(X_standardized, test_size=0.2, random_state=42)

# Raw data model
model_raw = LogisticRegression(random_state=42, max_iter=200)
model_raw.fit(X_train_raw, y_train)
y_pred_raw = model_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test, y_pred_raw)

# Standardized data model
model_std = LogisticRegression(random_state=42, max_iter=200)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

# Step 4: Print results
print(f'\nAccuracy (Raw Data): {acc_raw:.2f}')
print(f'Accuracy (Standardized): {acc_std:.2f}')

# Step 5: Visualize
plt.figure(figsize=(10, 6))

# Raw data
plt.subplot(2, 1, 1)
sns.boxplot(data=data)
plt.title('Raw Features')
plt.xticks(rotation=45)

# Standardized data
plt.subplot(2, 1, 2)
sns.boxplot(data=pd.DataFrame(X_standardized, columns=iris.feature_names))
plt.title('Standardized Features (StandardScaler)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('standardization_effect.png')
plt.close()

# Additional: Check mean and variance after standardization
standardized_data = pd.DataFrame(X_standardized, columns=iris.feature_names)
print("\nStandardized Data Statistics (Mean ~ 0, Std ~ 1):")
print(standardized_data.describe())

print("\nStandardization complete. Check 'standardization_effect.png' for visualization.")