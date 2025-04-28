import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Binning
# This script demonstrates binning continuous features.

# Tasks:
# 1. Generate synthetic data with continuous features.
# 2. Apply KBinsDiscretizer to bin a feature.
# 3. Train a Logistic Regression model with binned features.
# 4. Evaluate performance (accuracy).
# 5. Visualize binned feature distribution.

# Step 1: Generate synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 80, 100),
    'Income': np.random.randint(20000, 120000, 100),
    'Target': np.random.choice([0, 1], 100)
})

# Step 2: Apply binning
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
data['Age_Binned'] = binner.fit_transform(data[['Age']])

# Step 3: Train Logistic Regression
X = data[['Age', 'Income']]
X_binned = data[['Age_Binned', 'Income']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_binned, X_test_binned = train_test_split(X_binned, test_size=0.2, random_state=42)

# Original features
model_orig = LogisticRegression(random_state=42)
model_orig.fit(X_train, y_train)
y_pred_orig = model_orig.predict(X_test)
acc_orig = accuracy_score(y_test, y_pred_orig)

# Binned features
model_binned = LogisticRegression(random_state=42)
model_binned.fit(X_train_binned, y_train)
y_pred_binned = model_binned.predict(X_test_binned)
acc_binned = accuracy_score(y_test, y_pred_binned)

print(f'Accuracy (Original): {acc_orig:.2f}')
print(f'Accuracy (Binned): {acc_binned:.2f}')

# Step 4: Visualize binned feature distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', color='blue', alpha=0.5, label='Original Age', bins=20)
sns.histplot(data=data, x='Age_Binned', color='red', alpha=0.5, label='Binned Age', bins=5)
plt.xlabel('Age / Binned Age')
plt.ylabel('Count')
plt.title('Binning: Original vs Binned Age Distribution')
plt.legend()
plt.grid(True)
plt.savefig('binning.png')
plt.close()