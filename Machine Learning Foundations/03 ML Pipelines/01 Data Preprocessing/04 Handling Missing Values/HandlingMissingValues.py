import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Handling Missing Values
# This script demonstrates techniques to handle missing values in a dataset.

# Tasks:
# 1. Load the Iris dataset and introduce synthetic missing values.
# 2. Apply mean imputation and median imputation.
# 3. Train a Logistic Regression model on imputed data.
# 4. Compare model performance (accuracy).
# 5. Visualize data distribution before and after imputation.

# Step 1: Load data and introduce missing values
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)

# Introduce 10% missing values randomly
np.random.seed(42)
mask = np.random.rand(*X.shape) < 0.1
X_with_missing = X.copy()
X_with_missing[mask] = np.nan
data_missing = pd.DataFrame(X_with_missing, columns=iris.feature_names)

# Step 2: Apply imputation
# Mean imputation
mean_imputer = SimpleImputer(strategy='mean')
X_mean_imputed = mean_imputer.fit_transform(X_with_missing)

# Median imputation
median_imputer = SimpleImputer(strategy='median')
X_median_imputed = median_imputer.fit_transform(X_with_missing)

# Step 3: Train Logistic Regression on each dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_mean, X_test_mean = train_test_split(X_mean_imputed, test_size=0.2, random_state=42)
X_train_median, X_test_median = train_test_split(X_median_imputed, test_size=0.2, random_state=42)

# Raw data (no missing values)
model_raw = LogisticRegression(random_state=42)
model_raw.fit(X_train_raw, y_train)
y_pred_raw = model_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test, y_pred_raw)

# Mean imputed data
model_mean = LogisticRegression(random_state=42)
model_mean.fit(X_train_mean, y_train)
y_pred_mean = model_mean.predict(X_test_mean)
acc_mean = accuracy_score(y_test, y_pred_mean)

# Median imputed data
model_median = LogisticRegression(random_state=42)
model_median.fit(X_train_median, y_train)
y_pred_median = model_median.predict(X_test_median)
acc_median = accuracy_score(y_test, y_pred_median)

print(f'Accuracy (Raw): {acc_raw:.2f}')
print(f'Accuracy (Mean Imputed): {acc_mean:.2f}')
print(f'Accuracy (Median Imputed): {acc_median:.2f}')

# Step 4: Visualize data distribution
plt.figure(figsize=(12, 8))

# Original data with missing values
plt.subplot(3, 1, 1)
sns.boxplot(data=data_missing)
plt.title('Data with Missing Values')
plt.xticks(rotation=45)

# Mean imputed data
plt.subplot(3, 1, 2)
sns.boxplot(data=pd.DataFrame(X_mean_imputed, columns=iris.feature_names))
plt.title('Mean Imputed Data')
plt.xticks(rotation=45)

# Median imputed data
plt.subplot(3, 1, 3)
sns.boxplot(data=pd.DataFrame(X_median_imputed, columns=iris.feature_names))
plt.title('Median Imputed Data')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('handling_missing_values.png')
plt.close()