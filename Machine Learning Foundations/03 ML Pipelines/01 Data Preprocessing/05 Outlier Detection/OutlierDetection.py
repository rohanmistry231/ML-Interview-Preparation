import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Outlier Detection
# This script demonstrates outlier detection using Isolation Forest.

# Tasks:
# 1. Load the Iris dataset and introduce synthetic outliers.
# 2. Apply Isolation Forest to detect outliers.
# 3. Evaluate the number of detected outliers.
# 4. Visualize outliers in the dataset.

# Step 1: Load data and introduce outliers
iris = load_iris()
X = iris.data
data = pd.DataFrame(X, columns=iris.feature_names)

# Introduce 5% outliers
np.random.seed(42)
n_outliers = int(0.05 * X.shape[0])
outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, X.shape[1]))
X_with_outliers = np.vstack([X, outliers])
data_with_outliers = pd.DataFrame(X_with_outliers, columns=iris.feature_names)

# Step 2: Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(X_with_outliers)
# -1 indicates outlier, 1 indicates inlier
outliers_detected = X_with_outliers[outlier_labels == -1]
inliers = X_with_outliers[outlier_labels == 1]

# Step 3: Evaluate
n_outliers_detected = len(outliers_detected)
print(f'Number of Outliers Detected: {n_outliers_detected}')

# Step 4: Visualize outliers (using first two features for 2D plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_with_outliers.iloc[:, 0], y=data_with_outliers.iloc[:, 1], hue=outlier_labels, palette={1: 'blue', -1: 'red'}, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Outlier Detection with Isolation Forest')
plt.legend(['Inliers', 'Outliers'])
plt.grid(True)
plt.savefig('outlier_detection.png')
plt.close()