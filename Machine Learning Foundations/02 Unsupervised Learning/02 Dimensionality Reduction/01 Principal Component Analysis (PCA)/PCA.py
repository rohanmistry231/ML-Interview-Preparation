import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Principal Component Analysis (PCA)
# This script demonstrates PCA for dimensionality reduction on the Iris dataset.

# Tasks:
# 1. Load the Iris dataset.
# 2. Standardize the features.
# 3. Apply PCA to reduce to 2 dimensions.
# 4. Evaluate explained variance.
# 5. Visualize the reduced data.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['Target'] = y

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Evaluate explained variance
explained_variance = pca.explained_variance_ratio_
print(f'Explained Variance Ratio: {explained_variance}')
print(f'Total Explained Variance: {sum(explained_variance):.2f}')

# Step 5: Visualize reduced data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris.target_names[y], palette='viridis', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset Reduced to 2D')
plt.grid(True)
plt.savefig('pca.png')
plt.close()

# Scree plot for explained variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.savefig('pca_scree.png')
plt.close()