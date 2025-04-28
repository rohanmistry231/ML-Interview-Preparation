import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# This script demonstrates DBSCAN on non-spherical data.

# Tasks:
# 1. Generate synthetic moon-shaped data.
# 2. Apply DBSCAN clustering.
# 3. Evaluate clustering performance using silhouette score (if applicable).
# 4. Visualize clusters and noise points.

# Step 1: Generate synthetic data
np.random.seed(42)
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# Step 2: Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Step 3: Evaluate performance
# Silhouette score only if there are at least 2 clusters and no noise (-1 labels)
if len(set(labels)) > 1 and -1 not in labels:
    silhouette = silhouette_score(X, labels)
    print(f'Silhouette Score: {silhouette:.2f}')
else:
    print('Silhouette Score: Not applicable due to noise or single cluster')

# Step 4: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Feature_1'], y=data['Feature_2'], hue=labels, palette='viridis', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering (Noise points in black)')
plt.grid(True)
plt.savefig('dbscan_clustering.png')
plt.close()