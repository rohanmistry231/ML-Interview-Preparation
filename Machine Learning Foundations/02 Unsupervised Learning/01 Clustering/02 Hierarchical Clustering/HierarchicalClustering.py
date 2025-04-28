import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Hierarchical Clustering
# This script demonstrates Hierarchical Clustering with a dendrogram.

# Tasks:
# 1. Generate synthetic clustering data.
# 2. Apply Hierarchical Clustering (Agglomerative).
# 3. Evaluate clustering performance using silhouette score.
# 4. Visualize clusters and dendrogram.

# Step 1: Generate synthetic data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# Step 2: Apply Hierarchical Clustering
n_clusters = 4
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = hierarchical.fit_predict(X)

# Step 3: Evaluate performance
silhouette = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette:.2f}')

# Step 4: Visualize clusters and dendrogram
plt.figure(figsize=(12, 10))

# Clusters
plt.subplot(2, 1, 1)
sns.scatterplot(x=data['Feature_1'], y=data['Feature_2'], hue=labels, palette='viridis', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')

# Dendrogram
plt.subplot(2, 1, 2)
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('hierarchical_clustering.png')
plt.close()