import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
import seaborn as sns

# Mean Shift Clustering
# This script demonstrates Mean Shift clustering.

# Tasks:
# 1. Generate synthetic clustering data.
# 2. Apply Mean Shift clustering.
# 3. Evaluate clustering performance using silhouette score.
# 4. Visualize clusters.

# Step 1: Generate synthetic data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# Step 2: Apply Mean Shift clustering
mean_shift = MeanShift()
labels = mean_shift.fit_predict(X)
cluster_centers = mean_shift.cluster_centers_

# Step 3: Evaluate performance
silhouette = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette:.2f}')

# Step 4: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Feature_1'], y=data['Feature_2'], hue=labels, palette='viridis', s=100)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Mean Shift Clustering')
plt.legend()
plt.grid(True)
plt.savefig('mean_shift_clustering.png')
plt.close()