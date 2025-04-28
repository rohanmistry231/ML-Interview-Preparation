import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# t-Distributed Stochastic Neighbor Embedding (t-SNE)
# This script demonstrates t-SNE for dimensionality reduction on the Iris dataset.

# Tasks:
# 1. Load the Iris dataset.
# 2. Standardize the features.
# 3. Apply t-SNE to reduce to 2 dimensions.
# 4. Visualize the reduced data.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['Target'] = y

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Step 4: Visualize reduced data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=iris.target_names[y], palette='viridis', s=100)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE: Iris Dataset Reduced to 2D')
plt.grid(True)
plt.savefig('tsne.png')
plt.close()