import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Linear Discriminant Analysis (LDA)
# This script demonstrates LDA for supervised dimensionality reduction on the Iris dataset.

# Tasks:
# 1. Load the Iris dataset.
# 2. Standardize the features.
# 3. Apply LDA to reduce to 2 dimensions (since we have 3 classes).
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

# Step 3: Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Step 4: Visualize reduced data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=iris.target_names[y], palette='viridis', s=100)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Iris Dataset Reduced to 2D')
plt.grid(True)
plt.savefig('lda.png')
plt.close()