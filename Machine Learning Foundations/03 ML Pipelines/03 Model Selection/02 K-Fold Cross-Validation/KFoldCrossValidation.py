import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns

# K-Fold Cross-Validation
# This script demonstrates K-Fold Cross-Validation.

# Tasks:
# 1. Load the Iris dataset.
# 2. Apply K-Fold Cross-Validation (k=5).
# 3. Train a Logistic Regression model.
# 4. Evaluate performance (mean accuracy and standard deviation).
# 5. Visualize cross-validation scores.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Apply K-Fold Cross-Validation
model = LogisticRegression(random_state=42)
k = 5
scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')

# Step 3: Evaluate performance
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)
print(f'K-Fold CV Scores: {scores}')
print(f'Mean Accuracy: {mean_accuracy:.2f}')
print(f'Standard Deviation: {std_accuracy:.2f}')

# Step 4: Visualize CV scores
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(1, k+1), y=scores, palette='viridis')
plt.axhline(mean_accuracy, color='red', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.2f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('K-Fold Cross-Validation Scores')
plt.legend()
plt.grid(True)
plt.savefig('kfold_cross_validation.png')
plt.close()