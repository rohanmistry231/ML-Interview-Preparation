import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Grid Search
# This script demonstrates Grid Search for hyperparameter tuning.

# Tasks:
# 1. Load the Iris dataset.
# 2. Define a parameter grid for SVC.
# 3. Perform Grid Search with cross-validation.
# 4. Evaluate the best model (accuracy).
# 5. Visualize hyperparameter performance.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Step 3: Perform Grid Search
model = SVC(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Step 4: Evaluate best model
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {best_score:.2f}')

# Step 5: Visualize hyperparameter performance
results = pd.DataFrame(grid_search.cv_results_)
pivot_table = results.pivot_table(values='mean_test_score', index='param_C', columns='param_kernel')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f')
plt.xlabel('Kernel')
plt.ylabel('C')
plt.title('Grid Search: Mean Test Accuracy')
plt.savefig('grid_search.png')
plt.close()