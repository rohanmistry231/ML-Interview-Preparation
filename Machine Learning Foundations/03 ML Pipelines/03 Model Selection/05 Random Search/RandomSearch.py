import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import seaborn as sns

# Random Search
# This script demonstrates Random Search for hyperparameter tuning.

# Tasks:
# 1. Load the Iris dataset.
# 2. Define a parameter distribution for SVC.
# 3. Perform Random Search with cross-validation.
# 4. Evaluate the best model (accuracy).
# 5. Visualize hyperparameter performance.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Define parameter distribution
param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Step 3: Perform Random Search
model = SVC(random_state=42)
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

# Step 4: Evaluate best model
best_model = random_search.best_estimator_
best_score = random_search.best_score_
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {best_score:.2f}')

# Step 5: Visualize hyperparameter performance
results = pd.DataFrame(random_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results, x='param_C', y='mean_test_score', hue='param_kernel', style='param_gamma', size='mean_test_score')
plt.xlabel('C')
plt.ylabel('Mean Test Accuracy')
plt.title('Random Search: Hyperparameter Performance')
plt.grid(True)
plt.savefig('random_search.png')
plt.close()