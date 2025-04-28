import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Feature Selection
# This script demonstrates feature selection using SelectKBest on the Iris dataset.

# Tasks:
# 1. Load the Iris dataset.
# 2. Apply SelectKBest to select top features.
# 3. Train a Logistic Regression model on selected features.
# 4. Evaluate model performance (accuracy).
# 5. Visualize feature importance scores.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)

# Step 2: Apply feature selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
selected_features = [iris.feature_names[i] for i in selector.get_support(indices=True)]
feature_scores = selector.scores_

# Step 3: Train Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_selected, X_test_selected = train_test_split(X_selected, test_size=0.2, random_state=42)

# Full features
model_full = LogisticRegression(random_state=42)
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)

# Selected features
model_selected = LogisticRegression(random_state=42)
model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)
acc_selected = accuracy_score(y_test, y_pred_selected)

print(f'Accuracy (Full Features): {acc_full:.2f}')
print(f'Accuracy (Selected Features): {acc_selected:.2f}')
print(f'Selected Features: {selected_features}')

# Step 4: Visualize feature scores
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_scores, y=iris.feature_names, palette='viridis')
plt.xlabel('Feature Score (f_classif)')
plt.ylabel('Feature')
plt.title('Feature Importance Scores')
plt.grid(True)
plt.savefig('feature_selection.png')
plt.close()