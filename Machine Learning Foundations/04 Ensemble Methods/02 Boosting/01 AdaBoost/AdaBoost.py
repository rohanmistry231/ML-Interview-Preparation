import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

# AdaBoost
# This script demonstrates AdaBoost classification.

# Tasks:
# 1. Generate synthetic classification data.
# 2. Split data into training and testing sets.
# 3. Train an AdaBoost Classifier with Decision Trees.
# 4. Make predictions and evaluate performance (accuracy, classification report).
# 5. Visualize decision boundary.

# Step 1: Generate synthetic data
np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
data['Target'] = y

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train AdaBoost Classifier
base_estimator = DecisionTreeClassifier(max_depth=1)  # Weak learner
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Step 4: Make predictions and evaluate
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Step 5: Visualize decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Testing Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('AdaBoost: Decision Boundary')
plt.legend()
plt.grid(True)
plt.savefig('adaboost.png')
plt.close()