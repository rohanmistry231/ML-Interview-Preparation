import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Train-Test Split
# This script demonstrates splitting data into training and testing sets.

# Tasks:
# 1. Load the Iris dataset.
# 2. Split data into training and testing sets.
# 3. Train a Logistic Regression model.
# 4. Evaluate performance (accuracy).
# 5. Visualize training vs testing data distribution.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['Target'] = y

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 5: Visualize data distribution (using first feature as example)
plt.figure(figsize=(10, 6))
sns.histplot(data=X_train[:, 0], color='blue', label='Training Data', alpha=0.5, bins=20)
sns.histplot(data=X_test[:, 0], color='red', label='Testing Data', alpha=0.5, bins=20)
plt.xlabel(iris.feature_names[0])
plt.ylabel('Count')
plt.title('Train-Test Split: Distribution of First Feature')
plt.legend()
plt.grid(True)
plt.savefig('train_test_split.png')
plt.close()