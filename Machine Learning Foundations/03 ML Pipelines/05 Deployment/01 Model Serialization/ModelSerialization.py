import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Model Serialization
# This script demonstrates saving and loading a trained model.

# Tasks:
# 1. Load the Iris dataset.
# 2. Train a Logistic Regression model.
# 3. Save the model using joblib.
# 4. Load the model and make predictions.
# 5. Visualize prediction results.

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['Target'] = y

# Step 2: Train Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 3: Save the model
joblib.dump(model, 'logistic_model.pkl')

# Step 4: Load the model and predict
loaded_model = joblib.load('logistic_model.pkl')
y_pred = loaded_model.predict(X_test)

# Evaluate performance
accuracy = loaded_model.score(X_test, y_test)
print(f'Accuracy of Loaded Model: {accuracy:.2f}')

# Step 5: Visualize predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=iris.target_names[y_pred], style=iris.target_names[y_test], s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Model Serialization: Predictions from Loaded Model')
plt.grid(True)
plt.savefig('model_serialization.png')
plt.close()