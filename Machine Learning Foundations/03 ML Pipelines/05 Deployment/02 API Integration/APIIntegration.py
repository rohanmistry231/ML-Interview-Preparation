import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns

# API Integration
# This script demonstrates creating a Flask API for a trained model.

# Tasks:
# 1. Load the Iris dataset and train a Logistic Regression model.
# 2. Save the model using joblib.
# 3. Create a Flask API to serve predictions.
# 4. Test the API with sample data.
# 5. Visualize API predictions.

# Step 1: Load data and train model
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 2: Save the model
joblib.dump(model, 'logistic_model_api.pkl')

# Step 3: Create Flask API
app = Flask(__name__)
model = joblib.load('logistic_model_api.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0]), 'class': iris.target_names[prediction[0]]})

# Note: The Flask app would be run with app.run(debug=True) in a real environment.
# For demonstration, we'll simulate API testing.

# Step 4: Test the API (simulated)
sample_data = X_test[:5]
predictions = model.predict(sample_data)
class_names = [iris.target_names[pred] for pred in predictions]

print('Sample Predictions:')
for i, (features, pred, name) in enumerate(zip(sample_data, predictions, class_names)):
    print(f'Sample {i+1}: Features={features.tolist()}, Prediction={pred}, Class={name}')

# Step 5: Visualize predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test[:5, 0], y=X_test[:5, 1], hue=class_names, style=iris.target_names[y_test[:5]], s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('API Integration: Simulated API Predictions')
plt.grid(True)
plt.savefig('api_integration.png')
plt.close()

# To run the Flask API, use: app.run(debug=True)
# Then send POST requests to http://localhost:5000/predict with JSON like {"features": [5.1, 3.5, 1.4, 0.2]}