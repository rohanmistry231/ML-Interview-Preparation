import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Encoding Categorical Variables
# This script demonstrates Label Encoding and One-Hot Encoding.

# Tasks:
# 1. Create a synthetic dataset with categorical variables.
# 2. Apply Label Encoding and One-Hot Encoding.
# 3. Train a Logistic Regression model on encoded data.
# 4. Compare model performance (accuracy).
# 5. Visualize the effect of encoding.

# Step 1: Create synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 60, 100),
    'Income': np.random.randint(30000, 100000, 100),
    'Category': np.random.choice(['Low', 'Medium', 'High'], 100),
    'Target': np.random.choice([0, 1], 100)
})

# Step 2: Apply encoding
# Label Encoding
label_encoder = LabelEncoder()
data['Category_Label'] = label_encoder.fit_transform(data['Category'])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse=False)
category_ohe = one_hot_encoder.fit_transform(data[['Category']])
category_ohe_df = pd.DataFrame(category_ohe, columns=one_hot_encoder.get_feature_names_out(['Category']))

data_encoded = pd.concat([data[['Age', 'Income']], category_ohe_df, data['Target']], axis=1)

# Step 3: Train Logistic Regression
X_label = data[['Age', 'Income', 'Category_Label']]
X_ohe = data_encoded.drop('Target', axis=1)
y = data['Target']

X_train_label, X_test_label, y_train, y_test = train_test_split(X_label, y, test_size=0.2, random_state=42)
X_train_ohe, X_test_ohe = train_test_split(X_ohe, test_size=0.2, random_state=42)

# Label encoded model
model_label = LogisticRegression(random_state=42)
model_label.fit(X_train_label, y_train)
y_pred_label = model_label.predict(X_test_label)
acc_label = accuracy_score(y_test, y_pred_label)

# One-Hot encoded model
model_ohe = LogisticRegression(random_state=42)
model_ohe.fit(X_train_ohe, y_train)
y_pred_ohe = model_ohe.predict(X_test_ohe)
acc_ohe = accuracy_score(y_test, y_pred_label)

print(f'Accuracy (Label Encoded): {acc_label:.2f}')
print(f'Accuracy (One-Hot Encoded): {acc_ohe:.2f}')

# Step 4: Visualize data distribution
plt.figure(figsize=(12, 5))

# Label encoded
plt.subplot(1, 2, 1)
sns.countplot(x='Category_Label', hue='Target', data=data)
plt.title('Label Encoded Categories')
plt.xlabel('Category (Encoded)')

# One-Hot encoded (show distribution of original categories)
plt.subplot(1, 2, 2)
sns.countplot(x='Category', hue='Target', data=data)
plt.title('Original Categories')
plt.xlabel('Category')

plt.tight_layout()
plt.savefig('encoding_categorical_variables.png')
plt.close()