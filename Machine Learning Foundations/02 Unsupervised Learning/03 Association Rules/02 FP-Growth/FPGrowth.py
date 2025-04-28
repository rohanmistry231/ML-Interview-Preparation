import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import seaborn as sns

# FP-Growth Algorithm
# This script demonstrates the FP-Growth algorithm for association rule mining.

# Tasks:
# 1. Create synthetic transactional data.
# 2. Apply the FP-Growth algorithm to find frequent itemsets.
# 3. Generate association rules.
# 4. Evaluate rules using support, confidence, and lift.
# 5. Visualize rule metrics.

# Step 1: Create synthetic transactional data
transactions = [
    ['Bread', 'Milk', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Butter', 'Cheese'],
    ['Bread', 'Milk', 'Butter'],
    ['Bread', 'Milk', 'Eggs', 'Cheese'],
    ['Milk', 'Cheese'],
    ['Bread', 'Eggs'],
    ['Butter', 'Cheese'],
    ['Bread', 'Milk', 'Butter', 'Eggs'],
    ['Milk', 'Butter']
]

# Convert to one-hot encoded DataFrame
items = set(item for transaction in transactions for item in transaction)
data = pd.DataFrame([[item in transaction for item in items] for transaction in transactions], columns=items)

# Step 2: Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(data, min_support=0.3, use_colnames=True)

# Step 3: Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Step 4: Evaluate rules
print('Frequent Itemsets:')
print(frequent_itemsets)
print('\nAssociation Rules:')
print(rules)

# Step 5: Visualize rule metrics
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift', palette='viridis')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('FP-Growth: Association Rules (Size and Color by Lift)')
plt.grid(True)
plt.savefig('fpgrowth_rules.png')
plt.close()