# Machine Learning Interview Questions for AI/ML Roles

## Supervised Learning
Supervised learning trains models on labeled data to predict outcomes, crucial for tasks like sales forecasting or spam detection.

### Basic
1. **What is supervised learning, and how does it differ from unsupervised learning?**  
   Supervised learning uses labeled data to predict outputs (e.g., classifying emails), while unsupervised learning finds patterns without labels (e.g., clustering customers). In AI/ML, supervised learning is chosen when the goal is prediction with known targets.

2. **What are the two main types of supervised learning tasks?**  
   - **Regression**: Predicts continuous outputs (e.g., house prices).  
   - **Classification**: Predicts categorical outputs (e.g., spam or not). These define the task type in ML workflows.

3. **What are some common evaluation metrics for regression models?**  
   - **Mean Squared Error (MSE)**: Measures average squared error.  
   - **R² Score**: Indicates variance explained by the model.  
   - **Mean Absolute Error (MAE)**: Average absolute error. In AI/ML, these assess prediction accuracy for continuous data like sales forecasts.

4. **What is classification, and what are some common classification algorithms?**  
   Classification predicts discrete labels (e.g., positive/negative sentiment). Algorithms include Logistic Regression, Decision Trees, and SVMs—used in tasks like fraud detection or medical diagnosis.

### Intermediate
5. **Explain the concept of linear regression. What is its goal?**  
   Linear regression models the relationship between features and a continuous target using a straight line (y = mx + b). Its goal is to minimize prediction error, e.g., predicting house prices based on size and location.

6. **What is the difference between L1 and L2 regularization in regression?**  
   - **L1 (Lasso)**: Adds absolute value of coefficients to the loss, promoting sparsity (some weights become zero).  
   - **L2 (Ridge)**: Adds squared coefficients, shrinking weights evenly. In ML, they prevent overfitting in models like regression for noisy datasets.

7. **How does a decision tree make predictions? What are its pros and cons?**  
   It splits data based on feature thresholds, predicting via leaf nodes (e.g., customer segmentation).  
   - **Pros**: Interpretable, handles non-linear data.  
   - **Cons**: Prone to overfitting, sensitive to small changes.

8. **What is the confusion matrix, and how is it used to evaluate classification models?**  
   A confusion matrix shows true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). It evaluates accuracy, precision, and recall—key for tasks like diagnosing diseases.

9. **What is the Naive Bayes theorem, and why is it called "naive"?**  
   Naive Bayes uses Bayes’ theorem (P(A|B) = P(B|A)P(A)/P(B)) to predict class probabilities, assuming feature independence (hence "naive"). It’s effective for text classification despite this simplification.

### Advanced
10. **How does polynomial regression capture non-linear relationships?**  
    It extends linear regression by adding polynomial terms (e.g., x²), fitting curves to data like stock price trends. In ML, it balances flexibility and complexity for non-linear patterns.

11. **Explain the bias-variance tradeoff in model complexity.**  
    - **Bias**: Error from overly simple models (underfitting).  
    - **Variance**: Error from sensitivity to training data (overfitting).  
    In AI/ML, optimal complexity minimizes total error for robust predictions.

12. **How does Random Forest improve upon decision trees?**  
    Random Forest builds multiple trees on random data subsets and features, averaging predictions. It reduces overfitting and variance, enhancing accuracy for tasks like credit risk assessment.

13. **Explain the ROC curve and AUC score. Why are they useful?**  
    The ROC curve plots true positive rate vs. false positive rate at various thresholds. AUC (Area Under Curve) summarizes performance—higher is better. They’re vital for imbalanced data, e.g., fraud detection.

14. **How does the K-Nearest Neighbors algorithm work, and what are its strengths and weaknesses?**  
    KNN predicts by finding the K closest training points (e.g., via Euclidean distance) and voting/averaging.  
    - **Strengths**: Simple, captures local patterns.  
    - **Weaknesses**: Slow on large datasets, sensitive to irrelevant features.

15. **Explain support vectors in SVM and their role in finding the decision boundary.**  
    Support vectors are data points nearest the decision boundary in SVM. They define the maximum-margin hyperplane, optimizing separation for high-dimensional tasks like image recognition.

## Unsupervised Learning
Unsupervised learning uncovers patterns in unlabeled data, key for clustering or dimensionality reduction.

### Basic
16. **What is unsupervised learning, and what are its main applications?**  
    Unsupervised learning finds structure without labels, used for clustering (e.g., market segmentation) and dimensionality reduction (e.g., feature compression).

17. **What is clustering, and why is it used?**  
    Clustering groups similar data points (e.g., customers by behavior). It’s used for insights, anomaly detection, or preprocessing in ML workflows.

18. **What is dimensionality reduction, and why is it important?**  
    It reduces feature count while retaining key information, speeding up modeling and enabling visualization (e.g., compressing image data).

### Intermediate
19. **Explain how K-Means clustering works. How do you choose the number of clusters?**  
    K-Means assigns points to K clusters by minimizing distance to centroids, iteratively updating them. The elbow method (plotting inertia vs. K) helps select K—used for customer grouping.

20. **What is Principal Component Analysis (PCA), and how does it reduce dimensionality?**  
    PCA transforms data into principal components (directions of max variance), retaining top ones. It compresses features for efficient modeling, e.g., in image processing.

21. **Explain the Apriori algorithm and its key steps.**  
    Apriori finds frequent itemsets for association rules (e.g., in recommendation systems):  
    - Identify frequent items.  
    - Generate candidate itemsets.  
    - Prune infrequent ones iteratively.

22. **What is Linear Discriminant Analysis (LDA), and when is it used?**  
    LDA maximizes class separability for dimensionality reduction, unlike PCA’s variance focus. It’s used in supervised tasks (e.g., face recognition) with labeled data.

### Advanced
23. **What is hierarchical clustering, and how does it differ from K-Means?**  
    Hierarchical clustering builds a tree of clusters (agglomerative or divisive), offering flexibility without preset K. It’s ideal for gene expression analysis, unlike K-Means’ fixed clusters.

24. **How does DBSCAN handle clusters of varying density?**  
    DBSCAN groups points by density (core points, border points, noise), excelling at irregular clusters and anomaly detection (e.g., outlier identification).

25. **What is t-SNE, and how does it differ from PCA?**  
    t-SNE reduces dimensions for visualization by preserving local structure (e.g., word embeddings), while PCA focuses on global variance. t-SNE is non-linear and computationally intensive.

26. **How does FP-Growth improve upon the Apriori algorithm?**  
    FP-Growth uses a tree structure (FP-tree) to mine frequent patterns without candidate generation, making it faster for large transactional datasets in ML.

## ML Pipelines
ML pipelines streamline data preprocessing to deployment for reproducible, scalable models.

### Basic
27. **What is a machine learning pipeline, and why is it important?**  
    A pipeline sequences data preprocessing, modeling, and evaluation steps. It ensures consistency and scalability in ML workflows, e.g., automating sales prediction.

28. **What are some common data preprocessing techniques?**  
    - **Normalization/Scaling**: Adjusts feature ranges.  
    - **Encoding**: Converts categories to numbers.  
    - **Imputation**: Fills missing values. These prepare raw data for modeling.

29. **What is the purpose of splitting data into training and testing sets?**  
    It trains the model on one subset and evaluates generalization on another, ensuring real-world performance (e.g., 80/20 split).

30. **What is overfitting, and how can it be detected?**  
    Overfitting occurs when a model learns noise, not patterns, performing well on training data but poorly on test data. It’s detected by comparing train vs. test accuracy.

### Intermediate
31. **Explain feature scaling and its importance in machine learning.**  
    Feature scaling (e.g., standardization) normalizes feature ranges, ensuring equal contribution. It’s critical for distance-based algorithms like KNN or gradient descent.

32. **How do you handle missing values in a dataset?**  
    - **Imputation**: Fill with mean/median/mode or predictive models.  
    - **Deletion**: Remove rows/columns. In ML, this maintains data integrity for accurate predictions.

33. **What is encoding, and why is it necessary for categorical variables?**  
    Encoding (e.g., one-hot, label encoding) converts categories to numbers, enabling algorithms to process them—essential for tasks like sentiment analysis.

34. **Explain K-Fold Cross-Validation and its advantages.**  
    K-Fold splits data into K subsets, training on K-1 and testing on 1, rotating through all. It provides robust performance estimates, reducing overfitting risk.

35. **How does Grid Search help in hyperparameter tuning?**  
    Grid Search tests all combinations of hyperparameters (e.g., learning rate), selecting the best via cross-validation—optimizing models like SVMs.

### Advanced
36. **What is feature engineering, and how does it improve model performance?**  
    Feature engineering creates new features (e.g., interaction terms) or transforms existing ones, enhancing predictive power for tasks like sales forecasting.

37. **How do you create polynomial features, and when might you use them?**  
    Polynomial features add terms like x² or xy (e.g., via Scikit-learn’s `PolynomialFeatures`). They’re used for non-linear patterns, like financial modeling.

38. **What is stratified K-Fold, and when is it used?**  
    Stratified K-Fold ensures class proportions in each fold match the dataset, critical for imbalanced data (e.g., rare disease detection).

39. **What is Random Search, and how does it compare to Grid Search?**  
    Random Search samples hyperparameter combinations randomly, often faster and equally effective for large spaces vs. Grid Search’s exhaustive approach.

40. **How can you address overfitting in machine learning models?**  
    - **Regularization**: Penalizes complexity (L1/L2).  
    - **Cross-Validation**: Ensures generalization.  
    - **More Data**: Reduces noise impact. These improve robustness in ML.

41. **How do you serialize a machine learning model?**  
    Serialization saves models (e.g., using Python’s `pickle` or `joblib`) for reuse:  
    ```python
    import pickle
    model = LinearRegression()
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    ```

42. **Explain how to integrate a machine learning model into an API.**  
    Load a serialized model into a web framework (e.g., Flask), exposing an endpoint:  
    ```python
    from flask import Flask, request
    app = Flask(__name__)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        return {"prediction": model.predict([data["features"]])[0]}
    ```

## Ensemble Methods
Ensemble methods combine models for better accuracy and robustness, critical in high-stakes predictions.

### Basic
43. **What are ensemble methods in machine learning?**  
    Ensemble methods combine multiple models (e.g., trees) to improve predictions, leveraging diversity for tasks like medical diagnosis.

44. **What is bagging, and how does it work?**  
    Bagging (Bootstrap Aggregating) trains models on random data subsets, averaging predictions (e.g., Random Forest). It reduces variance in ML.

45. **What is boosting, and how does it differ from bagging?**  
    Boosting trains models sequentially, focusing on errors, unlike bagging’s parallel approach. It reduces bias for better accuracy.

### Intermediate
46. **Explain Bootstrap Aggregating (Bagging) and its role in reducing variance.**  
    Bagging samples data with replacement, training diverse models and aggregating outputs. It stabilizes predictions, e.g., in financial modeling.

47. **Explain AdaBoost and its working principle.**  
    AdaBoost assigns weights to misclassified samples, iteratively improving weak learners (e.g., stumps) by focusing on errors—effective for classification.

48. **Compare and contrast bagging and boosting.**  
    - **Bagging**: Parallel, reduces variance (e.g., Random Forest).  
    - **Boosting**: Sequential, reduces bias (e.g., AdaBoost).  
    In ML, choose based on overfitting vs. underfitting needs.

### Advanced
49. **How does Random Forest use bagging to improve model performance?**  
    Random Forest applies bagging to decision trees with random feature subsets, averaging outputs for robust predictions—used in risk assessment.

50. **What is Gradient Boosting, and how does it build upon previous models?**  
    Gradient Boosting minimizes a loss function (e.g., MSE) by adding models that correct residuals of prior ones—powers tools like XGBoost.

51. **Why are ensemble methods often more accurate than individual models?**  
    They reduce errors via diversity (bagging) or error correction (boosting), improving generalization for complex tasks.

## Additional Questions
52. **How can generative AI be used for data augmentation in supervised learning?**  
    Generative AI (e.g., GANs) creates synthetic data (e.g., images) to expand training sets, improving model robustness in tasks like image classification.

53. **What is AutoML, and how does it benefit ML pipelines?**  
    AutoML automates model selection, tuning, and preprocessing (e.g., Google AutoML), speeding up development and democratizing ML.

54. **How do you handle imbalanced datasets in classification tasks?**  
    - **SMOTE**: Oversamples minority class.  
    - **Class Weights**: Adjusts loss function.  
    - **Resampling**: Balances classes. These ensure fairness, e.g., in fraud detection.

55. **Write a Python function to implement linear regression using Scikit-learn.**  
    Demonstrates practical model building:  
    ```python
    from sklearn.linear_model import LinearRegression
    def fit_linear_regression(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model
    # Example usage
    X = [[1], [2], [3]]
    y = [2, 4, 6]
    model = fit_linear_regression(X, y)
    print(model.predict([[4]]))  # Predicts ~8
    ```