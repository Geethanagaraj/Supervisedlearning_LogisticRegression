# Supervised Learning – Logistic Regression on Breast Cancer Dataset

This is a **beginner-friendly machine learning project** using Python's `scikit-learn`.  
We use **Logistic Regression** to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on tumor features.

---

## **Dataset**
We are using the **Breast Cancer dataset** from `sklearn.datasets`, which contains:
- 569 samples
- 30 features (e.g., mean radius, texture, perimeter, area, smoothness)
- 2 classes: 0 = malignant, 1 = benign

---

## **Project Steps**

1. **Load the dataset** using `sklearn.datasets.load_breast_cancer()`.  
2. **Split the data** into training and testing sets using `train_test_split`.  
3. **Create a Logistic Regression model** (`LogisticRegression`) and train it.  
4. **Make predictions** on the test set.  
5. **Evaluate the model** using accuracy and confusion matrix.

---

## **Python Libraries Used**
- `scikit-learn` (`sklearn`) → For machine learning models and metrics  
- `numpy` → For array operations (optional)  

---

## **Code Example**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))