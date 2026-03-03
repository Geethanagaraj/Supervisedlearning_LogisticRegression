#step 1
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.Linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#step 2
data = load_breast_cancer()
x = data.data
y = data.targer

