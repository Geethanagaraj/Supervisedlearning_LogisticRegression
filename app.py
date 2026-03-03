#step 1
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.Linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#step 2
data = load_breast_cancer()
x = data.data
y = data.targer

#step 3
x_train, x_test, y_train, y_test = train_test_split(test_size = 0.2, random_state = 42)

#step 4
model = LogisticRegression(max_iter = 5000)
model.fit(x_train, y_train)

