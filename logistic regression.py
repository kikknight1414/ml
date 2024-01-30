import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 
3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X, y)
pred=model.predict(X)
print("Predicted values are : ",pred)
print("Actual values are : ",y)
print('Accuracy: ', accuracy_score(y, pred))
print('Recall: ', recall_score(y, pred))
print('Precision: ', precision_score(y, pred))
print('Confusion Matrix: \n', confusion_matrix(y, pred))
