from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
irisData = load_iris()
X = irisData.data
y = irisData.target

# Display dataset and target values
print("DataSet of Lengths and Breadth of sepals and petals:")
print(X)
print("Targets for the above data:")
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Display predicted values by the model
print("Predicted values by the model:")
print(y_pred)

# Display actual target values in the test dataset
print("Actual Target values in the Test Dataset:")
print(y_test)

# Display confusion matrix
print("Confusion Matrix of above y_pred and y_test:\n", confusion_matrix(y_test, y_pred))

# Display classification report
print("Classification report:\n", classification_report(y_test, y_pred))
