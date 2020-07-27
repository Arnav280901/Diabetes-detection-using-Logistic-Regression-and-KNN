# Importing the libraries
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
x_data = pd.read_csv('Diabetes_XTrain.csv')
y_data = pd.read_csv('Diabetes_YTrain.csv')
X = x_data.iloc[:, :].values
y = y_data.iloc[:, :].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("PREDICTED RESULTS, TEST SET RESULTS\n")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX")
print(cm)
print("\nACCURACY")
print(accuracy_score(y_test, y_pred))

# Visualizing the results
plt.scatter(X[432:577, 4], y_pred.reshape(len(y_pred), 1), color='red')
plt.title('RELATIONSHIP BETWEEN INSULIN AND DIABETIES')
plt.xlabel('Insulin')
plt.ylabel('Diabetic/Non-Diabetic (1/0)')
plt.show()
