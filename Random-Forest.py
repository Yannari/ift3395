# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X_train_file = pd.read_csv('train.csv')
Y_train_file = pd.read_csv('train_result.csv')
test_file = pd.read_csv('test.csv')

Y_train_file = Y_train_file['Class']
t = test_file.pop("Unnamed: 1568")
t = X_train_file.pop("Unnamed: 1568")
X_train_file = X_train_file.values
Y_train_file = Y_train_file.values
#test_file = test_file.values

from sklearn.model_selection import train_test_split
X_train_file, x_validation, Y_train_file, y_validation = train_test_split(X_train_file, Y_train_file, test_size=0.1, random_state=42)

print(X_train_file.shape)
print(Y_train_file.shape)
print(x_validation.shape)
print(y_validation.shape)

clf = RandomForestClassifier()
clf.fit(X_train_file, Y_train_file)

prediction_validation = clf.predict(x_validation)
print("Validation Accuracy: " + str(accuracy_score(y_validation, prediction_validation)))
print("Validation Confusion Matrix: \n" + str(confusion_matrix(y_validation, prediction_validation)))

prediction_test = clf.predict(test_file)
index = 5
print("Predicted " + str(prediction_test[index]))
plt.imshow(test_file.iloc[index].values.reshape((28, 56)), cmap='gray')
plt.show()
my_submission = pd.DataFrame({'Index': list(range(0, len(prediction_test))), 'Class': prediction_test})
my_submission.to_csv('submission2.csv', index=False)
