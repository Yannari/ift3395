# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sys


def main():
	# 2. LOAD TESTING AND TRAINING DATASET
    

    X_train_file = pd.read_csv(sys.argv[1])
    Y_train_file = pd.read_csv('train_result.csv')
    test_file = pd.read_csv(sys.argv[2])

    Y_train_file = Y_train_file['Class']
    t = test_file.pop("Unnamed: 1568")
    t = X_train_file.pop("Unnamed: 1568")
    X_train_file = X_train_file.values
    Y_train_file = Y_train_file.values
    #test_file = test_file.values


    ##########################################################
	# TRAIN RANDOM FOREST ALGORITHM
	##########################################################
	# Create a RandomForestClassifier object with the parameters over the data
    from sklearn.model_selection import train_test_split
    X_train_file, x_validation, Y_train_file, y_validation = train_test_split(X_train_file, Y_train_file, test_size=0.1, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train_file, Y_train_file)

    prediction_validation = clf.predict(x_validation)
    print("Validation Accuracy: " + str(accuracy_score(y_validation, prediction_validation)))
    print("Validation Confusion Matrix: \n" + str(confusion_matrix(y_validation, prediction_validation)))

    # APPLY THE TRAINED LEARNER TO TEST NEW DATA
    prediction_test = clf.predict(test_file)
    index = 5
    print("Predicted " + str(prediction_test[index]))
    plt.imshow(test_file.iloc[index].values.reshape((28, 56)), cmap='gray')
    plt.show()

    #PUT DATA IN A CSV
    my_submission = pd.DataFrame({'Index': list(range(0, len(prediction_test))), 'Class': prediction_test})
    my_submission.to_csv('./submission2.csv', index=False)
    print("data save in submission2.csv in the current folder")

print("LOADING...")
main()
