import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mnist_train = pd.read_csv("test.csv")
# mnist_train.head()
# mnist_train.info()

some_digit_index = 5241
some_digit_label = mnist_train.iloc[some_digit_index].loc["Feature 1"]
some_digit_data = mnist_train.iloc[some_digit_index].drop("Feature 1")

def plot_digit(mnist_digit_data):
    digit_img = some_digit_data.values.reshape(28, 28)
    plt.imshow(digit_img, interpolation='nearest', cmap=plt.cm.binary)
    plt.axis("off")
    plt.show()
    
# plot_digit(some_digit_data)
print(some_digit_data)