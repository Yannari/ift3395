from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import h5py
import pandas as pd
from keras import models
import tqdm

test_df = pd.read_csv('train.csv')
test_df.pop("Unnamed: 1568")
test_df.head()
print(test_df.shape)

X_test = test_df.values
print(X_test.shape)

# Scale pixel values
X_test = X_test.astype('float32') / 255

# Reshape to (batch, W, H, channels)
X_test = X_test.reshape(-1, 28, 56, 1)
# print(X_test.shape)

# plot image entry #42, reshape image to size 56x28 pixel
plt.imshow(X_test[43].reshape(28, 56), cmap=plt.cm.gray)

X_test = X_test.reshape(-1, 28, 28, 1)
print(X_test.shape)
model = models.load_model('digit-recognizer.h5')
y_pred = model.predict(X_test)
#y_pred.reshape(5264, 19)
y2_pred = np.zeros((5264, 19), dtype=float)
#y2_pred.reshape(5264, 19)
# print(y2_pred.shape)
y2_pred = y_pred
X_test = X_test.reshape(-1, 28, 56, 1)
print("shape: ", X_test.shape)
for i in range(10, 16):
    plt.subplot(280 + (i % 10 + 1))
    plt.imshow(X_test[i].reshape(28, 56), cmap=plt.cm.gray)
    print(y2_pred)
    print(y2_pred[i])
    plt.title(y2_pred[i].argmax())
plt.show()

output = []
for index, prediction in tqdm.tqdm(enumerate(y2_pred), total=y2_pred.shape[0]):
    image_id = index + 1
    label = prediction.argmax()
    output.append({"Index": image_id, "Class": label})
    print(i)
    i = i + 1

submission_df = pd.DataFrame(output)
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
