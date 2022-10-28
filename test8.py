from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import h5py
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Check how many examples do we have in our train and test sets
print(f"We have {len(X_train)} images in the training set and {len(X_test)} images in the test set.")

# Let's see the first sample of our training set
print(X_train[0].shape)

plt.imshow(X_train[0])
plt.figure(figsize=(3, 3))
plt.imshow(X_train[0], cmap="gray")
plt.title(y_train[0])
plt.axis(False)

random_image = random.randint(0, len(X_train))

plt.figure(figsize=(3, 3))
plt.imshow(X_train[random_image], cmap="gray")

plt.title(y_train[random_image])
plt.axis(False)

#X_train = X_train.reshape(X_train.shape + (1,))
X_train = X_train.reshape(-1, 28, 28, 1)
#X_test = X_test.reshape(X_test.shape + (1, ))
X_test = X_test.reshape(-1, 28, 28, 1)

#y_train = y_train.reshape(-1, 2)
print(X_train.shape)

print(y_train.shape)
X_train = X_train / 255.
X_test = X_test / 255.

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


model = tf.keras.Sequential([
    layers.Conv2D(filters=10,
                  kernel_size=3,
                  activation="relu",
                  input_shape=(28, 28, 1)),
    layers.Conv2D(10, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(10, 3, activation="relu"),
    layers.Conv2D(10, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])
print(model.summary())
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=250)
model.evaluate(X_test, y_test)
model.save("digit-recognizer.h5")
