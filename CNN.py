#1. import all packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tfhj


def main():
    # 2. LOAD TESTING AND TRAINING DATASET
    X_train = pd.read_csv(sys.argv[1])
    Y_train = pd.read_csv("train_result.csv")
    test = pd.read_csv(sys.argv[2])
    Y_train = Y_train['Class']

    t = X_train.pop("Unnamed: 1568")
    t = test.pop("Unnamed: 1568")

    X_train = X_train.values.reshape(-1, 28, 56, 1)
    idx = np.arange(X_train.shape[0])
    np.random.shuffle(idx)
    X_train = X_train[idx]
    Y_train = Y_train[idx]

    test = test.values.reshape(-1, 28, 56, 1)

    # one-hot encoding
    from keras.utils.np_utils import to_categorical
    Y_train = to_categorical(Y_train, num_classes=19)

    from sklearn.model_selection import train_test_split
    X_train, X_cv, Y_train, y_cv = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    from keras.layers import Input, InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
    from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
    from keras.models import Sequential, Model
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    import keras
    from keras import backend as K

    # Building a CNN model
    input_shape = (28, 56, 1)
    X_input = Input(input_shape)

    # layer 1
    x = Conv2D(64, (3, 3), strides=(1, 1), name='layer_conv1', padding='same')(X_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='maxPool1')(x)
    # layer 2
    x = Conv2D(32, (3, 3), strides=(1, 1), name='layer_conv2', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='maxPool2')(x)
    # layer 3
    x = Conv2D(32, (3, 3), strides=(1, 1), name='conv3', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='maxPool3')(x)
    # fc
    x = Flatten()(x)
    x = Dense(64, activation='relu', name='fc0')(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(19, activation='softmax', name='fc2')(x)

    conv_model = Model(inputs=X_input, outputs=x, name='Predict')
    conv_model.summary()

    conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    conv_model.fit(X_train, Y_train, epochs=250, batch_size=16, validation_data=(X_cv, y_cv))

    sgd = SGD(lr=1e-3, momentum=0.5, decay=1e-4, nesterov=False)
    conv_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    conv_model.fit(X_train, Y_train, epochs=250, validation_data=(X_cv, y_cv))

    # APPLY THE TRAINED LEARNER TO TEST NEW DATA
    y_pred = conv_model.predict(test)
    y_pred = np.argmax(y_pred, axis=1)
    plt.imshow(X_train[1][:, :, 0])
    plt.title(y_pred[1])
    plt.show()

    #PUT DATA IN A CSV
    my_submission = pd.DataFrame({'Index': list(range(0, len(y_pred))), 'Class': y_pred})
    my_submission.to_csv('./submission.csv', index=False)

print("LOADING...")
main()
