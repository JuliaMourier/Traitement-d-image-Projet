import numpy as np
import tensorflow as tf
import cv2

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical


class digitFinderModel:
    """
    CNN trained on mnist dataset to categorise numbers

    numbers are written in white with black background of size (28,28,1)

    don't forget to normalize the input image

    CNN is :
        - conv2D(32,(3,3), relu)
        - MaxPooling2D(2,2)
        - Flatten
        - Dense(100, relu)
        - Dense(10, softmax)

    out is one hot encoded
    """

    def __init__(self, toTrain: bool):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        if (toTrain):
            self.trainModel()

        self.model = load_model("digitModel")

    def trainModel(self):
        # load dataset
        (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
        # reshape input to one channel
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_valid = X_valid.reshape((X_valid.shape[0], 28, 28, 1))
        # normalize values
        X_train = X_train / 255
        X_valid = X_valid / 255
        # one hot encode target values
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

        # compile NN
        self.model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # callback to save best model
        checkpoint = ModelCheckpoint("digitModel", save_best_only=True)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid),
                       callbacks=[checkpoint])


def findDigit(image: np.ndarray):
    """
    function that takes an image and returns un number between 0 and 9 :
    number on image should be written black on white background

    - get pretrained model
    - prepare image to fit in NN input
    - get prediction from model
    - find associated number from one hot encoded prediction
    - return value

    function is to use on every part of sudoku found in processing scripts
    """

    # get model
    # MODEL SHOULD BE TRAINED - model is trained if "digitModel" directory exists
    model = digitFinderModel(False)

    # resize image to predict
    img = prepareImage(image)

    prediction = model.predict(img)

    number = getPredictionValue(prediction)

    return number


def prepareImage(image: np.ndarray) -> np.ndarray:
    img = image.copy()

    # check if image is gray scaled
    if not img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert black and white
    img = abs(img - 255)

    # reshape image to fit input of neural network (28,28,1)
    img.reshape(28, 28, 1)

    # normalize image values
    img = img / 255

    return img


def getPredictionValue(prediction: []) -> int:
    maxValue = 0
    idx = -1
    for i in range(0, len(prediction)):
        if prediction[i] > maxValue:
            maxValue = prediction[i]
            idx = i

    if idx >= 0:
        return idx
    else:
        print("error in finding prediction value, value is now 0")
        return 0
