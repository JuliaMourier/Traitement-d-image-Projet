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

    numbers are written in white with black background of size (50,50,1)

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

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
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

        new_X_train = []
        new_X_valid = []

        for train in X_train :
            #new_X_train.append(prepareImage(train))
            new_X_train.append(cv2.resize(train, (50,50), interpolation=cv2.INTER_AREA))

        for valid in X_valid :
            #new_X_valid.append(prepareImage(valid))
            new_X_valid.append(cv2.resize(valid, (50,50), interpolation=cv2.INTER_AREA))

        new_X_train = np.array(new_X_train)
        new_X_valid = np.array(new_X_valid)

        # reshape input to one channel
        new_X_train = new_X_train.reshape((new_X_train.shape[0], 50, 50, 1))
        new_X_valid = new_X_valid.reshape((new_X_valid.shape[0], 50, 50, 1))

        # normalize values
        new_X_train = new_X_train / 255
        new_X_valid = new_X_valid / 255

        # one hot encode target values
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

        # compile NN
        self.model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # callback to save best model
        checkpoint = ModelCheckpoint("digitModel", save_best_only=True)

        self.model.fit(new_X_train, y_train, epochs=10, batch_size=32, validation_data=(new_X_valid, y_valid),
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

    prediction = model.model.predict(img)

    number = getPredictionValue(prediction[0])

    return number


def prepareImage(image: np.ndarray) -> np.ndarray:
    img = image.copy()

    # check if image is gray scaled
    if not len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert black and white
    img = cv2.bitwise_not(img)
    # resize image to fit input of neural network (50,50,1)
    #kernel = np.ones((1, 1), np.uint8)
    #img_eroded = cv2.dilate(img, kernel)
    #img = cv2.resize(img_eroded, (50,50), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (50,50), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel)
    #img = cv2.resize(img, (50,50), interpolation=cv2.INTER_AREA)

    img = np.reshape(img, (1, 50, 50, 1))
    # normalize image values
    img = img.astype('float32')
    img = img / 255.0

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

#digitFinderModel(True)