import tensorflow.keras as tfk
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.models import Sequential, load_model
# Loading de MNIST data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(y_train[0:12])

plt.figure(figsize=(5, 5))
for k in range(12):
    plt.subplot(3, 4, k + 1)
    plt.imshow(X_train[k], cmap="Greys")
    plt.axis("off")
plt.tight_layout()
plt.show()
X_train = X_train.reshape(60000, 784).astype("float32")
X_valid = X_valid.reshape(10000, 784).astype("float32")
X_train /= 255
X_valid /= 255
n_classes = 10
y_train = tfk.utils.to_categorical(y_train, n_classes)
y_valid = tfk.utils.to_categorical(y_valid, n_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(len(X_train), activation='softmax'))

print(model.summary())

checkpoint = ModelCheckpoint("custom_model", save_best_only=True)
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['acc'])

# history = model.fit(train_generator,validation_data = validation_generator,epochs = 20, verbose = 1, callbacks = [checkpoint])
model = load_model("custom_model")

#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

#epochs = range(len(acc))

plt.figure(figsize=(6, 5))

plt.plot(epochs, acc, 'r', label='training_accuracy')
plt.plot(epochs, val_acc, 'b', label='validation_accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(figsize=(6, 5))

plt.plot(epochs, loss, 'r', label='training_loss')
plt.plot(epochs, val_loss, 'b', label='validation_loss')
plt.title('Training and Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
