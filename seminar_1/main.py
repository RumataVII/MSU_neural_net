import keras
import numpy as np
from keras import Input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam

# load the MNIST dataser
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x - images, y - respective numbers (human OCRed)

# transform the training and validation datasets to the categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# transform the training amnd validation datasets from grayscale (0.255) images to vectors in [0,1]
x_train = np.reshape(x_train, (-1, 28*28))
x_train = x_train.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28*28))
x_test = x_test.astype("float32") / 255.0

# preparing the nodel
model = Sequential([
    keras.Input(shape=(28*28,)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10),
    Activation('softmax'),
])

# preparing the metrics
train_acc_metric = CategoricalAccuracy()
val_acc_metric = CategoricalAccuracy()

# setting up the optimizer, the loss function, and metrics
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=[train_acc_metric]
)

model.summary()

# training the model eventually
batch_size = 32
epochs = 5
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
)

# evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

