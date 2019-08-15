import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model, Sequential
import miso.layers.cyclic as cyclic


def transfer_learning_dense_layers(nb_classes):
    model = Sequential()
    model.add(Dropout(0.05))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    return model
