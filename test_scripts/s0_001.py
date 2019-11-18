import tensorflow as tf
import numpy as np
from tensorflow import keras

class simple(tf.keras.layers.Layer):
    def __init__(self):
        super(simple, self).__init__()

    def build(self, input_shape):
        super(simple,self).build(input_shape)

    def call(self, input):
        input_shape = input.shape
        numChan = input_shape[-1]
        numBatch = input_shape[0]
        width = input_shape[1]
        height = input_shape[2]
        return input


mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    simple(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

test = model(train_images[0:10])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, verbose=2)