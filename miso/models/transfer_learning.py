import tensorflow as tf
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda
from tensorflow.keras.models import Model, Sequential


def resnet50_head(input_shape):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda y: tf.reverse(y, axis=[-1]))(inputs)
    x = Lambda(lambda y: y * tf.constant(255.0)
                         - tf.reshape(tf.constant([103.939, 116.779, 128.68]),
                                      [1, 1, 1, 3]))(x)
    x = resnet50.ResNet50(include_top=False,
                          weights='imagenet',
                          pooling='avg')(x)
    model = Model(inputs=inputs, outputs=x)
    model.get_layer('resnet50').trainable = False
    return model


def marchitto_tail(nb_classes):
    model = Sequential()
    model.add(Dropout(0.05))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    return model
