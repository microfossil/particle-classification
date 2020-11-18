import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Activation, \
                                    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

import numpy as np

from miso.layers import cyclic

def base_cyclic(input_shape,
                nb_classes,
                filters=4,
                blocks=None,
                dropout=0.5,
                dense=512,
                conv_padding='same',
                conv_activation='relu',
                use_batch_norm=True,
                global_pooling=None,
                use_depthwise_conv=True):
    # Number of blocks
    if blocks is None:
        blocks = int(np.log2(input_shape[0]) - 2)
    inputs = Input(shape=input_shape)
    x = cyclic.CyclicSlice4()(inputs)
    # Convolution blocks
    for i in range(blocks):
        conv_filters = filters * 2 ** i
        for j in range(2):
            x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
            if use_batch_norm is True:
                x = BatchNormalization()(x)
            x = Activation(conv_activation)(x)
        x = MaxPooling2D()(x)
        x = cyclic.CyclicRoll4()(x)
    if global_pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif global_pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    # Dense layers
    x = Flatten()(x)
    x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    x = Dropout(dropout)(x)
    x = Dense(dense, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    # Return model
    model = Model(inputs, x, name='base_cyclic')
    return model


def mirror_cyclic(input_shape,
                nb_classes,
                filters=4,
                blocks=4,
                dropout=0.5,
                dense=512,
                conv_padding='same',
                conv_activation='relu',
                use_batch_norm=True,
                global_pooling=None):

    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }

    inputs = Input(shape=input_shape)
    x = cyclic.CyclicSlice4()(inputs)
    for i in range(blocks):
        conv_filters = filters * 2 ** i
        # First layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        if use_batch_norm is True:
            # x = GroupNormalization(conv_filters)(x)
            x = BatchNormalization()(x)
            # x = LayerNormalization()(x)
            # x = BatchInstanceNormalisation()(x)
            # x = Lambda(lambda x: batch_instance_norm(x, scope="bin_{}_0".format(i)))(x)
            # x = Lambda((lambda x: tf.layers.batch_normalization(x, training=K.learning_phase())))(x)
        xa = Activation(conv_activation)(x)
        xb = Activation(conv_activation)(-x)
        x = tf.stack([xa, xb], 4)
        x = tf.reduce_max(x, axis=-1)
        # Second layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        if use_batch_norm is True:
            # x = GroupNormalization(conv_filters)(x)
            x = BatchNormalization()(x)
            # x = LayerNormalization()(x)
            # x = BatchInstanceNormalisation()(x)
            # x = Lambda(lambda x: batch_instance_norm(x, scope="bin_{}_1".format(i)))(x)
            # x = Lambda((lambda x: tf.layers.batch_normalization(x, training=K.learning_phase())))(x)
        # x = Activation(conv_activation)(x)
        xa = Activation(conv_activation)(x)
        xb = Activation(conv_activation)(-x)
        x = tf.stack([xa, xb], 4)
        x = tf.reduce_max(x, axis=-1)
        # Pool
        x = MaxPooling2D()(x)
        # Roll
        x = cyclic.CyclicRoll4()(x)
    if global_pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif global_pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    # Dense layers
    x = Flatten()(x)
    x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    x = Dropout(dropout)(x)
    x = Dense(dense, activation='relu')(x)
    # x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs, x, name='base_cyclic')
    return model

