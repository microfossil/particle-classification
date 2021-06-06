import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    return x


def sep_conv_block(x, nfilters, size=3, padding='same', initializer="he_normal"):
    x = DepthwiseConv2D(kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=nfilters, kernel_size=(1, 1), padding=padding, kernel_initializer=initializer)(x)
    x = Activation("elu")(x)
    x = DepthwiseConv2D(kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=nfilters, kernel_size=(1, 1), padding=padding, kernel_initializer=initializer)(x)
    x = Activation("elu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2), type='upsample'):
    if type == 'upsample':
        y = UpSampling2D()(tensor)
    else:
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def sep_deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2), type='upsample'):
    if type == 'upsample':
        y = UpSampling2D()(tensor)
    else:
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = sep_conv_block(y, nfilters)
    return y

def UNetOld(filters,
         num_classes,
         model_fn=Model,
         return_vector=False,
         activation='softmax'
         ):
    # Down
    input_layer = Input(shape=(None, None, 3), name='image')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters * 2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters * 4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters * 8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters * 16)
    conv5 = Dropout(0.5)(conv5)
    # Up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters * 8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
    deconv7 = Dropout(0.5)(deconv7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    output_layer = Conv2D(filters=num_classes, kernel_size=(1, 1), activation=activation)(deconv9)
    if return_vector:
        model = model_fn(input_layer, [output_layer, deconv9])
    else:
        model = model_fn(input_layer, output_layer)
    model.summary()
    return model
