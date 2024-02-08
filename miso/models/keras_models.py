import collections
import tensorflow as tf

from keras import applications as ka, Input, Model
from keras.layers import Lambda, Dropout, Dense, BatchNormalization

from miso.layers.cyclic import CyclicGainSlice12, CyclicDensePoolN, CyclicSlice4, CyclicDensePool4

KerasModelParameters = collections.namedtuple(
    'KerasModelParameters',
    ['model_func', 'prepro_func', 'default_input_shape']
)


def tf_prepro(input_shape):
    # (x / 127.5) - 1
    inputs = Input(shape=input_shape)
    x = Lambda(lambda y: tf.subtract(tf.multiply(y, 2), 1))(inputs)
    return inputs, x


def torch_prepro(input_shape):
    # (x / 255 - mean) / std_dev
    inputs = Input(shape=input_shape)
    x = Lambda(lambda y: tf.subtract(y, tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, 1, 3])))(inputs)
    x = Lambda(lambda y: tf.divide(y, tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, 1, 3])))(x)
    return inputs, x


def caffe_prepro(input_shape):
    # Convert to BGR then x - mean
    inputs = Input(shape=input_shape)
    x = Lambda(lambda y: tf.reverse(y, axis=[-1]))(inputs)
    x = Lambda(lambda y: tf.subtract(tf.multiply(y, 255.0), tf.reshape(tf.constant([103.939, 116.779, 128.68]), [1, 1, 1, 3])))(x)
    return inputs, x


def no_prepro(input_shape):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda y: tf.multiply(y, 255.0))(inputs)
    return inputs, x


def head(cnn_type, use_cyclic, use_gain, input_shape, weights='imagenet', add_prepro=False, trainable=True):
    if cnn_type not in KERAS_MODEL_PARAMETERS:
        raise ValueError(f"CNN type {cnn_type} is not supported, valid values are {KERAS_MODEL_PARAMETERS.keys()}")

    params = KERAS_MODEL_PARAMETERS[cnn_type]
    if add_prepro:
        inputs, x = params.prepro_func(input_shape)
    else:
        inputs = Input(shape=input_shape)
        x = inputs
    if use_cyclic:
        if use_gain:
            x = CyclicGainSlice12()(x)
        else:
            x = CyclicSlice4()(x)
    x = params.model_func(include_top=False, weights=weights, pooling='avg', input_shape=input_shape).call(x)
    if use_cyclic:
        if use_gain:
            x = CyclicDensePoolN(pool_op=tf.reduce_mean)(x)
        else:
            x = CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    model = Model(inputs=inputs, outputs=x)
    if not trainable:
        for layer in model.layers:
            layer.trainable = False
    return model


def tail(num_classes, input_shape, dropout=(0.4, 0.3)):
    # inp = Input(shape=input_shape)
    # x = Dropout(rate=dropout[0], name='dropout1')(inp)
    # x = BatchNormalization()(x)
    # x = Dense(512, activation='relu', bias_initializer='zeros')(x)
    # x = Dropout(rate=dropout[1], name='dropout2')(x)
    # x = BatchNormalization()(x)
    # outp = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

    # inp = Input(shape=input_shape)
    # x = Dropout(rate=0.5)(inp)
    # # x = BatchNormalization()(x)
    # x = Dense(512, activation='relu', bias_initializer='zeros')(x)
    # # x = Dropout(rate=dropout[1], name='dropout2')(x)
    # # x = BatchNormalization()(x)
    # outp = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

    inp = Input(shape=input_shape)
    outp = Dropout(dropout[0])(inp)
    outp = Dense(512, activation='relu')(outp)
    # outp = Dropout(dropout[1])(outp)
    # outp = Dense(512, activation='relu')(outp)
    outp = Dense(num_classes, activation='softmax')(outp)

    # outp = Dense(num_classes, activation='softmax')(inp)

    return Model(inp, outp)


def tail_vector(num_classes, input_shape, dropout=(0.5, 0.5)):
    inp = Input(shape=input_shape)
    outp = Dropout(dropout[0])(inp)
    outp = Dense(512, activation='relu')(outp)
    # outp = Dropout(dropout[1])(outp)
    # outp = Dense(512, activation='relu')(outp)
    return Model(inp, outp)


KERAS_MODEL_PARAMETERS = {
    'xception': KerasModelParameters(ka.xception.Xception, tf_prepro, [299, 299, 3]),
    'vgg16': KerasModelParameters(ka.vgg16.VGG16, caffe_prepro, [224, 224, 3]),
    'vgg19': KerasModelParameters(ka.vgg19.VGG19, caffe_prepro, [224, 224, 3]),
    'resnet50': KerasModelParameters(ka.resnet.ResNet50, caffe_prepro, [224, 224, 3]),
    'resnet50v2': KerasModelParameters(ka.resnet_v2.ResNet50V2, tf_prepro, [224, 224, 3]),
    'resnet101': KerasModelParameters(ka.resnet.ResNet101, caffe_prepro, [224, 224, 3]),
    'resnet101v2': KerasModelParameters(ka.resnet_v2.ResNet101V2, tf_prepro, [224, 224, 3]),
    'resnet152': KerasModelParameters(ka.resnet.ResNet152, caffe_prepro, [224, 224, 3]),
    'resnet152v2': KerasModelParameters(ka.resnet_v2.ResNet152V2, tf_prepro, [224, 224, 3]),
    'inceptionresnetV2': KerasModelParameters(ka.inception_resnet_v2.InceptionResNetV2, tf_prepro, [299, 299, 3]),
    'mobilenet': KerasModelParameters(ka.mobilenet.MobileNet, tf_prepro, [224, 224, 3]),
    'mobilenetV2': KerasModelParameters(ka.mobilenet_v2.MobileNetV2, tf_prepro, [224, 224, 3]),
    'densenet121': KerasModelParameters(ka.densenet.DenseNet121, torch_prepro, [224, 224, 3]),
    'densenet169': KerasModelParameters(ka.densenet.DenseNet169, torch_prepro, [224, 224, 3]),
    'densenet201': KerasModelParameters(ka.densenet.DenseNet201, torch_prepro, [224, 224, 3]),
    'nasnetmobile': KerasModelParameters(ka.nasnet.NASNetMobile, tf_prepro, [224, 224, 3]),
    'nasnetlarge': KerasModelParameters(ka.nasnet.NASNetLarge, tf_prepro, [331, 331, 3]),
    'efficientnetb0': KerasModelParameters(ka.efficientnet.EfficientNetB0, no_prepro, [224, 224, 3]),
    'efficientnetb1': KerasModelParameters(ka.efficientnet.EfficientNetB1, no_prepro, [240, 240, 3]),
    'efficientnetb2': KerasModelParameters(ka.efficientnet.EfficientNetB2, no_prepro, [260, 260, 3]),
    'efficientnetb3': KerasModelParameters(ka.efficientnet.EfficientNetB3, no_prepro, [300, 300, 3]),
    'efficientnetb4': KerasModelParameters(ka.efficientnet.EfficientNetB4, no_prepro, [380, 380, 3]),
    'efficientnetb5': KerasModelParameters(ka.efficientnet.EfficientNetB5, no_prepro, [456, 456, 3]),
    'efficientnetb6': KerasModelParameters(ka.efficientnet.EfficientNetB6, no_prepro, [528, 528, 3]),
    'efficientnetb7': KerasModelParameters(ka.efficientnet.EfficientNetB7, no_prepro, [600, 600, 3]),
    'convnexttiny': KerasModelParameters(ka.convnext.ConvNeXtTiny, no_prepro, [224, 224, 3]),
    'convnextsmall': KerasModelParameters(ka.convnext.ConvNeXtSmall, no_prepro, [224, 224, 3]),
    'convnextbase': KerasModelParameters(ka.convnext.ConvNeXtBase, no_prepro, [224, 224, 3]),
    'convnextlarge': KerasModelParameters(ka.convnext.ConvNeXtLarge, no_prepro, [224, 224, 3]),
    'convnextxlarge': KerasModelParameters(ka.convnext.ConvNeXtXLarge, no_prepro, [224, 224, 3]),
}
