import os
import collections
import tensorflow as tf
import tensorflow.keras as tfkeras
from miso.layers._common_blocks import ChannelSE
from miso.layers.cyclic import *
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if tfkeras.backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre', use_cyclic=False):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tfkeras.layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = tfkeras.layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = tfkeras.layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tfkeras.layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
        # if use_cyclic:
        #     x = CyclicRoll4()(x)

        x = tfkeras.layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = tfkeras.layers.Activation('relu', name=relu_name + '2')(x)
        x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tfkeras.layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
        if use_cyclic:
            x = CyclicRoll4()(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = tfkeras.layers.Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre', use_cyclic=False):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tfkeras.layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = tfkeras.layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = tfkeras.layers.Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = tfkeras.layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = tfkeras.layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = tfkeras.layers.Activation('relu', name=relu_name + '2')(x)
        x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tfkeras.layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = tfkeras.layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = tfkeras.layers.Activation('relu', name=relu_name + '3')(x)
        x = tfkeras.layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = tfkeras.layers.Add()([x, shortcut])

        return x

    return layer


# -------------------------------------------------------------------------
#   Residual Model Builder
# -------------------------------------------------------------------------


def ResNetCyclic(model_params,
                 input_shape=None,
                 input_tensor=None,
                 include_top=True,
                 classes=1000,
                 weights='imagenet',
                 **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `tfkeras.layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = tfkeras.layers.Input(input_shape)
    else:
        if not tfkeras.backend.is_keras_tensor(input_tensor):
            img_input = tfkeras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block
    if model_params.attention:
        Attention = model_params.attention(**kwargs)
    else:
        Attention = None

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = model_params.filters

    # resnet bottom
    if model_params.use_cyclic:
        x = CyclicSlice4()(img_input)
        # x = tfkeras.layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(x)
    else:
        x = img_input
        # x = tfkeras.layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    # x = tfkeras.layers.ZeroPadding2D(padding=(3, 3))(x)
    # x = tfkeras.layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)

    # 2 layers of 3 x 3 to start!
    x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tfkeras.layers.Conv2D(init_filters, (3, 3), strides=(1, 1), name='conv0a', **conv_params)(x)
    # if model_params.use_cyclic:
    #     x = CyclicRoll4()(x)
    x = tfkeras.layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = tfkeras.layers.Activation('relu', name='relu0')(x)

    x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tfkeras.layers.Conv2D(init_filters, (3, 3), strides=(1, 1), name='conv0b', **conv_params)(x)
    if model_params.use_cyclic:
        x = CyclicRoll4()(x)

    # x = tfkeras.layers.ZeroPadding2D(padding=(1, 1))(x)
    # x = tfkeras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** (stage + 1))

            # first block of first stage without strides because we have maxpooling before
            # if block == 0 and stage == 0:
            #     x = ResidualBlock(filters, stage, block, strides=(1, 1),
            #                       cut='post', attention=Attention)(x)

            if block == 0:
                x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                  cut='post', attention=Attention)(x)

            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='pre', attention=Attention)(x)

    x = tfkeras.layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = tfkeras.layers.Activation('relu', name='relu1')(x)

    if model_params.global_pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif model_params.global_pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    x = tfkeras.layers.Flatten()(x)
    x = CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    x = tfkeras.layers.Dropout(0.5)(x)
    x = tfkeras.layers.Dense(512, activation='relu')(x)
    x = tfkeras.layers.Dense(classes, activation='softmax')(x)

    # # resnet top
    # if include_top:
    #     x = tfkeras.layers.GlobalAveragePooling2D(name='pool1')(x)
    #     x = tfkeras.layers.Dense(classes, name='fc1')(x)
    #     x = tfkeras.layers.Activation('softmax', name='softmax')(x)
    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tfkeras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tfkeras.models.Model(inputs, x)
    return model


ResnetModelParameters = collections.namedtuple(
    'ModelParams',
    ['model_name', 'filters', 'repetitions', 'residual_block', 'attention', 'use_cyclic', 'global_pooling']
)

MODELS_PARAMS = {
    'resnet18': ResnetModelParameters('resnet18', 64, (2, 2, 2, 2), residual_conv_block, None, use_cyclic=True, global_pooling=None),
    'resnet34': ResnetModelParameters('resnet34', 64, (3, 4, 6, 3), residual_conv_block, None, use_cyclic=True, global_pooling=None),
    'resnet50': ResnetModelParameters('resnet50', 64, (3, 4, 6, 3), residual_bottleneck_block, None, use_cyclic=True, global_pooling=None),
    'resnet101': ResnetModelParameters('resnet101', 64, (3, 4, 23, 3), residual_bottleneck_block, None,
                                       use_cyclic=True, global_pooling=None),
    'resnet152': ResnetModelParameters('resnet152', 64, (3, 8, 36, 3), residual_bottleneck_block, None,
                                       use_cyclic=True, global_pooling=None),
    'seresnet18': ResnetModelParameters('seresnet18', 64, (2, 2, 2, 2), residual_conv_block, ChannelSE,
                                        use_cyclic=True, global_pooling=None),
    'seresnet34': ResnetModelParameters('seresnet34', 64, (3, 4, 6, 3), residual_conv_block, ChannelSE,
                                        use_cyclic=True, global_pooling=None),
}
