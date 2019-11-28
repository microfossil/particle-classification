import math
from tensorflow.keras.models import Model
from miso.models.transfer_learning import *
from miso.models.base_cyclic import *
from miso.models.resnet import *
from classification_models.tfkeras import Classifiers


def generate(params: dict):
    # Network
    type = params.get('type')

    # Input
    img_height = params.get('img_height')
    img_width = params.get('img_width')
    img_channels = params.get('img_channels')
    input_shape = (img_height, img_width, img_channels)

    # Base Cyclic
    # Create at CEREGE specifically for foraminifera by adding cyclic layers
    if type.startswith("base_cyclic"):
        blocks = int(math.log2(img_height) - 2)
        model = base_cyclic(input_shape=input_shape,
                            nb_classes=params['num_classes'],
                            filters=params['filters'],
                            blocks=blocks,
                            dropout=0.5,
                            dense=512,
                            conv_activation=params['activation'],
                            use_batch_norm=params['use_batch_norm'],
                            global_pooling=params['global_pooling'])
    # ResNet Cyclic
    elif type.startswith("resnet_cyclic"):
        blocks = int(math.log2(img_height) - 2)
        blocks -= 1     # Resnet has one block to start with already
        resnet_params = ModelParams('resnet_cyclic',
                                    params['filters'],
                                    [1 for i in range(4)],
                                    residual_conv_block,
                                    None,
                                    use_cyclic=True)
        model = ResNetCyclic(resnet_params, input_shape, None, True, params['num_classes'])

    # ResNet50 Transfer Learning
    # Uses the pre-trained ResNet50 network from tf.keras with full image input and augmentation
    # Has a lambda layer to rescale the normal image input (range 0-1) to that expected by the pre-trained network
    elif type.endswith('tl'):
        model_head, model_tail = generate_tl(params)
        outputs = model_tail(model_head.outputs[0])
        model = Model(model_head.inputs[0], outputs)
        return model
    # ResNet, SEResNet and DenseNet from the image-classifiers python package
    # Uses normal keras
    else:
        classifier, preprocess_input = Classifiers.get(type)
        model = classifier(input_shape=(img_height, img_width, img_channels),
                           weights=None,
                           classes=params['num_classes'])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def generate_tl(params: dict):
    # Network
    cnn_type = params.get('type')

    # Input
    img_height = params.get('img_height')
    img_width = params.get('img_width')
    img_channels = params.get('img_channels')
    input_shape = (img_height, img_width, img_channels)

    model_head = head(cnn_type, input_shape=input_shape)
    model_tail = tail(params['num_classes'])

    return model_head, model_tail


def generate_vector(model, params: dict):
    cnn_type = params['type']

    if cnn_type.endswith("tl"):
        vector_layer = model.layers[-1].layers[-2]
        vector_model = Model(model.inputs, vector_layer.output)
        # vector_model = model
    elif cnn_type.startswith("base_cyclic"):
        vector_layer = model.get_layer(index=-2)
        vector_model = Model(model.inputs, vector_layer.output)
    elif cnn_type.startswith("resnet_cyclic"):
        vector_layer = model.get_layer(index=-2)
        vector_model = Model(model.inputs, vector_layer.output)
    elif cnn_type.startswith("resnet") or cnn_type.startswith("seresnet"):
        vector_layer = model.get_layer(index=-3)
        vector_model = Model(model.inputs, vector_layer.output)
    elif cnn_type.startswith("vgg") or cnn_type.startswith("densenet"):
        vector_layer = model.get_layer(index=-2)
        vector_model = Model(model.inputs, vector_layer.output)
    else:
        raise ValueError("The network type, {}, is not valid".format(cnn_type))

    return vector_model
