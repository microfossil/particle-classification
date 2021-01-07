import math
from collections import namedtuple

from miso.models.transfer_learning import *
from miso.models.base_cyclic import *
from miso.models.resnet_cyclic import *
from classification_models.tfkeras import Classifiers, ModelsFactory
from miso.training.parameters import MisoParameters

try:
    from tensorflow.keras.applications.efficientnet import *
except ImportError:
    pass


def generate(tp: MisoParameters):
    #
    # Transfer learning
    #
    # Uses any of the available models in keras applications
    if tp.cnn.id.endswith('tl'):
        parts = tp.cnn.id.split("_")
        if parts[0] in TRANSFER_LEARNING_PARAMS.keys():
            model_head, model_tail = generate_tl(tp.cnn.id, tp.dataset.num_classes, tp.cnn.img_shape)
            return combine_tl(model_head, model_tail)
        else:
            raise ValueError("The CNN type {} is not supported, valid CNNs are {}".format(parts[0], TRANSFER_LEARNING_PARAMS.keys()))

    #
    # Full network training
    #
    # Base Cyclic - custom network created at CEREGE specifically for foraminifera by adding cyclic layers
    if tp.cnn.id.startswith("base_cyclic"):
        model = base_cyclic(input_shape=tp.cnn.img_shape,
                            nb_classes=tp.dataset.num_classes,
                            filters=tp.cnn.filters,
                            blocks=tp.cnn.blocks,
                            dropout=0.5,
                            dense=512,
                            conv_activation=tp.cnn.activation,
                            use_batch_norm=tp.cnn.use_batch_norm,
                            global_pooling=tp.cnn.global_pooling)
    # ResNet Cyclic - custom network created at CEREGE specifically for foraminifera by adding cyclic layers
    elif tp.cnn.id.startswith("resnet_cyclic"):
        blocks = int(math.log2(tp.cnn.img_shape[0]) - 2)
        blocks -= 1  # Resnet has one block to start with already
        resnet_params = ResnetModelParameters('resnet_cyclic',
                                    tp.cnn.filters,
                                    [1 for i in range(blocks)],
                                    residual_conv_block,
                                    None,
                                    use_cyclic=True,
                                    global_pooling=tp.cnn.global_pooling)
        model = ResNetCyclic(resnet_params, tp.cnn.img_shape, None, True, tp.dataset.num_classes)
    # EfficientNet
    elif tp.cnn.id.lower().startswith(("efficientnet")):
        if tp.cnn.id.lower() == "efficientnetb0":
            model_fn = EfficientNetB0
        elif tp.cnn.id.lower() == "efficientnetb1":
            model_fn = EfficientNetB1
        elif tp.cnn.id.lower() == "efficientnetb2":
            model_fn = EfficientNetB2
        elif tp.cnn.id.lower() == "efficientnetb3":
            model_fn = EfficientNetB3
        elif tp.cnn.id.lower() == "efficientnetb4":
            model_fn = EfficientNetB4
        elif tp.cnn.id.lower() == "efficientnetb5":
            model_fn = EfficientNetB5
        elif tp.cnn.id.lower() == "efficientnetb6":
            model_fn = EfficientNetB6
        elif tp.cnn.id.lower() == "efficientnetb7":
            model_fn = EfficientNetB7
        model = model_fn(weights=None,
                         input_shape=tp.cnn.img_shape,
                         classes=tp.dataset.num_classes)
    # ResNet, SEResNet, DenseNet and others from qubvel's image-classifiers python package
    elif tp.cnn.id in ModelsFactory().models.keys():
        classifier, preprocess_input = Classifiers.get(tp.cnn.id)
        model = classifier(input_shape=tp.cnn.img_shape,
                           weights=None,
                           classes=tp.dataset.num_classes)
    else:
        raise ValueError(
            "The CNN type {} is not supported, valid CNNs are base_cyclic, resnet_cyclic, efficientnetb[0-7] and {}".format(tp.cnn.id, ModelsFactory().models.keys()))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_tl_head(cnn_type, img_shape):
    model_head = head(cnn_type, input_shape=img_shape)
    return model_head


def generate_tl_tail(num_classes, input_shape):
    model_tail = tail(num_classes, input_shape)
    return model_tail


def generate_tl(cnn_type, num_classes, img_shape):
    model_head = generate_tl_head(cnn_type, img_shape)
    model_tail = tail(num_classes, [model_head.layers[-1].output.shape[-1], ])
    return model_head, model_tail

def combine_tl(model_head, model_tail):
    return Model(inputs=model_head.input, outputs=model_tail.call(model_head.output))


def generate_vector(model, cnn_type):
    if cnn_type.endswith("tl"):
        vector_tensor = model.get_layer(index=-2).get_output_at(1)
        vector_model = Model(model.inputs, vector_tensor)
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
        vector_layer = model.get_layer(index=-2)
        vector_model = Model(model.inputs, vector_layer.output)
    return vector_model
