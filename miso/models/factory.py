import math
from collections import namedtuple

from miso.models.transfer_learning import *
from miso.models.base_cyclic import *
from miso.models.resnet_cyclic import *
from classification_models.tfkeras import Classifiers
from miso.training.parameters import MisoParameters

try:
    from tensorflow.keras.applications.efficientnet import *
except ImportError:
    pass


FactoryModelParameters = namedtuple("FactoryModelParameters", "factory_fn default_size default_colour")


def generate(tp: MisoParameters):
    # Base Cyclic
    # Create at CEREGE specifically for foraminifera by adding cyclic layers
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
    # ResNet Cyclic
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
        model = ResNetCyclic(resnet_params, tp.cnn.img_shape, None, True, tp.cnn.num_classes)
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
    # ResNet50 Transfer Learning
    # Uses the pre-trained ResNet50 network from tf.keras with full image input and augmentation
    # Has a lambda layer to rescale the normal image input (range 0-1) to that expected by the pre-trained network
    elif tp.cnn.id.endswith('tl'):
        model_head, model_tail = generate_tl(tp.cnn.id, tp.dataset.num_classes, tp.cnn.img_shape, tp.cnn.use_msoftmax)
        outputs = model_tail(model_head.outputs[0])
        model = Model(model_head.inputs[0], outputs)
        return model
    # ResNet, SEResNet and DenseNet from the image-classifiers python package
    # Uses normal keras
    else:
        classifier, preprocess_input = Classifiers.get(tp.cnn.id)
        model = classifier(input_shape=tp.cnn.img_shape,
                           weights=None,
                           classes=tp.cnn.num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_tl_head(cnn_type, img_shape):
    model_head = head(cnn_type, input_shape=img_shape)
    return model_head


def generate_tl_tail(num_classes, input_shape, use_msoftmax):
    model_tail = tail(num_classes, input_shape, use_msoftmax=use_msoftmax)
    return model_tail


def generate_tl(cnn_type, num_classes, img_shape, use_msoftmax):
    model_head = generate_tl_head(cnn_type, img_shape)
    model_tail = tail(num_classes, [model_head.layers[-1].output.shape[-1], ], use_msoftmax=use_msoftmax)
    return model_head, model_tail


def generate_vector(model, cnn_type):
    if cnn_type.endswith("tl"):
        vector_tensor = model.get_layer(index=-2).get_output_at(0)
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
