import math
from miso.models.transfer_learning import *
from miso.models.base_cyclic import *
from miso.models.resnet_cyclic import *
from classification_models.tfkeras import Classifiers
from miso.training.parameters import TrainingParameters

try:
    from tensorflow.keras.applications.efficientnet import *
except ImportError:
    pass


def generate(tp: TrainingParameters):
    # Base Cyclic
    # Create at CEREGE specifically for foraminifera by adding cyclic layers
    if tp.type.startswith("base_cyclic"):
        blocks = int(math.log2(tp.img_shape[0]) - 2)
        model = base_cyclic(input_shape=tp.img_shape,
                            nb_classes=tp.num_classes,
                            filters=tp.filters,
                            blocks=blocks,
                            dropout=0.5,
                            dense=512,
                            conv_activation=tp.activation,
                            use_batch_norm=tp.use_batch_norm,
                            global_pooling=tp.global_pooling)
    # ResNet Cyclic
    elif tp.type.startswith("resnet_cyclic"):
        blocks = int(math.log2(tp.img_shape[0]) - 2)
        blocks -= 1  # Resnet has one block to start with already
        resnet_params = ResnetModelParameters('resnet_cyclic',
                                    tp.filters,
                                    [1 for i in range(blocks)],
                                    residual_conv_block,
                                    None,
                                    use_cyclic=True)
        model = ResNetCyclic(resnet_params, tp.img_shape, None, True, tp.num_classes)
    # EfficientNet
    elif tp.type.lower().startswith(("efficientnet")):
        if tp.type.lower() == "efficientnetb0":
            model_fn = EfficientNetB0
        elif tp.type.lower() == "efficientnetb1":
            model_fn = EfficientNetB1
        elif tp.type.lower() == "efficientnetb2":
            model_fn = EfficientNetB2
        elif tp.type.lower() == "efficientnetb3":
            model_fn = EfficientNetB3
        elif tp.type.lower() == "efficientnetb4":
            model_fn = EfficientNetB4
        elif tp.type.lower() == "efficientnetb5":
            model_fn = EfficientNetB5
        elif tp.type.lower() == "efficientnetb6":
            model_fn = EfficientNetB6
        elif tp.type.lower() == "efficientnetb7":
            model_fn = EfficientNetB7
        model = model_fn(weights=None,
                         input_shape=tp.img_shape,
                         classes=tp.num_classes)
    # ResNet50 Transfer Learning
    # Uses the pre-trained ResNet50 network from tf.keras with full image input and augmentation
    # Has a lambda layer to rescale the normal image input (range 0-1) to that expected by the pre-trained network
    elif tp.type.endswith('tl'):
        model_head, model_tail = generate_tl(tp.type, tp.num_classes, tp.img_shape)
        outputs = model_tail(model_head.outputs[0])
        model = Model(model_head.inputs[0], outputs)
        return model
    # ResNet, SEResNet and DenseNet from the image-classifiers python package
    # Uses normal keras
    else:
        classifier, preprocess_input = Classifiers.get(type)
        model = classifier(input_shape=tp.img_shape,
                           weights=None,
                           classes=tp.num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
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
        raise ValueError("The network type, {}, is not valid".format(cnn_type))
    return vector_model
