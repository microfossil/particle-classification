import math

from keras.optimizers import SGD, Adam

from miso.models.keras_models import head, tail
from miso.models.transfer_learning import *
from miso.models.base_cyclic import *
from miso.models.resnet_cyclic import *
from miso.training.parameters import MisoConfig

try:
    from tensorflow.keras.applications.efficientnet import *
except ImportError:
    pass


def generate(tp: MisoConfig):
    # Base Cyclic - custom network created at CEREGE specifically for foraminifera by adding cyclic layers
    if tp.cnn.type.startswith("base_cyclic"):
        model = base_cyclic(input_shape=tp.cnn.img_shape,
                            nb_classes=tp.dataset.num_classes,
                            filters=tp.cnn.filters,
                            blocks=tp.cnn.blocks,
                            dropout=0.5,
                            dense=512,
                            conv_activation=tp.cnn.activation,
                            use_batch_norm=tp.cnn.use_batch_norm,
                            global_pooling=tp.cnn.global_pooling)
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # ResNet Cyclic - custom network created at CEREGE specifically for foraminifera by adding cyclic layers
    elif tp.cnn.type.startswith("resnet_cyclic"):
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
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Standard networks in keras applications (with options for transfer learning or cyclic gain
    else:
        # Legacy support where transfer learning, cyclic and gain were part of the cnn type
        parts = tp.cnn.type.split("_")
        if len(parts) > 0:
            tp.cnn.type = parts[0]
            if "cyclic" in parts:
                tp.cnn.use_tl_cyclic = True
            if "cyclicgain" in parts:
                tp.cnn.use_tl_cyclic_gain = True
        if tp.cnn.type in KERAS_MODEL_PARAMETERS.keys():
            model_head = head(tp.cnn.type,
                              use_cyclic=tp.cnn.use_tl_cyclic,
                              use_gain=tp.cnn.use_tl_cyclic_gain,
                              input_shape=tp.cnn.img_shape,
                              weights='imagenet',
                              add_prepro=True,
                              trainable=True)
            model_tail = tail(tp.dataset.num_classes, [model_head.layers[-1].output.shape[-1], ])
            model = combine_head_and_tail(model_head, model_tail)
        else:
            raise ValueError(
                "The CNN type {} is not supported, valid CNNs are {}".format(tp.cnn.type, KERAS_MODEL_PARAMETERS.keys()))
        # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_tl(tp: MisoConfig):
    # Legacy support where transfer learning, cyclic and gain were part of the cnn type
    parts = tp.cnn.type.split("_")
    if len(parts) > 0:
        tp.cnn.type = parts[0]
        if "cyclic" in parts:
            tp.cnn.use_tl_cyclic = True
        if "cyclicgain" in parts:
            tp.cnn.use_tl_cyclic_gain = True
    if tp.cnn.type in KERAS_MODEL_PARAMETERS.keys():
        model_head = head(tp.cnn.type,
                          use_cyclic=tp.cnn.use_tl_cyclic,
                          use_gain=tp.cnn.use_tl_cyclic_gain,
                          input_shape=tp.cnn.img_shape,
                          weights='imagenet',
                          add_prepro=True,
                          trainable=False)
        model_tail = tail(tp.dataset.num_classes, [model_head.layers[-1].output.shape[-1], ])
    else:
        raise ValueError(
            "The CNN type {} is not supported, valid CNNs are {}".format(tp.cnn.type, KERAS_MODEL_PARAMETERS.keys()))
    opt = Adam(lr=0.001)
    model_tail.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_head, model_tail


def combine_head_and_tail(model_head, model_tail):
    return Model(inputs=model_head.input, outputs=model_tail.call(model_head.output))


def generate_vector_from_model(model, cnn_type):
    vector_tensor = model.get_layer(index=-2).get_output_at(1)
    vector_model = Model(model.inputs, vector_tensor)
    return vector_model
