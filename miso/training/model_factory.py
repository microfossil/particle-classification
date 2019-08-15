import math

from tensorflow.keras.models import Model
from miso.models.marchitto_transfer import transfer_learning_dense_layers
from miso.models.resnet50_tl import resnet50_transfer_learning, resnet50_transfer_learning_head
from miso.models.base_cyclic import *
from classification_models import Classifiers

def generate(params: dict()):
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
    # ResNet50 Transfer Learning
    # Uses the pre-trained ResNet50 network from tf.keras with full image input and augmentation
    # Has a lambda layer to rescale the normal image input (range 0-1) to that expected by the pre-trained network
    elif type == 'resnet50_tl':
        model_tl = resnet50_transfer_learning_head(input_shape=(img_height, img_width, img_channels),
                                                   nb_classes=params['num_classes'])
        model_dense = transfer_learning_dense_layers(params['num_classes'])
        outputs = model_dense(model_tl.outputs[0])
        model = Model(model_tl.inputs[0], outputs)
        return model
    # ResNet, SEResNet and DenseNet from the image-classifiers python package
    # Uses normal keras
    else:
        classifier, preprocess_input = Classifiers.get(type)
        model = classifier(input_shape=(img_height, img_width, 1),
                           weights=None,
                           classes=params['num_classes'])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
