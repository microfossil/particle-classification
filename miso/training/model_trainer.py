"""
Creates and trains a generic network
"""
import os
import time
import datetime
from collections import OrderedDict

import tensorflow.keras.backend as K
import keras.backend as J

from tensorflow.keras.models import Model as ModelK
from keras.models import Model as ModelJ
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from miso.models.marchitto_transfer import marchitto_transfer
from miso.stats.mislabelling import plot_mislabelled

from miso.data.datasource import DataSource
from miso.data.generators import *
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from miso.training.augmentation import *
from miso.export.freezing import freeze_or_save, convert_to_inference_mode
from miso.training.model_info import ModelInfo
from miso.training.model_factory import generate

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def train_image_classification_model(params: dict, data_source: DataSource = None):
    # Make both backends use the same session
    J.set_session(K.get_session())

    if params['type'] == 'resnet50_tl':
        return train_image_classification_model_transfer_learning(params)

    # Params -----------------------------------------------------------------------------------------------------------
    name = params.get('name')
    description = params.get('description')

    # Network
    cnn_type = params.get('type')

    # Input
    img_height = params.get('img_height')
    img_width = params.get('img_width')
    img_channels = params.get('img_channels')

    # Training
    batch_size = params.get('batch_size')
    max_epochs = params.get('max_epochs')
    alr_epochs = params.get('alr_epochs')
    alr_drops = params.get('alr_drops')

    # Input data
    input_dir = params.get('input_dir')
    data_min_count = params.get('data_min_count')
    data_split = params.get('data_split')

    # Output
    output_dir = params.get('output_dir')

    # Data -------------------------------------------------------------------------------------------------------------
    # print("@Loading images...")
    if img_channels == 3:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'

    if data_source is None:
        data_source = DataSource()
        data_source.set_directory_source(input_dir, data_min_count)
        data_source.load_images(img_size=(img_height, img_width),
                                rescale_params=(255, 0, 1),
                                color_mode=color_mode,
                                split=data_split,
                                seed=params['seed'],
                                print_status=True)
    else:
        pass
    # Need to add "resplit"

    if params['use_class_weights'] is True:
        params['class_weights'] = data_source.get_class_weights()
    else:
        params['class_weights'] = None
    params['num_classes'] = data_source.num_classes

    # Model ------------------------------------------------------------------------------------------------------------
    print("@Generating model")
    if cnn_type.startswith("base_cyclic") or cnn_type == 'resnet50_tl':
        model_uses_tf_keras = True
    else:
        model_uses_tf_keras = False
    model = generate(params)

    # Vector -----------------------------------------------------------------------------------------------------------
    if cnn_type.startswith("base_cyclic"):
        vector_layer = model.get_layer(index=-2)
        vector_model = ModelK(model.inputs, vector_layer.output)
    elif cnn_type == "resnet50_tl":
        vector_layer = model.get_layer(index=-2)
        vector_model = ModelK(model.inputs, vector_layer.output)
    elif cnn_type.startswith("resnet") or cnn_type.startswith("seresnet"):
        vector_layer = model.get_layer(index=-3)
        vector_model = ModelJ(model.inputs, vector_layer.output)
    elif cnn_type.startswith("vgg") or cnn_type.startswith("densenet"):
        vector_layer = model.get_layer(index=-2)
        vector_model = ModelJ(model.inputs, vector_layer.output)
    else:
        raise ValueError("The network type, {}, is not valid".format(cnn_type))

    # Augmentation -----------------------------------------------------------------------------------------------------
    if params['aug_rotation'] is True:
        rotation_range = [0, 360]
    else:
        rotation_range = None

    def augment(x):
        return augmentation_complete(x,
                                     rotation=rotation_range,
                                     gain=params['aug_gain'],
                                     gamma=params['aug_gamma'],
                                     zoom=params['aug_zoom'],
                                     gaussian_noise=params['aug_gaussian_noise'],
                                     bias=params['aug_bias'])

    if params['use_augmentation'] is True:
        augment_fn = augment
    else:
        augment_fn = None

    # Generator --------------------------------------------------------------------------------------------------------
    train_gen = tf_augmented_image_generator(data_source.train_images,
                                             data_source.train_onehots,
                                             batch_size,
                                             augment_fn)
    test_gen = image_generator(data_source.test_images,
                               data_source.test_onehots,
                               batch_size)

    # Training ---------------------------------------------------------------------------------------------------------
    # tensorboard_cb = TensorBoard(log_dir='./tensorboard',
    #                              histogram_freq=0,
    #                              batch_size=32,
    #                              write_graph=True,
    #                              write_grads=False,
    #                              write_images=False,
    #                              embeddings_freq=0,
    #                              embeddings_layer_names=None,
    #                              embeddings_metadata=None,
    #                              embeddings_data=None,
    #                              update_freq='epoch')
    alr_cb = AdaptiveLearningRateScheduler(nb_epochs=alr_epochs,
                                           nb_drops=alr_drops,
                                           tf_keras=model_uses_tf_keras)
    print("@Training")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=math.ceil(len(data_source.train_images) // batch_size),
        validation_data=test_gen,
        validation_steps=math.ceil(len(data_source.test_images) // batch_size),
        epochs=max_epochs,
        verbose=0,
        shuffle=False,
        max_queue_size=1,
        class_weight=params['class_weights'],
        callbacks=[alr_cb])
    end = time.time()
    training_time = end - start
    print("@Training time: {}s".format(training_time))
    time.sleep(3)

    # Graphs -----------------------------------------------------------------------------------------------------------
    print("@Generating results")
    # Calculate test set scores
    y_true = data_source.test_cls
    y_prob = model.predict(data_source.test_images)
    y_pred = y_prob.argmax(axis=1)

    # Store results
    result = TrainingResult(params,
                            history,
                            y_true,
                            y_pred,
                            y_prob,
                            data_source.cls_labels,
                            training_time)

    # Save the results
    now = datetime.datetime.now()
    save_dir = os.path.join(output_dir, "{0}_{1:%Y-%m-%d_%H%M%S}".format(name, now))
    os.makedirs(save_dir, exist_ok=True)

    # Plot the graphs
    plot_loss_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
    plot_accuracy_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plot_confusion_accuracy_matrix(y_true, y_pred, data_source.cls_labels)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

    if params['save_mislabeled'] is True:
        print("@Estimating mislabeled")
        vectors = vector_model.predict(np.concatenate((data_source.train_images, data_source.test_images), axis=0))
        plot_mislabelled(data_source.get_images(),
                         vectors,
                         data_source.get_classes(),
                         data_source.cls_labels,
                         data_source.get_short_filenames(),
                         save_dir,
                         11)

    # Save model -------------------------------------------------------------------------------------------------------
    print("@Saving model")
    # Convert if necessary to fix TF batch normalisation issues
    model = convert_to_inference_mode(model, lambda: generate(params))

    # Generate description
    if description is None:
        description = "{}: {} model trained on data from {} ({} images in {} classes).\n" \
                      "Accuracy: {:.1f} (P: {:.1f}, R: {:.1f}, F1 {:.1f})".format(
            name,
            cnn_type,
            input_dir,
            len(data_source.data_df),
            len(data_source.cls_labels),
            result.accuracy * 100,
            result.mean_precision() * 100,
            result.mean_recall() * 100,
            result.mean_f1_score() * 100)

    # Create model info with all the parameters
    inputs = OrderedDict()
    inputs["image"] = model.inputs[0]
    outputs = OrderedDict()
    outputs["pred"] = model.outputs[0]
    outputs["vector"] = vector_layer.output
    info = ModelInfo(name,
                     description,
                     cnn_type,
                     now,
                     "frozen_model.pb",
                     params,
                     inputs,
                     outputs,
                     data_source.cls_labels,
                     [255, 0, 1],
                     input_dir,
                     result.accuracy,
                     result.mean_precision(),
                     result.mean_recall(),
                     result.mean_f1_score())

    # Freeze and save graph
    if params['save_model'] is not None:
        freeze_or_save(model,
                       os.path.join(save_dir, "model"),
                       info,
                       params['save_model'] == 'frozen',
                       model_uses_tf_keras)

    # Save info
    info.save(os.path.join(save_dir, "model", "network_info.xml"))

    print("@Complete")
    return model, data_source, result


def train_image_classification_model_transfer_learning(params: dict):
    # Make both backends use the same session
    J.set_session(K.get_session())

    # Params -----------------------------------------------------------------------------------------------------------
    name = params.get('name')
    description = params.get('description')

    # Network
    cnn_type = params.get('type')
    # - channels must be 3 for resnet50 transfer learning
    if cnn_type is 'resnet50_tl':
        params['img_height'] = 224
        params['img_width'] = 224
        params['img_channels'] = 3

    # Input
    img_height = params.get('img_height')
    img_width = params.get('img_width')
    img_channels = params.get('img_channels')

    # Training
    batch_size = params.get('batch_size')
    max_epochs = params.get('max_epochs')
    alr_epochs = params.get('alr_epochs')
    alr_drops = params.get('alr_drops')

    # Input data
    input_dir = params.get('input_dir')
    data_min_count = params.get('data_min_count')
    data_split = params.get('data_split')

    # Output
    output_dir = params.get('output_dir')

    # Data -------------------------------------------------------------------------------------------------------------
    print("@Loading images...")
    if img_channels == 3:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'
    data_source = DataSource()
    data_source.set_directory_source(input_dir, data_min_count)
    params['num_classes'] = data_source.num_classes

    data_source.load_images(img_size=(img_height, img_width),
                            rescale_params=(255, 0, 1),
                            color_mode=color_mode,
                            split=data_split,
                            seed=params['seed'],
                            print_status=True)

    if params['use_class_weights'] is True:
        params['class_weights'] = data_source.get_class_weights()
    else:
        params['class_weights'] = None

    # Model ------------------------------------------------------------------------------------------------------------
    print("@Generating model")
    model_uses_tf_keras = True
    # if cnn_type == "resnet50_tl":
    inputs = Input(shape=[img_height, img_width, img_channels])
    x = Lambda(lambda y: tf.reverse(y, axis=[-1]))(inputs)
    x = Lambda(lambda y: y * tf.constant(255.0)
                         - tf.reshape(tf.constant([103.939, 116.779, 128.68]),
                                      [1, 1, 1, 3]))(x)
    x = ResNet50(include_top=False,
                 weights='imagenet',
                 pooling='avg')(x)
    model_tl = Model(inputs, x)
        # model_tl = ResNet50(input_shape=(img_height, img_width, 3), include_top=False, pooling='avg')
    model_dense = marchitto_transfer(data_source.num_classes)
    model_dense.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_dense.summary()

    # Create Vectors ---------------------------------------------------------------------------------------------------
    # Rescale the images back to 0-255 so we can use the model preprocessing function
    # modified_train_images = preprocess_input(data_source.train_images * 255.0)
    # modified_test_images = preprocess_input(data_source.test_images * 255.0)
    train_vector = model_tl.predict(data_source.train_images)
    test_vector = model_tl.predict(data_source.test_images)
    data_source.train_vectors = train_vector
    data_source.test_vectors = test_vector

    # Augmentation -----------------------------------------------------------------------------------------------------
    # No augmentation - there is a bug in the batch normalisation layer for tensorflow v1.xx where the mean and variance
    # are still calculated even when the layer is set to not trainable. This means the vectors produced are not the
    # vary according to the batch. For augmentation we need to include the ResNet network (with its batch normalisation
    # layers) in the graph, and because of this bug, the training performance is poor.

    # Generator --------------------------------------------------------------------------------------------------------
    # No generator needed

    # Training ---------------------------------------------------------------------------------------------------------
    alr_cb = AdaptiveLearningRateScheduler(nb_epochs=alr_epochs,
                                           nb_drops=alr_drops,
                                           tf_keras=model_uses_tf_keras)
    print("@Training")
    start = time.time()
    history = model_dense.fit(train_vector,
                              data_source.train_onehots,
                              validation_data=(test_vector, data_source.test_onehots),
                              epochs=max_epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              verbose=0,
                              class_weight=params['class_weights'],
                              callbacks=[alr_cb])
    end = time.time()
    training_time = end - start
    print("@Training time: {}s".format(training_time))
    time.sleep(3)

    # Generator --------------------------------------------------------------------------------------------------------
    # Now we be tricky and join the trained dense layers to the resnet model to create a model that accepts images
    # as input
    # inputs = Input(shape=[img_height, img_width, img_channels])
    # x = Lambda(lambda y: tf.reverse(y, axis=[-1]))(inputs)
    # x = Lambda(lambda y: y * tf.constant(255.0)
    #                      - tf.reshape(tf.constant([103.939, 116.779, 128.68]),
    #                                   [1, 1, 1, 3]))(x)
    # x = ResNet50(include_top=False,
    #              weights='imagenet',
    #              pooling='avg')(x)
    outputs = model_dense(model_tl.outputs[0])
    model = Model(model_tl.inputs[0], outputs)
    model.summary()

    # Later we will need another version of this for freezing
    def create_model():
        inputs = Input(shape=[img_height, img_width, img_channels])
        x = Lambda(lambda y: tf.reverse(y, axis=[-1]))(inputs)
        x = Lambda(lambda y: y * tf.constant(255.0)
                             - tf.reshape(tf.constant([103.939, 116.779, 128.68]),
                                          [1, 1, 1, 3]))(x)
        x = ResNet50(include_top=False,
                     weights='imagenet',
                     pooling='avg')(x)
        outputs = marchitto_transfer(data_source.num_classes)(x)
        return Model(inputs, outputs)

    # Vector -----------------------------------------------------------------------------------------------------------
    if cnn_type == "resnet50_tl":
        vector_layer = model.layers[-1].layers[-2]
        vector_model = ModelK(model.inputs, vector_layer.output)
    else:
        raise ValueError("The network type, {}, is not valid".format(cnn_type))

    # Graphs -----------------------------------------------------------------------------------------------------------
    print("@Generating results")
    # Calculate test set scores
    y_true = data_source.test_cls
    y_prob = model.predict(data_source.test_images)
    y_pred = y_prob.argmax(axis=1)

    # Store results
    result = TrainingResult(params,
                            history,
                            y_true,
                            y_pred,
                            y_prob,
                            data_source.cls_labels,
                            training_time)

    # Save the results
    now = datetime.datetime.now()
    save_dir = os.path.join(output_dir, "{0}_{1:%Y-%m-%d_%H%M%S}".format(name, now))
    os.makedirs(save_dir, exist_ok=True)

    # Plot the graphs
    plot_loss_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
    plot_accuracy_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plot_confusion_accuracy_matrix(y_true, y_pred, data_source.cls_labels)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

    if params['save_mislabeled'] is True:
        print("@Estimating mislabeled")
        vectors = vector_model.predict(np.concatenate((data_source.train_images, data_source.test_images), axis=0))
        plot_mislabelled(data_source.get_images(),
                         vectors,
                         data_source.get_classes(),
                         data_source.cls_labels,
                         data_source.get_short_filenames(),
                         save_dir,
                         11)

    # Save model -------------------------------------------------------------------------------------------------------
    print("@Saving model")
    # Generate description
    if description is None:
        description = "{}: {} model trained on data from {} ({} images in {} classes).\n" \
                      "Accuracy: {:.1f} (P: {:.1f}, R: {:.1f}, F1 {:.1f})".format(
            name,
            cnn_type,
            input_dir,
            len(data_source.data_df),
            len(data_source.cls_labels),
            result.accuracy * 100,
            result.mean_precision() * 100,
            result.mean_recall() * 100,
            result.mean_f1_score() * 100)

    # Convert if necessary to fix TF batch normalisation issues
    model = convert_to_inference_mode(model, create_model)

    # Create model info with all the parameters
    inputs = OrderedDict()
    inputs["image"] = model.inputs[0]
    outputs = OrderedDict()
    outputs["pred"] = model.outputs[0]
    outputs["vector"] = model.layers[-1].layers[-2].output
    info = ModelInfo(name,
                     description,
                     cnn_type,
                     now,
                     "frozen_model.pb",
                     params,
                     inputs,
                     outputs,
                     data_source.cls_labels,
                     [255, 0, 1],
                     input_dir,
                     result.accuracy,
                     result.mean_precision(),
                     result.mean_recall(),
                     result.mean_f1_score())

    # Freeze and save graph
    if params['save_model'] is not None:
        freeze_or_save(model,
                       os.path.join(save_dir, "model"),
                       info,
                       params['save_model'] == 'frozen',
                       model_uses_tf_keras)

    print("@Complete")
    return model, data_source, result
