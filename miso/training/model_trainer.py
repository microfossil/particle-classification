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

from miso.stats.mislabelling import plot_mislabelled
from miso.data.datasource import DataSource
from miso.data.generators import *
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.accuracy import *
from miso.stats.training import *
from miso.training.augmentation import *
from miso.save.freezing import freeze_or_save, convert_to_inference_mode
from miso.training.model_info import ModelInfo
from miso.training.model_factory import *


def train_image_classification_model(params: dict, data_source: DataSource = None):
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
    input_dir = params.get('input_source')
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

    data_source = DataSource()
    data_source.set_source(input_dir, data_min_count)
    data_source.load_images(img_size=(img_height, img_width),
                            prepro_type=None,
                            prepro_params=(255, 0, 1),
                            color_mode=color_mode,
                            split=data_split,
                            seed=params['seed'],
                            print_status=True)

    if params['use_class_weights'] is True:
        params['class_weights'] = data_source.get_class_weights()
    else:
        params['class_weights'] = None
    params['num_classes'] = data_source.num_classes

    if cnn_type == 'resnet50_tl':
        # Model --------------------------------------------------------------------------------------------------------
        print("@Generating model")
        model_uses_tf_keras = True
        # Get head and tail
        model_head, model_tail = generate_tl(params)
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Create Vectors -----------------------------------------------------------------------------------------------
        # Note that the images are scaled internally in the network to match the ResNet50 ImageNet expected scaling.
        train_vector = model_head.predict(data_source.train_images)
        test_vector = model_head.predict(data_source.test_images)
        data_source.train_vectors = train_vector
        data_source.test_vectors = test_vector

        # Augmentation -------------------------------------------------------------------------------------------------
        # No augmentation - there is a bug in the batch normalisation layer for tensorflow v1.xx where the mean and variance
        # are still calculated even when the layer is set to not trainable. This means the vectors produced are not the
        # vary according to the batch. For augmentation we need to include the ResNet network (with its batch normalisation
        # layers) in the graph, and because of this bug, the training performance is poor.

        # Generator ----------------------------------------------------------------------------------------------------
        # No generator needed

        # Training -----------------------------------------------------------------------------------------------------
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=alr_epochs,
                                               nb_drops=alr_drops,
                                               tf_keras=model_uses_tf_keras)
        print("@Training")
        start = time.time()
        history = model_tail.fit(train_vector,
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

        # Generator ----------------------------------------------------------------------------------------------------
        # Now we be tricky and join the trained dense layers to the resnet model to create a model that accepts images
        # as input
        outputs = model_tail(model_head.outputs[0])
        model = ModelK(model_head.inputs[0], outputs)
        model.summary()

        # Vector -------------------------------------------------------------------------------------------------------
        vector_model = generate_vector(model, params)

    else:
        # Model --------------------------------------------------------------------------------------------------------
        print("@Generating model")
        if cnn_type.startswith("base_cyclic"):
            model_uses_tf_keras = True
        else:
            model_uses_tf_keras = False
        model = generate(params)

        # Vector -------------------------------------------------------------------------------------------------------
        vector_model = generate_vector(model, params)

        # Augmentation -------------------------------------------------------------------------------------------------
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

        # Generator ----------------------------------------------------------------------------------------------------
        train_gen = tf_augmented_image_generator(data_source.train_images,
                                                 data_source.train_onehots,
                                                 batch_size,
                                                 augment_fn)
        test_gen = image_generator(data_source.test_images,
                                   data_source.test_onehots,
                                   batch_size)

        # Training -----------------------------------------------------------------------------------------------------

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
    save_dir = os.path.join(output_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(name, now))
    os.makedirs(save_dir, exist_ok=True)

    # Plot the graphs
    plot_loss_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
    plot_accuracy_vs_epochs(history)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plot_confusion_accuracy_matrix(y_true, y_pred, data_source.cls_labels)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plot_precision_recall(y_true, y_pred, data_source.cls_labels)
    plt.savefig(os.path.join(save_dir, "precision_recall.png"))

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
    vector_model = generate_vector(model, params)

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
    outputs["vector"] = vector_model.outputs[0]
    info = ModelInfo(name,
                     description,
                     cnn_type,
                     now,
                     "frozen_model.pb",
                     params,
                     inputs,
                     outputs,
                     input_dir,
                     data_source.cls_labels,
                     data_source.cls_counts,
                     "rescale",
                     [255, 0, 1],
                     result.accuracy,
                     result.precision,
                     result.recall,
                     result.f1_score,
                     result.support)

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
    return model, vector_model, data_source, result

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
