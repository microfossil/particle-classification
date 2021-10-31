"""
Creates and trains a generic network
"""
import time
import datetime
from collections import OrderedDict

from miso.stats.mislabelling import find_and_save_mislabelled
from miso.archive.datasource import DataSource
from miso.archive.generators import *
from miso.utils.wave import *
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from archive.augmentation import *
from miso.deploy.saving import freeze, convert_to_inference_mode
from miso.deploy.model_info import ModelInfo
from miso.models.model_factory import *


def train_image_classification_model(params: dict, data_source: DataSource = None):
    K.clear_session()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    intro()

    # Params -----------------------------------------------------------------------------------------------------------
    name = params.get('name')
    description = params.get('description')

    # Network
    cnn_type = params.get('type')

    # Input
    img_size = params.get('img_size')
    if img_size is not None:
        [img_height, img_width, img_channels] = params.get('img_size')
    else:
        img_height = params.get('img_height')
        img_width = params.get('img_width')
        img_channels = params.get('img_channels')

    # Training
    batch_size = params.get('batch_size', 64)
    max_epochs = params.get('max_epochs', 1000)
    alr_epochs = params.get('alr_epochs', 10)
    alr_drops = params.get('alr_drops', 4)

    # Input data
    input_dir = params.get('input_source', None)
    data_min_count = params.get('data_min_count', 40)
    data_split = params.get('data_split', 0.2)
    data_split_offset = params.get('data_split_offset', 0)
    seed = params.get('seed', None)

    # Output
    output_dir = params.get('save_dir')

    # Type
    # - rgb
    # - greyscale
    # - greyscale3
    # - rgbd
    # - greyscaled
    img_type = params.get('img_type', None)
    if img_type is None:
        if img_channels == 3:
            img_type = 'rgb'
        elif img_channels == 1:
            if cnn_type.endswith('tl'):
                img_type = 'greyscale3'
                params['img_channels'] = 3
            else:
                img_type = 'greyscale'
        else:
            raise ValueError("Number of channels must be 1 or 3")
    elif img_type == 'rgbd':
        params['img_channels'] = 4
    elif img_type == 'greyscaled':
        params['img_channels'] = 2
    elif img_type == 'greyscaledm':
        params['img_channels'] = 3

    print('@ Image type: {}'.format(img_type))

    # Data -------------------------------------------------------------------------------------------------------------
    if data_source is None:
        data_source = DataSource()
        data_source.use_mmap = params['use_mmap']
        data_source.set_source(input_dir,
                               data_min_count,
                               mapping=params['class_mapping'],
                               min_count_to_others=params['data_map_others'],
                               mmap_directory=params['mmap_directory'])
        data_source.load_dataset(img_size=(img_height, img_width), img_type=img_type)
    data_source.split(data_split, seed)

    if params['use_class_weights'] is True:
        params['class_weights'] = data_source.get_class_weights()
        print("@ Class weights are {}".format(params['class_weights']))
    else:
        params['class_weights'] = None
    params['num_classes'] = data_source.num_classes

    if cnn_type.endswith('tl'):
        start = time.time()

        # Generate vectors
        model_head = generate_tl_head(params)
        print("@ Calculating train vectors")
        t = time.time()
        train_vector = model_head.predict(data_source.train_images)
        print("! {}s elapsed".format(time.time() - t))
        print("@ Calculating test vectors")
        t = time.time()
        test_vector = model_head.predict(data_source.test_images)
        print("! {}s elapsed".format(time.time() - t))
        # Clear
        K.clear_session()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        data_source.train_vectors = train_vector
        data_source.test_vectors = test_vector

        # Augmentation -------------------------------------------------------------------------------------------------
        # No augmentation as we pre-calculate vectors

        # Generator ----------------------------------------------------------------------------------------------------
        # No generator needed

        # Model --------------------------------------------------------------------------------------------------------
        print("@ Generating tail")
        # Get  tail
        model_tail = generate_tl_tail(params, [train_vector.shape[1], ])
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Generator ----------------------------------------------------------------------------------------------------
        train_gen = tf_vector_generator(train_vector,
                                        data_source.train_onehots,
                                        batch_size)
        test_gen = tf_vector_generator(test_vector,
                                       data_source.test_onehots,
                                       batch_size)

        # Training -----------------------------------------------------------------------------------------------------
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=alr_epochs,
                                               nb_drops=alr_drops)
        print("@ Training")
        if data_split > 0:
            validation_data = test_gen
        else:
            validation_data = None
        # log_dir = "C:\\logs\\profile\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)
        history = model_tail.fit_generator(
            train_gen,
            steps_per_epoch=math.ceil(len(train_vector) // batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(len(test_vector) // batch_size),
            epochs=max_epochs,
            verbose=0,
            shuffle=False,
            max_queue_size=1,
            class_weight=params['class_weights'],
            callbacks=[alr_cb])
        end = time.time()
        training_time = end - start
        print("@ Training time: {}s".format(training_time))
        time.sleep(3)

        # Generator ----------------------------------------------------------------------------------------------------
        # Now we be tricky and join the trained dense layers to the resnet model to create a model that accepts images
        # as input
        model_head = generate_tl_head(params)
        outputs = model_tail(model_head.output)
        model = Model(model_head.input, outputs)
        model.summary()

        # Vector -------------------------------------------------------------------------------------------------------
        vector_model = generate_vector(model, params)

    else:
        # Model --------------------------------------------------------------------------------------------------------
        print("@ Generating model")
        start = time.time()
        model = generate(params)

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
                                               nb_drops=alr_drops)
        print("@ Training")
        if data_split > 0:
            validation_data = test_gen
        else:
            validation_data = None
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=math.ceil(len(data_source.train_images) // batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(len(data_source.test_images) // batch_size),
            epochs=max_epochs,
            verbose=0,
            shuffle=False,
            max_queue_size=1,
            class_weight=params['class_weights'],
            callbacks=[alr_cb])
        end = time.time()
        training_time = end - start
        print("@ Training time: {}s".format(training_time))
        time.sleep(3)

        # Vector -------------------------------------------------------------------------------------------------------
        vector_model = generate_vector(model, params)

    # Graphs -----------------------------------------------------------------------------------------------------------
    print("@ Generating results")
    if data_split > 0:
        # Calculate test set scores
        y_true = data_source.test_cls
        y_prob = model.predict(data_source.test_images)
        y_pred = y_prob.argmax(axis=1)
    else:
        y_true = np.asarray([])
        y_prob = np.asarray([])
        y_pred = np.asarray([])

    # Inference time
    max_count = np.min([1000, len(data_source.images)])
    to_predict = np.copy(data_source.images[0:max_count])

    inf_times = []
    for i in range(3):
        start = time.time()
        model.predict(to_predict)
        end = time.time()
        diff = (end - start) / max_count * 1000
        inf_times.append(diff)
        print("@ Calculating inference time {}/10: {:.3f}ms".format(i + 1, diff))
    inference_time = np.median(inf_times)

    # Store results
    result = TrainingResult(params,
                            history,
                            y_true,
                            y_pred,
                            y_prob,
                            data_source.cls_labels,
                            training_time,
                            inference_time)

    # Save the results
    now = datetime.datetime.now()
    save_dir = os.path.join(output_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(name, now))
    os.makedirs(save_dir, exist_ok=True)

    # Plot the graphs
    # plot_model(model, to_file=os.path.join(save_dir, "model_plot.pdf"), show_shapes=True)
    if data_split > 0:
        plot_loss_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "loss_vs_epoch.pdf"))
        plot_accuracy_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.pdf"))
        plot_confusion_accuracy_matrix(y_true, y_pred, data_source.cls_labels)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.pdf"))
        plt.close('all')

    if params['save_mislabeled'] is True:
        print("@ Estimating mislabeled")
        vectors = vector_model.predict(data_source.images)
        find_and_save_mislabelled(data_source.images,
                                  vectors,
                                  data_source.cls,
                                  data_source.cls_labels,
                                  data_source.get_short_filenames(),
                                  save_dir,
                                  11)

    # Save model -------------------------------------------------------------------------------------------------------
    print("@ Saving model")
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
                     result.support,
                     result.epochs[-1],
                     training_time,
                     params['data_split'],
                     inference_time)

    # Freeze and save graph
    if params['save_model'] is not None:
        freeze(model,
               os.path.join(save_dir, "model"),
               info)

    # Save info
    info.save(os.path.join(save_dir, "model", "network_info.xml"))

    print("@ Deleting temporary files")
    data_source.delete_memmap_files(del_split=True, del_source=params['delete_mmap_files'])

    wave()

    print("@ Complete")
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
