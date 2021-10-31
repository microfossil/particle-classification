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
    # Make both backends use the same session
    K.clear_session()

    intro()

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
    input_dir = params.get('input_source')
    data_min_count = params.get('data_min_count')
    data_split = params.get('data_split')
    data_split_offset = params.get('data_split_offset')
    seed = params.get('seed')

    # Output
    output_dir = params.get('save_dir')

    # Data -------------------------------------------------------------------------------------------------------------
    # print("@Loading images...")
    if img_channels == 3:
        color_mode = 'rgb'
    else:
        if cnn_type.endswith('tl'):
            color_mode = 'greyscale3'
            params['img_channels'] = 3
        else:
            color_mode = 'greyscale'
    print('Color mode: {}'.format(color_mode))

    if data_source is None:
        data_source = DataSource()
        data_source.use_mmap = params['use_mmap']
        data_source.set_source(input_dir,
                               data_min_count,
                               mapping=params['class_mapping'],
                               min_count_to_others=params['data_map_others'],
                               mmap_directory=params['mmap_directory'])
        data_source.load_dataset(img_size=(img_height, img_width),
                                 prepro_type=None,
                                 prepro_params=(255, 0, 1),
                                 img_type=color_mode,
                                 print_status=True)
    data_source.split(data_split, data_split_offset, seed)

    if params['use_class_weights'] is True:
        params['class_weights'] = data_source.get_class_weights()
        print("@Class weights are {}".format(params['class_weights']))
    else:
        params['class_weights'] = None
    params['num_classes'] = data_source.num_classes

    if cnn_type.endswith('tl'):

        # mnist = tf.keras.datasets.mnist
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train, x_test = x_train / 255.0, x_test / 255.0
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Flatten(input_shape=(28, 28)),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(10)
        # ])
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # model.compile(optimizer='adam',
        #               loss=loss_fn,
        #               metrics=['accuracy'])
        # model.fit(x_train, y_train, epochs=5)
        # model.evaluate(x_test, y_test, verbose=2)
        # probability_model = tf.keras.Sequential([
        #     model,
        #     tf.keras.layers.Softmax()
        # ])

        start = time.time()
        # Create Vectors -----------------------------------------------------------------------------------------------
        # Note that the images are scaled internally in the network to match the expected preprocessing
        # print(time.time())
        # model_head.predict(data_source.train_images[0:1024])
        # print(time.time())
        # model_head.predict(data_source.train_images[0:1024])
        # print(time.time())
        # test = data_source.train_images[0:1024].copy()
        # model_head.predict(test)
        # print(time.time())
        # model_head.predict(data_source.train_images[0:1024])
        # print(time.time())
        # test = data_source.train_images[0:1024].copy()
        # model_head.predict(test)
        # print(time.time())

        # Generate vectors
        model_head = generate_tl_head(params)
        print("@Calculating train vectors")
        t = time.time()
        train_vector = model_head.predict(data_source.train_images)
        print("!{}s elapsed".format(time.time() - t))
        print("@Calculating test vectors")
        t = time.time()
        test_vector = model_head.predict(data_source.test_images)
        print("!{}s elapsed".format(time.time() - t))
        # Clear
        K.clear_session()

        # print(train_vector.dtype)
        # print(test_vector.dtype)

        # Generate vectors (random!)
        # train_vector = np.random.random(size=[data_source.train_images.shape[0], 2048])
        # test_vector = np.random.random(size=[data_source.test_images.shape[0], 2048])

        # train_vector = []
        # test_vector = []
        # step = 64
        #
        # for i in range(0, len(data_source.train_images), step):
        #     train_vector.append(model_head.predict(data_source.train_images[i:i+step]))
        #     print("@Calculating train vectors - {} of {}".format(i, len(data_source.train_images)))
        # train_vector = np.concatenate(train_vector, axis=0)
        #
        # for i in range(0, len(data_source.test_images), step):
        #     test_vector.append(model_head.predict(data_source.test_images[i:i + step]))
        #     print("@Calculating test vectors - {} of {}".format(i, len(data_source.test_images)))
        # test_vector = np.concatenate(test_vector, axis=0)

        data_source.train_vectors = train_vector
        data_source.test_vectors = test_vector

        # Augmentation -------------------------------------------------------------------------------------------------
        # No augmentation - there is a bug in the batch normalisation layer for tensorflow v1.xx where the mean and variance
        # are still calculated even when the layer is set to not trainable. This means the vectors produced are not the
        # vary according to the batch. For augmentation we need to include the ResNet network (with its batch normalisation
        # layers) in the graph, and because of this bug, the training performance is poor.

        # Generator ----------------------------------------------------------------------------------------------------
        # No generator needed

        # Model --------------------------------------------------------------------------------------------------------
        print("@Generating tail")
        # Get  tail
        model_tail = generate_tl_tail(params, [train_vector.shape[1], ])
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Training -----------------------------------------------------------------------------------------------------
        # alr_cb = AdaptiveLearningRateScheduler(nb_epochs=alr_epochs,
        #                                        nb_drops=alr_drops)
        # print("@Training")
        # if data_split > 0:
        #     validation_data = (test_vector, data_source.test_onehots)
        # else:
        #     validation_data = None
        # history = model_tail.fit(train_vector,
        #                          data_source.train_onehots,
        #                          validation_data=validation_data,
        #                          epochs=max_epochs,
        #                          batch_size=batch_size,
        #                          shuffle=True,
        #                          verbose=0,
        #                          class_weight=params['class_weights'],
        #                          callbacks=[alr_cb])
        # end = time.time()
        # training_time = end - start
        # print("@Training time: {}s".format(training_time))
        # time.sleep(3)

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
        print("@Training")
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
        print("@Training time: {}s".format(training_time))
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
        print("@Generating model")
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
        print("@Training")
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
        print("@Training time: {}s".format(training_time))
        time.sleep(3)

        # Vector -------------------------------------------------------------------------------------------------------
        vector_model = generate_vector(model, params)

    # Graphs -----------------------------------------------------------------------------------------------------------
    print("@Generating results")
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
        print("@Calculating inference time {}/10: {:.3f}ms".format(i + 1, diff))
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
        print("@Estimating mislabeled")
        vectors = vector_model.predict(data_source.images)
        find_and_save_mislabelled(data_source.images,
                                  vectors,
                                  data_source.cls,
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

    print("@Deleting temporary files")
    data_source.delete_memmap_files(del_split=True, del_source=params['delete_mmap_files'])

    wave()

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
