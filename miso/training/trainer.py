"""
Creates and trains a generic network
"""
import time
import datetime
from collections import OrderedDict

from miso.data.training_dataset import TrainingDataset
from miso.stats.mislabelling import plot_mislabelled
from miso.archive.datasource import DataSource
from miso.archive.generators import *
from miso.utils.wave import *
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from miso.training.augmentation import *
from miso.deploy.saving import freeze, convert_to_inference_mode
from miso.deploy.model_info import ModelInfo
from miso.models.factory import *


def train_image_classification_model(tp: TrainingParameters):
    K.clear_session()

    tp.sanitise()

    print("+------------------------------------------------------------------------------+")
    print("| MISO Particle Classification Library                                         |")
    print("+------------------------------------------------------------------------------+")
    print("| To update library:                                                           |")
    print("| pip install -U git+http://www.github.com/microfossil/particle-classification |")
    print("+------------------------------------------------------------------------------+")
    print("@ Name: {}".format(tp.name))
    print("@ {}".format(tp.description))
    print("")
    print("@ CNN type: {}".format(tp.type))
    print("@ Image type: {}".format(tp.img_type))
    print("@ Image shape: {}".format(tp.img_shape))
    print("")

    # Load data
    ds = TrainingDataset(tp.source,
                         tp.img_shape,
                         tp.img_type,
                         tp.min_count,
                         tp.map_others,
                         tp.test_split,
                         tp.random_seed,
                         tp.memmap_directory)
    ds.load(tp.batch_size)
    tp.num_classes = ds.num_classes

    # ------------------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------------------
    if tp.type.endswith('tl'):
        print('-' * 80)
        start = time.time()
        # Generate head model and predict vectors
        model_head = generate_tl_head(tp.type, tp.img_shape)
        print("@ Calculating train vectors")
        t = time.time()
        train_vectors = model_head.predict(ds.train.data)
        print("! {}s elapsed".format(time.time() - t))
        print("@ Calculating test vectors")
        t = time.time()
        test_vectors = model_head.predict(ds.test.data)
        print("! {}s elapsed".format(time.time() - t))

        # Clear session
        K.clear_session()

        # Generate tail model and compile
        model_tail = generate_tl_tail(tp.num_classes, [train_vectors.shape[-1], ])
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.alr_epochs,
                                               nb_drops=tp.alr_drops,
                                               verbose=1)
        print('-' * 80)
        print("@ Training")
        if tp.test_split > 0:
            validation_data = (test_vectors, ds.test_cls_onehot)
        else:
            validation_data = None
        if tp.use_class_weights is True:
            class_weights = ds.class_weights
            print("@ Class weights: {}".format(class_weights))
        else:
            class_weights = None
        # log_dir = "C:\\logs\\profile\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)
        history = model_tail.fit(x=train_vectors,
                                 y=ds.train_cls_onehot,
                                 batch_size=tp.batch_size,
                                 epochs=tp.max_epochs,
                                 verbose=0,
                                 callbacks=[alr_cb],
                                 validation_data=validation_data,
                                 class_weight=class_weights)
        end = time.time()
        training_time = end - start
        print("@ Training time: {}s".format(training_time))
        time.sleep(3)

        # Now we join the trained dense layers to the resnet model to create a model that accepts images as input
        model_head = generate_tl_head(tp.type, tp.img_shape)
        outputs = model_tail(model_head.output)
        model = Model(model_head.input, outputs)
        model.summary()

        # Vector model
        vector_model = generate_vector(model, tp.type)

    # ------------------------------------------------------------------------------
    # Full network train
    # ------------------------------------------------------------------------------
    else:
        print('-' * 80)
        start = time.time()

        # Generate model
        model = generate(tp)

        # Augmentation
        if tp.aug_rotation is True:
            rotation_range = [0, 360]
        else:
            rotation_range = None

        def augment(x):
            return augmentation_complete(x,
                                         rotation=rotation_range,
                                         gain=tp.aug_gain,
                                         gamma=tp.aug_gamma,
                                         zoom=tp.aug_zoom,
                                         gaussian_noise=tp.aug_gaussian_noise,
                                         bias=tp.aug_bias)

        if tp.use_augmentation is True:
            augment_fn = augment
        else:
            augment_fn = None

        # Training
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.alr_epochs,
                                               nb_drops=tp.alr_drops,
                                               verbose=1)
        print('-' * 80)
        print("@ Training")
        if tp.test_split > 0:
            validation_data = ds.test_tfdataset
        else:
            validation_data = None
        if tp.use_class_weights is True:
            class_weights = ds.class_weights
            print("@ Class weights: {}".format(class_weights))
        else:
            class_weights = None

        def create_gen(tfdataset):
            iterator = tfdataset.make_one_shot_iterator()
            next_val = iterator.get_next()
            with K.get_session().as_default() as sess:
                while True:
                    inputs, labels = sess.run(next_val)
                    yield inputs, labels

        history = model.fit_generator(
            ds.train_generator.tf1_compat_generator(),
            steps_per_epoch=ds.train_batches_per_epoch,
            validation_data=ds.test_generator.tf1_compat_generator(),
            validation_steps=ds.test_batches_per_epoch,
            epochs=tp.max_epochs,
            verbose=0,
            shuffle=False,
            max_queue_size=1,
            class_weight=class_weights,
            callbacks=[alr_cb])
        end = time.time()
        training_time = end - start
        print("@ Training time: {}s".format(training_time))
        time.sleep(3)

        # Vector mode.
        vector_model = generate_vector(model, tp.type)

    # ------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------
    print("@ Generating results")
    now = datetime.datetime.now()
    save_dir = os.path.join(tp.output_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(tp.name, now))
    os.makedirs(save_dir, exist_ok=True)
    # Accuracy
    if tp.test_split > 0:
        y_true = ds.test_cls
        y_prob = model.predict(ds.test.data, batch_size=32)
        y_pred = y_prob.argmax(axis=1)
    else:
        y_true = np.asarray([])
        y_prob = np.asarray([])
        y_pred = np.asarray([])
    # Inference time
    print("@ Calculating inference time")
    max_count = np.min([1000, len(ds.train.data)])
    to_predict = np.copy(ds.train.data[0:max_count])
    inf_times = []
    for i in range(3):
        start = time.time()
        model.predict(to_predict)
        end = time.time()
        diff = (end - start) / max_count * 1000
        inf_times.append(diff)
        print("@ - {}/3: {:.3f}ms".format(i + 1, diff))
    inference_time = np.median(inf_times)
    print("@ - average: {}".format(inference_time))
    # Store results
    result = TrainingResult(tp,
                            history,
                            y_true,
                            y_pred,
                            y_prob,
                            ds.cls_labels,
                            training_time,
                            inference_time)
    # ------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------
    if tp.description is None:
        tp.description = "{}: {} model trained on data from {} ({} images in {} classes).\n" \
                         "Accuracy: {:.1f} (P: {:.1f}, R: {:.1f}, F1 {:.1f})".format(
            tp.name,
            tp.type,
            tp.source,
            len(ds.filenames_dataset.filenames),
            len(ds.cls_labels),
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
    info = ModelInfo(tp.name,
                     tp.description,
                     tp.type,
                     now,
                     "frozen_model.pb",
                     tp,
                     inputs,
                     outputs,
                     tp.source,
                     ds.cls_labels,
                     ds.filenames_dataset.cls_counts,
                     "rescale",
                     [255, 0, 1],
                     result.accuracy,
                     result.precision,
                     result.recall,
                     result.f1_score,
                     result.support,
                     result.epochs[-1],
                     training_time,
                     tp.test_split,
                     inference_time)
    info.save(os.path.join(save_dir, "model", "network_info.xml"))
    # ------------------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------------------
    print("@ Saving model")
    # Convert if necessary to fix TF batch normalisation issues
    model = convert_to_inference_mode(model, lambda: generate(tp))
    vector_model = generate_vector(model, tp.type)
    # Freeze and save graph
    if tp.save_model is not None:
        freeze(model, os.path.join(save_dir, "model"), info)
    # ------------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------------
    # Plot the graphs
    # plot_model(model, to_file=os.path.join(save_dir, "model_plot.pdf"), show_shapes=True)
    if tp.test_split > 0:
        print("@ Generating graphs")
        plot_loss_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "loss_vs_epoch.pdf"))
        plot_accuracy_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.pdf"))
        plot_confusion_accuracy_matrix(y_true, y_pred, ds.cls_labels)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.pdf"))
        plt.close('all')
    # Mislabeled
    if tp.save_mislabeled is True:
        print("@ Estimating mislabeled")
        vectors = vector_model.predict(ds.train.data)
        plot_mislabelled(ds.train.data,
                         vectors,
                         ds.train_cls,
                         ds.cls_labels,
                         [os.path.basename(f) for f in ds.filenames_dataset.train.filenames],
                         save_dir,
                         11)
        vectors = vector_model.predict(ds.test.data)
        plot_mislabelled(ds.test.data,
                         vectors,
                         ds.test_cls,
                         ds.cls_labels,
                         [os.path.basename(f) for f in ds.filenames_dataset.test.filenames],
                         save_dir,
                         11)

    # ------------------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------------------
    print("@ Cleaning up")
    ds.release()
    print("@ Complete")
    print('-' * 80)
    print()
    # return model, vector_model, ds, result

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
