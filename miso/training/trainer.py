"""
Creates and trains a generic network
"""
import time
import datetime
from collections import OrderedDict
import tensorflow.keras.backend as K

from miso.data.training_dataset import TrainingDataset
from miso.stats.mislabelling import plot_mislabelled
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from miso.training.augmentation import *
from miso.deploy.saving import freeze, convert_to_inference_mode, save_frozen_model_tf2
from miso.deploy.model_info import ModelInfo
from miso.models.factory import *


def train_image_classification_model(tp: TrainingParameters):
    tf_version = int(tf.__version__[0])

    if tf_version == 2:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    K.clear_session()
    tp.sanitise()

    print("+------------------------------------------------------------------------------+")
    print("| MISO Particle Classification Library                                         |")
    print("+------------------------------------------------------------------------------+")
    print("| Stable version:                                                              |")
    print("| pip install -U miso2                                                         |")
    print("| Development version:                                                         |")
    print("| pip install -U git+http://www.github.com/microfossil/particle-classification |")
    print("+------------------------------------------------------------------------------+")
    print("Tensorflow version: {}".format(tf.__version__))
    print()
    print("-" * 80)
    print("Train information:")
    print("- name: {}".format(tp.name))
    print("- description: {}".format(tp.description))
    print("- CNN type: {}".format(tp.type))
    print("- image type: {}".format(tp.img_type))
    print("- image shape: {}".format(tp.img_shape))
    print()

    # Load data
    ds = TrainingDataset(tp.source,
                         tp.img_shape,
                         tp.img_type,
                         tp.min_count,
                         tp.map_others,
                         tp.test_split,
                         tp.random_seed,
                         tp.memmap_directory)
    ds.load()
    tp.num_classes = ds.num_classes

    # ------------------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------------------
    if tp.type.endswith('tl'):
        print('-' * 80)
        print("@ Transfer learning network training")
        start = time.time()

        # Generate head model and predict vectors
        model_head = generate_tl_head(tp.type, tp.img_shape)
        print("@ Calculating vectors")
        t = time.time()
        # vectors = model_head.predict(ds.images.data, batch_size=32, verbose=1)
        gen = ds.images.create_generator(32, shuffle=False, one_shot=True)
        if tf_version == 2:
            vectors = model_head.predict(gen.to_tfdataset())
        else:
            vectors = model_head.predict_generator(gen.tf1_compat_generator(), steps=len(gen))
        print("! {}s elapsed, ({}/{} vectors)".format(time.time() - t, len(vectors), len(ds.images.data)))

        # Clear session
        K.clear_session()

        # Generate tail model and compile
        model_tail = generate_tl_tail(tp.num_classes, [vectors.shape[-1], ])
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.alr_epochs,
                                               nb_drops=tp.alr_drops,
                                               verbose=2)
        print('-' * 80)
        print("@ Training")
        if tp.test_split > 0:
            validation_data = (vectors[ds.test_idx], ds.cls_onehot[ds.test_idx])
        else:
            validation_data = None
        if tp.use_class_weights is True:
            class_weights = ds.class_weights
            print("@ Class weights: {}".format(class_weights))
            if tf_version == 2:
                class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None
        # log_dir = "C:\\logs\\profile\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)
        history = model_tail.fit(x=vectors[ds.train_idx],
                                 y=ds.cls_onehot[ds.train_idx],
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
        print("@ Full network training")
        start = time.time()

        # Generate model
        model = generate(tp)

        # Augmentation
        if tp.aug_rotation is True:
            rotation_range = [0, 360]
        else:
            rotation_range = None

        def augment(x):
            if tf_version == 2:
                return augmentation_complete_tf2(x,
                                                 rotation=rotation_range,
                                                 gain=tp.aug_gain,
                                                 gamma=tp.aug_gamma,
                                                 zoom=tp.aug_zoom,
                                                 gaussian_noise=tp.aug_gaussian_noise,
                                                 bias=tp.aug_bias)
            else:
                return augmentation_complete(x,
                                             rotation=rotation_range,
                                             gain=tp.aug_gain,
                                             gamma=tp.aug_gamma,
                                             zoom=tp.aug_zoom,
                                             gaussian_noise=tp.aug_gaussian_noise,
                                             bias=tp.aug_bias)

        if tp.use_augmentation is True:
            print("@ - using augmentation")
            augment_fn = augment
        else:
            print("@ - NOT using augmentation")
            augment_fn = None

        # Training
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.alr_epochs,
                                               nb_drops=tp.alr_drops,
                                               verbose=1)
        if tp.test_split > 0:
            if tf_version == 2:
                validation_data = ds.test.create_generator(tp.batch_size, one_shot=True)
            else:
                validation_data = ds.test.create_generator(tp.batch_size, one_shot=False)
        else:
            validation_data = None
        if tp.use_class_weights is True:
            class_weights = ds.class_weights
            print("@ Class weights: {}".format(class_weights))
        else:
            class_weights = None

        train_gen = ds.images.create_generator(tp.batch_size, map_fn=augment_fn)
        if tf_version == 2:
            history = model.fit(train_gen.to_tfdataset(),
                                          steps_per_epoch=len(train_gen),
                                          validation_data=validation_data.to_tfdataset(),
                                          epochs=tp.max_epochs,
                                          verbose=0,
                                          shuffle=False,
                                          max_queue_size=1,
                                          class_weight=dict(enumerate(class_weights)),
                                          callbacks=[alr_cb])
        else:
            history = model.fit_generator(train_gen.tf1_compat_generator(),
                                          steps_per_epoch=len(train_gen),
                                          validation_data=validation_data.tf1_compat_generator(),
                                          validation_steps=len(validation_data),
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

        # Vector model
        vector_model = generate_vector(model, tp.type)

    # ------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------
    print("Evaluating model")
    now = datetime.datetime.now()
    save_dir = os.path.join(tp.output_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(tp.name, now))
    os.makedirs(save_dir, exist_ok=True)
    # Accuracy
    if tp.test_split > 0:
        y_true = ds.cls[ds.test_idx]
        gen = ds.test_generator(1, shuffle=False, one_shot=True)
        if tf_version == 2:
            y_prob = model.predict(gen.to_tfdataset())
        else:
            y_prob = model.predict_generator(gen.tf1_compat_generator(), len(gen))
        y_pred = y_prob.argmax(axis=1)
    else:
        y_true = np.asarray([])
        y_prob = np.asarray([])
        y_pred = np.asarray([])
    # Inference time
    print("- calculating inference time:", end='')
    max_count = np.min([128, len(ds.images.data)])
    inf_times = []
    for i in range(3):
        gen = ds.images.create_generator(32, idxs=np.arange(max_count), shuffle=False, one_shot=True)
        start = time.time()
        if tf_version == 2:
            model.predict(gen.to_tfdataset())
        else:
            model.predict_generator(gen.tf1_compat_generator(), len(gen))
        end = time.time()
        diff = (end - start) / max_count * 1000
        inf_times.append(diff)
        print(" {:.3f}ms".format(diff), end='')
    inference_time = np.median(inf_times)
    print(", median: {}".format(inference_time))
    # Store results
    # - fix to make key same for tensorflow 1 and 2
    if 'accuracy' in history.history:
        history.history['acc'] = history.history.pop('accuracy')
        history.history['val_acc'] = history.history.pop('val_accuracy')
    result = TrainingResult(tp,
                            history,
                            y_true,
                            y_pred,
                            y_prob,
                            ds.cls_labels,
                            training_time,
                            inference_time)
    print("- accuracy {:.2f}".format(result.accuracy * 100))
    print("- mean precision {:.2f}".format(result.mean_precision * 100))
    print("- mean recall {:.2f}".format(result.mean_recall * 100))
    # ------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------
    if tp.description is None:
        tp.description = "{}: {} model trained on data from {} ({} images in {} classes).\n" \
                         "Accuracy: {:.1f} (P: {:.1f}, R: {:.1f}, F1 {:.1f})".format(
            tp.name,
            tp.type,
            tp.source,
            len(ds.filenames.filenames),
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
                     ds.filenames.cls_counts,
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
        gen = ds.images.create_generator(1, shuffle=False, one_shot=True)
        if tf_version == 2:
            vectors = vector_model.predict(gen.to_tfdataset(), steps=len(gen))
        else:
            vectors = vector_model.predict_generator(gen.tf1_compat_generator(), steps=len(gen))
        plt.matshow(vectors)
        plt.show()
        plot_mislabelled(ds.images.data,
                         vectors,
                         ds.cls,
                         ds.cls_labels,
                         [os.path.basename(f) for f in ds.filenames.filenames],
                         save_dir,
                         11)
    # ------------------------------------------------------------------------------
    # Save model (has to be last thing it seems)
    # ------------------------------------------------------------------------------
    print("@ Saving model")
    # Convert if necessary to fix TF batch normalisation issues
    inference_model = convert_to_inference_mode(model, lambda: generate(tp))
    # Freeze and save graph
    if tp.save_model is not None:
        if tf_version == 2:
            tf.saved_model.save(inference_model, os.path.join(os.path.join(save_dir, "model_keras")))
            save_frozen_model_tf2(inference_model, os.path.join(save_dir, "model"), "frozen_model.pb")
        else:
            tf.saved_model.save(inference_model, os.path.join(os.path.join(save_dir, "model_keras")))
            freeze(inference_model, os.path.join(save_dir, "model"))
    info.save(os.path.join(save_dir, "model", "network_info.xml"))
    # ------------------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------------------
    print("@ Cleaning up")
    ds.release()
    print("@ Complete")
    print('-' * 80)
    print()
    return model, vector_model, ds, result


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

if __name__ == "__main__":
    tp = TrainingParameters()
    tp.source = "https://1drv.ws/u/s!AiQM7sVIv7fah4MNU5lCmgcx4Ud_dQ?e=nPpUmT"
    # tp.source = "/Users/chaos/OneDrive/Datasets/DeepWeeds/"
    tp.source = "https://1drv.ws/u/s!AiQM7sVIv7fak98qYjFt5GELIEqSMQ?e=EUiUIX"
    tp.output_dir = "/media/ross/DATA/tmp"
    tp.type = "resnet50_cyclic_tl"
    tp.img_shape = [224, 224, 3]
    tp.img_type = 'rgb'
    tp.save_mislabeled = False
    train_image_classification_model(tp)
