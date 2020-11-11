"""
Creates and trains a generic network
"""
import time
import datetime
from collections import OrderedDict
import tensorflow.keras.backend as K
from sklearn.manifold import TSNE

import miso
from miso.data.tf_generator import TFGenerator

from miso.data.training_dataset import TrainingDataset
from miso.stats.embedding import plot_embedding
from miso.stats.mislabelling import plot_mislabelled
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.parameters import MisoParameters
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from miso.training.augmentation import *
from miso.training.tf_augmentation import aug_all_fn
from miso.deploy.saving import freeze, convert_to_inference_mode, save_frozen_model_tf2
from miso.deploy.model_info import ModelInfo
from miso.models.factory import *

import matplotlib.pyplot as plt
import pandas as pd


def train_image_classification_model(tp: MisoParameters):
    tf_version = int(tf.__version__[0])

    if tf_version == 2:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    K.clear_session()

    # Clean the training parameters
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
    print("-" * 80)
    print("Train information:")
    print("- name: {}".format(tp.name))
    print("- description: {}".format(tp.description))
    print("- CNN type: {}".format(tp.cnn.id))
    print("- image type: {}".format(tp.cnn.img_type))
    print("- image shape: {}".format(tp.cnn.img_shape))
    print()

    # Load data
    ds = TrainingDataset(tp.dataset.source,
                         tp.cnn.img_shape,
                         tp.cnn.img_type,
                         tp.dataset.min_count,
                         tp.dataset.map_others,
                         tp.dataset.test_split,
                         tp.dataset.random_seed,
                         tp.dataset.memmap_directory)
    ds.load()
    tp.dataset.num_classes = ds.num_classes

    # ------------------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------------------
    if tp.cnn.id.endswith('tl'):
        print('-' * 80)
        print("Transfer learning network training")
        start = time.time()

        # Generate head model and predict vectors
        model_head = generate_tl_head(tp.cnn.id, tp.cnn.img_shape)

        # Calculate vectors
        print("- calculating vectors")
        t = time.time()
        gen = ds.images.create_generator(32, shuffle=False, one_shot=True)
        if tf_version == 2:
            vectors = model_head.predict(gen.create())
        else:
            vectors = model_head.predict_generator(gen.create(), steps=len(gen))
        print("! {}s elapsed, ({}/{} vectors)".format(time.time() - t, len(vectors), len(ds.images.data)))

        # Clear session
        K.clear_session()

        # Generate tail model and compile
        model_tail = generate_tl_tail(tp.dataset.num_classes, [vectors.shape[-1], ], tp.cnn.use_msoftmax)
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Learning rate scheduler
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.training.alr_epochs,
                                               nb_drops=tp.training.alr_drops,
                                               verbose=1)
        print('-' * 80)
        print("Training")

        # Validation data
        if tp.dataset.test_split > 0:
            if tp.cnn.use_msoftmax:
                validation_data = ((vectors[ds.test_idx], ds.cls_onehot[ds.test_idx]), ds.cls_onehot[ds.test_idx])
            else:
                validation_data = (vectors[ds.test_idx], ds.cls_onehot[ds.test_idx])
        else:
            validation_data = None

        # Class weights
        if tp.training.use_class_weights is True:
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            if tf_version == 2:
                class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None
        # log_dir = "C:\\logs\\profile\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)

        # Train
        history = model_tail.fit(x=(vectors[ds.train_idx], ds.cls_onehot[ds.train_idx]),
                                 y=ds.cls_onehot[ds.train_idx],
                                 batch_size=tp.training.batch_size,
                                 epochs=tp.training.max_epochs,
                                 verbose=0,
                                 callbacks=[alr_cb],
                                 validation_data=validation_data,
                                 class_weight=class_weights)
        # Elapsed time
        end = time.time()
        training_time = end - start
        print("- training time: {}s".format(training_time))
        time.sleep(3)

        # Now we join the trained dense layers to the resnet model to create a model that accepts images as input
        model_head = generate_tl_head(tp.cnn.id, tp.cnn.img_shape)
        outputs = model_tail(model_head.output)
        model = Model(model_head.input, outputs)
        model.summary()

        # Vector model
        vector_model = generate_vector(model, tp.cnn.id)

    # ------------------------------------------------------------------------------
    # Full network train
    # ------------------------------------------------------------------------------
    else:
        print('-' * 80)
        print("Full network training")
        start = time.time()

        # Generate model
        model = generate(tp)

        # Augmentation
        if tp.augmentation.rotation is True:
            rotation_range = [0, 360]
        else:
            rotation_range = None
        if tp.augmentation.use_augmentation is True:
            print("- using augmentation")
            augment_fn = aug_all_fn(rotation=rotation_range,
                                    gain=tp.augmentation.gain,
                                    gamma=tp.augmentation.gamma,
                                    zoom=tp.augmentation.zoom,
                                    gaussian_noise=tp.augmentation.gaussian_noise,
                                    bias=tp.augmentation.bias,
                                    random_crop=tp.augmentation.random_crop,
                                    divide=255)
        else:
            print("- NOT using augmentation")
            augment_fn = TFGenerator.map_fn_divide_255

        # Learning rate scheduler
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.training.alr_epochs,
                                               nb_drops=tp.training.alr_drops,
                                               verbose=1)

        # Training generator
        train_gen = ds.images.create_generator(tp.training.batch_size, map_fn=augment_fn)

        # Validation generator
        if tp.dataset.test_split > 0:
            if tf_version == 2:
                validation_gen = ds.test_generator(tp.training.batch_size, shuffle=False, one_shot=True)
            else:
                validation_gen = ds.test_generator(tp.training.batch_size, shuffle=False, one_shot=True)
        else:
            validation_gen = None

        # Class weights
        if tp.training.use_class_weights is True:
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            if tf_version == 2:
                class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None



        # Train the model
        history = model.fit_generator(train_gen.create(),
                                      steps_per_epoch=len(train_gen),
                                      validation_data=validation_gen.create(),
                                      validation_steps=len(validation_gen),
                                      epochs=tp.training.max_epochs,
                                      verbose=0,
                                      shuffle=False,
                                      max_queue_size=1,
                                      class_weight=class_weights,
                                      callbacks=[alr_cb])

        # Elapsed time
        end = time.time()
        training_time = end - start
        print("@ Training time: {}s".format(training_time))
        time.sleep(3)

        # Vector model
        vector_model = generate_vector(model, tp.cnn.id)

    # ------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------
    print("Evaluating model")
    now = datetime.datetime.now()
    save_dir = os.path.join(tp.output.output_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(tp.name, now))
    os.makedirs(save_dir, exist_ok=True)
    # Accuracy
    if tp.dataset.test_split > 0:
        y_true = ds.cls[ds.test_idx]
        gen = ds.test_generator(1, shuffle=False, one_shot=True)
        if tf_version == 2:
            y_prob = model.predict(gen.to_tfdataset())
        else:
            y_prob = model.predict_generator(gen.tf1_compat_generator(), len(gen))
        print(y_prob)
        y_pred = y_prob.argmax(axis=1)
        print(y_pred)
        print(y_true)
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
            tp.cnn.id,
            tp.dataset.source,
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
                     tp.cnn.id,
                     now,
                     "frozen_model.pb",
                     tp,
                     inputs,
                     outputs,
                     tp.dataset.source,
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
                     tp.dataset.test_split,
                     inference_time)
    # ------------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------------
    # Plot the graphs
    # plot_model(model, to_file=os.path.join(save_dir, "model_plot.pdf"), show_shapes=True)
    print("-" * 80)
    print("Plotting")
    if tp.dataset.test_split > 0:
        print("- loss")
        plot_loss_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "loss_vs_epoch.pdf"))
        print("- accuracy")
        plot_accuracy_vs_epochs(history)
        plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.pdf"))
        print("- confusion matrix")
        plot_confusion_accuracy_matrix(y_true, y_pred, ds.cls_labels)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.pdf"))
        plt.close('all')
    # Mislabeled
    print("- mislabeled")
    gen = ds.images.create_generator(1, shuffle=False, one_shot=True)
    if tf_version == 2:
        vectors = vector_model.predict(gen.create(), steps=len(gen))
    else:
        vectors = vector_model.predict_generator(gen.create(), steps=len(gen))
    if tp.output.save_mislabeled is True:
        plot_mislabelled(ds.images.data,
                         vectors,
                         ds.cls,
                         ds.cls_labels,
                         [os.path.basename(f) for f in ds.filenames.filenames],
                         save_dir,
                         11)
    # t-SNE
    print("- t-SNE")
    idx = np.random.choice(range(len(vectors)), np.min((len(vectors), 1024)), replace=False)
    vec_subset = vectors[idx]
    X = TSNE(n_components=2).fit_transform(vec_subset)
    counts = np.unique(ds.cls, return_counts=True)
    print(counts)
    counts = np.unique(ds.cls[idx], return_counts=True)
    print(counts)
    plot_embedding(X, ds.cls[idx], ds.num_classes)
    plt.savefig(os.path.join(save_dir, "tsne.pdf"))
    # Legend
    info = pd.DataFrame({"index": range(ds.num_classes), "label": ds.cls_labels})
    info.to_csv(os.path.join(save_dir, "legend.csv"), sep=';')

    # ------------------------------------------------------------------------------
    # Save model (has to be last thing it seems)
    # ------------------------------------------------------------------------------
    print('-' * 80)
    print("Saving model")
    # Convert if necessary to fix TF batch normalisation issues
    inference_model = convert_to_inference_mode(model, lambda: generate(tp))
    # Freeze and save graph
    if tp.output.save_model is not None:
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
    print("- cleaning up")
    ds.release()
    print("- complete")
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
    tp = MisoParameters()
    # tp.dataset.source = "https://1drv.ws/u/s!AiQM7sVIv7fah4MNU5lCmgcx4Ud_dQ?e=nPpUmT"
    # tp.source = "/Users/chaos/OneDrive/Datasets/DeepWeeds/"
    tp.dataset.source = "https://1drv.ws/u/s!AiQM7sVIv7fak98qYjFt5GELIEqSMQ?e=EUiUIX"
    tp.dataset.source = "https://1drv.ws/u/s!AiQM7sVIv7falskYWoLgrbSD2RC-Fg?e=4yhC9b"
    tp.dataset.source = "/media/mar76c/DATA/Datasets/Pollen/pollen_all"
    tp.output.output_dir = "/media/mar76c/DATA/TrainedNetworks/"
    tp.cnn.id = "efficientnetb0"
    tp.training.batch_size = 32
    tp.cnn.img_shape = None
    tp.cnn.img_type = 'rgb'
    tp.cnn.use_msoftmax = False
    tp.output.save_mislabeled = False
    train_image_classification_model(tp)
