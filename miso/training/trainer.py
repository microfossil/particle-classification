"""
Creates and trains a generic network
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import datetime
from collections import OrderedDict
import tensorflow.keras.backend as K
from sklearn.manifold import TSNE

from miso.data.tf_generator import TFGenerator
from miso.data.training_dataset import TrainingDataset
from miso.stats.embedding import plot_embedding
from miso.stats.mislabelling import find_and_save_mislabelled
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.training_result import TrainingResult
from miso.stats.confusion_matrix import *
from miso.stats.training import *
from miso.training.tf_augmentation import aug_all_fn
from miso.deploy.saving import freeze, convert_to_inference_mode, save_frozen_model_tf2
from miso.deploy.model_info import ModelInfo
from miso.models.factory import *

import matplotlib.pyplot as plt
import pandas as pd


def train_image_classification_model(tp: MisoParameters):
    tf_version = int(tf.__version__[0])

    # Hack to make RTX cards work
    if tf_version == 2:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    K.clear_session()

    # Clean the training parameters
    tp.sanitise()

    print("+---------------------------------------------------------------------------+")
    print("| MISO Particle Classification Library                                      |")
    print("+---------------------------------------------------------------------------+")
    print("| Stable version:                                                           |")
    print("| pip install miso2                                                         |")
    print("| Development version:                                                      |")
    print("| pip install git+http://www.github.com/microfossil/particle-classification |")
    print("+---------------------------------------------------------------------------+")
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
                         tp.dataset.val_split,
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
        # K.clear_session()

        # Generate tail model and compile
        model_tail = generate_tl_tail(tp.dataset.num_classes, [vectors.shape[-1], ])
        model_tail.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Learning rate scheduler
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.training.alr_epochs,
                                               nb_drops=tp.training.alr_drops,
                                               verbose=1)
        print('-' * 80)
        print("Training")

        # Training generator
        train_gen = TFGenerator(vectors,
                                ds.cls_onehot,
                                ds.train_idx,
                                tp.training.batch_size,
                                shuffle=True,
                                one_shot=False,
                                oversample=tp.training.use_class_balancing)

        # Validation generator
        if tp.dataset.val_split > 0:
            val_gen = TFGenerator(vectors,
                                         ds.cls_onehot,
                                         ds.test_idx,
                                         tp.training.batch_size,
                                         shuffle=False,
                                         one_shot=True)
        else:
            val_gen = None

        # Class weights (only if over sampling is not used)
        if tp.training.use_class_weights is True and tp.training.use_class_balancing is False:
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            if tf_version == 2:
                class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None
        if tp.training.use_class_balancing:
            print("- class balancing using random over sampling")

        v = model_tail.predict(vectors[0:1])
        print(v[0, :10])

        model = Model(inputs=model_head.input, outputs=model_tail(model_head.output))
        vector_model = Model(model.inputs, model.get_layer(index=-2).get_output_at(0))
        v = vector_model.predict(ds.images.data[0:1] / 255)
        print(v[0, :10])
        v = vector_model.predict(ds.images.data[0:1])
        print(v[0, :10])

        # Train
        history = model_tail.fit_generator(train_gen.create(),
                                           steps_per_epoch=len(train_gen),
                                           validation_data=val_gen.create(),
                                           validation_steps=len(val_gen),
                                           epochs=tp.training.max_epochs,
                                           verbose=0,
                                           shuffle=False,
                                           max_queue_size=1,
                                           class_weight=class_weights,
                                           callbacks=[alr_cb])
        # Elapsed time
        end = time.time()
        training_time = end - start
        print("- training time: {}s".format(training_time))
        time.sleep(3)

        # Now we join the trained dense layers to the resnet model to create a model that accepts images as input
        # model_head = generate_tl_head(tp.cnn.id, tp.cnn.img_shape)
        model = Model(inputs=model_head.input, outputs=model_tail(model_head.output))
        model.summary()

        vector_model = Model(model.inputs, model.get_layer(index=-2).get_output_at(0))
        v = vector_model.predict(ds.images.data[0:1] / 255)
        print(v[0, :10])
        v = vector_model.predict(ds.images.data[0:1])
        print(v[0, :10])

        vector_model.summary()
        print(vector_model.get_layer(index=-1))
        # print(vector_model.get_layer(index=-1).get_weights())
        model_tail.summary()
        print(model_tail.get_layer(index=-2))

        model = Model(inputs=model_head.input, outputs=model_tail(model_head.layers[-1].layers[-1].output))
        model.summary()

        # print(model_tail.get_layer(index=-2).get_weights())

        vector_tensor = model_tail.get_layer(index=-2).get_output_at(0)
        vector_model = Model(model_tail.inputs, vector_tensor)
        v = vector_model.predict(vectors[0:1])
        print(v[0, :10])

        vectors = model_head.predict(next(iter(gen.create())))
        v = vector_model.predict(vectors[0:1])
        print(v[0, :10])


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
        if tp.dataset.val_split > 0:
            val_gen = ds.test_generator(tp.training.batch_size, shuffle=False, one_shot=True)
        else:
            val_gen = None

        # Class weights
        if tp.training.use_class_weights is True and tp.training.use_class_balancing is False:
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            if tf_version == 2:
                class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None

        # Train the model
        history = model.fit_generator(train_gen.create(),
                                      steps_per_epoch=len(train_gen),
                                      validation_data=val_gen.create(),
                                      validation_steps=len(val_gen),
                                      epochs=tp.training.max_epochs,
                                      verbose=0,
                                      shuffle=False,
                                      max_queue_size=1,
                                      class_weight=class_weights,
                                      callbacks=[alr_cb])

        # Elapsed time
        end = time.time()
        training_time = end - start
        print()
        print("Total training time: {}s".format(training_time))
        time.sleep(3)

        # Vector model
        vector_model = generate_vector(model, tp.cnn.id)

    # ------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------
    print('-' * 80)
    print("Evaluating model")
    now = datetime.datetime.now()
    save_dir = os.path.join(tp.output.save_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(tp.name, now))
    os.makedirs(save_dir, exist_ok=True)
    # Accuracy
    if tp.dataset.val_split > 0:
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
                     tp.dataset.val_split,
                     inference_time)
    # ------------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------------
    # Plot the graphs
    # plot_model(model, to_file=os.path.join(save_dir, "model_plot.pdf"), show_shapes=True)
    print("-" * 80)
    print("Plotting")
    if tp.dataset.val_split > 0:
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
    print("vector length {}", len(vectors))
    print(vectors[0,:10])
    v = vector_model.predict(ds.images.data[0:1] / 255)
    print(v[0, :10])
    if tp.output.save_mislabeled is True:
        find_and_save_mislabelled(ds.images.data,
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
    cls_info = pd.DataFrame({"index": range(ds.num_classes), "label": ds.cls_labels})
    cls_info.to_csv(os.path.join(save_dir, "legend.csv"), sep=';')

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

    # Save model info
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
