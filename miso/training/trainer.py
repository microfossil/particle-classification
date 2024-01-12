"""
Creates and trains a generic network
"""
import os
import skimage.io
import traceback
import warnings
import matplotlib.pyplot as plt

plt.switch_backend('agg')

warnings.simplefilter(action="ignore", category=FutureWarning)

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
from miso.deploy.saving import (
    freeze,
    convert_to_inference_mode,
    save_frozen_model_tf2,
    convert_to_inference_mode_tf2,
    load_from_xml,
    save_model_as_onnx,
)
from miso.deploy.model_info import ModelInfo
from miso.models.factory import *

import matplotlib.pyplot as plt
import pandas as pd


def predict_in_batches(model, generator):
    results = []
    try:
        for x, y in iter(generator):
            results.append(model.predict(x))
    except tf.errors.OutOfRangeError:
        pass
    results = np.concatenate(results, axis=0)
    return results


def train_image_classification_model(tp: MisoParameters):

    # Hack to make RTX cards work
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

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
    ds = TrainingDataset(
        tp.dataset.source,
        tp.cnn.img_shape,
        tp.cnn.img_type,
        tp.dataset.min_count,
        tp.dataset.map_others,
        tp.dataset.val_split,
        tp.dataset.random_seed,
        tp.dataset.memmap_directory,
    )
    ds.load()
    tp.dataset.num_classes = ds.num_classes

    # Create save lodations
    now = datetime.datetime.now()
    save_dir = os.path.join(
        tp.output.save_dir, "{0}_{1:%Y%m%d-%H%M%S}".format(tp.name, now)
    )
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------------------
    if tp.cnn.id.endswith("tl"):
        print("-" * 80)
        print("Transfer learning network training")
        start = time.time()

        # Generate head model and predict vectors
        model_head = generate_tl_head(tp.cnn.id, tp.cnn.img_shape)

        # Calculate vectors
        print("- calculating vectors")
        t = time.time()
        gen = ds.images.create_generator(
            tp.training.batch_size, shuffle=False, one_shot=True
        )
        vectors = model_head.predict(gen.create())
        print(f"! {time.time() - t}s elapsed, ({len(vectors)}/{len(ds.images.data)} vectors)")

        # Clear session
        # K.clear_session()

        # Generate tail model and compile
        model_tail = generate_tl_tail(tp.dataset.num_classes, [vectors.shape[-1]])
        model_tail.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

        # Learning rate scheduler
        alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.training.alr_epochs,
                                               nb_drops=tp.training.alr_drops,
                                               verbose=1)
        print("-" * 80)
        print("Training")

        # Training generator
        train_gen = TFGenerator(vectors,
                                ds.cls_onehot,
                                ds.train_idx,
                                tp.training.batch_size,
                                shuffle=True,
                                one_shot=False,
                                undersample=tp.training.use_class_undersampling)

        # Validation generator
        val_one_shot = True
        if tp.dataset.val_split > 0:
            val_gen = TFGenerator(
                vectors,
                ds.cls_onehot,
                ds.test_idx,
                tp.training.batch_size,
                shuffle=False,
                one_shot=val_one_shot,
            )
            val_data = val_gen.create()
            val_steps = len(val_gen)
        else:
            val_gen = None
            val_data = None
            val_steps = None

        # Class weights (only if over sampling is not used)
        if tp.training.use_class_weights is True and tp.training.use_class_undersampling is False:
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None
        if tp.training.use_class_undersampling:
            print("- class balancing using random under sampling")

        # Train
        train_fn = model_tail.fit
        history = train_fn(train_gen.create(),
                           steps_per_epoch=len(train_gen),
                           validation_data=val_data,
                           validation_steps=val_steps,
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
        model = combine_tl(model_head, model_tail)
        vector_model = generate_vector(model, tp.cnn.id)
    # ------------------------------------------------------------------------------
    # Full network train
    # ------------------------------------------------------------------------------
    else:
        print("-" * 80)
        print("Full network training")
        start = time.time()

        # Generate model
        model = generate(tp)
        # model.summary()

        # Augmentation
        if tp.augmentation.rotation is True:
            tp.augmentation.rotation = [0, 360]
        elif tp.augmentation.rotation is False:
            tp.augmentation.rotation = None
        if tp.training.use_augmentation is True:
            print("- using augmentation")
            augment_fn = aug_all_fn(
                rotation=tp.augmentation.rotation,
                gain=tp.augmentation.gain,
                gamma=tp.augmentation.gamma,
                zoom=tp.augmentation.zoom,
                gaussian_noise=tp.augmentation.gaussian_noise,
                bias=tp.augmentation.bias,
                random_crop=tp.augmentation.random_crop,
                divide=255,
            )
        else:
            print("- NOT using augmentation")
            augment_fn = TFGenerator.map_fn_divide_255

        # Learning rate scheduler
        alr_cb = AdaptiveLearningRateScheduler(
            nb_epochs=tp.training.alr_epochs, nb_drops=tp.training.alr_drops, verbose=1
        )

        # Training generator
        train_gen = ds.train_generator(
            batch_size=tp.training.batch_size,
            map_fn=augment_fn,
            undersample=tp.training.use_class_undersampling,
        )

        # Save example of training data
        print("- saving example training batch")
        training_examples_dir = os.path.join(save_dir, "examples", "training")
        os.makedirs(training_examples_dir)
        images, labels = next(iter(train_gen.create()))
        for t_idx, im in enumerate(images):
            im = im.numpy() * 255
            im[im > 255] = 255
            if np.ndim(im) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=-1)
            elif im.shape[-1] == 1:
                im = np.repeat(im, 3, axis=-1)
            skimage.io.imsave(
                os.path.join(training_examples_dir, "{:03d}.jpg".format(t_idx)),
                im.astype(np.uint8),
            )

        # Validation generator
        val_one_shot = True
        if tp.dataset.val_split > 0:
            # Maximum 8 in batch otherwise validation results jump around a bit because
            val_gen = ds.test_generator(
                min(tp.training.batch_size, 16), shuffle=False, one_shot=val_one_shot
            )
            # val_gen = ds.test_generator(tp.training.batch_size, shuffle=False, one_shot=val_one_shot)
            val_data = val_gen.create()
            val_steps = len(val_gen)
        else:
            val_gen = None
            val_data = None
            val_steps = None

        # Class weights
        if (
                tp.training.use_class_weights is True
                and tp.training.use_class_undersampling is False
        ):
            class_weights = ds.class_weights
            print("- class weights: {}".format(class_weights))
            class_weights = dict(enumerate(class_weights))
        else:
            class_weights = None
        if tp.training.use_class_undersampling:
            print("- class balancing using random under sampling")

        # Train the model
        train_fn = model.fit
        history = train_fn(
            train_gen.create(),
            steps_per_epoch=len(train_gen),
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=tp.training.max_epochs,
            verbose=0,
            shuffle=False,
            max_queue_size=1,
            class_weight=class_weights,
            callbacks=[alr_cb],
        )

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
    print("-" * 80)
    print("Evaluating model")
    # Accuracy
    if tp.dataset.val_split > 0:
        y_true = ds.cls[ds.test_idx]
        gen = ds.test_generator(tp.training.batch_size, shuffle=False, one_shot=True)
        y_prob = model.predict(gen.create())
        y_pred = y_prob.argmax(axis=1)
    else:
        y_true = np.asarray([])
        y_prob = np.asarray([])
        y_pred = np.asarray([])
    # Inference time
    print("- calculating inference time:", end="")
    max_count = np.min([128, len(ds.images.data)])
    inf_times = []
    for i in range(3):
        gen = ds.images.create_generator(
            tp.training.batch_size,
            idxs=np.arange(max_count),
            shuffle=False,
            one_shot=True,
        )
        start = time.time()
        model.predict(gen.create())
        end = time.time()
        diff = (end - start) / max_count * 1000
        inf_times.append(diff)
        print(" {:.3f}ms".format(diff), end="")
    inference_time = np.median(inf_times)
    print(", median: {}".format(inference_time))
    # Store results
    # - fix to make key same for tensorflow 1 and 2
    if "accuracy" in history.history:
        history.history["acc"] = history.history.pop("accuracy")
        history.history["val_acc"] = history.history.pop("val_accuracy")
    result = TrainingResult(
        tp,
        history,
        y_true,
        y_pred,
        y_prob,
        ds.cls_labels,
        training_time,
        inference_time,
    )
    print("- accuracy {:.2f}".format(result.accuracy * 100))
    print("- mean precision {:.2f}".format(result.mean_precision * 100))
    print("- mean recall {:.2f}".format(result.mean_recall * 100))
    # ------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------
    if tp.description is None:
        tp.description = (
            "{}: {} model trained on data from {} ({} images in {} classes).\n"
            "Accuracy: {:.1f} (P: {:.1f}, R: {:.1f}, F1 {:.1f})".format(
                tp.name,
                tp.cnn.id,
                tp.dataset.source,
                len(ds.filenames.filenames),
                len(ds.cls_labels),
                result.accuracy * 100,
                result.mean_precision * 100,
                result.mean_recall * 100,
                result.mean_f1_score * 100,
            )
        )

    # Create model info with all the parameters
    inputs = OrderedDict()
    inputs["image"] = model.inputs[0]
    outputs = OrderedDict()
    outputs["pred"] = model.outputs[0]
    outputs["vector"] = vector_model.outputs[0]
    info = ModelInfo(
        tp.name,
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
        inference_time,
    )
    # ------------------------------------------------------------------------------
    # Cleanlab health report
    # ------------------------------------------------------------------------------
    print("-" * 80)
    print("Health report")
    import cleanlab

    report = cleanlab.dataset.health_summary(y_true, y_prob, class_names=ds.cls_labels)
    with open(os.path.join(save_dir, "health_summary.txt"), "w") as fp:
        fp.write("Dataset health report\n")
        fp.write("\n")
        fp.write("This is calculated using the test data\n")
        fp.write("\n" + "-" * 80 + "\n")
        fp.write(f"Overall label heath: {report['overall_label_health_score']}")
        fp.write("\n" + "-" * 80 + "\n")
        fp.write("Classes by label quality\n")
        fp.write(report["classes_by_label_quality"].to_string())
        fp.write("\n" + "-" * 80 + "\n")
        fp.write("Overlapping classes\n")
        fp.write(report["overlapping_classes"].to_string())

    # ------------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------------
    # Plot the graphs
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
        plt.close("all")

    if tp.output.save_mislabeled is True:
        print("- mislabeled")
        print("- calculating vectors... ", end="")
        gen = ds.images.create_generator(
            tp.training.batch_size, shuffle=False, one_shot=True
        )
        vectors = vector_model.predict(gen.create())
        print("{} total".format(len(vectors)))
        find_and_save_mislabelled(
            ds.images.data,
            vectors,
            ds.cls,
            ds.cls_labels,
            [os.path.basename(f) for f in ds.filenames.filenames],
            save_dir,
            11,
        )

    # t-SNE
    print("- t-SNE (1024 vectors max)")
    print("- calculating vectors... ", end="")
    idxs = np.random.choice(
        np.arange(len(ds.images.data)),
        np.min((1024, len(ds.images.data))),
        replace=False,
    )
    gen = ds.images.create_generator(
        tp.training.batch_size, idxs=idxs, shuffle=False, one_shot=True
    )
    vec_subset = vector_model.predict(gen.create())
    X = TSNE(n_components=2).fit_transform(vec_subset)
    plot_embedding(X, ds.cls[idxs], ds.num_classes)
    plt.savefig(os.path.join(save_dir, "tsne.pdf"))
    cls_info = pd.DataFrame({"index": range(ds.num_classes), "label": ds.cls_labels})
    cls_info.to_csv(os.path.join(save_dir, "legend.csv"), sep=";")

    # ------------------------------------------------------------------------------
    # Save model (has to be last thing it seems)
    # ------------------------------------------------------------------------------
    print("-" * 80)
    print("Saving model")
    # Convert if necessary to fix TF batch normalisation issues

    # Freeze and save graph
    if tp.output.save_model is not None:
        inference_model = convert_to_inference_mode_tf2(model, lambda: generate(tp))
        frozen_func = save_frozen_model_tf2(
            inference_model, os.path.join(save_dir, "model"), "model_tf2.pb"
        )
        info.protobuf = "model_tf2.pb"
        info.inputs["image"] = frozen_func.inputs[0]
        info.outputs["pred"] = frozen_func.outputs[0]
        info.save(os.path.join(save_dir, "model", "network_info.xml"))

        try:
            save_model_as_onnx(
                inference_model,
                inference_model.inputs[0].name, [None, ] + tp.cnn.img_shape,
                os.path.join(os.path.join(save_dir, "model_onnx")),
            )
            info.protobuf = "model.onnx"
            info.inputs["image"] = inference_model.inputs[0]
            info.outputs["pred"] = inference_model.outputs[0]
            info.save(os.path.join(save_dir, "model_onnx", "network_info.xml"))
        except Exception as e:
            print("Failed to save ONNX model!")
            print(e)
            traceback.print_exc()

    # ------------------------------------------------------------------------------
    # Confirm model save
    # ------------------------------------------------------------------------------
    if tp.output.save_model is not None and tp.dataset.val_split > 0:
        print("-" * 80)
        print("Validate saved model")
        y_pred_old = y_pred
        y_true = ds.cls[ds.test_idx]
        gen = ds.test_generator(32, shuffle=False, one_shot=True)
        y_prob = []
        model, img_size, cls_labels = load_from_xml(
            os.path.join(save_dir, "model", "network_info.xml")
        )
        for b in iter(gen.to_tfdataset()):
            y_prob.append(model(b[0]).numpy())
        y_prob = np.concatenate(y_prob, axis=0)
        y_pred = y_prob.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred)
        print(
            "Saved model on test set: acc {:.2f}, prec {:.2f}, rec {:.2f}, f1 {:.2f}".format(
                acc, np.mean(p), np.mean(r), np.mean(f1)
            )
        )
        acc = accuracy_score(y_pred_old, y_pred)
        if acc == 1.0:
            print("Overlap: {:.2f}% - PASSED".format(acc * 100))
        else:
            print("Overlap: {:.2f}% - FAILED".format(acc * 100))

    # ------------------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------------------
    print("- cleaning up")
    ds.release()
    print("- complete")
    print("-" * 80)
    print()
    return model, vector_model, ds, result
