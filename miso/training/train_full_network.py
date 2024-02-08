import os
import time

import skimage
import tensorflow as tf
import numpy as np

from miso.data.tf_generator import TFGenerator
from miso.data.training_dataset import TrainingDataset
from miso.models.factory import generate
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.parameters import MisoParameters
from miso.training.augmentation import aug_all_fn


def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for t_idx, im in enumerate(images):
        im = im.numpy() * 255
        im[im > 255] = 255
        if np.ndim(im) == 2:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=-1)
        elif im.shape[-1] == 1:
            im = np.repeat(im, 3, axis=-1)
        skimage.io.imsave(
            os.path.join(output_dir, "{:03d}.jpg".format(t_idx)),
            im.astype(np.uint8),
        )


def train_full_network(tp: MisoParameters, ds: TrainingDataset, save_dir: str):
    # Tensorflow version
    tf_version = int(tf.__version__[0])

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
    training_examples_dir = os.path.join(save_dir, "examples", "train")
    images, labels = next(iter(train_gen.create()))
    save_images(images, training_examples_dir)

    # Validation generator
    if tf_version == 2:
        val_one_shot = True
    else:
        # One repeat for validation for TF1 otherwise we get end of dataset errors
        val_one_shot = False
    if tp.dataset.val_split > 0:
        # Maximum 8 in batch otherwise validation results jump around a bit because
        val_gen = ds.test_generator(
            min(tp.training.batch_size, 16), shuffle=False, one_shot=val_one_shot
        )
        val_data = val_gen.create()
        val_steps = len(val_gen)

        # Save example of validation data
        print("- saving example validation batch")
        training_examples_dir = os.path.join(save_dir, "examples", "val")
        images, labels = next(iter(val_gen.create()))
        save_images(images, training_examples_dir)

    else:
        val_data = None
        val_steps = None

    # Class weights
    if (tp.training.use_class_weights is True and tp.training.use_class_undersampling is False):
        class_weights = ds.class_weights
        print("- class weights: {}".format(class_weights))
        if tf_version == 2:
            class_weights = dict(enumerate(class_weights))
    else:
        class_weights = None
    if tp.training.use_class_undersampling:
        print("- class balancing using random under sampling")

    # Train the model
    if tf_version == 2:
        train_fn = model.fit
    else:
        train_fn = model.fit_generator
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
    history.history["training_time"] = training_time

    return model, history