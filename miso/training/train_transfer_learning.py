import time
import tensorflow as tf
import numpy as np

from miso.data.tf_generator import TFGenerator
from miso.data.training_dataset import TrainingDataset
from miso.models.factory import generate_tl, combine_head_and_tail
from miso.training.adaptive_learning_rate import AdaptiveLearningRateScheduler
from miso.training.helpers import predict_in_batches
from miso.training.parameters import MisoParameters
from miso.training.augmentation import aug_all_fn


def train_transfer_learning(tp: MisoParameters, ds: TrainingDataset):

    print("-" * 80)
    print("Transfer learning network training")
    start = time.time()

    # Generate head and tail models separately
    model_head, model_tail = generate_tl(tp)
    # model_head.summary()

    # Calculate vectors using head model
    print("- calculating vectors")
    t = time.time()
    gen = ds.images.create_generator(
        tp.training.batch_size, shuffle=False, one_shot=True
    )
    vectors = model_head.predict(gen.create())
    print(f"! {time.time() - t}s elapsed, ({len(vectors)}/{len(ds.images.data)} vectors)")

    # Calculate augmented vectors
    train_idx = ds.train_idx
    cls_onehot = ds.cls_onehot
    if tp.training.use_augmentation is True and tp.training.transfer_learning_augmentation_factor > 0:
        print("- calculating augmentated vectors")
        t = time.time()
        if tp.augmentation.rotation is True:
            tp.augmentation.rotation = [0, 360]
        elif tp.augmentation.rotation is False:
            tp.augmentation.rotation = None

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

        aug_gen = ds.train_generator(
            batch_size=tp.training.batch_size,
            shuffle=False,
            one_shot=True,
            map_fn=augment_fn,
            undersample=tp.training.use_class_undersampling,
        )

        for aug_i in range(tp.training.transfer_learning_augmentation_factor):
            print(f"- {aug_i + 1}/{tp.training.transfer_learning_augmentation_factor}")
            aug_vectors = model_head.predict(aug_gen.create())
            len_vectors = len(vectors)
            train_idx = np.concatenate((train_idx, np.arange(len(aug_vectors)) + len_vectors))
            vectors = np.concatenate((vectors, aug_vectors))
            cls_onehot = np.concatenate((cls_onehot, ds.cls_onehot[ds.train_idx]))
        print(f"! {time.time() - t}s elapsed, ({len(vectors)}/{len(ds.images.data)} normal + augmented vectors)")

    print("-" * 80)
    print("Training")

    # Learning rate scheduler
    alr_cb = AdaptiveLearningRateScheduler(nb_epochs=tp.training.alr_epochs,
                                           nb_drops=tp.training.alr_drops,
                                           verbose=1)

    # Training generator
    train_gen = TFGenerator(vectors,
                            cls_onehot,
                            train_idx,
                            tp.training.batch_size,
                            shuffle=True,
                            one_shot=False,
                            undersample=tp.training.use_class_undersampling)

    # Validation generator
    if tp.dataset.val_split > 0:
        val_gen = TFGenerator(
            vectors,
            ds.cls_onehot,
            ds.test_idx,
            tp.training.batch_size,
            shuffle=False,
            one_shot=True,
        )
        val_data = val_gen.create()
        val_steps = len(val_gen)
    else:
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
    history = model_tail.fit(train_gen.create(),
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
    history.history["training_time"] = training_time

    # Combine the head model with the tail model to create the full model
    model = combine_head_and_tail(model_head, model_tail)

    # Return the model and history
    return model, history
