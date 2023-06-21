import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def image_generator(images,
                    onehots,
                    batch_size):
    """
    Create a simple generator without modifying the images

    :param images: Images to generate
    :param onehots: Onehot encoding of target class
    :param batch_size: Batch size for training
    :return:
    """
    datagen = ImageDataGenerator()
    return keras_augmented_image_generator(images, onehots, batch_size, datagen)


def keras_augmented_image_generator(images,
                                    onehots,
                                    batch_size,
                                    datagen):
    """
    Create a generator from with augmentation via an ImageDataGenerator
    The generator can then be used for training in model.fit_generator

    The ImageDataGenerator should be configured with the desired augmentations.
    Note: when brightness is used, rescale=1./255 must also be used
    (assuming a desired output range of [0,1] as currently brightness augmentation internally rescales to [0,255]

    :param images: Images to augment
    :param onehots: Onehot encoding of target class
    :param batch_size: Batch size for training
    :param datagen: The ImageDataGenerator with augmentation options configured
    :return:
    """
    return datagen.flow(images,
                        onehots,
                        batch_size=batch_size,
                        shuffle=True)


def tf_augmented_image_generator(images,
                                 onehots,
                                 batch_size,
                                 map_fn,
                                 shuffle_size=1000,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """
    Create a generator suing a tf.data.Dataframe with augmentation via a map function.
    The generator can then be used for training in model.fit_generator

    The map function must consist of tensorflow operators (not numpy).

    On Windows machines this will lead to faster augmentation, as there are some
    problems performing augmentation in parallel when multiprocessing is enabled in
    in model.fit / model.fit_generator and the default Keras numpy-based augmentated is used,
    e.g. in ImageDataGenerator

    :param images: Images to augment
    :param onehots: Onehot encoding of target class
    :param batch_size: Batch size for training
    :param map_fn: The augmentation map function
    :param shuffle_size: Batch size of images shuffled. Smaller values reduce memory consumption.
    :param num_parallel_calls: Number of calls in parallel, default is automatic tuning.
    :return:
    """
    # Get shapes from input data
    img_size = images.shape
    img_size = (None, img_size[1], img_size[2], img_size[3])
    onehot_size = onehots.shape
    onehot_size = (None, onehot_size[1])
    images_tensor = tf.placeholder(tf.float32, shape=img_size)
    onehots_tensor = tf.placeholder(tf.float32, shape=onehot_size)
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, onehots_tensor))
    if map_fn is not None:
        dataset = dataset.map(lambda x, y: (map_fn(x), y), num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True).repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_val = iterator.get_next()
    with K.get_session().as_default() as sess:
        sess.run(init_op, feed_dict={images_tensor: images, onehots_tensor: onehots})
        while True:
            inputs, labels = sess.run(next_val)
            yield inputs, labels

def tf_vector_generator(vectors,
                         onehots,
                         batch_size,
                         shuffle_size=1000,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE):
    # Get shapes from input data
    vec_size = vectors.shape
    vec_size = (None, vec_size[1])
    onehot_size = onehots.shape
    onehot_size = (None, onehot_size[1])
    vectors_tensor = tf.placeholder(tf.float32, shape=vec_size)
    onehots_tensor = tf.placeholder(tf.float32, shape=onehot_size)
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((vectors_tensor, onehots_tensor))
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True).repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_val = iterator.get_next()
    with K.get_session().as_default() as sess:
        sess.run(init_op, feed_dict={vectors_tensor: vectors, onehots_tensor: onehots})
        while True:
            inputs, labels = sess.run(next_val)
            yield inputs, labels


def image_generator_from_dataframe(dataframe,
                                   img_size,
                                   batch_size,
                                   cls_labels,
                                   datagen,
                                   color_mode='rgb'):
    """
    Creates a generator that loads images from information in a pandas dataframe.

    The dataframe must have at least two columns:
    - "filename" with the absolute path to the file
    - "cls" with the class label of each image (text)

    Images will be preprocessed using an ImageDataGenerator, resized to a fixed shape and converted to grayscale if desired.

    :param dataframe: Pandas dataframe with the image information
    :param img_size: Shape of to resize the images to, e.g. (128, 128)
    :param batch_size: Size of the generator batch
    :param cls_labels: List containing each class label
    :param datagen: The ImageDataGenerator for preprocessing
    :param color_mode: 'rgb' or 'grayscale' to produce 3 or 1 channel images, respectively
    :return:
    """
    return datagen.flow_from_dataframe(
        dataframe,
        x_col="filenames",
        y_col="cls",
        classes=cls_labels,
        target_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        interpolation='bilinear',
        class_mode='categorical')


def tf_augmented_image_generator_for_segmentation(images,
                                                  targets,
                                                  batch_size,
                                                  map_fn,
                                                  augment_targets=False,
                                                  shuffle_size=1000,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE):
    # Get shapes from input data
    img_size = images.shape
    img_size = (None,) + img_size[1:]
    target_size = targets.shape
    target_size = (None,) + target_size[1:]
    images_tensor = tf.compat.v1.placeholder(tf.float32, shape=img_size)
    targets_tensor = tf.compat.v1.placeholder(tf.float32, shape=target_size)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, targets_tensor))
    if map_fn is not None:
        if augment_targets is False:
            dataset = dataset.map(lambda x, y: (map_fn(x), y), num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(lambda x, y: map_fn(x, y), num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True).repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        sess.run(init_op, feed_dict={images_tensor: images, targets_tensor: targets})
        while True:
            inputs, labels = sess.run(next_val)
            yield inputs, labels
