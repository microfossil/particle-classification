import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from imblearn.under_sampling import RandomUnderSampler


class TFGenerator(object):
    def __init__(self,
                 data,
                 cls=None,
                 idxs=None,
                 batch_size=32,
                 shuffle=True,
                 prefetch=4,
                 map_fn=None,
                 one_shot=False,
                 undersample=False,
                 data_dtype=tf.float32,
                 labels_dtype=tf.float32):
        """
        Class to create a tf.data.Dataset given a set of data and associated labels.
        Use the create() function to return the dataset
        :param data: Input data
        :param cls: Input class data (e.g. onehot vector or index), can be none
        :param idxs: Indices of the data to use, if None all data are used
        :param batch_size: Batch size for training / inference
        :param shuffle: Whether to shuffle the data each time
        :param prefetch: How many batches to prefetch
        :param map_fn: Function applied to the data when creating a batch. Must take a tensor as input
        :param one_shot: If True, dataset will only iterate through the data once. (Use for validation / inference etc)
        """
        self.data = data
        self.data_dtype = data_dtype
        if isinstance(cls, list):
            cls = np.asarray(cls)
        self.labels = cls
        self.labels_dtype = labels_dtype
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.map_fn = map_fn
        self.one_shot = one_shot
        self.undersample = undersample
        if idxs is None:
            self.idxs = np.arange(len(data))
        else:
            self.idxs = idxs.copy()
        self.on_epoch_end()

    def __len__(self):
        if self.one_shot:
            return int(np.ceil(len(self.idxs) / self.batch_size))
        else:
            return int(np.floor(len(self.idxs) / self.batch_size))

    def on_epoch_end(self):
        if self.undersample:
            rus = RandomUnderSampler()
            if isinstance(self.labels[0], np.ndarray):
                c = np.argmax(self.labels, axis=-1)
            else:
                c = self.labels
            x, y = rus.fit_sample(self.idxs.reshape(-1, 1), c[self.idxs])
            x = x.flatten()
            #y = y.flatten()
            np.random.shuffle(x)
            #print(len(x))
            #print(np.unique(y, return_counts=True))
            #np.random.shuffle(y)
            #for i in range(100):
            #    print(y[i], end="")
            #print()
            self.idxs = x
        elif self.shuffle:
            np.random.shuffle(self.idxs)

    def generator(self):
        """
        Generates pairs of data and optionally, cls. After all data is processed, the index is randomised
        :return: generator of (data[i], cls[i])
        """
        i = 0
        while i < len(self.idxs):
            idx = self.idxs[i]
            if self.labels is None:
                yield self.data[idx]
            else:
                yield (self.data[idx], self.labels[idx])
            i += 1
        self.on_epoch_end()

    def to_tfdataset(self):
        """
        Creates the tf.data.Dataset
        :return: A tf.data.Dataset that iterates through batches of (data, label) pairs
        """
        if isinstance(self.labels, list) or np.ndim(self.labels) <= 1:
            label_shape = None
        else:
            label_shape = self.labels[0].shape
        if self.labels is not None:
            ds = tf.data.Dataset.from_generator(self.generator,
                                                output_types=(self.data_dtype, self.labels_dtype),
                                                output_shapes=(self.data[0].shape, label_shape))
        else:
            ds = tf.data.Dataset.from_generator(self.generator,
                                                output_types=self.data_dtype,
                                                output_shapes=self.data[0].shape)
        """
        From docs:
        Performance can often be improved by setting num_parallel_calls so that map will use multiple threads to process elements. 
        If deterministic order isn't required, it can also improve performance to set deterministic=False.

        Note that the map function has to take a Tensor input
        """
        if self.map_fn is not None:
            ds = ds.map(lambda x, y: (self.map_fn(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.one_shot is False:
            ds = ds.repeat()
        ds = ds.batch(self.batch_size).prefetch(self.prefetch)
        return ds

    def tf1_compat_generator_error(self):
        # Get shapes from input data
        images = self.data
        onehots = self.labels
        shuffle_size = 1000
        img_size = images.shape
        img_size = (None, *img_size[1:])
        onehot_size = onehots.shape
        onehot_size = (None, onehot_size[1])
        images_tensor = tf.placeholder(tf.float32, shape=img_size)
        onehots_tensor = tf.placeholder(tf.float32, shape=onehot_size)
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images_tensor, onehots_tensor))
        if self.map_fn is not None:
            dataset = dataset.map(lambda x, y: (self.map_fn(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.one_shot:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True).repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        next_val = iterator.get_next()
        with K.get_session().as_default() as sess:
            sess.run(init_op, feed_dict={images_tensor: images, onehots_tensor: onehots})
            while True:
                inputs, labels = sess.run(next_val)
                yield inputs, labels

    def tf1_compat_generator(self):
        if isinstance(self.labels, list) or np.ndim(self.labels) <= 1:
            label_shape = None
        else:
            label_shape = self.labels[0].shape
        if self.labels is not None:
            ds = tf.data.Dataset.from_generator(self.generator,
                                                output_types=(self.data_dtype, self.labels_dtype),
                                                output_shapes=(self.data[0].shape, label_shape))
        else:
            ds = tf.data.Dataset.from_generator(self.generator,
                                                output_types=self.data_dtype,
                                                output_shapes=self.data[0].shape)
        if self.map_fn is not None:
            ds = ds.map(lambda x, y: (self.map_fn(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.one_shot is False:
            ds = ds.repeat()
        ds = ds.batch(self.batch_size).prefetch(self.prefetch)
        iterator = ds.make_initializable_iterator()
        init_op = iterator.initializer
        next_val = iterator.get_next()
        with K.get_session().as_default() as sess:
            sess.run(init_op)
            while True:
                inputs, labels = sess.run(next_val)
                yield inputs, labels

    def create(self):
        if int(tf.__version__[0]) == 2:
            return self.to_tfdataset()
        else:
            return self.tf1_compat_generator()

    @staticmethod
    def map_fn_divide_255(t):
        t = tf.cast(t, tf.float32)
        return tf.divide(t, 255.0)

    @staticmethod
    def map_fn_divide_255_and_rotate_fn(k):
        def wrapper(t):
            t = tf.cast(t, tf.float32)
            t = tf.image.rot90(t, k)
            return tf.divide(t, 255.0)
        return wrapper

