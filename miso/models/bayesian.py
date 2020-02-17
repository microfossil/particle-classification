import tensorflow_probability as tfp

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Activation, \
                                    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from miso.layers import cyclic


def neg_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def get_neg_log_likelihood_fn(bayesian=False):
    """
    Get the negative log-likelihood function
    # Arguments
        bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        a negative log-likelihood function
    """
    if bayesian:
        def neg_log_likelihood_bayesian(y_true, y_pred):
            labels_distribution = tfp.distributions.Categorical(logits=y_pred)
            log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
            loss = -tf.reduce_mean(input_tensor=log_likelihood)
            return loss
        return neg_log_likelihood_bayesian
    else:
        def neg_log_likelihood(y_true, y_pred):
            y_pred_softmax = tf.keras.layers.Activation('softmax')(y_pred)  # logits to softmax
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred_softmax)
            return loss
        return neg_log_likelihood


def base_bayes(input_shape,
                nb_classes,
               nb_training_examples,
                filters=4,
                blocks=4,
                dropout=0.5,
                dense=512,
                conv_padding='same',
                conv_activation='relu',
                use_batch_norm=True,
                global_pooling=None):

    # Based on: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py

    kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                              tf.cast(nb_training_examples, dtype=tf.float32))

    inputs = Input(shape=input_shape)
    x = inputs
    # x = cyclic.CyclicSlice4()(inputs)
    for i in range(blocks):
        conv_filters = filters * 2 ** i
        # First layer
        x = tfp.layers.Convolution2DFlipout(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_divergence_fn=kl_divergence_function)(x)
        # if use_batch_norm is True:
        #     x = BatchNormalization()(x)
        x = Activation(conv_activation)(x)
        # Second layer
        x = tfp.layers.Convolution2DReparameterization(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_divergence_fn=kl_divergence_function)(x)
        # if use_batch_norm is True:
            # x = BatchNormalization()(x)
        x = Activation(conv_activation)(x)
        # Pool
        x = MaxPooling2D()(x)
        # Roll
        # x = cyclic.CyclicRoll4()(x)
    # if global_pooling == 'avg':
    #     x = GlobalAveragePooling2D()(x)
    # elif global_pooling == 'max':
    #     x = GlobalMaxPooling2D()(x)
    # Dense layers
    x = Flatten()(x)
    # x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    # x = Dropout(dropout)(x)
    x = tfp.layers.DenseFlipout(dense, activation='relu', kernel_divergence_fn=kl_divergence_function)(x)
    # x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    x = tfp.layers.DenseFlipout(nb_classes, activation='softmax', kernel_divergence_fn=kl_divergence_function)(x)

    model = Model(inputs, x, name='base_bayes')

    return model


def build_bayesian_cnn_model_1(input_shape, nb_classes):
    model_in = Input(shape=input_shape)
    x = tfp.layers.Convolution2DFlipout(32, kernel_size=3, padding="same", strides=2)(model_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tfp.layers.Convolution2DFlipout(64, kernel_size=3, padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = tfp.layers.DenseFlipout(512, activation='relu')(x)
    model_out = tfp.layers.DenseFlipout(nb_classes, activation=None)(x)  # logits
    model = Model(model_in, model_out)
    return model

def bayes2(img_width, nb_classes, nb_examples):
    """Creates a Keras model using the LeNet-5 architecture.

    Returns:
        model: Compiled Keras model.
    """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                              tf.cast(nb_examples, dtype=tf.float32))

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    model = tf.keras.models.Sequential([
        tfp.layers.Convolution2DFlipout(
            6, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            16, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            120, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            84,
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tfp.layers.DenseFlipout(
            nb_classes,
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.softmax)
    ])

    # Model compilation.
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # We use the categorical_crossentropy loss since the MNIST dataset contains
    # ten labels. The Keras API will then automatically add the
    # Kullback-Leibler divergence (contained on the individual layers of
    # the model), to the cross entropy loss, effectively
    # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model