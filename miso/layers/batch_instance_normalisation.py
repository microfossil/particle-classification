import tensorflow as tf

def batch_instance_norm(x, scope='batch_instance_norm'):
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class BatchInstanceNormalisation(Layer):

    def __init__(self, **kwargs):
        super(BatchInstanceNormalisation, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        ch = input_shape[-1]
        self.rho = self.add_weight(name='rho',
                                   shape=[ch],
                                   trainable=True,
                                   initializer=tf.constant_initializer(1.0),
                                   constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        self.gamma = self.add_weight(name='gamma',
                                   shape=[ch],
                                   trainable=True,
                                   initializer=tf.constant_initializer(1.0),
                                   constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        self.beta = self.add_weight(name='beta',
                                   shape=[ch],
                                   trainable=True,
                                   initializer=tf.constant_initializer(0.0),
                                   constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        super(BatchInstanceNormalisation, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        x_hat = self.rho * x_batch + (1 - self.rho) * x_ins
        x_hat = x_hat * self.gamma + self.beta

        return x_hat