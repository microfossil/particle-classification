"""
https://towardsdatascience.com/enhancing-the-power-of-softmax-for-image-classification-4f8f85141739

https://github.com/christk1/MSH_tensorflow_keras/blob/master/mnist.py

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(params.embedding_size, activation='relu', name='feats0')(x)
x = tf.keras.layers.Dense(params.embedding_size, name='features')(x)
aux_input = tf.keras.Input(shape=(params.num_classes,))
predictions = MSoftMaxLayer(n_classes=params.num_classes, m=params.m, s=params.s, name='MSoftMaxLayer')(
    [x, aux_input])
model = tf.keras.models.Model(inputs=[base_model.input, aux_input], outputs=predictions)


REMOVE RELU FROM FINAL LAYER!
"""
import tensorflow as tf


class MSoftMaxLayer(tf.keras.layers.Layer):

    def __init__(self, n_classes, m=0.5, s=64., **kwargs):
        self.num_classes = n_classes
        self.m = m
        self.s = s
        super(MSoftMaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], self.num_classes),
                                      initializer=tf.random_normal_initializer(stddev=0.01),
                                      trainable=True)
        # input_shape[0] contains the batch
        super(MSoftMaxLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inps):
        # emb (N x embs), labels (N, 10) = labels one hot
        emb, labels = inps

        # normalize feature
        emb = tf.nn.l2_normalize(emb, axis=1) * self.s  # (n, 512)
        # normalize weights
        W = tf.nn.l2_normalize(self.kernel, axis=0)  # (512, 10)
        fc7 = tf.matmul(emb, W)  # n x 10

        # pick elements along axis 1
        zy = tf.reduce_max(input_tensor=tf.multiply(fc7, labels), axis=1)  # (n, 1)

        cos_t = zy / self.s
        t = tf.acos(cos_t)
        body = tf.cos(t + self.m)
        new_zy = body * self.s
        diff = new_zy - zy
        diff = tf.expand_dims(diff, 1)
        body = tf.multiply(labels, diff)
        fc7 = fc7 + body
        fc7 = tf.nn.softmax(fc7)

        return fc7

    def get_config(self):
        config = super(MSoftMaxLayer, self).get_config()
        config.update({'num_classes': self.num_classes, 'm': self.m, 's': self.s})
        return config

    def compute_output_shape(self, input_shape):
        return None, self.num_classes