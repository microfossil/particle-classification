import tensorflow as tf
import numpy as np


class CyclicSlice4(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CyclicSlice4, self).__init__(**kwargs)

    def call(self, input):
        F = slice_4(input)
        return F

class CyclicGainSlice12(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CyclicGainSlice12, self).__init__(**kwargs)

    def call(self, input):
        F = slice_gain_12(input)
        return F


class CyclicRoll4(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CyclicRoll4, self).__init__(**kwargs)

    def call(self, input):
        return roll_4(input)


class CyclicPool4(tf.keras.layers.Layer):
    def __init__(self, pool_op, **kwargs):
        super(CyclicPool4, self).__init__(**kwargs)
        self.pool_op = pool_op

    def call(self, input):
        return pool_4(input, self.pool_op)


class CyclicDensePool4(tf.keras.layers.Layer):
    def __init__(self, pool_op):
        super(CyclicDensePool4, self).__init__()
        self.pool_op = pool_op

    def call(self, input):
        return dense_pool_4(input, self.pool_op)


class CyclicDensePoolN(tf.keras.layers.Layer):
    def __init__(self, pool_op, split_count=12):
        super(CyclicDensePoolN, self).__init__()
        self.pool_op = pool_op
        self.n = split_count

    def call(self, input):
        return dense_pool_n(input, self.pool_op, self.n)

def gain_3(X):
    Y = []
    Y.append(X)
    Y.append(tf.multiply(X, 0.5))
    Y.append(tf.multiply(X, 2))
    return Y

def rotate_4(X):
    Y = []
    Y.append(X)
    Y.append(tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [1]))
    Y.append(tf.reverse(X, [1, 2]))
    Y.append(tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [2]))
    return Y

def unrotate_4(X):
    Y = []
    Y.append(X)
    Y.append(tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [2]))
    Y.append(tf.reverse(X, [1, 2]))
    Y.append(tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [1]))
    return Y

def reorder_4(X, order):
    Y = tf.split(X, 4)
    Z = [Y[i] for i in order]
    return tf.concat(Z, 0)


def slice_4(X):
    Y = rotate_4(X)
    return tf.concat(Y, 0)

def slice_gain_12(X):
    Y = gain_3(X)
    Y = tf.concat(Y, 0)
    Y = rotate_4(Y)
    return tf.concat(Y, 0)

# Also try role that is a bit different
def roll_4(X):
    Y = unrotate_4(X)
    Z = []
    Z.append(X)
    for i in range(1, 4):
        Z.append(reorder_4(Y[i], np.roll(range(4), shift=-i)))
    return tf.concat(Z, 3)


def stack_4(X):
    Y = tf.split(X, 4)
    Z = []
    Z.append(Y[0])
    Z.append(tf.reverse(tf.transpose(Y[1], [0, 2, 1, 3]), [2]))
    Z.append(tf.reverse(Y[2], [1, 2]))
    Z.append(tf.reverse(tf.transpose(Y[3], [0, 2, 1, 3]), [1]))
    return tf.concat(3)


def pool_4(X, pool_op):
    Y = tf.split(X, 4)
    Z = []
    Z.append(Y[0])
    Z.append(tf.reverse(tf.transpose(Y[1], [0, 2, 1, 3]), [2]))
    Z.append(tf.reverse(Y[2], [1, 2]))
    Z.append(tf.reverse(tf.transpose(Y[3], [0, 2, 1, 3]), [1]))
    W = tf.stack(Z, 4)
    return pool_op(W, (4))


def dense_pool_4(X, pool_op):
    Y = tf.split(X, 4)
    W = tf.stack(Y, 2)
    return pool_op(W, 2)

def dense_pool_n(X, pool_op, n):
    Y = tf.split(X, n)
    W = tf.stack(Y, 2)
    return pool_op(W, 2)
