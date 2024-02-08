import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def aug_all_fn(rotation=(0, 360),
               gain=(0.8, 1.0, 1.2),
               gamma=(0.5, 1.0, 2),
               zoom=(0.9, 1.0, 1.1),
               gaussian_noise=None,
               bias=None,
               random_crop=None,
               divide=255):
    def wrapper(im_x):
        im_x = tf.cast(im_x, tf.float32)
        if divide is not None:
            im_x = tf.divide(im_x, tf.constant(divide, dtype=tf.float32))
        im_x = aug_rotation(im_x, rotation)
        im_x = aug_zoom(im_x, zoom)
        im_x = aug_gain_gamma(im_x, gain, gamma)
        im_x = aug_gaussian_noise(im_x, gaussian_noise)
        im_x = aug_bias(im_x, bias)
        im_x = aug_random_crop(im_x, random_crop)
        return im_x
    return wrapper


def aug_rotation(im_x, rotation=(0, 360)):
    if rotation is not None:
        if len(rotation) > 2:
            rotation_factor = tf.random.shuffle(tf.constant(rotation))[0]
        elif len(rotation) == 2:
            rotation_factor = tf.random.uniform([], rotation[0] / 180 * np.pi, rotation[1] / 180 * np.pi)
        else:
            raise ValueError("Rotation needs at least 2 values")
        im_x = tfa.image.rotate(im_x, rotation_factor, interpolation='bilinear')
    return im_x


def aug_zoom(im_x, zoom=(0.9, 1.0, 1.1)):
    if zoom is not None:
        if len(zoom) > 2:
            zoom_value = tf.random.shuffle(tf.constant(zoom))[0]
        elif len(zoom) == 2:
            zoom_value = tf.random.uniform([], zoom[0], zoom[1])
        else:
            raise ValueError("Zoom needs at least 2 values")
        zoom_start_factor = tf.divide(tf.subtract(1.0, zoom_value), 2)
        zoom_end_factor = tf.subtract(1.0, zoom_start_factor)
        zoom_factor = [[zoom_start_factor, zoom_start_factor, zoom_end_factor, zoom_end_factor]]
        im_x = tf.image.crop_and_resize([im_x],
                                        boxes=zoom_factor,
                                        box_indices=tf.constant([0]),
                                        crop_size=tf.shape(im_x)[0:2],
                                        method='bilinear',
                                        extrapolation_value=0)[0]
    return im_x


def aug_gain_gamma(im_x, gain=(0.8, 1.0, 1.2), gamma=(0.5, 1.0, 2)):
    # Gain
    if gain is not None:
        if len(gain) > 2:
            gain_factor = tf.random.shuffle(tf.constant(gain))[0]
        elif len(gain) == 2:
            gain_factor = tf.random.uniform([], gain[0], gain[1])
        else:
            raise ValueError("Gain needs at least 2 values")
    else:
        gain_factor = tf.constant(1.0)
    # Gamma
    if gamma is not None:
        if len(gamma) > 2:
            gamma_factor = tf.random.shuffle(tf.constant(gamma))[0]
        elif len(gamma) == 2:
            gamma_factor = tf.random.uniform([], gamma[0], gamma[1])
        else:
            raise ValueError("Gamma needs at least 2 values")
    else:
        gamma_factor = tf.constant(1.0)
    if gamma is not None or gain is not None:
        im_x = tf.image.adjust_gamma(im_x, gamma_factor, gain_factor)
    return im_x


def aug_gaussian_noise(im_x, gaussian_noise=None):
    shape = tf.shape(im_x)
    if gaussian_noise is not None:
        if len(gaussian_noise) > 2:
            gaussian_noise_factor = tf.random.shuffle(tf.constant(gaussian_noise))[0]
        elif len(gaussian_noise) == 2:
            gaussian_noise_factor = tf.random.uniform([], gaussian_noise[0], gaussian_noise[1])
        else:
            raise ValueError("Noise needs at least 2 values")
        noise_tensor = tf.random.normal(shape,
                                        mean=0.0,
                                        stddev=gaussian_noise_factor,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None)
        im_x = tf.add(im_x, noise_tensor)
    return im_x


def aug_bias(im_x, bias=(-0.5, 0.5)):
    if bias is not None:
        if len(bias) > 2:
            bias_factor = tf.random.shuffle(tf.constant(bias))[0]
        elif len(bias) == 2:
            bias_factor = tf.random.uniform([], bias[0], bias[1])
        else:
            raise ValueError("Offset needs at least 2 values")
        im_x = tf.add(im_x, bias_factor)
    return im_x


def aug_random_crop(im_x, crop_size):
    if crop_size is not None:
        im_x = tf.image.random_crop(im_x, crop_size)
    return im_x
