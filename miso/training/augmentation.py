import math
import tensorflow as tf

try:
    import tensorflow_addons as tfa
except ImportError:
    pass


def augmentation_complete_tf2(im_x,
                          rotation=[0, 360],
                          gain=[0.8, 1.0, 1.2],
                          gamma=[0.5, 1.0, 2],
                          zoom=[0.9, 1.0, 1.1],
                          gaussian_noise=None,
                          bias=None,
                          divide_by_255=True):
    """
    Method used with tf.data.Dataset to implement pre-processing
    """
    im_x = tf.cast(im_x, tf.float32)
    if divide_by_255:
        im_x = tf.divide(im_x, 255)
    shape = tf.shape(im_x)
    # ROTATION
    if rotation is not None:
        if len(rotation) > 2:
            rotation_factor = tf.random.shuffle(tf.constant(rotation))[0]
        elif len(rotation) == 2:
            rotation_factor = tf.random.uniform([], rotation[0] / 180 * math.pi, rotation[1] / 180 * math.pi)
        im_x = tfa.image.rotate(im_x, rotation_factor, interpolation='BILINEAR')
    # ZOOM
    if zoom is not None:
        if len(zoom) > 2:
            zoom_value = tf.random.shuffle(tf.constant(zoom))[0]
        elif len(zoom) == 2:
            zoom_value = tf.random.uniform([], zoom[0], zoom[1])
        zoom_start_factor = tf.divide(tf.subtract(1.0, zoom_value), 2)
        zoom_end_factor = tf.subtract(1.0, zoom_start_factor)
        zoom_factor = [[zoom_start_factor, zoom_start_factor, zoom_end_factor, zoom_end_factor]]
        im_x = \
            tf.image.crop_and_resize([im_x], boxes=zoom_factor, box_indices=tf.constant([0]), crop_size=tf.shape(im_x)[0:2],
                                     method='bilinear', extrapolation_value=0)[0]
    # GAIN
    if gain is not None:
        if len(gain) > 2:
            gain_factor = tf.random.shuffle(tf.constant(gain))[0]
        elif len(gain) == 2:
            gain_factor = tf.random.uniform([], gain[0], gain[1])
    else:
        gain_factor = tf.constant(1.0)
    # GAMMA
    if gamma is not None:
        if len(gamma) > 2:
            gamma_factor = tf.random.shuffle(tf.constant(gamma))[0]
        elif len(gamma) == 2:
            gamma_factor = tf.random.uniform([], gamma[0], gamma[1])
    else:
        gamma_factor = tf.constant(1.0)
    if gamma is not None or gain is not None:
        im_x = tf.image.adjust_gamma(im_x, gamma_factor, gain_factor)
    # NOISE
    if gaussian_noise is not None:
        if len(gaussian_noise) > 2:
            gaussian_noise_factor = tf.random.shuffle(tf.constant(gaussian_noise))[0]
        elif len(gaussian_noise) == 2:
            gaussian_noise_factor = tf.random.uniform([], gaussian_noise[0], gaussian_noise[1])
        noise_tensor = tf.random.normal(shape,
                                        mean=0.0,
                                        stddev=gaussian_noise_factor,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None)
        im_x = tf.add(im_x, noise_tensor)
    # OFFSET
    if bias is not None:
        if len(bias) > 2:
            bias_factor = tf.random.shuffle(tf.constant(bias))[0]
        elif len(bias) == 2:
            bias_factor = tf.random.uniform([], bias[0], bias[1])
        im_x = tf.add(im_x, bias_factor)
    return im_x


def augmentation_complete(im_x,
                          rotation=[0, 360],
                          gain=[0.8, 1.0, 1.2],
                          gamma=[0.5, 1.0, 2],
                          zoom=[0.9, 1.0, 1.1],
                          gaussian_noise=None,
                          bias=None,
                          divide_by_255=True):
    """
    Method used with tf.data.Dataset to implement pre-processing
    """
    im_x = tf.cast(im_x, tf.float32)
    if divide_by_255:
        im_x = tf.divide(im_x, 255)
    shape = tf.shape(im_x)
    # ROTATION
    if rotation is not None:
        if len(rotation) > 2:
            rotation_factor = tf.random.shuffle(tf.constant(rotation))[0]
        elif len(rotation) == 2:
            rotation_factor = tf.random.uniform([], rotation[0] / 180 * math.pi, rotation[1] / 180 * math.pi)
        im_x = tf.contrib.image.rotate(im_x, rotation_factor, interpolation='BILINEAR')
    # ZOOM
    if zoom is not None:
        if len(zoom) > 2:
            zoom_value = tf.random.shuffle(tf.constant(zoom))[0]
        elif len(zoom) == 2:
            zoom_value = tf.random.uniform([], zoom[0], zoom[1])
        zoom_start_factor = tf.divide(tf.subtract(1.0, zoom_value), 2)
        zoom_end_factor = tf.subtract(1.0, zoom_start_factor)
        zoom_factor = [[zoom_start_factor, zoom_start_factor, zoom_end_factor, zoom_end_factor]]
        im_x = \
            tf.image.crop_and_resize([im_x], boxes=zoom_factor, box_ind=tf.constant([0]), crop_size=tf.shape(im_x)[0:2],
                                     method='bilinear', extrapolation_value=0)[0]
    # GAIN
    if gain is not None:
        if len(gain) > 2:
            gain_factor = tf.random_shuffle(tf.constant(gain))[0]
        elif len(gain) == 2:
            gain_factor = tf.random_uniform([], gain[0], gain[1])
    else:
        gain_factor = tf.constant(1.0)
    # GAMMA
    if gamma is not None:
        if len(gamma) > 2:
            gamma_factor = tf.random_shuffle(tf.constant(gamma))[0]
        elif len(gamma) == 2:
            gamma_factor = tf.random_uniform([], gamma[0], gamma[1])
    else:
        gamma_factor = tf.constant(1.0)
    if gamma is not None or gain is not None:
        im_x = tf.image.adjust_gamma(im_x, gamma_factor, gain_factor)
    # NOISE
    if gaussian_noise is not None:
        if len(gaussian_noise) > 2:
            gaussian_noise_factor = tf.random_shuffle(tf.constant(gaussian_noise))[0]
        elif len(gaussian_noise) == 2:
            gaussian_noise_factor = tf.random_uniform([], gaussian_noise[0], gaussian_noise[1])
        noise_tensor = tf.random.normal(shape,
                                        mean=0.0,
                                        stddev=gaussian_noise_factor,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None)
        im_x = tf.add(im_x, noise_tensor)
    # OFFSET
    if bias is not None:
        if len(bias) > 2:
            bias_factor = tf.random_shuffle(tf.constant(bias))[0]
        elif len(bias) == 2:
            bias_factor = tf.random_uniform([], bias[0], bias[1])
        im_x = tf.add(im_x, bias_factor)
    return im_x


def segmentation_augmentation(im_x,
                              mask,
                              rotation=[0, 360],
                              gain=[0.8, 1.0, 1.2],
                              gamma=[0.5, 1.0, 2],
                              zoom=[0.9, 1.0, 1.1],
                              gaussian_noise=None,
                              bias=None):
    """
    Method used with tf.data.Dataset to implement pre-processing
    """
    im_x = tf.cast(im_x, tf.float32)
    mask = tf.cast(mask, tf.float32)
    shape = tf.shape(im_x)
    # Rotation
    # - rotate image
    # - rotate mask
    if rotation is not None:
        if len(rotation) > 2:
            rotation_factor = tf.random_shuffle(tf.constant(rotation))[0]
        elif len(rotation) == 2:
            rotation_factor = tf.random_uniform([], rotation[0] / 180 * math.pi, rotation[1] / 180 * math.pi)
        im_x = tf.contrib.image.rotate(im_x, rotation_factor, interpolation='BILINEAR')
        mask = tf.contrib.image.rotate(mask, rotation_factor, interpolation='BILINEAR')
    # Zoom
    # - zoom image
    # - zoom mask
    if zoom is not None:
        if len(zoom) > 2:
            zoom_value = tf.random_shuffle(tf.constant(zoom))[0]
        elif len(zoom) == 2:
            zoom_value = tf.random_uniform([], zoom[0], zoom[1])
        zoom_start_factor = tf.divide(tf.subtract(1.0, zoom_value), 2)
        zoom_end_factor = tf.subtract(1.0, zoom_start_factor)
        zoom_factor = [[zoom_start_factor, zoom_start_factor, zoom_end_factor, zoom_end_factor]]
        im_x = tf.image.crop_and_resize([im_x],
                                        boxes=zoom_factor,
                                        box_ind=tf.constant([0]),
                                        crop_size=tf.shape(im_x)[0:2],
                                        method='bilinear',
                                        extrapolation_value=0)[0]
        mask = tf.image.crop_and_resize([mask],
                                        boxes=zoom_factor,
                                        box_ind=tf.constant([0]),
                                        crop_size=tf.shape(mask)[0:2],
                                        method='bilinear',
                                        extrapolation_value=0)[0]

    # Gain
    # - only apply gain to image
    if gain is not None:
        if len(gain) > 2:
            gain_factor = tf.random_shuffle(tf.constant(gain))[0]
        elif len(gain) == 2:
            gain_factor = tf.random_uniform([], gain[0], gain[1])
    else:
        gain_factor = tf.constant(1.0)
    # Gamma
    # - only apply gamma to image
    if gamma is not None:
        if len(gamma) > 2:
            gamma_factor = tf.random_shuffle(tf.constant(gamma))[0]
        elif len(gamma) == 2:
            gamma_factor = tf.random_uniform([], gamma[0], gamma[1])
    else:
        gamma_factor = tf.constant(1.0)
    if gamma is not None or gain is not None:
        im_x = tf.image.adjust_gamma(im_x, gamma_factor, gain_factor)
    # Noise
    # - only apply noise to image
    if gaussian_noise is not None:
        if len(gaussian_noise) > 2:
            gaussian_noise_factor = tf.random_shuffle(tf.constant(gaussian_noise))[0]
        elif len(gaussian_noise) == 2:
            gaussian_noise_factor = tf.random_uniform([], gaussian_noise[0], gaussian_noise[1])
        noise_tensor = tf.random.normal(shape,
                                        mean=0.0,
                                        stddev=gaussian_noise_factor,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None)
        im_x = tf.add(im_x, noise_tensor)
    # Offset
    # - only apply offset to image
    if bias is not None:
        if len(bias) > 2:
            bias_factor = tf.random_shuffle(tf.constant(bias))[0]
        elif len(bias) == 2:
            bias_factor = tf.random_uniform([], bias[0], bias[1])
        im_x = tf.add(im_x, bias_factor)
    return (im_x, mask)


def transform(images,
              transforms,
              interpolation="NEAREST",
              output_shape=None,
              name=None):
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns,
        num_channels) (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW). The rank must be statically known (the
        shape is not `TensorShape(None)`.
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
        the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      interpolation: Interpolation mode.
        Supported values: "NEAREST", "BILINEAR".
      output_shape: Output dimesion after the transform, [height, width].
        If None, output is the same size as input image.

      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with tf.name_scope(name or "transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        transform_or_transforms = tf.convert_to_tensor(
            transforms, name="transforms", dtype=tf.dtypes.float32)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        elif image_or_images.get_shape().ndims is None:
            raise TypeError("image_or_images rank must be statically known")
        elif len(image_or_images.get_shape()) == 2:
            images = image_or_images[None, :, :, None]
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images[None, :, :, :]
        elif len(image_or_images.get_shape()) == 4:
            images = image_or_images
        else:
            raise TypeError("Images should have rank between 2 and 4.")

        if output_shape is None:
            output_shape = tf.shape(images)[1:3]

        output_shape = tf.convert_to_tensor(
            output_shape, tf.dtypes.int32, name="output_shape")

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width")

        if len(transform_or_transforms.get_shape()) == 1:
            transforms = transform_or_transforms[None]
        elif transform_or_transforms.get_shape().ndims is None:
            raise TypeError(
                "transform_or_transforms rank must be statically known")
        elif len(transform_or_transforms.get_shape()) == 2:
            transforms = transform_or_transforms
        else:
            raise TypeError("Transforms should have rank 1 or 2.")

        output = _image_ops_so.image_projective_transform_v2(
            images,
            output_shape=output_shape,
            transforms=transforms,
            interpolation=interpolation.upper())
        if len(image_or_images.get_shape()) == 2:
            return output[0, :, :, 0]
        elif len(image_or_images.get_shape()) == 3:
            return output[0, :, :, :]
        else:
            return output


def angles_to_projective_transforms(angles,
                                    image_height,
                                    image_width,
                                    name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of
        images) a vector with an angle to rotate each image in the batch. The
        rank must be statically known (the shape is not `TensorShape(None)`.
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
      given to `transform` op.
    """
    with tf.name_scope(name or "angles_to_projective_transforms"):
        angle_or_angles = tf.convert_to_tensor(
            angles, name="angles", dtype=tf.dtypes.float32)
        if len(angle_or_angles.get_shape()) == 0:
            angles = angle_or_angles[None]
        elif len(angle_or_angles.get_shape()) == 1:
            angles = angle_or_angles
        else:
            raise TypeError("Angles should have rank 0 or 1.")
        # yapf: disable
        x_offset = ((image_width - 1) -
                    (tf.math.cos(angles) * (image_width - 1) -
                     tf.math.sin(angles) * (image_height - 1))) / 2.0
        y_offset = ((image_height - 1) -
                    (tf.math.sin(angles) * (image_width - 1) +
                     tf.math.cos(angles) * (image_height - 1))) / 2.0
        # yapf: enable
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.math.cos(angles)[:, None],
                -tf.math.sin(angles)[:, None],
                x_offset[:, None],
                tf.math.sin(angles)[:, None],
                tf.math.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.dtypes.float32),
            ],
            axis=1)


def rotate(images, angles, interpolation="NEAREST", name=None):
    """Rotate image(s) counterclockwise by the passed angle(s) in radians.

    Args:
      images: A tensor of shape
        (num_images, num_rows, num_columns, num_channels)
        (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW). The rank must be statically known (the
        shape is not `TensorShape(None)`.
      angles: A scalar angle to rotate all images by, or (if images has rank 4)
        a vector of length num_images, with an angle for each image in the
        batch.
      interpolation: Interpolation mode. Supported values: "NEAREST",
        "BILINEAR".
      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, rotated by the given
      angle(s). Empty space due to the rotation will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
    """
    with tf.name_scope(name or "rotate"):
        image_or_images = tf.convert_to_tensor(images)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        if image_or_images.get_shape().ndims is None:
            raise TypeError("image_or_images rank must be statically known")
        elif len(image_or_images.get_shape()) == 2:
            images = image_or_images[None, :, :, None]
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images[None, :, :, :]
        elif len(image_or_images.get_shape()) == 4:
            images = image_or_images
        else:
            raise TypeError("Images should have rank between 2 and 4.")

        image_height = tf.cast(tf.shape(images)[1], tf.dtypes.float32)[None]
        image_width = tf.cast(tf.shape(images)[2], tf.dtypes.float32)[None]
        output = transform(
            images,
            angles_to_projective_transforms(angles, image_height, image_width),
            interpolation=interpolation)
        if image_or_images.get_shape().ndims is None:
            raise TypeError("image_or_images rank must be statically known")
        elif len(image_or_images.get_shape()) == 2:
            return output[0, :, :, 0]
        elif len(image_or_images.get_shape()) == 3:
            return output[0, :, :, :]
        else:
            return output
