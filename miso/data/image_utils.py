import numpy as np
from skimage import io as skio, transform as skt, color as skc


def load_image(filename, img_size=None, img_type='rgb'):
    """
    Loads an image, converting it if necessary
    :param filename: Filename of image to load
    :param img_size: Size of image, e.g. (224, 224). If None, the image dimensions are preserved
    :param img_type: 'rgb' (colour) or 'k' (greyscale)
    :return:
    """
    # Colour
    if img_type == 'rgb':
        im = skio.imread(filename)
        # im = np.asarray(im, dtype=np.float)
        # If it was a single channel image, make into 3-channel
        if im.ndim == 2:
            im = np.repeat(im[..., np.newaxis], repeats=3, axis=-1)
    # Greyscale
    elif img_type == 'k' or img_type == 'greyscale':
        im = skio.imread(filename, as_gray=True)
        im = im[..., np.newaxis]
    else:
        raise ValueError("img_type must be 'rgb' or 'k'")
    # Resize and pad
    if img_size is not None:
        im = resize_and_pad_image(im, img_size)
    return im


def resize_and_pad_image(im, img_size):
    # Add last dim if greyscale
    if np.ndim(im) == 2:
        im = im[..., np.newaxis]
    # Get the ratio of width to height for each
    current_whratio = im.shape[1] / im.shape[0]
    desired_whratio = img_size[1] / img_size[0]
    # Check if the image has roughly the same ratio, else pad it
    if np.round(im.shape[0] * desired_whratio) != im.shape[1]:
        height = im.shape[0]
        width = im.shape[1]
        # Desired shape is wider than current one
        if desired_whratio > current_whratio:
            half = np.round(height * desired_whratio)
            height_pad_start = 0
            height_pad_end = 0
            width_pad_start = int(abs(np.floor((width - half) / 2)))
            width_pad_end = int(abs(np.ceil((width - half) / 2)))
        # Desired shape is taller than current
        else:
            half = np.round(width / desired_whratio)
            height_pad_start = int(abs(np.floor((height - half) / 2)))
            height_pad_end = int(abs(np.ceil((height - half) / 2)))
            width_pad_start = 0
            width_pad_end = 0
        # Constant value to pad with
        consts = [np.median(np.concatenate((im[0, :, i], im[-1, :, i], im[:, 0, i], im[:, -1, i]))) for i in range(im.shape[2])]
        # Pad
        im = np.stack(
            [np.pad(im[:, :, c],
                    ((height_pad_start, height_pad_end), (width_pad_start, width_pad_end)),
                    mode='constant',
                    constant_values=consts[c])
             for c in range(im.shape[2])], axis=2)
        # # Revert if was greyscale
        # if im.shape[2] == 1:
        #     im = im[..., 0]
    # Resize
    if im.dtype == np.uint8:
        im = im / 255
    if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
        im = skt.resize(im, img_size)
    return im


def to_channels(im, out_channels, to_greyscale=False):
    # Number of channels in image
    if np.ndim(im) == 3:
        img_channels = im.shape[2]
    else:
        im = im[..., np.newaxis]
        img_channels = 1
    # Convert to greyscale if needed:
    if img_channels == 3 and to_greyscale is True:
        im = skc.rgb2gray(im)[..., np.newaxis]
        img_channels = 1
    # Repeat if needed
    if img_channels == 1 and out_channels == 3:
        im = np.repeat(im, 3, axis=-1)
    return im


def resize_transform(im, shape, to_greyscale=False):
    im = to_channels(im, shape[2], to_greyscale)
    if im.shape[0] == shape[0] and im.shape[1] == shape[1]:
        return im
    im = skt.resize(im, shape)
    return (im * 255).astype(np.uint8)


def resize_with_pad_transform(im, shape, to_greyscale=False):
    im = to_channels(im, shape[2], to_greyscale)
    if im.shape[0] == shape[0] and im.shape[1] == shape[1]:
        return im
    im = resize_and_pad_image(im, shape)
    return (im * 255).astype(np.uint8)


def null_transform(im, args):
    return im
