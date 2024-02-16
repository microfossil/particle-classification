import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import skimage.io
from PIL import Image
from tqdm import tqdm

from miso.data.image_utils import load_image
from miso.deploy.model_info import ModelInfo
from miso.deploy.saving import load_frozen_model_tf2, load_from_xml, load_onnx_from_xml
import tensorflow as tf

from miso.inference.morphology import MorphologyProcessor


def get_image_paths_and_samples(base_path, sample_name="unknown"):
    base_path = Path(base_path)
    image_paths = []
    samples = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            for file in subdir.rglob('*'):
                if file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                    image_paths.append(str(file))
                    samples.append(subdir.name)
        else:
            file = subdir
            if file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                image_paths.append(str(file))
                samples.append(sample_name)
    return image_paths, samples


def classify_folder(model_info_path,
                    images_path,
                    output_path,
                    batch_size,
                    sample_name="unknown",
                    unsure_threshold=0.0):
    model, img_size, img_type, labels = load_from_xml(model_info_path)

    # Create a dataset of image paths
    image_paths, sample_names = get_image_paths_and_samples(images_path, sample_name)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Image loading functions
    def load_and_preprocess_image(image_path):
        def _load_image(image_path):
            image_path = image_path.numpy().decode('utf-8')
            image = load_image(image_path, img_size, img_type)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            return image
        image = tf.py_function(_load_image, [image_path], tf.float32)
        image.set_shape(img_size)
        return image

    # Map using the image dataset
    image_dataset = image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    image_dataset = image_dataset.batch(batch_size)

    # Run predictions
    predictions = []
    idxs = []
    cls = []
    scores = []
    for batch in tqdm(image_dataset):
        preds = model(batch).numpy()
        batch_idxs = np.argmax(preds, axis=1)
        batch_labels = [labels[idx] for idx in batch_idxs]
        batch_scores = np.max(preds, axis=1)
        predictions.extend(preds)
        idxs.extend(batch_idxs)
        cls.extend(batch_labels)
        scores.extend(batch_scores)

    idxs = [idx if score > unsure_threshold else -1 for idx, score in zip(idxs, scores)]
    cls = [cls if score > unsure_threshold else "unsure" for cls, score in zip(cls, scores)]

    df = pd.DataFrame({
        "filename": image_paths,
        "short_filename": [Path(f).relative_to(images_path) for f in image_paths],
        "sample": sample_names,
        "class_index": idxs,
        "class": cls,
        "score": scores
    })
    df.to_csv(output_path, index=False)


import onnxruntime as rt


def segment_folder(model_info_path,
                   images_path,
                   output_path,
                   batch_size,
                   sample_name="unknown",
                   threshold=0.5,
                   save_contours=False):

    sess, img_size, img_type, _ = load_onnx_from_xml(model_info_path)

    # Make sure the number of input channels is correct
    num_channels = sess.get_inputs()[0].shape[-1]
    img_size[-1] = num_channels
    if num_channels == 1:
        img_type = "k"

    image_paths, sample_names = get_image_paths_and_samples(images_path, sample_name)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_and_preprocess_image(image_path):
        def _load_image(image_path):
            image_path = image_path.numpy().decode('utf-8')
            image = load_image(image_path, img_size, img_type)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            return image
        image = tf.py_function(_load_image, [image_path], tf.float32)
        image.set_shape(img_size)
        return image

    image_dataset = image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.batch(batch_size)

    contours = []
    for batch in tqdm(image_dataset):
        batch = batch.numpy()
        masks = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
        # masks = model(batch).numpy()
        for mask in masks:
            mask = (mask > threshold).astype(np.uint8)  # Threshold the mask
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                # Find the largest contour based on area
                areas = [cv2.contourArea(cnt) for cnt in cnts]
                largest_contour = cnts[np.argmax(areas)]
                contours.append(largest_contour)
            else:
                contours.append(None)

    # Loop through the images again, loading greyscale and then calculating the morphology
    datas = []
    idx = 0
    for image_path, sample_name, contour in tqdm(list(zip(image_paths, sample_names, contours))):
        idx += 1
        image = Image.open(image_path)
        greyscale = np.array(image.convert('L'), dtype=np.float32)

        # Make square
        border_pixels = np.concatenate(
            [greyscale[0, :],
             greyscale[-1, :],
             greyscale[:, 0],
             greyscale[:, -1]]
        )
        border_mean = np.median(border_pixels)
        orig_size = greyscale.shape[:2]
        diff = abs(orig_size[0] - orig_size[1])
        d1 = diff // 2
        d2 = diff - d1
        padding = ((d1, d2), (0, 0)) if orig_size[0] < orig_size[1] else ((0, 0), (d1, d2))
        greyscale = np.pad(greyscale, padding, mode='constant', constant_values=border_mean)

        # Get max size
        max_size = max(greyscale.shape)

        # Get scale factor
        scale_factor = max_size / img_size[0]

        # Scale contour
        if contour is not None:
            contour = contour * scale_factor

        m = MorphologyProcessor.calculate_morphology(None, greyscale, contour)

        data = {}

        data["filename"] = image_path
        data["sample"] = sample_name
        data.update(m.__dict__)

        datas.append(data)

        # Save the contour
        if contour is not None and save_contours:
            # Save path
            output_path = Path(output_path)
            contour_image_path = output_path.parent / "contours" / sample_name
            contour_image_path.mkdir(parents=True, exist_ok=True)
            rgb = cv2.cvtColor(greyscale, cv2.COLOR_GRAY2RGB).astype(np.uint8)
            cv2.drawContours(rgb, [contour.astype(int)], -1, (255, 0, 0), 1)
            skimage.io.imsave(contour_image_path / f"{Path(image_path).stem}_contour.png", rgb, check_contrast=False)

        if idx % 1000 == 0:
            df = pd.DataFrame(datas)
            df.to_csv(output_path, index=False)


if __name__ == "__main__":
    pass
    # def _load_image(image_path):
    #     image_path = image_path
    #     image = Image.open(image_path)
    #     greyscale = np.array(image.convert('L'), dtype=np.float32)[..., np.newaxis]
    #     greyscale = greyscale[:,50:,:]
    #     border_pixels = np.concatenate(
    #         [greyscale[0, :, 0],
    #          greyscale[-1, :, 0],
    #          greyscale[:, 0, 0],
    #          greyscale[:, -1, 0]]
    #     )
    #     border_mean = np.median(border_pixels)
    #     orig_size = greyscale.shape[:2]
    #     diff = abs(orig_size[0] - orig_size[1])
    #     d1 = diff // 2
    #     d2 = diff - d1
    #     padding = ((d1, d2), (0, 0), (0, 0)) if orig_size[0] < orig_size[1] else ((0, 0), (d1, d2), (0, 0))
    #     image = np.pad(greyscale, padding, mode='constant', constant_values=border_mean)
    #     image = tf.image.resize(image, [224, 224])
    #     image = image / 255.0
    #     return image
    #
    # # Test load image
    # image_path = r"C:\Users\ross.marchant\data\_Zooscan_centering_NOprocess\_inputs\_images\train_processed\crust_amphipoda\013-crus_amphipoda.png"
    # image = _load_image(image_path)
    #
    # plt.imshow(image[..., 0], cmap='gray')
    # plt.show()

    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    #
    # segment_folder(
    #     r"C:\Users\ross.marchant\code\Microfossil\particle-trieur\target\classes\trained_networks\plankton_segmenter\model_info.xml",
    #     r"C:\Users\ross.marchant\data\_Zooscan_centering_NOprocess\_inputs\_images\train_processed",
    #     r"F:\morphology.csv",
    #     64,
    #     sample_name="unknown",
    #     threshold=0.8,
    #     save_contours=True)

