import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
from PIL import Image
from tqdm import tqdm

from miso.deploy.model_info import ModelInfo
from miso.deploy.saving import load_frozen_model_tf2, load_from_xml
import tensorflow as tf


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
    model, img_size, labels = load_from_xml(model_info_path)

    # Create a dataset of image paths
    image_paths, sample_names = get_image_paths_and_samples(images_path, sample_name)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Map the preprocessing function to each element, and set the number of parallel calls
    def load_and_preprocess_image(image_path):
        # image = tf.io.read_file(image_path)
        # image = tf.image.decode_image(image, channels=img_size[2], expand_animations=False)
        # image = tf.image.resize(image, [img_size[0], img_size[1]])
        # image = image / 255.0  # Normalize to [0,1] range
        # return image
        # Use tf.py_function to wrap the Pillow-based loading and preprocessing
        def _load_image(image_path):
            # Convert image_path from tensor to string
            image_path = image_path.numpy().decode('utf-8')
            # Use Pillow to open the TIFF image and convert it to an RGB array
            image = Image.open(image_path)
            if img_size[2] == 3:
                image = np.array(image.convert('RGB'), dtype=np.float32)
            else:
                image = np.array(image.convert('L'), dtype=np.float32)
            # Resize and normalize the image
            image = tf.image.resize(image, [img_size[0], img_size[1]])
            image = image / 255.0  # Normalize to [0,1] range
            return image
        # Wrap the custom loading function using tf.py_function
        # The output_types argument is important for TensorFlow to manage the tensor's dtype
        image = tf.py_function(_load_image, [image_path], tf.float32)
        # Set the shape of the tensor after the tf.py_function since it loses shape information
        image.set_shape([img_size[0], img_size[1], img_size[2]])
        return image

    image_dataset = image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    image_dataset = image_dataset.batch(batch_size)

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
