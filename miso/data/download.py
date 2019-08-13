import tensorflow as tf
import os


def download_images(origin, directory):
    os.makedirs(directory, exist_ok=True)
    tf.keras.utils.get_file(
        "tmp_download.zip",
        origin,
        untar=False,
        md5_hash=None,
        file_hash=None,
        cache_subdir=directory,
        hash_algorithm='auto',
        extract=True,
        archive_format='auto',
        cache_dir=None)
    os.remove(os.path.join(directory, "tmp_download.zip"))


