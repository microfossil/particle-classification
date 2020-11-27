import tensorflow as tf
import os
from zipfile import ZipFile
import hashlib


def download_images(origin, directory):
    # Fix URLs
    # - OneDrive
    if origin.startswith("https://1drv.ms/"):
        origin = origin.replace("https://1drv.ms/", "https://1drv.ws/")

    hash = hashlib.md5(origin.encode()).hexdigest()[:10]
    directory = os.path.join(directory, hash)
    zip_path = os.path.join(directory, "download.zip")
    if os.path.exists(directory) is False or os.path.exists(zip_path) is True:
        os.makedirs(directory, exist_ok=True)
        outp = tf.keras.utils.get_file(
            "download.zip",
            origin,
            untar=False,
            md5_hash=None,
            file_hash=None,
            cache_subdir=directory,
            hash_algorithm='auto',
            extract=True,
            archive_format='auto',
            cache_dir=None)
        print(outp)
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(zip_path, 'r') as zip_obj:
            # Get list of files names in zip
            paths = [path for path in zip_obj.namelist() if not path.startswith("_")]
            dir_name = paths[0].replace(',', '/').replace(',', '\\')
            folder_path = os.path.join(directory, dir_name)
        os.remove(os.path.join(directory, "download.zip"))
    else:
        paths = [path for path in os.listdir(directory) if not path.startswith("_")]
        folder_path = os.path.join(directory, paths[0])
        print("Already downloaded at: {}".format(folder_path))
    return folder_path


