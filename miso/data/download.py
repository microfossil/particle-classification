import tensorflow as tf
import os
from zipfile import ZipFile
import hashlib


def download_images(origin, directory):
    hash = hashlib.md5(origin.encode()).hexdigest()[:10]
    directory = os.path.join(directory, hash)
    if os.path.exists(directory) is False:
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
        zip_path = os.path.join(directory, "download.zip")
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(zip_path, 'r') as zip_obj:
            # Get list of files names in zip
            dir_name = zip_obj.namelist()[0].replace(',','/').replace(',','\\')
            folder_path = os.path.join(directory, dir_name)
        os.remove(os.path.join(directory, "download.zip"))
    else:
        folder_path = os.path.join(directory, os.listdir(directory)[0])
        print("@ Already downloaded at: {}".format(folder_path))
    return folder_path


