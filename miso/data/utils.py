import glob
import os
from collections import OrderedDict
import pandas as pd
import platform


def list_directory(source_dir):
    return sorted(glob.glob(os.path.join(source_dir, "*")))


def parse_directory(source_dir, verbose=True, skip='~', has_classes=True, sub_directories=[]):
    """
    Parses a directory consisting of subdirectories named by class and stores the image filenames in a dictionary
    :param source_dir: The directory to parse
    :param verbose: Print out the progress
    :param skip: If a subdirectory starts with this character it will be skipped ('_' and '.' are skipped always, unless skip is None)
    :param has_classes: The files are arranged in subsdirectories by class name
    :return: Dictionary with class names for the keys and lists of filenames for the values
    """
    if has_classes:
        sub_dirs = sorted(glob.glob(os.path.join(source_dir, "*")))
    else:
        sub_dirs = [source_dir]
    filenames = OrderedDict()
    if verbose:
        print("Parsing directory {}".format(source_dir))
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir) is False:
            continue
        # Get the directory name
        sub_name = os.path.basename(sub_dir)
        # Skip directories starting with ~
        if sub_name.startswith(skip) or \
            sub_name.startswith('_') or \
            sub_name.startswith('.'):
            continue
        # Get the files
        if verbose:
            print("- {}".format(sub_name), end='')
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
            sub_files1 = sorted(glob.glob(os.path.join(sub_dir, *sub_directories, ext)))
            files.extend(sub_files1)
            if platform.system() == 'Linux':
                sub_files2 = sorted(glob.glob(os.path.join(sub_dir, *sub_directories, ext.upper())))
                files.extend(sub_files2)
        # Add to dictionary
        if verbose:
            print(" ({} files)".format(len(files)))
        if has_classes:
            filenames[sub_name] = files
        else:
            filenames['null'] = files
    return filenames


def flatten_to_list(filenames_dict: dict):
    filenames = []
    for value in filenames_dict.values():
        filenames.extend(value)
    return filenames


def parse_csv(csv_file, source_dir, file_idx=0, cls_idx=1, cls_label_idx=2, verbose=True):
    if verbose:
        print("Parsing csv file {} for directory {}".format(csv_file, source_dir))
    df = pd.read_csv(csv_file)
    num_classes = np.max(df.iloc[:, cls_idx]) + 1
    cls_labels = [df.loc[df.iloc[:, cls_idx] == i].iloc[0, cls_label_idx] for i in range(num_classes)]
    filenames = OrderedDict()
    for i in range(num_classes):
        if verbose:
            print("- {}".format(cls_labels[i]), end='')
        names = df.loc[df.iloc[:, cls_idx] == i].iloc[:, file_idx]
        paths = [os.path.join(source_dir, n) for n in names]
        filenames[cls_labels[i]] = paths
        if verbose:
            print(" ({} files)".format(len(paths)))
    return filenames


if __name__ == "__main__":
    from mml.data.filenames_dataset import FilenamesDataset
    import numpy as np

    result = parse_csv(r"D:\Datasets\Weeds\labels.csv",
                       r"D:\Datasets\Weeds\DeepWeeds")

    # fs = FilenamesDataset(r"C:\data\SeagrassFrames")
    # fs.split(0.2)
    # arr = np.zeros(len(fs.train_filenames))
    # load_images(fs.train_filenames, arr, None)