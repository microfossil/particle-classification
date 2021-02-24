from typing import NamedTuple

import numpy as np
from glob import glob
import os
from collections import OrderedDict
import platform
import pandas as pd
import xml.etree.ElementTree as ET

from miso.data.download import download_images

from pathlib import Path



def ls(source_dir):
    return sorted(glob(os.path.join(source_dir, "*")))


def parse_directory(source_dir, skip='~', has_classes=True):
    """
    Parses a directory consisting of subdirectories named by class and stores the image filenames in a dictionary
    :param source_dir: The directory to parse
    :param verbose: Print out the progress
    :param skip: If a subdirectory starts with this character it will be skipped ('_' and '.' are skipped always)
    :param has_classes: The files are arranged in subdirectories by class name
    :return: Dictionary with class names for the keys and lists of filenames for the values
    """
    if has_classes:
        sub_dirs = sorted(glob(os.path.join(source_dir, "*")))
    else:
        sub_dirs = [source_dir]
    filenames = OrderedDict()
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
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
            sub_files1 = sorted(glob(os.path.join(sub_dir, ext)))
            files.extend(sub_files1)
            if platform.system() == 'Linux':
                sub_files2 = sorted(glob(os.path.join(sub_dir, ext.upper())))
                files.extend(sub_files2)
        # Add to dictionary
        if has_classes:
            filenames[sub_name] = files
        else:
            filenames['null'] = files
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


def parse_xml(xml_filename):
    project = ET.parse(xml_filename).getroot()

    filenames = []
    cls = []
    cls_labels = []

    filenames_dict = OrderedDict()

    images_xml = project.find('images')
    for i, image_xml in enumerate(images_xml.iter('image')):
        relfile = image_xml.find('source').find('filename').text
        if os.path.isabs(relfile):
            absfile = relfile
        else:
            absfile = os.path.abspath(os.path.join(os.path.dirname(xml_filename), relfile))
        if os.path.isfile(absfile) is False:
            continue
        filenames.append(absfile)

        cls_names = []
        cls_scores = []
        cls_base = image_xml.find('classifications')
        for cls_val in cls_base.iter('classification'):
            cls_names.append(cls_val.find('code').text)
            cls_scores.append(float(cls_val.find('value').text))
        cls.append(cls_names[np.argmax(cls_scores)])

    for taxon_xml in project.find('taxons').iter('taxon'):
        if taxon_xml.find('isClass').text == 'true':
            cls_labels.append(taxon_xml.find('code').text)
    cls_labels = sorted(cls_labels)

    df = pd.DataFrame({'filenames': filenames, 'cls': cls})
    for label in cls_labels:
        filenames_dict[label] = df.filenames[df.cls == label]
    return filenames_dict


class FilenamesDataset(object):
    def __init__(self,
                 source: str,
                 csv_idxs=(0, 1, 2),
                 has_classes=True):
        self.source = source
        self.csv_idxs = csv_idxs
        self.has_classes = has_classes
        self.cls_filenames = OrderedDict()
        self.filenames = None
        self.cls_labels = None
        self.cls = None
        self.cls_counts = None
        self.num_classes = None

    def load(self, min_count=0, map_others=False):
        print('-' * 80)
        print("Parsing source {}".format(self.source))
        print()

        # Load filenames
        if self.source.endswith(".xml"):
            self.cls_filenames = parse_xml(self.source)
        elif self.source.endswith(".csv"):
            self.cls_filenames = parse_csv(self.source, self.source, self.csv_idxs[0], self.csv_idxs[1], self.csv_idxs[2])
        elif self.source.startswith("http"):
            dir_for_download = os.path.join(str(Path.home()), 'miso_datasets')
            os.makedirs(dir_for_download, exist_ok=True)
            dir_path = download_images(self.source, dir_for_download)
            self.cls_filenames = parse_directory(dir_path, has_classes=self.has_classes)
        else:
            self.cls_filenames = parse_directory(self.source, has_classes=self.has_classes)

        # Check we found some images
        filenames = [v for key, val in self.cls_filenames.items() for v in val]
        if len(filenames) == 0:
            raise ValueError("! Did not find any images")

        # Display how many found
        print("Classes:")
        for k, v in self.cls_filenames.items():
            print("- {}: {} images".format(k, len(v)))
        print()

        # Apply min count
        if min_count > 0:
            print("Removing classes with less than {} images:".format(min_count))
            others = []
            filtered_cls_filenames = OrderedDict()
            was_min = False
            for k, v in self.cls_filenames.items():
                if len(v) < min_count:
                    others.extend(v)
                    print("- {}: {} images".format(k, len(v)))
                    was_min = True
                else:
                    filtered_cls_filenames[k] = v
            if was_min is False:
                print("- none found")
            # Put in others
            if map_others is True and len(others) > min_count:
                if 'others' in filtered_cls_filenames:
                    filtered_cls_filenames['others'].extend(others)
                else:
                    filtered_cls_filenames['others'] = others
                print("- these images have been places in the 'others' class, {} total".format(len(others)))
            self.cls_filenames = filtered_cls_filenames
            print()

        # Flatten filename and class dictionary to a list
        self.filenames = [v for key, val in self.cls_filenames.items() for v in val]
        self.cls_labels = list(self.cls_filenames.keys())
        self.cls = np.asarray([self.cls_labels.index(key) for key, val in self.cls_filenames.items() for v in val])
        self.cls_counts = np.asarray([len(val) for key, val in self.cls_filenames.items()])
        self.num_classes = len(self.cls_filenames.keys())

