import glob
import os
import hashlib
import gc
import numpy as np
import pandas as pd
from PIL import Image
import skimage.color as skcolor
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import OrderedDict
from scipy.stats.mstats import gmean
from miso.data.download import download_images
from numpy.lib.format import open_memmap
import lxml.etree as ET
from pathlib import Path


class DataSource:

    def __init__(self):
        self.num_classes = 0
        self.cls_labels = []
        self.cls_counts = []
        self.source_name = ""

        self.data_df = None
        self.train_df = None
        self.test_df = None

        self.random_idx = None
        self.random_idx_init = None

        self.images = None
        self.cls = None
        self.onehots = None
        self.vectors = None

        self.train_images = None
        self.train_cls = None
        self.train_onehots = None
        self.train_vectors = None

        self.test_images = None
        self.test_cls = None
        self.test_onehots = None
        self.test_vectors = None

        self.use_mmap = False
        self.images_mmap_filename = None
        self.mmap_directory = None

    def get_class_weights(self):
        count = np.bincount(self.data_df['cls']).astype(np.float64)
        weights = gmean(count) / count
        weights[weights < 0.1] = 0.1
        weights[weights > 10] = 10
        return weights

    def get_short_filenames(self):
        return pd.concat((self.train_df.short_filenames, self.test_df.short_filenames))

    @staticmethod
    def preprocess_image(im, prepro_method='rescale', prepro_params=(255, 0, 1)):
        # TODO divide according to image depth (8 or 16 bits etc)
        if prepro_method == 'rescale':
            im = np.divide(im, prepro_params[0])
            im = np.subtract(im, prepro_params[1])
            im = np.multiply(im, prepro_params[2])
        return im

    @staticmethod
    def load_image(filename, img_size, img_type):
        if img_type == 'rgb':
            im = Image.open(filename)
            im = np.asarray(im, dtype=np.float)
            if im.ndim == 2:
                im = np.expand_dims(im, -1)
                im = np.repeat(im, repeats=3, axis=-1)
        elif img_type == 'greyscale':
            im = Image.open(filename).convert('L')
            im = np.asarray(im, dtype=np.float)
            im = np.expand_dims(im, -1)
        elif img_type == 'greyscale3':
            im = Image.open(filename).convert('L')
            im = np.asarray(im, dtype=np.float)
            im = np.expand_dims(im, -1)
            im = np.repeat(im, repeats=3, axis=-1)
        elif img_type == 'greyscaled':
            ims = DataSource.read_tiff(filename, [0, 2])
            g = skcolor.rgb2grey(ims[0]) * 255      # Scales to 0 - 1 for some reason
            if ims[1].ndim == 3:
                d = skcolor.rgb2grey(ims[1])
            else:
                d = ims[1].astype(float)
            im = np.concatenate((g[:, :, np.newaxis], d[:, :, np.newaxis]), 2)
        elif img_type == 'greyscaledm':
            ims = DataSource.read_tiff(filename, [0, 2, 4])
            g = skcolor.rgb2grey(ims[0]) * 255  # Scales to 0 - 1 for some reason
            if ims[1].ndim == 3:
                d = skcolor.rgb2grey(ims[1])
            else:
                d = ims[1].astype(float)
            if ims[2].ndim == 3:
                m = skcolor.rgb2grey(ims[2])
            else:
                m = ims[2].astype(float)
            im = np.concatenate((g[:, :, np.newaxis], d[:, :, np.newaxis], m[:, :, np.newaxis]), 2)
        elif img_type == 'rgbd':
            ims = DataSource.read_tiff(filename, [0, 2])
            rgb = ims[0]
            # print("rgbd {}".format(filename))
            # print(rgb.shape)
            # print(ims[1].shape)
            if rgb.ndim == 2:
                rgb = np.expand_dims(rgb, -1)
                rgb = np.repeat(rgb, repeats=3, axis=-1)
            d = skcolor.rgb2grey(ims[1])
            im = np.concatenate((rgb, d[:,:, np.newaxis]), 2)
        # print(im.shape)
        im = DataSource.make_image_square(im)
        # print(im.shape)
        im = resize(im, (img_size[0], img_size[1]), order=1)
        # print(im.shape)
        return im

    @staticmethod
    def read_tiff(filename, indices):
        img = Image.open(filename)
        images = []
        # print("num frames: {}".format(img.n_frames))
        for i in range(img.n_frames):
            img.seek(i)
            # print("- frame {} shape {}".format(i, np.array(img).shape))
            # print(img)
        for i, idx in enumerate(indices):
            img.seek(idx)
            if len(np.array(img).shape) == 0:
                #print("Bad")
                img.mode = 'L'
            images.append(np.array(img))
        return images

    def load_dataset(self,
                     img_size,
                     img_type='rgb',
                     dtype=np.float16):

        # Image filenames
        filenames = self.data_df['filenames']
        image_count = len(filenames)

        # Color mode:
        # - rgb: normal RGB
        # - greyscale: convert to greyscale (single channel) if necessary
        # - greyscale3: convert to greyscale then repeat across 3 channels
        #               (for inputting greyscale images into networks that take three channels)
        if img_type == 'rgb' or img_type == 'greyscale3':
            channels = 3
        elif img_type == 'greyscale':
            channels = 1
        elif img_type == 'rgbd':
            channels = 4
        elif img_type == 'greyscaled':
            channels = 2
        elif img_type == 'greyscaledm':
            channels = 3
        else:
            raise ValueError("Unknown image type")

        # float16 is used be default to save memory
        if dtype is np.float16:
            byte_count = 2
        elif dtype is np.float32:
            byte_count = 4
        elif dtype is np.float64:
            byte_count = 8
        else:
            byte_count = 'X'

        # Sometimes the data set is too big to be saved into memory.
        # In this case, we can memory map the numpy array onto disk.
        # Make sure to delete the files afterwards
        if self.use_mmap:
            # Unique hash id is used for the filename
            hashstr = hashlib.sha256(pd.util.hash_pandas_object(self.data_df, index=True).values).hexdigest()[0:16]
            unique_id = "{}_{}_{}_{}_{}.npy".format(hashstr, img_size[0], img_size[1], img_type, byte_count)
            self.images_mmap_filename = os.path.join(self.mmap_directory, unique_id)
            print(self.images_mmap_filename)
            # If the memmap file already exists, simply load it
            if os.path.exists(self.images_mmap_filename):
                self.images = open_memmap(self.images_mmap_filename, dtype=dtype, mode='r+', shape=(image_count, img_size[0], img_size[1], channels))
                return
            self.images = open_memmap(self.images_mmap_filename, dtype=dtype, mode='w+',  shape=(image_count, img_size[0], img_size[1], channels))
        else:
            self.images = np.zeros(shape=(image_count, img_size[0], img_size[1], channels), dtype=dtype)

        # Load each image
        idx = 0
        print("@ Loading images... ")
        for filename in filenames:
            try:
                im = self.load_image(filename, img_size, img_type)
                im = self.preprocess_image(im)
                # Convert to format
                im = im.astype(dtype)
                if im.ndim == 2:
                    im = im[:, :, np.newaxis]
                self.images[idx] = im
            except:
                print("@ Error loading image {}".format(filename))
            idx += 1
            if idx % 100 == 0:
                print("\r@ Loading images {}%".format((int)(idx / len(filenames) * 100)))
        if self.use_mmap:
            self.images.flush()

    def delete_memmap_files(self, del_split=True, del_source=True):
        if self.use_mmap is False:
            return
        if self.mmap_directory is None:
            return
        if del_split:
            train_filename = os.path.join(self.mmap_directory, "train.npy")
            test_filename = os.path.join(self.mmap_directory, "test.npy")
            if os.path.exists(train_filename):
                if self.train_images is not None:
                    self.train_images._mmap.close()
                del self.train_images
                gc.collect()
                os.remove(train_filename)
            if os.path.exists(test_filename):
                if self.test_images is not None:
                    self.test_images._mmap.close()
                del self.test_images
                gc.collect()
                os.remove(test_filename)
        if del_source:
            if os.path.exists(self.images_mmap_filename):
                if self.images is not None:
                    self.images._mmap.close()
                del self.images
                gc.collect()
                os.remove(self.images_mmap_filename)

    def split(self, split=0.20, seed=None):
        dtype=self.images.dtype
        if split > 0.0:
            # Split with stratify
            train_idx, test_idx = train_test_split(range(len(self.images)), test_size=split, random_state=seed, shuffle=True, stratify=self.cls)
            self.random_idx = train_idx + test_idx
        else:
            train_idx = np.random.permutation(range(len(self.images)))
            test_idx = []
            self.random_idx = train_idx
        print("@ Split mapping...")
        img_size = self.images.shape[1:]
        # Memmap splitting
        if self.use_mmap:
            print("@ Split mapping - deleting old memmap files")
            train_filename = os.path.join(self.mmap_directory, "train.npy")
            test_filename = os.path.join(self.mmap_directory, "test.npy")
            self.delete_memmap_files(del_split=True, del_source=False)
            print("@ Split mapping - creating new memmap files")
            self.train_images = open_memmap(train_filename, dtype=dtype, mode='w+', shape=(len(train_idx), ) + img_size)
            self.test_images = open_memmap(test_filename, dtype=dtype, mode='w+', shape=(len(test_idx), ) + img_size)
            print("@ Split mapping - copying train images")
            for i in range(len(train_idx)):
                self.train_images[i] = self.images[train_idx[i]]
            print("@ Split mapping - copying test images")
            for i in range(len(test_idx)):
                self.test_images[i] = self.images[test_idx[i]]
        # Normal splitting
        else:
            self.train_images = self.images[train_idx]
            self.test_images = self.images[test_idx]
        # Remainder
        self.train_cls = self.cls[train_idx]
        self.test_cls = self.cls[test_idx]
        self.train_onehots = self.onehots[train_idx]
        self.test_onehots = self.onehots[test_idx]
        self.train_df = self.data_df.iloc[train_idx,:]
        self.test_df = self.data_df.iloc[test_idx,:]
        print("@ Split mapping - done")

    def set_source(self,
                   source,
                   min_count,
                   max_count=None,
                   min_count_to_others=False,
                   extension=None,
                   mapping: dict = None,
                   map_others=True,
                   must_contain: str = None,
                   ignore_list: list = None,
                   mmap_directory = None):
        """
        Loads images from from a directory where each sub-directories contains images for a single class, e.g.:

        directory
          |-- class 1 directory
          |-- class 2 directory
          |-- class 3 directory
          `-- ...

        The cls for the class are taken as the sub-directory names

        :param source: Path to the directory containing sub-directories of classes
        :param extension: Extension of the images in directory (e.g. "jpg"). If `None`, it looks for jpg, png and tiff
        :param min_count: Minimum number of images in a sub-directory for that class to be included
        :param max_count: Maximum number of images in a sub-directory to be used (If `None` all images are used)
        :param mapping: Dictionary mapping classes to final classes. E.g. {"cat": "animal", "dog",:"animal"} maps "cat" and "dog" both to animal.
        If mapping is `None`, the original classes are used. If mapping is not `None` then only the classes in the map are used.
        :param map_others: If `True` then classes not in the mapping will be mapped to an "Others" class
        :param must_contain: The image filenames must contain this string
        :param ignore_list: List of classes that will be ignored, and their images not loaded
        :return:
        """
        self.mmap_directory = mmap_directory
        if source.startswith("http"):
            print("@ Downloading dataset " + source + "...")
            dir_for_download = os.path.join(os.getcwd(), 'datasets')
            os.makedirs(dir_for_download, exist_ok=True)
            dir_path = download_images(source, dir_for_download)
            self.source_name = dir_path
            if mmap_directory is None:
                self.mmap_directory = dir_for_download
        else:
            self.source_name = source
            if mmap_directory is None:
                self.mmap_directory = str(Path(self.source_name).parent)

        if self.source_name.endswith("xml"):
            print("@ Parsing project file " + self.source_name)
            filenames = self.parse_xml(self.source_name)
            if mmap_directory is None:
                self.mmap_directory = str(Path(self.source_name).parent)
        else:
            print("@ Parsing image directory...")
            # Get alphabetically sorted list of class directories
            class_dirs = sorted(glob.glob(os.path.join(self.source_name, "*")))

            # Load images from each class and place into a dictionary
            filenames = OrderedDict()
            for class_dir in class_dirs:
                if os.path.isdir(class_dir) is False:
                    continue

                # Get the class name
                class_name = os.path.basename(class_dir)

                # Skip directories starting with ~
                if class_name.startswith('~'):
                    continue

                # Get the files
                files = []
                if extension is None:
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
                        if must_contain is not None:
                            files.extend(sorted(glob.glob(os.path.join(class_dir, "*" + must_contain + ext))))
                        else:
                            files.extend(sorted(glob.glob(os.path.join(class_dir, ext))))
                else:
                    if must_contain is not None:
                        files = sorted(glob.glob(os.path.join(class_dir, "*" + must_contain + "*." + extension)))
                    else:
                        files = sorted(glob.glob(os.path.join(class_dir, "*." + extension)))

                # Add to dictionary
                filenames[class_name] = files

        # Map the classes into overall classes if mapping is enabled
        if mapping is not None:
            print("@ Applying mapping...")
            mapped_filenames = OrderedDict()
            # Sort the map
            sorted_map = OrderedDict()
            for key in sorted(mapping.keys()):
                sorted_map[key] = mapping[key]

            # Keep a record of which classes have already been mapped
            already_mapped_cls = list()

            # Iterate through the map
            for cls, sub_cls_list in sorted_map.items():
                print(" - {} <= ".format(cls), end='')
                # Create list entry for this class
                mapped_filenames[cls] = list()
                # Add all the component classes
                for sub_cls in sub_cls_list:
                    if sub_cls in filenames and sub_cls not in already_mapped_cls:
                        print("{} ".format(sub_cls), end='')
                        mapped_filenames[cls].extend(filenames[sub_cls])
                        already_mapped_cls.append(sub_cls)
                # Number of classes
                print("({} images)".format(len(mapped_filenames[cls])))

            # Add others
            if map_others is True:
                mapped_filenames['other'] = list()
                # Iterate though the filenames dictionary and add any classes not already mapped to others
                for cls in filenames.keys():
                    if cls not in already_mapped_cls:
                        mapped_filenames['other'].extend(filenames[cls])

            # Save the mapped filenames as the current filenames dictionary
            filenames = mapped_filenames

        # Remove any classes that do not have enough images and put in 'other'
        print("@ Moving classes with not enough images to 'other'...")
        not_enough_list = list()
        enough_dict = OrderedDict()
        for cls, cls_filenames in filenames.items():
            print(" - ({:5d} images) {}".format(len(cls_filenames), cls), end='')
            if len(cls_filenames) < min_count:
                not_enough_list.extend(cls_filenames)
                print(" => other".format(len(cls_filenames)))
            else:
                enough_dict[cls] = cls_filenames
                print()

        # Put the others in the list if there is also enough for them
        if min_count_to_others is True:
            if len(not_enough_list) > min_count:
                if 'other' in enough_dict:
                    enough_dict['other'].extend(not_enough_list)
                else:
                    enough_dict['other'] = not_enough_list
                print(" - {} images in other".format(len(not_enough_list)))
            else:
                print(" - other not included ({} images)".format(len(not_enough_list)))

        filenames = enough_dict
        # print(enough_dict.keys())

        # Finally, create a list for each (make sure 'other' is last class
        cls_index = []
        cls_labels = []
        cls_counts = []
        long_filenames = []
        short_filenames = []
        self.cls_labels = []

        if 'other' in filenames:
            other_index = len(filenames) - 1
        else:
            other_index = len(filenames)

        index = 0
        for cls, cls_filenames in filenames.items():
            for filename in cls_filenames:
                if cls != 'other':
                    cls_index.append(index)
                else:
                    cls_index.append(other_index)
                cls_labels.append(cls)
                long_filenames.append(filename)
                short_filenames.append(os.path.basename(filename))

            if cls != 'other':
                self.cls_labels.append(cls)
                index += 1

        if 'other' in filenames:
            self.cls_labels.append('other')
        df = {"filenames": long_filenames, "short_filenames": short_filenames, "cls": cls_labels, "cls": cls_index}

        self.data_df = pd.DataFrame(df)
        self.num_classes = len(self.cls_labels)
        print("@ {} images in {} classes".format(len(cls_index), self.num_classes))

        for idx in range(self.num_classes):
            cls_counts.append(len(self.data_df[self.data_df['cls'] == idx]))
            # print(cls_counts)
        self.cls_counts = cls_counts
        self.cls = self.data_df['cls'].to_numpy()
        self.onehots = to_categorical(self.data_df['cls'])

        # print(self.data_df)

    @staticmethod
    def make_image_square(im):
        if im.shape[0] == im.shape[1]:
            return im
        height = im.shape[0]
        width = im.shape[1]
        half = max(height, width)
        height_pad_start = int(abs(np.floor((height - half) / 2)))
        height_pad_end = int(abs(np.ceil((height - half) / 2)))
        width_pad_start = int(abs(np.floor((width - half) / 2)))
        width_pad_end = int(abs(np.ceil((width - half) / 2)))
        consts = [np.median(np.concatenate((im[0, :, i], im[-1, :, i], im[:, 0, i], im[:, -1, i]))) for i in
                  range(im.shape[2])]
        im = np.stack(
            [np.pad(im[:, :, c],
                    ((height_pad_start, height_pad_end), (width_pad_start, width_pad_end)),
                    mode='constant',
                    constant_values=consts[c])
             for c in range(im.shape[2])], axis=2)
        return im

    @staticmethod
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

    @staticmethod
    def parse_directory(source_dir):
        sub_dirs = sorted(glob.glob(os.path.join(source_dir, "*")))

        filenames = OrderedDict()
        for sub_dir in sub_dirs:
            if os.path.isdir(sub_dir) is False:
                continue
            # Get the directory name
            sub_name = os.path.basename(sub_dir)
            # Skip directories starting with ~
            if sub_name.startswith('~'):
                continue
            # Get the files
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
                files.extend(sorted(glob.glob(os.path.join(sub_dir, ext))))
            # Add to dictionary
            filenames[sub_name] = files
        return filenames


if __name__ == "__main__":
    ds = DataSource()
    ds.set_source(r"C:\Users\rossm\Documents\Data\Foraminifera\BenthicPlanktic\Benthic_Planktic_Source_v2\project.xml", 40)
