import glob
import os
import hashlib
import gc
import dill
import numpy as np
import pandas as pd
import bz2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import OrderedDict
from miso.data.generators import image_generator_from_dataframe
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

        self.images_mmap_filename = None
        self.mmap_directory = None

    def get_class_weights(self):
        count = np.bincount(self.data_df['cls'])
        weights = gmean(count) / count
        return weights

    def get_short_filenames(self):
        return pd.concat((self.train_df.short_filenames, self.test_df.short_filenames))

    # def get_dataframe_hash(self, img_size, color_mode):
    #     hash_str = hashlib.sha256(pd.util.hash_pandas_object(self.data_df, index=True).values).hexdigest()
    #     return "{}_{}_{}_{}".format(hash_str, img_size[0], img_size[1], color_mode)

    def load_images(self,
                    img_size,
                    prepro_type=None,
                    prepro_params=(255, 0, 1),
                    color_mode='rgb',
                    print_status=False,
                    dtype=np.float16):

        # hashed_filename = os.path.join(self.source_directory, self.get_dataframe_hash(img_size, color_mode) + ".pkl")
        # try:
        #     with open(hashed_filename, 'rb') as file:
        #         images = dill.load(file)
        #         self.split(images, split, seed)
        #         print("@Found existing processed images at " + hashed_filename)
        #         return
        # except:
        #     pass
        filenames = self.data_df['filenames']
        image_count = len(filenames)
        if color_mode == 'rgb' or color_mode == 'greyscale3':
            channels = 3
        else:
            channels = 1

        if dtype is np.float16:
            byte_count = 2
        elif dtype is np.float32:
            byte_count = 4
        elif dtype is np.float64:
            byte_count = 8
        else:
            byte_count = 'X'

        # Memory map
        hashstr = hashlib.sha256(pd.util.hash_pandas_object(self.data_df, index=True).values).hexdigest()[0:16]
        unique_id = "{}_{}_{}_{}_{}.npy".format(hashstr, img_size[0], img_size[1], color_mode, byte_count)
        self.images_mmap_filename = os.path.join(self.mmap_directory, unique_id)
        print(self.images_mmap_filename)

        if os.path.exists(self.images_mmap_filename):
            self.images = open_memmap(self.images_mmap_filename, dtype=dtype, mode='r+', shape=(image_count, img_size[0], img_size[1], channels))
            return

        self.images = open_memmap(self.images_mmap_filename, dtype=dtype, mode='w+',  shape=(image_count, img_size[0], img_size[1], channels))
        idx = 0
        # Load each image
        print("@Loading images...")
        print()
        for filename in filenames:
            if color_mode == 'rgb':
                im = Image.open(filename)
            else:
                im = Image.open(filename).convert('L')
            im = np.asarray(im, dtype=np.float)
            if color_mode == 'greyscale3':
                im = np.expand_dims(im, -1)
                im = np.repeat(im, repeats=3, axis=-1)
            if im.ndim == 2:
                im = np.expand_dims(im, -1)
                if color_mode == 'rgb':
                    im = np.repeat(im, repeats=3, axis=-1)

            # height = im.shape[0]
            # width = im.shape[1]
            # new_height = img_size[0]
            # new_width = img_size[1]
            # if height > width:
            #     new_width = int(np.round(width * new_height / height))
            # elif width > height:
            #     new_height = int(np.round(height * new_width / width))
            im = self.make_image_square(im)
            # im = resize(im, (new_height, new_width), order=1)
            im = resize(im, (img_size[0], img_size[1]), order=1)

            # im = np.ones((224,224,3))

            # Rescale according to params
            # e.g. to take a [0,255] range image to [-1,1] params would be 255,-0.5,2
            # I' = (I / 255 - 0.5) * 2
            #TODO divide according to image depth (8 or 16 bits etc)
            im = np.divide(im, prepro_params[0])
            im = np.subtract(im, prepro_params[1])
            im = np.multiply(im, prepro_params[2])
            # Convert to format
            im = im.astype(dtype)
            if im.ndim == 2:
                im = im[:, :, np.newaxis]
            self.images[idx] = im
            idx += 1
            if idx % 100 == 0:
                if print_status:
                    print("@Loading images {}%".format((int)(idx / len(filenames) * 100)))
                else:
                    print("\r{} of {} processed".format(idx, len(filenames)), end='')
        # Convert all to a numpy array
        # self.images = np.asarray(self.images)
        # if self.images.ndim == 3:
        #     images = self.images[:, :, :, np.newaxis]
        # Split into test and training sets
        #self.split(images, split, split_index, seed)
        self.images.flush()

    # def load_images_using_datagen(self, img_size, datagen, color_mode='rgb', split=0.25, split_offset=0, seed=None):
    #     """
    #     Loads images from disk using an ImageDataGenerator. Images are resize and the colour changed if necessary.
    #     :param img_size: Images will be transformed to this size
    #     :param datagen: Keras ImageDataGenerator to used. If None, default generator using rescale=1./255 will be used
    #     :param color_mode: Convert image to 'rgb' (3 channel) or 'grayscale' (1 channel)
    #     :param test_size: If not None, will call train_test_split(test_size) before loading
    #     :return: tuple of training images, training classes, test images and test classes
    #     """
    #     # Data generator
    #     gen = image_generator_from_dataframe(self.data_df,
    #                                          img_size,
    #                                          10000000,
    #                                          self.cls_labels,
    #                                          datagen,
    #                                          color_mode)
    #     gen.shuffle = False
    #     self.images, _ = next(gen)
    #     self.split(split, split_offset, seed)

    def delete_memmap_files(self, del_split=True, del_source=True):
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

    def split(self, split=0.25, split_offset=0, seed=None, dtype=np.float16):
        # Create new random index if necessary
        if self.random_idx_init is None:
            np.random.seed(seed)
            self.random_idx_init = np.random.permutation(len(self.images))
        # Roll according to offset
        roll = int(np.round(len(self.images) * split_offset))
        self.random_idx = np.roll(self.random_idx_init, roll)
        # Now split
        test_len = np.round(len(self.images) * split)
        test_idx = self.random_idx[0:int(test_len)]
        train_idx = self.random_idx[int(test_len):]
        # Memmap
        print("Split mapping")
        img_size = self.images.shape[1:]
        train_filename = os.path.join(self.mmap_directory, "train.npy")
        test_filename = os.path.join(self.mmap_directory, "test.npy")
        self.delete_memmap_files(True, False)
        self.train_images = open_memmap(train_filename, dtype=dtype, mode='w+', shape=(len(train_idx), img_size[0], img_size[1], img_size[2]))
        self.test_images = open_memmap(test_filename, dtype=dtype, mode='w+', shape=(len(test_idx), img_size[0], img_size[1], img_size[2]))
        for i in range(len(train_idx)):
            self.train_images[i] = self.images[train_idx[i]]
        for i in range(len(test_idx)):
            self.test_images[i] = self.images[test_idx[i]]
        # self.test_images = self.images[test_idx]
        self.train_cls = self.cls[train_idx]
        self.test_cls = self.cls[test_idx]
        self.train_onehots = self.onehots[train_idx]
        self.test_onehots = self.onehots[test_idx]
        self.train_df = self.data_df.iloc[train_idx,:]
        self.test_df = self.data_df.iloc[test_idx,:]

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

        The labels for the class are taken as the sub-directory names

        :param directory: Path to the directory containing sub-directories of classes
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
            print("@Downloading dataset " + source + "...")
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
            print("@Parsing project file " + self.source_name)
            filenames = self.parse_xml(self.source_name)
            if mmap_directory is None:
                self.mmap_directory = str(Path(self.source_name).parent)
        else:
            print("@Parsing image directory...")
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
            print("@Applying mapping...")
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
        print("@Moving classes with not enough images to 'other'...")
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
        df = {"filenames": long_filenames, "short_filenames": short_filenames, "labels": cls_labels, "cls": cls_index}

        self.data_df = pd.DataFrame(df)
        self.num_classes = len(self.cls_labels)
        print("@{} images in {} classes".format(len(cls_index), self.num_classes))

        for idx in range(self.num_classes):
            cls_counts.append(len(self.data_df[self.data_df['cls'] == idx]))
            # print(cls_counts)
        self.cls_counts = cls_counts
        self.cls = self.data_df['cls'].to_numpy()
        self.onehots = to_categorical(self.data_df['cls'])

        # print(self.data_df)

    def make_image_square(self, im):
        if im.shape[0] == im.shape[1]:
            return im
        #print("not_square {} {}".format(im.shape[0],im.shape[1]))
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
             for c in range(im.shape[2])],
            axis=2)
        return im

    def parse_xml(self, xml_filename):
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
            # print(label)
            # print(len(filenames))
            # print(len(cls))
            # print(np.sum(cls == label))
            filenames_dict[label] = df.filenames[df.cls == label]

        return filenames_dict


if __name__ == "__main__":
    ds = DataSource()
    ds.set_source(r"C:\Users\rossm\Documents\Data\Foraminifera\BenthicPlanktic\Benthic_Planktic_Source_v2\project.xml", 40)
