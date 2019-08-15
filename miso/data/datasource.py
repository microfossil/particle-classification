import glob
import os
import hashlib
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


class DataSource:

    def __init__(self):
        self.num_classes = 0
        self.cls_labels = []
        self.source_directory = ""

        self.data_df = None
        self.train_df = None
        self.test_df = None

        self.train_images = None
        self.train_cls = None
        self.train_onehots = None
        self.train_vectors = None
        self.test_images = None
        self.test_cls = None
        self.test_onehots = None
        self.test_vectors = None

    def get_images(self):
        return np.concatenate((self.train_images, self.test_images), axis=0)

    def get_classes(self):
        return np.concatenate((self.train_cls, self.test_cls), axis=0)

    def get_class_weights(self):
        count = np.bincount(self.data_df['cls'])
        weights = gmean(count) / count
        return weights

    def get_short_filenames(self):
        return pd.concat((self.train_df.short_filenames, self.test_df.short_filenames))

    def get_dataframe_hash(self, img_size, color_mode):
        hash_str = hashlib.sha256(pd.util.hash_pandas_object(self.data_df, index=True).values).hexdigest()
        return "{}_{}_{}_{}".format(hash_str, img_size[0], img_size[1], color_mode)

    def load_images(self,
                    img_size,
                    prepro_type=None,
                    prepro_params=(255, 0, 1),
                    color_mode='rgb',
                    split=0.25,
                    print_status=False,
                    seed=None):

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
        images = []
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
            if im.ndim == 2:
                im = np.expand_dims(im, -1)
                if color_mode == 'rgb':
                    im = np.repeat(im, repeats=3, axis=-1)
            im = self.make_image_square(im)
            im = resize(im, img_size, order=1)
            # Rescale according to params
            # e.g. to take a [0,255] range image to [-1,1] params would be 255,-0.5,2
            # I' = (I / 255 - 0.5) * 2
            im = np.divide(im, prepro_params[0])
            im = np.subtract(im, prepro_params[1])
            im = np.multiply(im, prepro_params[2])
            # Convert to float
            im = im.astype(np.float32)
            images.append(im)
            idx += 1
            if idx % 100 == 0:
                if print_status:
                    print("@Loading images {}%".format((int)(idx / len(filenames) * 100)))
                else:
                    print("\r{} of {} processed".format(idx, len(filenames)), end='')
        # Convert all to a numpy array
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[:, :, :, np.newaxis]
        # Split into test and training sets
        self.split(images, split, seed)
        # with bz2.BZ2File(hashed_filename, 'w') as file:
        # with open(hashed_filename, 'wb') as file:
        #     print("@Saving processed images to " + hashed_filename)
        #     dill.dump(images, file)

    def load_images_using_datagen(self, img_size, datagen, color_mode='rgb', split=0.25, seed=None):
        """
        Loads images from disk using an ImageDataGenerator. Images are resize and the colour changed if necessary.
        :param img_size: Images will be transformed to this size
        :param datagen: Keras ImageDataGenerator to used. If None, default generator using rescale=1./255 will be used
        :param color_mode: Convert image to 'rgb' (3 channel) or 'grayscale' (1 channel)
        :param test_size: If not None, will call train_test_split(test_size) before loading
        :return: tuple of training images, training classes, test images and test classes
        """
        # Data generator
        gen = image_generator_from_dataframe(self.data_df,
                                             img_size,
                                             10000000,
                                             self.cls_labels,
                                             datagen,
                                             color_mode)
        gen.shuffle = False
        images, _ = next(gen)
        self.split(images, split, seed)

    def split(self, images, split=0.25, seed=None):
        self.train_images, self.test_images, \
        self.train_cls, self.test_cls, \
        self.train_onehots, self.test_onehots, \
        self.train_df, self.test_df = \
            train_test_split(images,
                             self.data_df['cls'],
                             to_categorical(self.data_df['cls']),
                             self.data_df,
                             test_size=split,
                             random_state=seed)

    def set_directory_source(self,
                             source,
                             min_count,
                             max_count=None,
                             min_count_to_others=False,
                             extension=None,
                             mapping: dict = None,
                             map_others=True,
                             must_contain: str = None,
                             ignore_list: list = None):
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
        if source.startswith("http"):
            print("@Downloading dataset " + source + "...")
            dir_for_download = os.path.join(os.getcwd(), 'datasets')
            os.makedirs(dir_for_download, exist_ok=True)
            dir_path = download_images(source, dir_for_download)
            self.source_directory = dir_path
        else:
            self.source_directory = source

        print("@Parsing image directory...")
        # Get alphabetically sorted list of class directories
        class_dirs = sorted(glob.glob(os.path.join(self.source_directory, "*")))
        # print(class_dirs)

        # Load images from each class and place into a dictionary
        filenames = OrderedDict()
        for class_dir in class_dirs:
            if os.path.isdir(class_dir) is False:
                continue

            # Get the class name
            class_name = os.path.basename(class_dir)
            # print(class_name)

            # print(glob.glob(os.path.join(class_dir, "*.png")))

            # Get the files
            files = []
            if extension is None:
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
                    if must_contain is not None:
                        files.extend(glob.glob(os.path.join(class_dir, "*" + must_contain + ext)))
                    else:
                        files.extend(glob.glob(os.path.join(class_dir, ext)))
                files = sorted(files)
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

        # print(self.data_df)

    def make_image_square(self, im):
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
             for c in range(im.shape[2])],
            axis=2)
        return im

    def set_xml_source(self, filename):
        pass

# def load_from_project_xml(xml_file, width):
#     project = ET.parse(xml_file).getroot()
#
#     images = []
#     cls = []
#     filenames = []
#
#     images_xml = project.find('images')
#     morphology_xml = []
#     for i, image_xml in enumerate(images_xml.iter('image')):
#         relfile = image_xml.find('source').find('filename').text
#         absfile = os.path.abspath(os.path.join(os.path.dirname(xml_file), relfile))
#         clsname = os.path.basename(os.path.dirname(absfile))
#
#         if os.path.exists(absfile):
#             im, sz = process_image(absfile, width)
#             images.append(im)
#             cls.append(clsname)
#             filenames.append(absfile)
#
#             values = dict()
#             for m in image_xml.find('morphology').findall('./'):
#                 values[m.tag] = m.text
#             morphology_xml.append(values)
#
#             if i % 100 == 0:
#                 print("{} images processed".format(i))
#
#     morphology_pd = pd.DataFrame.from_records(morphology_xml)
#     morphology = morphology_pd.values
#     morphology = normalize(morphology, norm='max', axis=0)
#
#     le = LabelEncoder()
#     le.fit(cls)
#     labels = le.classes_
#     cls = le.transform(cls)
#     images = np.asarray(images)
#     num_classes = len(labels)
#
#     return images, cls, labels, num_classes, filenames, morphology
