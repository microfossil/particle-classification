from typing import NamedTuple

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from scipy.stats.mstats import gmean

from miso.data.filenames_dataset import FilenamesDataset
from miso.data.image_dataset import ImageDataset
from miso.data.tf_generator import TFGenerator


class TrainingDataset(object):
    def __init__(self,
                 source,
                 img_size=(224, 224, 3),
                 img_type='rgb',
                 min_count=0,
                 map_others=False,
                 train_split=None,
                 test_split=0.2,
                 random_seed=0,
                 memmap_directory=None):
        if len(img_size) != 3:
            raise ValueError("img_size must be in format [height, width, num_channels]")
        self.source = source
        self.img_size = img_size
        self.img_type = img_type
        self.min_count = min_count
        self.map_others = map_others
        self.test_split = test_split
        self.train_split = train_split
        self.random_seed = random_seed
        self.memmap_directory = memmap_directory

        self.filenames: FilenamesDataset = None
        self.images: ImageDataset = None
        self.train_idx = None
        self.test_idx = None
        self.cls = None
        self.cls_onehot = None
        self.cls_labels = None
        self.num_classes = None
        self.class_weights = None

    def get_class_weights(self):
        count = np.bincount(self.data_df['cls'])
        weights = gmean(count) / count
        weights[weights < 0.1] = 0.1
        weights[weights > 10] = 10
        return weights

    def load(self):
        # Get filenames
        fs = FilenamesDataset(self.source, has_classes=True)
        fs.load(self.min_count, self.map_others)
        self.filenames = fs
        self.cls = self.filenames.cls
        self.cls_labels = fs.cls_labels
        self.num_classes = fs.num_classes

        # Create one hot
        self.cls_onehot = to_categorical(fs.cls)

        # Class weights
        weights = gmean(fs.cls_counts) / fs.cls_counts
        weights[weights < 0.1] = 0.1
        weights[weights > 10] = 10
        self.class_weights = weights

        # Create split
        if self.test_split > 0:
            self.train_idx, self.test_idx = train_test_split(np.arange(len(self.filenames.filenames)),
                                                             stratify=self.cls,
                                                             test_size=self.test_split,
                                                             train_size=self.train_split,
                                                             random_state=self.random_seed)
        else:
            self.train_idx = np.arange(len(self.filenames.filenames))
            self.test_idx = []

        # Load images
        to_greyscale = False
        if self.img_type == 'k' or self.img_type == 'greyscale':
            to_greyscale = True
        # print(self.img_size)
        self.images = ImageDataset(self.filenames.filenames,
                                   self.cls_onehot,
                                   transform_fn='resize_with_pad',
                                   transform_args=[self.img_size, to_greyscale],
                                   memmap_directory=self.memmap_directory)
        self.images.load()

    def train_generator(self, batch_size=32, shuffle=True, one_shot=False, undersample=False, map_fn=TFGenerator.map_fn_divide_255):
        return self.images.create_generator(batch_size, self.train_idx, map_fn=map_fn, shuffle=shuffle, one_shot=one_shot, undersample=undersample)

    def test_generator(self, batch_size=32, shuffle=True, one_shot=False, undersample=False, map_fn=TFGenerator.map_fn_divide_255):
        return self.images.create_generator(batch_size, self.test_idx, map_fn=map_fn, shuffle=shuffle, one_shot=one_shot, undersample=undersample)

    def release(self):
        self.images.release()


if __name__ == "__main__":
    source = "/Users/chaos/Documents/Development/Data/Modern_Coretop_Source/project.xml"
    import matplotlib.pyplot as plt
    # XML
    ts = TrainingDataset(source,
                         img_size=[224, 224, 1],
                         img_type='k',
                         min_count=10,
                         map_others=False,
                         test_split=0.2,
                         random_seed=0,
                         memmap_directory=None)
    ts.load(32)
    im = ts.images.data[0]
    plt.imshow(im)
    plt.title("shape: {}, max: {}, min: {}".format(im.shape, im.max(), im.min()))
    plt.show()

    ts = TrainingDataset(source,
                         img_size=[224, 224, 3],
                         img_type='k',
                         min_count=10,
                         map_others=False,
                         test_split=0.2,
                         random_seed=0,
                         memmap_directory=None)
    ts.load(32)
    im = ts.images.data[0]
    plt.imshow(im)
    plt.title("shape: {}, max: {}, min: {}".format(im.shape, im.max(), im.min()))
    plt.show()

    ts = TrainingDataset(source,
                         img_size=[224, 224, 3],
                         img_type='rgb',
                         min_count=10,
                         map_others=False,
                         test_split=0.2,
                         random_seed=0,
                         memmap_directory=None)
    ts.load(32)
    im = ts.images.data[0]
    plt.imshow(im)
    plt.title("shape: {}, max: {}, min: {}".format(im.shape, im.max(), im.min()))
    plt.show()
