from tensorflow.keras.utils import to_categorical

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
        self.random_seed = random_seed
        self.memmap_directory = memmap_directory

        self.filenames_dataset: FilenamesDataset = None
        self.train: ImageDataset = None
        self.test: ImageDataset = None
        self.cls_labels = None

    def load(self, batch_size=32):
        # Get filenames
        fs = FilenamesDataset(self.source, has_classes=True)
        fs.load(self.min_count, self.map_others)
        fs.split(self.test_split, stratify=True, seed=self.random_seed)
        self.filenames_dataset = fs
        self.cls_labels = fs.cls_labels

        # Load images and labels as onehot vectors
        to_greyscale = False
        if self.img_type == 'k' or self.img_type == 'greyscale':
            to_greyscale = True
        self.train_cls = fs.train.cls
        self.test_cls = fs.test.cls
        self.train_cls_onehot = to_categorical(fs.train.cls)
        self.test_cls_onehot = to_categorical(fs.test.cls)
        self.train = ImageDataset(fs.train.filenames,
                                  self.train_cls_onehot,
                                  transform_fn='resize_with_pad',
                                  transform_args=[self.img_size, to_greyscale],
                                  memmap_directory=self.memmap_directory)
        self.train.load()
        self.test = ImageDataset(fs.test.filenames,
                                 self.test_cls_onehot,
                                 transform_fn='resize_with_pad',
                                 transform_args=[self.img_size, to_greyscale],
                                 memmap_directory=self.memmap_directory)
        self.test.load()

        # Create generators for training
        self.train_generator = TFGenerator(self.train.data,
                                           self.train.cls,
                                           batch_size=batch_size,
                                           map_fn=TFGenerator.map_fn_divide_255,
                                           one_shot=False)
        self.test_generator = TFGenerator(self.test.data,
                                          self.test.cls,
                                          batch_size=batch_size,
                                          map_fn=TFGenerator.map_fn_divide_255,
                                          one_shot=True)
        self.train_batches_per_epoch = len(self.train_generator)
        self.test_batches_per_epoch = len(self.test_generator)
        # Also create tf datasets
        self.train_tfdataset = self.train_generator.to_tfdataset()
        self.test_tfdataset = self.test_generator.to_tfdataset()

    def release(self):
        self.train.release()
        self.test.release()


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
    im = ts.train.data[0]
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
    im = ts.train.data[0]
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
    im = ts.train.data[0]
    plt.imshow(im)
    plt.title("shape: {}, max: {}, min: {}".format(im.shape, im.max(), im.min()))
    plt.show()
