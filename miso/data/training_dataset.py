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

        self.ds_train: ImageDataset = None
        self.ds_test: ImageDataset = None

    def load(self, batch_size=32):
        # Get filenames
        fs = FilenamesDataset(self.source, has_classes=True)
        fs.load(self.min_count, self.map_others)
        fs.split(self.test_split, stratify=True, seed=self.random_seed)

        # Load images and labels as onehot vectors
        to_greyscale = False
        if self.img_type == 'k' or self.img_type == 'greyscale':
            to_greyscale = True
        train_oh = to_categorical(fs.train.cls)
        test_oh = to_categorical(fs.test.cls)
        self.ds_train = ImageDataset(fs.train.filenames,
                                train_oh,
                                transform_fn='resize_with_pad',
                                transform_args=[self.img_size, to_greyscale],
                                memmap_directory=self.memmap_directory)
        self.ds_test = ImageDataset(fs.test.filenames,
                                test_oh,
                                transform_fn='resize_with_pad',
                                transform_args=[self.img_size, to_greyscale],
                                memmap_directory=self.memmap_directory)
        self.ds_train.load()
        self.ds_test.load()

        # Create generators for training
        self.ds_train_generator = TFGenerator(self.ds_train.data,
                                              self.ds_train.cls,
                                              batch_size=batch_size,
                                              map_fn=TFGenerator.map_fn_divide_255,
                                              one_shot=False)
        self.ds_test_generator = TFGenerator(self.ds_test.data,
                                              self.ds_test.cls,
                                              batch_size=batch_size,
                                              map_fn=TFGenerator.map_fn_divide_255,
                                              one_shot=True)
        self.train_batches_per_epoch = len(self.ds_train_generator)
        self.test_batches_per_epoch = len(self.ds_test_generator)

        # Also create tf datasets
        self.ds_train_tfdataset = self.ds_train_generator.to_tfdataset()
        self.ds_test_tfdataset = self.ds_test_generator.to_tfdataset()



if __name__ == "__main__":
    