import numpy as np
import skimage.io as skio

from miso.data.dataset import DatasetBase
from miso.data.image_loader import ParallelImageLoader
from miso.data.image_utils import resize_transform, resize_with_pad_transform, null_transform


class ImageDataset(DatasetBase):
    def __init__(self,
                 filenames,
                 cls=None,
                 transform_fn=None,
                 transform_args=None,
                 img_size=None,
                 memmap_directory=None,
                 overwrite_memmap=False,
                 unique_id=None,
                 dtype=np.uint8):
        self.filenames = filenames
        self.cls = cls
        self.rescale = rescale_transform
        self.transform_fn = transform_fn
        self.transform_args = transform_args
        self.unique_id = unique_id

        # Pre-made transforms
        if self.transform_fn == 'resize':
            self.transform_fn = resize_transform
        elif self.transform_fn == 'resize_with_pad':
            self.transform_fn = resize_with_pad_transform

        if self.transform_fn is None:
            self.transform_fn = null_transform
            self.transform_args = [0]

        # Get dataset unique identification hash
        super().__init__(memmap_directory=memmap_directory, overwrite_memmap=overwrite_memmap, dtype=dtype)
        self.hash_data = ["ImageDataset", self.filenames, str(self.dtype), self.unique_id]

        print('-' * 60)
        print("@ Image dataset, id: {}".format(self.get_hash_id()))
        if memmap_directory is None:
            print("@ - stored in RAM")
        else:
            print("@ - will be saved at {}".format(self.memmap_file))

        if img_size is None:
            # Read first image to see the size
            im = skio.imread(self.filenames[0])
            if self.transform_args is not None:
                im = self.transform_fn(im, *self.transform_args)
            else:
                im = self.transform_fn(im)
            self.img_size = im.shape
        else:
            self.img_size = img_size
        self.arr_size = (len(self.filenames), ) + self.img_size
        print("Array size is {}".format(self.arr_size))

    def load(self):
        if self.read_or_create_data(self.arr_size, self.dtype) is not True:
            loader = ParallelImageLoader(self.filenames,
                                         self.data,
                                         transform_fn=self.transform_fn,
                                         transform_args=self.transform_args)
            loader.load()



