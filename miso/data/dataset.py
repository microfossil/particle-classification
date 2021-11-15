import hashlib
import os
import numpy as np
from numpy.lib.format import open_memmap
import gc


class DatasetBase:
    def __init__(self, memmap_directory=None, overwrite_memmap=False, dtype=np.uint8):
        self.memmap_directory = memmap_directory
        self.overwrite_memmap = overwrite_memmap
        self.memmap_file = None
        self.hash_data = None
        self.data = None
        self.shape = None
        self.dtype = dtype

    def read_or_create_data(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

        if isinstance(shape, list):
            raise ValueError("Shape must be a tuple, e.g. (100,32,32,3) not a list, e.g. [100,32,32,3]")

        if self.memmap_directory:
            if self.hash_data is None:
                raise ValueError("Please set the hash_data (as a numpy array) to use memory mapping")
            self.memmap_file = os.path.join(self.memmap_directory, self.get_hash_id() + ".npy")
            if self.overwrite_memmap is False and os.path.exists(self.memmap_file):
                print("Existing data found at {}".format(self.memmap_file))
                self.data = open_memmap(self.memmap_file, mode='r+', dtype=self.dtype, shape=self.shape)
                # Check if not all zeros
                # If all zeros, usually indication of an error creating the memmap file previously, therefore recreate
                if np.count_nonzero(self.data[0]) > 0 and np.count_nonzero(self.data[-1]) > 0:
                    return True
                else:
                    self.data._mmap.close()
                    print("File was likely corrupted, recreating.")
            print("Creating memmap file at {}".format(self.memmap_file))
            os.makedirs(self.memmap_directory, exist_ok=True)
            self.data = open_memmap(self.memmap_file, mode='w+', dtype=self.dtype, shape=self.shape)
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)

        return False

    def get_hash_id(self):
        """
        Creates a 16 character hash id based on the hash data. This is used to create a unique
        filename for the numpy memmap array
        :return: 16 character hash id
        """
        if self.hash_data is None:
            raise ValueError("Please set the hash_data before getting the hash")
        else:
            return hashlib.sha256(repr(self.hash_data).encode('UTF-8')).hexdigest()[0:16]

    def release(self):
        if self.memmap_file is None:
            del self.data
        else:
            if os.path.exists(self.memmap_file):
                self.data._mmap.close()
                del self.data
                gc.collect()
                os.remove(self.memmap_file)
            self.memmap_file = None


