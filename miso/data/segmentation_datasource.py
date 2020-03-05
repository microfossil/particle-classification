import numpy as np
from PIL import Image, ImageSequence
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd


class MaskDataSource:

    def __init__(self):
        self.data_df = None
        self.train_df = None
        self.test_df = None
        self.train_images = None
        self.train_masks = None
        self.test_images = None
        self.test_masks = None

    def load_images(self,
                    img_size,
                    rescale_params=(255, 0, 1),
                    color_mode='rgb',
                    split=0.25,
                    flatten=True,
                    seed=None):
        filenames = self.data_df['filenames']
        images = []
        masks = []
        # Load each image
        print("Loading images...")
        print()
        for idx, filename in enumerate(filenames):
            if idx % 100 == 0:
                print(idx)
            im = Image.open(filename)

            for i, page in enumerate(ImageSequence.Iterator(im)):
                # Colour image
                if i == 0:
                    page = np.array(page)
                    if color_mode == 'grayscale':
                        page = rgb2gray(page)
                    else:
                        page = page / 255
                    page = resize(page, img_size, order=1)
                    images.append(page)
                if i == 4:
                    page = np.array(page)
                    page = rgb2gray(page)
                    page = resize(page, img_size, order=1)
                    page = page > 0.5
                    page = page.astype(int)
                    if flatten:
                        encoded = np.zeros((img_size[0], img_size[1], 2))
                        for j in range(2):
                            temp = page == 0
                            temp = temp.astype(int)
                            encoded[:, :, j] = temp
                        encoded = np.reshape(encoded, (img_size[0] * img_size[1], 2))
                        masks.append(encoded)
                    else:
                        page = page[:,:,np.newaxis]
                        masks.append(page)
                if i == 5:
                    break
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[:, :, :, np.newaxis]
        masks = np.asarray(masks)
        self.split(images, masks, split, seed)

    def split(self, images, masks, split=0.25, seed=None):
        self.train_images, self.test_images, \
        self.train_masks, self.test_masks, \
        self.train_df, self.test_df = \
            train_test_split(images,
                             masks,
                             self.data_df,
                             test_size=split,
                             random_state=seed)

    def set_directory_source(self,
                             directory):
        files = glob.glob(directory + "\\*.tiff")
        files.extend(glob.glob(directory + "\\*.tif"))
        files = sorted(files)

        filenames = []
        short_filenames = []
        for file in files:
            filenames.append(file)
            short_filenames.append(
                os.path.basename(os.path.dirname(file)) + os.path.sep + os.path.basename(file))

        df = {"filenames": filenames, "short_filenames": short_filenames}
        self.data_df = pd.DataFrame(df)
        print("{} files found".format(len(filenames)))

        #     if img_type == 'rgb':
        #         im = Image.open(filename)
        #     else:
        #         im = Image.open(filename).convert('L')
        #     im = np.asarray(im, dtype=np.float)
        #     im = resize(im, img_size, order=1)
        #     # Rescale according to params
        #     # e.g. to take a [0,255] range image to [-1,1] params would be 255,-0.5,2
        #     # I' = (I / 255 - 0.5) * 2
        #     im = np.divide(im, rescale_params[0])
        #     im = np.subtract(im, rescale_params[1])
        #     im = np.multiply(im, rescale_params[2])
        #     # Convert to float
        #     im = im.astype(np.float32)
        #     images.append(im)
        #     idx += 1
        #     if idx % 100 == 0:
        #         print("\r{} of {} processed".format(idx, len(filenames)), end='')
        # # Convert all to a numpy array
        # images = np.asarray(images)
        # if images.ndim == 3:
        #     images = images[:, :, :, np.newaxis]
        # # Split into test and training sets
        # self.split(images, split, seed)

# d = DataSource()
# d.set_directory_source(r"C:\Users\rossm\Documents\Data\Foraminifera\Collated\MD042722_200-201_()_18")
# d.load_images((128,128))
