import os
from glob import glob

import cv2
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import skimage.morphology as skm

from miso.utils.flowcam import parse_image_list

source_dir = r"C:\Users\rossm\OneDrive\Datasets\Plankton\F44 80 micron"

cal_filename = os.path.join(source_dir, "cal_image_000001.tif")
raw_filenames = sorted(glob(os.path.join(source_dir, "rawfile_*.tif")))
lst_filename = sorted(glob(os.path.join(source_dir, "*.lst")))[0]
df = parse_image_list(lst_filename)
print(df)

im_cal = skio.imread(cal_filename).astype(np.float32)

for raw_filename in tqdm(raw_filenames[:4]):
    im_raw = skio.imread(raw_filename)
    plt.imshow(im_raw)
    plt.show()

ims = []
for raw_filename in tqdm(raw_filenames[:40]):
    im_raw = skio.imread(raw_filename)
    ims.append(im_raw)

ims = np.asarray(ims)

bg = np.median(ims, axis=0)

plt.imshow(bg/255)
plt.show()

plt.imshow(np.abs(im_raw - bg) / 255)
plt.colorbar()
plt.show()

plt.imshow((np.abs(im_raw - bg) > 5).astype(np.float32))
plt.show()

plt.imshow(im_raw), plt.show()


im_raw = ims[37]

plt.imshow(im_raw), plt.show()

gr = np.max(np.abs(im_raw - bg), axis=-1) > 20

plt.imshow(gr.astype(np.float32))
plt.show()

from scipy import ndimage

grc = skm.binary_closing(gr, skm.disk(5))
grc = skm.area_opening(grc, 256)
grc = ndimage.binary_fill_holes(grc)

plt.imshow(grc.astype(np.float32))
plt.show()


# Group the results by image
# df_grouped = df.groupby("collage_file")

# Extra info to save
# df_filename = [""] * len(df)
# df_cls = [""] * len(df)
# df_campaign = [campaign_name] * len(df)
# df_sample = [run_name] * len(df)

im_save_dir = os.path.join(source_dir, "new_images")
os.makedirs(im_save_dir, exist_ok=True)

mask_save_dir = os.path.join(source_dir, "new_masks")
os.makedirs(mask_save_dir, exist_ok=True)

# Process each image
for fi, filename in tqdm(enumerate(raw_filenames)):
    # Load the image
    im_filename = os.path.join(source_dir, filename)
    im = skio.imread(im_filename)

    # Calculate mask
    gr = np.max(np.abs(im - bg), axis=-1) > 20
    grc = skm.binary_closing(gr, skm.disk(5))
    grc = skm.area_opening(grc, 256)
    mask = ndimage.binary_fill_holes(grc)
    # cv2.imshow("im", mask.astype(np.float32))
    # cv2.waitKey(10)

    # Find contours
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)



    # is_mask = False
    # if save_mask:
    #     if os.path.exists(im_filename):
    #         try:
    #             mask = skio.imread(mask_filename)
    #             is_mask = True
    #         except:
    #             print("Error opening {}".format(mask_filename))
    #     else:
    #         print("Mask not found {}".format(im_filename))

    # Cut each image out
    for ci, contour in enumerate(contours[0]):
        bb = cv2.boundingRect(contour)
        print(bb)
        # row_id = row[0]
        # row = row[1]
        # # Get image coordinates
        # id = row['id']
        # x = row['image_x']
        # y = row['image_y']
        # width = row['image_w']
        # height = row['image_h']
        # Get the segmented mask
        x = bb[0]
        y = bb[1]
        width = bb[2]
        height = bb[3]

        seg_im = im[y:y + height, x:x + width, ...]
        seg_im_filename = os.path.join(im_save_dir, "{:04d}_{:04d}.png".format(fi, ci))
        skio.imsave(seg_im_filename, seg_im)

        seg_mask = mask[y:y + height, x:x + width, ...]
        seg_mask = seg_mask.astype(np.uint8) * 255
        seg_mask_filename = os.path.join(mask_save_dir, "{:04d}_{:04d}.png".format(fi, ci))
        skio.imsave(seg_mask_filename, seg_mask)
        # df_cls[id-1] = cls
        # df_filename[id-1] = seg_im_filename
