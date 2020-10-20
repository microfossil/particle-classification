import pandas as pd
import os
import skimage.io as skio
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import argparse


def process_dir(input_dir, save_dir, species_filename, campaign_name, save_csv=True):
    # Find all lst files
    lst_filenames = sorted(glob(os.path.join(input_dir, "**", "*.lst")))
    # Extract the base directories
    dirs = [os.path.dirname(fn) for fn in lst_filenames]
    # Only take unique ones
    dirs = sorted(list(set(dirs)))
    # dirs = [d for d in sorted(glob(os.path.join(input_dir, "*"))) if os.path.isdir(d)]
    df_master = None
    for d in dirs:
        # Last part of directory
        run_save_dir = os.path.join(save_dir, d[len(input_dir)+1:])
        run_name = d[len(input_dir)+1:].replace("/", "_").replace(" ", "_").replace("\\", "_")
        df = process(d, run_save_dir, species_filename, campaign_name, run_name, save_csv=True)
        if df_master is None:
            df_master = df
        else:
            df_master.append(df)
    if save_csv:
        df_master.to_csv(os.path.join(save_dir, "{}_data.csv".format(campaign_name)))
    return df_master


def process(input_dir, save_dir, species_filename, campaign_name, run_name=None, save_csv=True):
    # Load species conversion list
    noms = pd.read_excel(species_filename, sheet_name=0).dropna()
    doublons = pd.read_excel(species_filename, sheet_name=1).dropna()
    feuil1 = pd.read_excel(species_filename, sheet_name=2).dropna()

    # The directory where we will save the images
    if run_name is None:
        run_name = os.path.basename(input_dir).replace(" ", "_")
    os.makedirs(save_dir, exist_ok=True)

    print('-' * 80)
    # print("Flowcam segmenter")
    # print('-' * 80)
    # print("- species XLSX: {}".format(species_filename))
    print("Campaign: {}".format(campaign_name))
    print("Sample: {}".format(run_name))
    print("- input directory: {}".format(input_dir))

    # Image data
    lst_filename = glob(os.path.join(input_dir, "*.lst"))
    if len(lst_filename) == 0:
        print("! No .lst file found in {}, skipping !".format(input_dir))
        return
    lst_filename = lst_filename[0]
    df = parse_image_list(lst_filename)

    # Classification data
    cla_filename = glob(os.path.join(input_dir, "*.cla"))
    if len(cla_filename) == 0 or os.path.getsize(cla_filename[0]) == 0:
        print("- classification (.cla) file is missing or empty, all images will be placed in \"unlabeled\" class")
        cls_dict = dict()
    else:
        cla_filename = cla_filename[0]
        cls_dict = parse_classifications(cla_filename)
        print("- classification filename: {}".format(cla_filename))
    print("- image data filename: {}".format(lst_filename))
    print("- output directory: {}".format(save_dir))
    print("Processing...")

    # Group the results by image
    df_grouped = df.groupby("collage_file")

    # Extra info to save
    df_cls = [""] * len(df)
    df_campaign = [campaign_name] * len(df)
    df_sample = [run_name] * len(df)

    # Process each image
    for filename, group in tqdm(df_grouped):
        # Load the image
        im_filename = os.path.join(input_dir, filename)
        im = skio.imread(im_filename)
        # Cut each image out
        for row in group.iterrows():
            row_id = row[0]
            row = row[1]
            # Get image number
            id = row['id']
            # Get class
            if id in cls_dict:
                cls = cls_dict[id]
            else:
                cls = "unlabeled"
            # Modify class to correct if needed
            if cls in doublons.iloc[:, 0].values:
                vals = doublons[doublons.iloc[:, 0] == cls]
                new_cls = vals.iloc[0, 1]
                cls = new_cls
            # Get image coordinates
            x = row['image_x']
            y = row['image_y']
            width = row['image_w']
            height = row['image_h']
            # Get the segmented image
            seg_im = im[y:y + height, x:x + width, ...]
            # Save it!
            seg_im_filename = os.path.join(save_dir, cls, "{}_{}_{:08d}.png".format(campaign_name, run_name, id))
            os.makedirs(os.path.dirname(seg_im_filename), exist_ok=True)
            skio.imsave(seg_im_filename, seg_im)
            df_cls[id-1] = cls
    df.insert(0, 'campaign', df_campaign)
    df.insert(1, 'sample', df_sample)
    df.insert(3, 'class', df_cls)
    if save_csv:
        df.to_csv(os.path.join(save_dir, "{}_{}_data.csv".format(campaign_name, run_name)))
    print("Complete!")
    return df


def parse_classifications(filename):
    cls_dict = OrderedDict()
    with open(filename, "r") as f:
        f.readline()
        f.readline()
        f.readline()
        num_classes = int(f.readline())
        for i in range(num_classes):
            cls_name = f.readline()[:-1]
            f.readline()
            f.readline()
            num_images = int(f.readline())
            for j in range(num_images):
                idx = int(f.readline())
                cls_dict[idx] = cls_name
    return cls_dict


def parse_image_list(filename):
    field_names = []
    with open(filename, "r") as f:
        f.readline()
        numfields = int(f.readline().split('|')[1])
        for i in range(numfields):
            field_names.append(f.readline().split('|')[0])
        # print(field_names)
    df = pd.read_csv(filename, '|', skiprows=numfields + 2, header=None)
    df.columns = field_names
    return df

#
# def process(folder):
#     cla_filename = glob(os.path.join(folder, "*.cla"))[0]
#     lst_filename = glob(os.path.join(folder, "*.lst"))[0]
#     run_id = os.path.basename(cla_filename)[:-4]
#
#     cls_dict = class_list(cla_filename)
#
#     field_names = []
#     with open(lst_filename, "r") as f:
#         numfields = int(f.readline().split('|')[1])
#         for i in range(numfields):
#             field_names.append(f.readline().split('|')[0])
#         print(field_names)
#     df = pd.read_csv(lst_filename, '|', skiprows=numfields + 2, header=None)
#     df.columns = field_names
#
#     last_image_name = None
#     last_image = None
#     for row in df.iterrows():
#         row_id = row[0]
#         row = row[1]
#         if row['collage_file'] != last_image_name:
#             last_image_name = row['collage_file']
#             last_image = skio.imread(os.path.join(folder, last_image_name))
#         id = row['id']
#         if id in cls_dict:
#             cls_name = cls_dict[id]
#         else:
#             cls_name = "sans_etiquette"
#         save_dir = os.path.join(folder, os.path.basename(cla_filename)[:-4] + "_images_individuelles", cls_name)
#         os.makedirs(save_dir, exist_ok=True)
#         im = last_image[row['image_y']:row['image_y'] + row['image_h'], row['image_x']:row['image_x'] + row['image_w'], :]
#         print(os.path.join(save_dir, run_id + "_{:08d}.png".format(id)))
#         sub_filename = os.path.join(save_dir, run_id + "_{:08d}.png".format(id))
#         skio.imsave(sub_filename, im)
#         df.loc[row_id, 'file'] = os.path.basename(sub_filename)
#     df.to_csv(os.path.join(folder, run_id + "_info.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment flowcam images into individual images sorted by class')
    parser.add_argument("-i", "--input", type=str, help="Input directory containing the flowcam data CSV file and collated images")
    # parser.add_argument("-d", "--dir", type=str, help="Input directory containing directories of samples of the flowcam data CSV file and collated images")
    parser.add_argument("-s", "--species", type=str, required=True,  help="XLSX file with the map of class names to true species identifiers")
    parser.add_argument("-o", "--output", type=str, default=None, required=True,
                        help="Output directory to save images (if not used, images will be saved in a directory alongside the data CSV file)")
    parser.add_argument("-n", "--name", required=True, help="Name of the campaign this sample is from")
    args = parser.parse_args()

    print('-' * 80)
    print("Flowcam segmenter")
    print("- species XLSX: {}".format(args.species))
    process_dir(args.input, args.output, args.species, args.name)
