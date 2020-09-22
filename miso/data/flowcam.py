import pandas as pd
import os
import skimage.io as skio
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import argparse


def process_dir(input_dir, species_filename, name, save_dir=None, save_csv=True):
    dirs = [d for d in glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    df_master = None
    for d in dirs:
        df = process(d, species_filename, name, save_dir, save_csv=False)
        if df_master is None:
            df_master = df
        else:
            df_master.append(df)
    if save_csv:
        df_master.to_csv(os.path.join(save_dir, "{}_data.csv".format(name)))
    return df_master


def process(input_dir, species_filename, name, save_dir=None, save_csv=True):
    # Load species conversion list
    noms = pd.read_excel(species_filename, sheet_name=0).dropna()
    doublons = pd.read_excel(species_filename, sheet_name=1).dropna()
    feuil1 = pd.read_excel(species_filename, sheet_name=2).dropna()

    # Data csv
    # data_csv_filename = glob(os.path.join(input_dir, "*_data.csv"))[0]
    cla_filename = glob(os.path.join(input_dir, "*.cla"))
    lst_filename = glob(os.path.join(input_dir, "*.lst"))
    if len(cla_filename) == 0 or len(lst_filename) == 0:
        print("No .cla or .lst file found in {}".format(input_dir))
        return
    cla_filename = cla_filename[0]
    lst_filename = lst_filename[0]

    cls_dict = get_class_list(cla_filename)
    df = get_classifications(lst_filename)

    # The directory the data csv is located in
    base_dir = input_dir
    run_name = os.path.basename(base_dir).replace(' ', '_')
    # The directory where we will save the images
    if save_dir is None:
        save_dir = os.path.join(base_dir, os.path.basename(base_dir) + "_images_individuelles")
    os.makedirs(save_dir, exist_ok=True)

    # Group the results by image
    df_grouped = df.groupby("collage_file")

    print('-' * 80)
    print("Flowcam segmenter")
    print('-' * 80)
    print("Dataset: {}".format(run_name))
    print("Directory: {}".format(base_dir))
    print("Classifications: {}".format(cla_filename))
    print("Image data: {}".format(lst_filename))
    print("Species XLSX: {}".format(species_filename))
    print("Processing...")
    # Process each image
    for filename, group in tqdm(df_grouped):
        # Load the image
        im_filename = os.path.join(base_dir, filename)
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
                cls = "sans_etiquette"
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
            seg_im_filename = os.path.join(save_dir, cls, "{}_{}_{:08d}.png".format(name, run_name, id))
            os.makedirs(os.path.dirname(seg_im_filename), exist_ok=True)
            skio.imsave(seg_im_filename, seg_im)
    if save_csv:
        df.to_csv(os.path.join(save_dir, "{}_{}_data.csv".format(name, run_name)))
    print("Complete!")
    return df


def get_class_list(filename):
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


def get_classifications(filename):
    field_names = []
    with open(filename, "r") as f:
        f.readline()
        numfields = int(f.readline().split('|')[1])
        for i in range(numfields):
            field_names.append(f.readline().split('|')[0])
        print(field_names)
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
    parser.add_argument("-d", "--dir", type=str, help="Input directory containing directories of samples of the flowcam data CSV file and collated images")
    parser.add_argument("-s", "--species", type=str, required=True,  help="XLSX file with the map of class names to true species identifiers")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory to save images (if not used, images will be saved in a directory alongside the data CSV file)")
    parser.add_argument("-n", "--name", required=True, help="Name of the campaign this sample is from")
    args = parser.parse_args()
    if args.input is not None:
        process(args.input, args.species, args.name, args.output)
    else:
        process_dir(args.dir, args.species, args.name, args.output)
