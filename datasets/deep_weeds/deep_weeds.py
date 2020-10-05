import pandas as pd
import os
from glob import glob
import shutil
from tqdm import tqdm

SOURCE_DIR = "/Users/chaos/Downloads/images-3"
SOURCE_CSV = "/Users/chaos/Downloads/images-3/labels.csv"
OUTPUT_DIR = "/Users/chaos/OneDrive/Datasets/DeepWeeds"

df = pd.read_csv(SOURCE_CSV)

for idx, row in tqdm(df.iterrows()):
    fn = row.Filename
    sp = row.Species.replace(' ', '_')

    path = os.path.join(OUTPUT_DIR, sp)
    os.makedirs(path, exist_ok=True)

    shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(path, fn))