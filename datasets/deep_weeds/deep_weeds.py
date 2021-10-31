import pandas as pd
import os
from glob import glob
import shutil
from tqdm import tqdm

SOURCE_DIR = r"C:\Users\rossm\Documents\Data\Weeds\NavuaSedge_Malanda_May2020"
SOURCE_CSV = r"C:\Users\rossm\Documents\Data\Weeds\NavuaSedge_Malanda_May2020.csv"
OUTPUT_DIR = r"C:\Users\rossm\Documents\Data\Weeds\NavuaSedge_Malanda_May2020_sorted"

df = pd.read_csv(SOURCE_CSV)

for idx, row in tqdm(df.iterrows()):
    fn = row.Filename
    sp = row.Species.replace(' ', '_')

    path = os.path.join(OUTPUT_DIR, sp)
    os.makedirs(path, exist_ok=True)

    shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(path, fn))