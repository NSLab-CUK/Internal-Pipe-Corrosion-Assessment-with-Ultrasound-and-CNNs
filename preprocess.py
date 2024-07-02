from preprocess_fn import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch

ROOT_DIR = "D:\water_quality_prediction\data"
ROOT_DIR_SUB = "D:\minseo"
ROOT_DATA_DIR = os.path.join(ROOT_DIR, "origin_Fe203")
ROOT_TARGET_DIR = os.path.join(ROOT_DIR, "specto_512")
SAVE_CSV = False  # set save format CSV or np file


csv_paths = get_filePaths_arr(
    root=ROOT_DATA_DIR
)  # [sub_dirs, data],[sub_dirs, data],...
# 실제 파일은 ROOT_DIR + sub_dirs + data에 존재
cnt = 0
for i in csv_paths:
    cnt += 1
    absolute_csv_paths = os.path.join(ROOT_DATA_DIR, i[0], i[1])
    df = pd.read_csv(absolute_csv_paths)
    # print(df)

    for j in tqdm(df.columns, desc=f"{cnt}/{len(csv_paths)}({i})"):
        # j=columns of csv
        if j == "Time":
            continue

        #'Unnamed: 500' 별도 처리 위한 구문
        if not j == "Unnamed: 500":
            continue

        data_np = df[j].values  # changed
        specto_np = get_spectrogram_dn(data_np)
        set_folder(ROOT_TARGET_DIR, i[0], i[1])
        j = "500"
        target_path = os.path.join(ROOT_TARGET_DIR, i[0], i[1], j)
        if SAVE_CSV:
            specto_df = pd.DataFrame(specto_np)
            specto_df.to_csv(f"{target_path}.csv", index=False)
        else:
            np.save(f"{target_path}.npy", specto_np)
