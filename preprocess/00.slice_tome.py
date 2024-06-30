import os
import argparse  # commandline arguments parsing
import pandas as pd
import re
from tqdm import tqdm

def extract_number(s):
    return int(re.search(r'\d+', s).group())

def extract_number_from_filename(filename):
    return int(re.search(r'\d+', filename).group())


config = argparse.ArgumentParser()  # commandline arguments parsing

config.add_argument("--start", default=1.5, type=float)
config.add_argument("--end", default=2.1, type=float)

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

start_time = config['start']
end_time = config['end']

# 디렉토리 지정
ROOT_DIR = "../data"
ROOT_DATA_DIR = os.path.join(ROOT_DIR, "data_origin", "JL_230829_ML6")
SAVE_DATA_DIR = os.path.join(ROOT_DIR, "data_slice")

os.makedirs(SAVE_DATA_DIR, exist_ok=True)


for idx, class_dirs in tqdm(sorted(enumerate(os.listdir(ROOT_DATA_DIR)), key=lambda x: extract_number(x[1])), leave=False):
    class_dir_path = os.path.join(ROOT_DATA_DIR, class_dirs)
    # print(class_dirs[:-7])
    file_path_arr = []
    for data in sorted(os.listdir(class_dir_path), key=extract_number_from_filename):
        file_path = os.path.join(class_dir_path, data)
        file_path_arr.append(file_path)

    SAVE_CLASS_DIR = os.path.join(SAVE_DATA_DIR, class_dirs[:-7])
    os.makedirs(SAVE_CLASS_DIR, exist_ok=True)

    for idx_, data_path in tqdm(enumerate(file_path_arr), leave=False):
        df = pd.read_csv(data_path, index_col='Time')
        df = df.rename(columns={'Unnamed: 170': '170'})
        # df_slice = df.iloc[int(start_time * 500):int(end_time * 500), :]
        df_slice = df.iloc[int(start_time * 500):int(start_time * 500)+256, :]
        for i in range(1, 171):
            df_slice[str(i)].to_csv(os.path.join(SAVE_CLASS_DIR, f"{idx_+1}_{i}.csv"))
