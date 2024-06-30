import os
import argparse  # commandline arguments parsing
import pandas as pd
import re
from tqdm import tqdm
import shutil
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def extract_number_from_filename(filename):
    return int(re.search(r'\d+', filename).group())


config = argparse.ArgumentParser()  # commandline arguments parsing

config.add_argument("--n_fft", default=256, type=int, help="FFT window size for STFT")
config.add_argument("--hop_length", default=2, type=int, help="Hop length for STFT")
config.add_argument("--win_length", default=256, type=int, help="Window length for STFT")
config.add_argument("--tol_ratio", type=float, default=0.6, help="Tolerance factor for data cleansing")

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

n_fft = config['n_fft']
hop_length = config['hop_length']
win_length = config['win_length']
tol_ratio = config['tol_ratio']

# 디렉토리 지정
ROOT_DIR = "./../data"
DATA_DIR = os.path.join(ROOT_DIR, "data_slice_256piece")
AFTER_CLEANSING_TARGET_DIR = os.path.join(ROOT_DIR, f"data_slice_256piece_tol_ratio_{tol_ratio}")
# ROOT_TARGET_DIR = os.path.join(ROOT_DIR, "spectrogram", f"librosa_stft_data_vec_tol{tol_factor}")
# SPECTO_ROOT_TARGET_DIR = os.path.join(ROOT_DIR, "spectrogram_png", f"librosa_stft_data_vec_tol{tol_factor}")
#
os.makedirs(AFTER_CLEANSING_TARGET_DIR, exist_ok=True)
# os.makedirs(SPECTO_ROOT_TARGET_DIR, exist_ok=True)

class_folder_list = sorted(os.listdir(DATA_DIR), key=extract_number_from_filename)

# 각 클래스별 값을 적재해서 평균내기 위한 임시 변수
class_mean = {}

for class_idx, class_folder in enumerate(tqdm(class_folder_list, desc="preProcessing Classes")):
    class_path = os.path.join(DATA_DIR, class_folder)
    after_class_path = os.path.join(AFTER_CLEANSING_TARGET_DIR, class_folder)
    os.makedirs(after_class_path, exist_ok=True)
    # 각 클래스 폴더에 대한 빈 데이터프레임 생성
    df_class = pd.DataFrame()

    # 클래스별 각 파일 순회
    for file_name in tqdm(os.listdir(class_path), desc=f"Scanning Files in {class_folder}", leave=False):
        file_path = os.path.join(class_path, file_name)
        df = pd.read_csv(file_path, index_col='Time')
        # 가로로 붙이기
        df_class = pd.concat([df_class, df], axis=1)

    # 각 클래스별 평균 구하기
    class_df_mean = df_class.mean(axis=1)
    class_mean[class_idx] = class_df_mean

    ### 각 파일을 순회하며 file_path와 diff_max를 저장할 데이터프레임 선언
    class_diff_df = pd.DataFrame(columns=['file_name', 'diff_max'])

    # 클래스별 각 파일 재 순회하며 평균 값과의 차이를 저장
    for file_name in tqdm(os.listdir(class_path), desc=f"Calculation Files in {class_folder}", leave=False):
        file_path = os.path.join(class_path, file_name)
        after_file_path = os.path.join(after_class_path, file_name)
        df = pd.read_csv(file_path, index_col='Time')
        data_np = df.to_numpy()
        # 넘파이 배열로 변경
        each_values = data_np.flatten()
        # 넘파이 배열로 변경
        mean_values = class_mean[class_idx].values

        diff = each_values - mean_values
        # 최대로 차이가 많이나는 한 포인트의 스칼라값 저장
        diff_max = np.abs(diff).max()
        diff_df = pd.DataFrame({'file_name': [file_name], 'diff_max': [diff_max]})
        class_diff_df = pd.concat([class_diff_df, diff_df])

    ### diff의 값을 기준으로 정렬하여 diff가 작은 상위 tol_ratio의 데이터를 저장

    # diff_max를 기준으로 오름차순 정렬
    class_diff_df = class_diff_df.sort_values('diff_max')
    class_diff_df.to_csv(os.path.join(AFTER_CLEANSING_TARGET_DIR, f"diff_{class_folder}_class.csv"))
    # 상위 tol_ratio 비율의 데이터만 선택
    top_data = class_diff_df.iloc[:int(tol_ratio * len(class_diff_df))]

    # 선택된 데이터의 파일을 새로운 위치로 이동
    for file_name in tqdm(top_data['file_name'], desc=f"Copy Files in {class_folder}", leave=False):
        file_path = os.path.join(class_path, file_name)
        new_path = os.path.join(after_class_path, file_name)
        shutil.copyfile(file_path, new_path)
