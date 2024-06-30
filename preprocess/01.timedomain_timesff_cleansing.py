import os
import argparse  # commandline arguments parsing
import pandas as pd
import re
from tqdm import tqdm
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import shutil

def extract_number_from_filename(filename):
    return int(re.search(r'\d+', filename).group())


config = argparse.ArgumentParser()  # commandline arguments parsing

config.add_argument("--n_fft", default=256, type=int, help="FFT window size for STFT")
config.add_argument("--hop_length", default=2, type=int, help="Hop length for STFT")
config.add_argument("--win_length", default=256, type=int, help="Window length for STFT")
config.add_argument("--tol_factor", type=float, default=0.4, help="Tolerance factor for data cleansing") # 바꿀때는 변경해라잉

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

n_fft = config['n_fft']
hop_length = config['hop_length']
win_length = config['win_length']
tol_factor = config['tol_factor']

# 디렉토리 지정
ROOT_DIR = "./../data"
DATA_DIR = os.path.join(ROOT_DIR, "data_slice_256piece")
AFTER_CLEANSING_TARGET_DIR = os.path.join(ROOT_DIR, f"data_slice_256piece_tol_ratio_{tol_factor}")
# SPECTO_ROOT_TARGET_DIR = os.path.join(ROOT_DIR, "spectrogram_png", f"librosa_stft_data_vec_tol{tol_factor}")
#
os.makedirs(AFTER_CLEANSING_TARGET_DIR, exist_ok=True)
# os.makedirs(SPECTO_ROOT_TARGET_DIR, exist_ok=True)


class_folder_list = sorted(os.listdir(DATA_DIR), key=extract_number_from_filename)
print(class_folder_list)
print("# of classes: ", len(class_folder_list))

# 각 클래스별 값을 적재해서 평균내기 위한 임시 변수
class_mean = {}

for class_idx, class_folder in enumerate(tqdm(class_folder_list, desc="preProcessing Classes")): # desc: describe(설명)

    class_folder_path = os.path.join(DATA_DIR, class_folder)
    after_class_path = os.path.join(AFTER_CLEANSING_TARGET_DIR, class_folder)
    fft_results = []
    os.makedirs(after_class_path, exist_ok=True)
    # 각 클래스 폴더에 대한 빈 데이터프레임 생성
    df_class = pd.DataFrame()

    # 클래스별 각 파일 순회
    for file_name in tqdm(os.listdir(class_folder_path), desc=f"Processing Files in {class_folder}", leave=False):
        file_path = os.path.join(class_folder_path, file_name)
        df = pd.read_csv(file_path, index_col='Time')
        data_np = df.to_numpy().flatten()
        # FFT 적용
        fft_result = fft(data_np)
        selected_freqs = np.abs(fft_result[:len(fft_result) // 2])
        fft_results.append(selected_freqs)

    # 클래스별 FFT 결과의 평균 계산
    fft_results = np.array(fft_results)
    mean_vector = np.mean(fft_results, axis=0)

    class_mean[class_folder] = mean_vector

    # 각 파일을 순회하며 file_path와 distance를 저장할 데이터프레임 선언
    class_distance_df = pd.DataFrame(columns=['file_name', 'distance'])

    # 클래스별 각 파일 재 순회하며 평균 값과의 차이를 저장
    for file_name in tqdm(os.listdir(class_folder_path), desc=f"Calculation Files in {class_folder}", leave=False):
        file_path = os.path.join(class_folder_path, file_name)
        after_file_path = os.path.join(after_class_path, file_name)
        df = pd.read_csv(file_path, index_col='Time')
        data_np = df.to_numpy().flatten()
        # FFT 적용
        fft_result = np.abs(fft(data_np)[:len(data_np) // 2])

        # 클래스별 평균 FFT 결과로부터의 유클리드 거리 계산
        # mean_vector = class_mean[class_folder]  # 클래스별 평균 벡터
        distance = np.linalg.norm(fft_result - mean_vector)

        # 최대로 차이가 많이나는 한 포인트의 스칼라값 저장
        distance_df = pd.DataFrame({'file_name': [file_name], 'distance': [distance]})
        class_distance_df = pd.concat([class_distance_df, distance_df])

    # distance의 값을 기준으로 정렬하여 distance가 작은 상위 tol_factor의 데이터를 저장

    # distance를 기준으로 오름차순 정렬
    class_distance_df = class_distance_df.sort_values('distance')
    class_distance_df.to_csv(os.path.join(AFTER_CLEANSING_TARGET_DIR, f"distance_{class_folder}_class.csv"))
    # 상위 tol_ratio 비율의 데이터만 선택
    top_data = class_distance_df.iloc[:int(tol_factor * len(class_distance_df))]

    # 선택된 데이터의 파일을 새로운 위치로 이동
    for file_name in tqdm(top_data['file_name'], desc=f"Copy Files in {class_folder}", leave=False):
        file_path = os.path.join(class_folder_path, file_name)
        new_path = os.path.join(after_class_path, file_name)
        shutil.copyfile(file_path, new_path)

