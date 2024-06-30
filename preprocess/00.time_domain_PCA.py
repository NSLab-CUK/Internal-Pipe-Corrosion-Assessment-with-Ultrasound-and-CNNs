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

def extract_number_from_filename(filename):
    return int(re.search(r'\d+', filename).group())

config = argparse.ArgumentParser()  # commandline arguments parsing

config.add_argument("--n_fft", default=256, type=int, help="FFT window size for STFT")
config.add_argument("--hop_length", default=2, type=int, help="Hop length for STFT")
config.add_argument("--win_length", default=256, type=int, help="Window length for STFT")
config.add_argument("--tol_factor", type=float, default=0.28, help="Tolerance factor for data cleansing")

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

n_fft = config['n_fft']
hop_length = config['hop_length']
win_length = config['win_length']
tol_factor = config['tol_factor']

# 디렉토리 지정
ROOT_DIR = "./../data"
DATA_DIR = os.path.join(ROOT_DIR, "data_slice_256piece")

# PCA 객체 생성
pca = PCA(n_components=2)  # 2차원으로 축소


class_folder_list = sorted(os.listdir(DATA_DIR), key=extract_number_from_filename)

# 색상 지정을 위한 colormap 생성
colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(class_folder_list)))

for class_idx, class_folder in enumerate(tqdm(class_folder_list, desc="preProcessing Classes")):
    class_folder_path = os.path.join(DATA_DIR, class_folder)
    fft_results = []

    for file_name in tqdm(os.listdir(class_folder_path), desc=f"Processing Files in {class_folder}", leave=False):
        file_path = os.path.join(class_folder_path, file_name)
        df = pd.read_csv(file_path, index_col='Time')
        data_np = df.to_numpy().flatten()
        # FFT 적용
        fft_result = fft(data_np)
        selected_freqs = np.abs(fft_result[:len(fft_result) // 2])
        fft_results.append(selected_freqs)

    # 클래스별 FFT 결과의 평균 및 표준편차 계산
    fft_results = np.array(fft_results)
    mean_vector = np.mean(fft_results, axis=0)

    # PCA 적용
    pca_result = pca.fit_transform(fft_results)

    # 산점도 그리기
    plt.scatter(pca_result[:, 0], pca_result[:, 1], color=colors[class_idx], label=class_folder, alpha=0.3)

# 범례 표시
plt.legend()
# 그래프 표시
plt.show()
# 그래프 저장
plt.savefig('Time-domain_PCA.png')