import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import argparse  # commandline arguments parsing
import librosa.display
import matplotlib.pyplot as plt
import re


def extract_number_from_filename(filename):
    return int(re.search(r'\d+', filename).group())


config = argparse.ArgumentParser()  # commandline arguments parsing

config.add_argument("--n_fft", default=256, type=int, help="FFT window size for STFT")
config.add_argument("--hop_length", default=2, type=int, help="Hop length for STFT")
config.add_argument("--win_length", default=256, type=int, help="Window length for STFT")

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

# 변수 지정
n_fft = config['n_fft']
hop_length = config['hop_length']
win_length = config['win_length']

# 디렉토리 지정
ROOT_DIR = "./../data"
DATA_FOLDER = "data_slice_256piece_tol_ratio_0.4"
DATA_DIR = os.path.join(ROOT_DIR, DATA_FOLDER)
ROOT_TARGET_DIR = os.path.join(ROOT_DIR, "spectrogram", f"{DATA_FOLDER}_sftf_n_fft{n_fft}_hop_length{hop_length}_win_length{win_length}")
ROOT_PNG_TARGET_DIR = os.path.join(ROOT_DIR, "spectrogram_png", f"{DATA_FOLDER}_sftf_n_fft{n_fft}_hop_length{hop_length}_win_length{win_length}")

# 폴더 생성
os.makedirs(ROOT_TARGET_DIR, exist_ok=True)
os.makedirs(ROOT_PNG_TARGET_DIR, exist_ok=True)

# 클래스 폴더 리스트 정렬한 리스트
# class_folder_list = sorted(os.listdir(DATA_DIR), key=extract_number_from_filename)

# 폴더내에 클래스 폴더 말고 오차 정렬파일도 있으므로 폴더만 반환하는 코드
class_folder_list = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
class_folder_list = sorted(class_folder_list, key=extract_number_from_filename)


for class_idx, class_folder in enumerate(class_folder_list):
    class_path = os.path.join(DATA_DIR, class_folder)
    sftf_class_path = os.path.join(ROOT_TARGET_DIR, class_folder)
    sftf_png_class_path = os.path.join(ROOT_PNG_TARGET_DIR, class_folder)
    os.makedirs(sftf_class_path, exist_ok=True)
    os.makedirs(sftf_png_class_path, exist_ok=True)

    # 클래스별 각 파일 순회
    for file_name in tqdm(os.listdir(class_path), desc=f"Scanning Files in {class_idx}/{len(class_folder_list)}", leave=False):
        file_path = os.path.join(class_path, file_name)
        df = pd.read_csv(file_path, index_col = 'Time')
        data_np = df.values.flatten()
        # print(data_np)

        D = librosa.stft(data_np, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        magnitude = np.abs(D)
        log_spectrogram = librosa.amplitude_to_db(magnitude)
        # log_spectrogram의 형태가 (256, 129, 1)인데, pandas.DataFrame은 2차원 데이터를 입력으로 받기 때문에 오류가 발생
        # log_spectrogram의 차원을 줄이려면, np.squeeze 함수를 사용하여 불필요한 차원을 제거
        log_spectrogram = np.squeeze(log_spectrogram)

        # 결과 저장 경로 설정
        target_specto_file_path = os.path.join(sftf_class_path, file_name)
        target_spectopng_file_path = os.path.join(sftf_png_class_path, f"{file_name[:-4]}.png")

        # CSV 파일로 저장
        pd.DataFrame(log_spectrogram).to_csv(target_specto_file_path, index=False, header=False)

        # 스펙트로그램 시각화 및 저장
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(log_spectrogram, sr=500, hop_length=hop_length, x_axis='time',
                                 y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram of {file_name}")
        plt.savefig(target_spectopng_file_path)
        plt.close()