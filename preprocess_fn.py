import numpy as np
import pandas as pd
import pickle
import os
import librosa
import time

def get_spectrogram_dn(data):
    amplitudes = data

    y = librosa.stft(amplitudes, n_fft=512, hop_length=4, win_length=32)
    #n_fft: fft point-주파수 1024개로 쪼개기
    # hop_length: shift size
    # win_length: frame length
    # 입력신호 길이 / hop_length= 출력 열수?
    # 출력 열은 특정시간 구간을 의미
    # 출력 행은 주파수정보를 의미, 1024지만 대칭이기때문에 513
    magnitude = np.abs(y)
    log_spectrogram = librosa.amplitude_to_db(magnitude)

    return log_spectrogram

def get_filePaths_arr(root):
    file_paths = []

    for sub_dirs in os.listdir(root):
        sub_dir_path = os.path.join(root, sub_dirs)
        for data in os.listdir(sub_dir_path):
            # file_path = os.path.join(sub_dir_path, data)
            file_path=[sub_dirs, data]
            file_paths.append(file_path)

    return file_paths #[sub_dirs, data],[sub_dirs, data],...

def set_folder(root, sub_dir1,sub_dir2):
    try:
        os.mkdir(root)
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(root,sub_dir1))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(root,sub_dir1,sub_dir2))
    except FileExistsError:
        pass



def test():
    for i in tqdm(range(100), desc="tqdm description"):
        time.sleep(0.01)

if __name__ == '__main__':
    test()