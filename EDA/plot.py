import os
import argparse  # commandline arguments parsing
import pandas as pd
import matplotlib.pyplot as plt

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

# os.makedirs(path, exist_ok=True)


file_path_arr = [[] for _ in range(30)]


for idx, class_dirs in enumerate(os.listdir(ROOT_DATA_DIR)):
    class_dir_path = os.path.join(ROOT_DATA_DIR, class_dirs)
    for data in os.listdir(class_dir_path):
        file_path = os.path.join(class_dir_path, data)
        file_path_arr[idx].append(file_path)

# 원하는 클래스 지정
class_idx = 2

plt.figure(figsize=(10, 6))
# 원하는 클래스의 모든 파일을 순회
for file_path in file_path_arr[class_idx]:
    # CSV 파일을 DataFrame으로 읽기
    df = pd.read_csv(file_path, index_col='Time')

    # 인덱스가 시간인 경우, 시간 순서대로 정렬
    df = df.sort_index()

    # DataFrame의 모든 열을 그림
    for col in df.columns:
        plt.plot(df.index, df[col], alpha = 0.12)

plt.title(f"Time Series Data for Class {class_idx}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()