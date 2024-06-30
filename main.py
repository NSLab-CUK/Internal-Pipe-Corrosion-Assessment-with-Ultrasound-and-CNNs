import os
import gc
import argparse  # commandline arguments parsing
import torch
import torch.nn as nn  # Layer, Active function, loss function
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from training_mmloss import Trainer
from data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import random
from utils import plot_confusion_matrix
from utils import get_data_list
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from model.ResNet_swish1_weightI import ResNet, ResidualBlock
from model.DenseNet import DenseNet
from model.EfficientNet import EfficientNet
from model.InceptionNet import Inception3, InceptionOutputs
from model.VGG import VGG
from model.AlexNet import AlexNet

"""
여러 파일 생성을 위한 저장위치 변경
"""


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


labels = [i + 1 for i in range(6)]


# torch.set_float32_matmul_precision("high")


config = argparse.ArgumentParser()  # commandline arguments parsing

# arguments parsing
config.add_argument("--batch_size", default=8, type=int)  # 3번에서 256->512
config.add_argument("--num_workers", default=4, type=int)
config.add_argument(
    "--lr", default=0.0005, type=float
)  # 2번에서 임시 10배 # 3번에서 원래대로
config.add_argument("--gpus", default="0")
config.add_argument("--epoch", default=200, type=int)
config.add_argument("--patience", default=50, type=int)  # 갑자기 20-> 50으로 바꿨음
config.add_argument("--train_samples", default=8000, type=int)
config.add_argument("--val_samples", default=2000, type=int)
config.add_argument("--num_classes", default=6, type=int)
config.add_argument("--model", default="resnet", type=str)
config.add_argument("--num", default=1, type=int)

config = config.parse_args()

config = vars(config)  # 딕셔너리 변환

# 하나의 GPU를 사용하는 경우 무조건 0번에만 할당되는 문제에 대한 실험적 코드
# torch.cuda.set_device(config['gpus'][0])
# device = torch.device(f"cuda:{config['gpus'][0]}" if torch.cuda.is_available() else "cpu")


num = config["num"]
plot = f"6class/{num}/plot"
output = f"6class/{num}/output"
logs = f"6class/{num}/logs"
confusion = f"6class/{num}/confusion_matrix"
fix_seed(num)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # usable deive setting

config["gpus"] = [
    int(x) for x in config["gpus"].strip("[]").split(",")
]  # config['gpus'] 를 리스트형태로 변환

k_folds = 5  # 왜 K-fold?

print("loading data ...")

# original code
# for label in labels:  # labels:1~30
#     for data_name in os.listdir(f"./data4/{label}"):
#         data_name_list.append(data_name)
#         y.append(label)


specto_data_path = "./data/spectrogram/6class_slice_256piece_tol_ratio_0.4_sftf_n_fft256_hop_length2_win_length256"
# specto_data_path = "./data/data_specto/JL_230829_ML6"
data_name_list, data_path_list, y = get_data_list(specto_data_path)
print(len(data_name_list))
print(len(data_path_list))
print(len(y))

print(f"get {len(data_name_list)} random files")
print("loading data from directory ...")

# np.load 대신 pd.read_csv 사용하여 데이터 불러오기
x = np.stack(
    [pd.read_csv(data_path).to_numpy() for data_path in tqdm(data_path_list)],
    axis=0,
)


def add_gaussian_noise_np(x, mean=0.0, std=1.0):
    """
    NumPy 배열 데이터에 가우시안 노이즈를 추가합니다.

    파라미터:
    - x: NumPy 배열 데이터
    - mean: 노이즈의 평균
    - std: 노이즈의 표준편차

    반환값:
    - 노이즈가 추가된 데이터
    """
    noise = np.random.normal(mean, std, x.shape)
    return x + noise


# x = np.expand_dims(x, 1)

print(f"get {len(x)} gaussian files")

print(f"done. tensor shape : {x.shape}")

device = torch.device("cuda")

models_dict = {  # 사전에 사용가능한 모델을 dictionary화
    "vgg": VGG,
    "alexnet": AlexNet,
    "resnet": ResNet,
    # "densenet": DenseNet,
    "efficientnet": EfficientNet,
    "inceptionnet": Inception3,
}
model_class = models_dict.get(config["model"].lower())
if not model_class:
    raise ValueError(f"Model {config['model']} not recognized")

if model_class == AlexNet or model_class == VGG or model_class == EfficientNet:
    model = model_class(num_class=config["num_classes"]).double().to(device)
elif model_class == Inception3:
    model = model_class(num_classes=config["num_classes"]).double().to(device)
elif model_class == ResNet:
    layers = [3, 4, 6, 3]  # ResNet-50에 대한 각 레이어의 블록 수
    block = ResidualBlock  # 사용할 residual block의 종류
    model = ResNet(block, layers, num_class=config["num_classes"]).double().to(device)

# model = model_class(num_class=config["num_classes"]).double().to(device)

if len(config["gpus"]) > 1:
    model = nn.DataParallel(model, device_ids=config["gpus"])
# 이부분 데이터 학습하기
trainer = Trainer(config, model, device)

# 데이터를 학습 데이터와 검증 데이터로 나눔
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # 가우시안 노이즈 추가
# x_noised = add_gaussian_noise_np(x_train, np.mean(x_train), np.std(x_train))
# # x_noised = add_gaussian_noise_np(X_train, 0, 0.08)
# # 원본 데이터와 노이즈가 추가된 데이터를 합침
# x_combined = np.concatenate((x_train, x_noised), axis=0)
#
# # 레이블도 동일하게 복제하여 합침
# y_combined = np.concatenate((y_train, y_train), axis=0)
# x_train, y_train = shuffle(x_combined, y_combined, random_state=0)


stopping_check = torch.inf
patience = 0
checkpoint_score = 0

writer = SummaryWriter(
    "./{}/{}_batch{}_lr{}".format(
        logs,
        config["model"],
        config["batch_size"],
        config["lr"],
    )
)

# print(x_train)
print(len(x_train))
# print(x_combined)
for epoch in range(config["epoch"]):
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_result = trainer.train(epoch, train_dataset)
    # print(train_result)
    eval_result, c_mat = trainer.eval(epoch, test_dataset)

    # tensorboard for training result
    writer.add_scalar("loss/train", train_result["loss"], epoch)
    writer.add_scalar("acc/train", train_result["acc"], epoch)
    writer.add_scalar("precision/train", train_result["precision"], epoch)
    writer.add_scalar("recall/train", train_result["recall"], epoch)
    writer.add_scalar("f1-score/train", train_result["f1"], epoch)

    # tensorboard for validation result
    writer.add_scalar("loss/val", eval_result["loss"], epoch)
    writer.add_scalar("acc/val", eval_result["acc"], epoch)
    writer.add_scalar("precision/val", eval_result["precision"], epoch)
    writer.add_scalar("recall/val", eval_result["recall"], epoch)
    writer.add_scalar("f1-score/val", eval_result["f1"], epoch)

    # tensorboard for confusion matrix
    ax = plot_confusion_matrix(c_mat.cpu().numpy())
    cm = ax.get_figure()
    if not os.path.exists(f"./{confusion}"):
        os.makedirs(f"./{confusion}")

    plt.savefig(
        "./{}/epoch{}_{}_batch{}_lr{}_acc{:.2f}.png".format(
            confusion,
            epoch,
            config["model"],
            config["batch_size"],
            config["lr"],
            eval_result["acc"],
        )
    )
    writer.add_figure("Confusion Matrix", cm, epoch)

    print(f"{epoch} train result:", train_result)
    print(f"{epoch} val result:", eval_result)
    if stopping_check < eval_result["loss"]:
        patience += 1

    stopping_check = eval_result["loss"]
    if not os.path.exists(f"./{output}"):
        os.makedirs(f"./{output}")
    if checkpoint_score < eval_result["acc"]:
        torch.save(
            trainer.model.state_dict(),
            "./{}/epoch{}_{}_batch{}_lr{}_acc{:.2f}.ckpt".format(
                output,
                epoch,
                config["model"],
                config["batch_size"],
                config["lr"],
                eval_result["acc"],
            ),
        )
        torch.save(
            c_mat,
            "./{}/epoch{}_{}_batch{}_lr{}_acc{:.2f}.pth".format(
                confusion,
                epoch,
                config["model"],
                config["batch_size"],
                config["lr"],
                eval_result["acc"],
            ),
        )

    if patience == config["patience"]:
        print("early stopping at", epoch)
        break

    writer.close()

    # del x_train, x_test, y_train, y_test
    gc.collect()
