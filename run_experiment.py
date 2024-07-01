import os
import gc
import argparse  # commandline arguments parsing
import torch
import torch.nn as nn  # Layer, Active function, loss function
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from training import Trainer
from data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import random
from utils import plot_confusion_matrix
from utils import get_data_list
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.ResNet import ResNet, ResidualBlock
from model.DenseNet import DenseNet, Bottleneck
from model.EfficientNet import EfficientNet
from model.InceptionNet import Inception3, InceptionOutputs
from model.VGG import VGG
from model.AlexNet import AlexNet


labels = [i + 1 for i in range(30)]

# torch.set_float32_matmul_precision("high")


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


fix_seed(42)


config = argparse.ArgumentParser()  # commandline arguments parsing

# arguments parsing
config.add_argument("--batch_size", default=16, type=int)
config.add_argument("--num_workers", default=4, type=int)
config.add_argument("--lr", default=0.000005, type=float)
config.add_argument("--gpus", default=[0])
config.add_argument("--epoch", default=200, type=int)
config.add_argument("--patience", default=20, type=int)
config.add_argument("--train_samples", default=8000, type=int)
config.add_argument("--val_samples", default=2000, type=int)
config.add_argument("--num_classes", default=3, type=int)
config.add_argument("--model", default="efficientnet", type=str)

config = config.parse_args()

config = vars(config)  # 딕셔너리 변환

# 하나의 GPU를 사용하는 경우 무조건 0번에만 할당되는 문제에 대한 실험적 코드
# torch.cuda.set_device(config['gpus'][0])
# device = torch.device(f"cuda:{config['gpus'][0]}" if torch.cuda.is_available() else "cpu")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # usable deive setting

# config['gpus'] = [int(x) for x in config['gpus'].strip('[]').split(',')] #config['gpus'] 를 리스트형태로 변환

# print(torch.cuda.current_device())
torch.cuda.set_device(config["gpus"][0])
# torch.cuda.set_device(3) #메모리 할당할 GPU
# print(torch.cuda.current_device())


k_folds = 5  # 왜 K-fold?

print("loading data ...")

sss = StratifiedShuffleSplit(
    n_splits=k_folds, test_size=0.2, random_state=0
)  # sklearn 잘 이해가 되지않음...


# original code
# for label in labels:  # labels:1~30
#     for data_name in os.listdir(f"./data4/{label}"):
#         data_name_list.append(data_name)
#         y.append(label)


specto_data_path = "./data/specto"
data_name_list, data_path_list, y = get_data_list(specto_data_path)


# stride = 100
stride = 100
selected_data_name_list = []
selected_data_path_list = []
selected_y = []
for i in range(0, len(data_name_list), stride):
    select = np.random.randint(i, i + 100)
    selected_data_name_list.append(data_name_list[select])
    selected_data_path_list.append(data_path_list[select])
    selected_y.append(y[select])
data_name_list = selected_data_name_list
data_path_list = selected_data_path_list
y = selected_y
print(f"get {len(data_name_list)} random files")

print("loading data from directory ...")

# origin code
# x = np.stack(
#     [np.load(f"./data4/{y[idx]}/{name}") for idx, name in enumerate(data_name_list)],
#     axis=0,
# )

x = np.stack(
    [np.load(data_path) for data_path in tqdm(data_path_list)],
    axis=0,
)
# x = np.expand_dims(x, 1)

print(f"done. tensor shape : {x.shape}")


device = torch.device("cuda")

models_dict = {  # 사전에 사용가능한 모델을 dictionary화
    "vgg": VGG,
    "alexnet": AlexNet,
    "resnet": ResNet,
    "densenet": DenseNet,
    "efficientnet": EfficientNet,
    "inceptionnet": Inception3,
}
model_class = models_dict.get(config["model"].lower())
if not model_class:
    raise ValueError(f"Model {config['model']} not recognized")


for i, (train_index, test_index) in enumerate(sss.split(x, y)):
    if model_class == AlexNet or model_class == VGG or model_class == EfficientNet:
        model = model_class(num_class=config["num_classes"]).double().to(device)
    elif model_class == Inception3:
        model = model_class(num_classes=config["num_classes"]).double().to(device)
    elif model_class == ResNet:
        layers = [2, 2, 2, 2]  # ResNet-18에 대한 각 레이어의 블록 수
        block = ResidualBlock  # 사용할 residual block의 종류
        model = ResNet(block, layers, num_class=config["num_classes"]).double().to(device)
    elif model_class == DenseNet:
        block_DenseNet_60 = [3, 6, 12, 8]
        block_DenseNet_121 = [6, 12, 24, 16]
        model = DenseNet(Bottleneck, block_DenseNet_60, num_class=config["num_classes"]).double().to(device)
    # if type(config["gpus"]) == list: ##GPU 여러개 일때 대응

    if len(config["gpus"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpus"])
    # 이부분 데이터 학습하기
    trainer = Trainer(config, model, device)

    print("TRAIN:\n", train_index, "\nTEST:\n", test_index)
    X_train, X_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    print("=========================================")
    print("====== K Fold Validation step => %d/%d =======" % (i + 1, k_folds))
    print("=========================================")

    stopping_check = torch.inf
    patience = 0
    checkpoint_score = 0

    writer = SummaryWriter(
        "./logs/{}_fold{}_batch{}_lr{}".format(
            config["model"],
            i + 1,
            config["batch_size"],
            config["lr"],
            config["train_samples"],
        )
    )
    for epoch in range(config["epoch"]):
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_result = trainer.train(epoch, train_dataset)
        print(train_result)
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
        cm = plot_confusion_matrix(c_mat.cpu().numpy())

        # 컬러바 객체 얻기
        cbar = cm.collections[0].colorbar

        # 컬러바 눈금 라벨의 글자 크기 설정
        cbar.ax.tick_params(labelsize=18)

        plt.xticks(fontsize=18)   # x축 틱 레이블의 글자 크기를 14로 설정
        plt.yticks(fontsize=18)   # y축 틱 레이블의 글자 크기를 14로 설정


        plt.xlabel("Predicted labels", labelpad=20, fontsize=25)
        plt.ylabel("True labels", labelpad=20, fontsize=25)
        # plt.title("Confusion Matrix", fontsize=16)   # 제목 글자 크기 조절
        plt.tight_layout()

        if not os.path.exists("./confusion_matrix"):
            os.makedirs("./confusion_matrix")

        plt.savefig(
            "./confusion_matrix/epoch{}_{}_fold{}_batch{}_lr{}_acc{:.2f}.png".format(
                epoch,
                config["model"],
                i + 1,
                config["batch_size"],
                config["lr"],
                eval_result["acc"],
            )
        )
        writer.add_figure("Confusion Matrix", cm.get_figure(), epoch)

        print(f"{epoch} train result:", train_result)
        print(f"{epoch} val result:", eval_result)
        if stopping_check < eval_result["loss"]:
            patience += 1

        stopping_check = eval_result["loss"]
        if not os.path.exists("./output"):
            os.makedirs("./output")
        if checkpoint_score < eval_result["acc"]:
            torch.save(
                trainer.model.state_dict(),
                "./output/epoch{}_{}_fold{}_batch{}_lr{}_acc{:.2f}.ckpt".format(
                    epoch,
                    config["model"],
                    i + 1,
                    config["batch_size"],
                    config["lr"],
                    eval_result["acc"],
                ),
            )
            torch.save(
                c_mat,
                "./confusion_matrix/epoch{}_{}_fold{}_batch{}_lr{}_acc{:.2f}.pth".format(
                    epoch,
                    config["model"],
                    i + 1,
                    config["batch_size"],
                    config["lr"],
                    eval_result["acc"],
                ),
            )

        if patience == config["patience"]:
            print("early stopping at", epoch)
            break

    writer.close()

    del X_train, X_test, y_train, y_test
    gc.collect()
