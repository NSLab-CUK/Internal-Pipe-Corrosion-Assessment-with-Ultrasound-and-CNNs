import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from tqdm import tqdm


# def plot_confusion_matrix(cf_matrix):
#     classes = [
#         "15mg/1L",
#         # "15mg/50mL",
#         "1mg/1L",
#         # "1mg/50mL",
#         "5mg/1L",
#         # "5mg/50mL",
#     ]
#     # classes = [
#     #     "Control Group",
#     #     "Hypoglycemia",
#     #     "Normal",
#     #     "Impaired fasting glucose",
#     #     "Diabetes mellitus",
#     # ]
#     dpi_val = 68.84
#     plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)
#
#     sn.set_context(font_scale=1)
#     df_cm = pd.DataFrame(
#         cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=classes, columns=classes
#     )
#
#     return sn.heatmap(df_cm, annot=False).get_figure()

def plot_confusion_matrix(cf_matrix):
    classes = [
        # "15mg/1L",
        # "1mg/1L",
        # "5mg/1L",
        "15mg/50mL",
        "1mg/50mL",
        "5mg/50mL",
    ]
    # classes = [
    #     "Control Group",
    #     "Hypoglycemia",
    #     "Normal",
    #     "Impaired fasting glucose",
    #     "Diabetes mellitus",
    # ]
    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)
    sn.set_context(font_scale=1)
    cm_numpy = cf_matrix
    df_cm = pd.DataFrame(
        cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
        index=classes,
        columns=classes,
        )

    return sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 20}, cbar=True)


def get_data_list(data_path):
    data_name_list = []
    y = []
    data_path_list = []
    data_path = "./data/specto"
    clss = os.listdir(data_path)
    for cls in clss:
        print(f"class {clss.index(cls)} : {cls}")
        labels = os.listdir(f"./data/specto/{cls}")
        for label in labels:
            for data_name in os.listdir(f"./data/specto/{cls}/{label}"):
                if data_name == "Unnamed":
                    continue
                data_name_list.append(data_name)
                data_path_list.append(f"./data/specto/{cls}/{label}/{data_name}")
                y.append(clss.index(cls))
    print(f"find {len(data_name_list)} files")
    return [data_name_list, data_path_list, y]


if __name__ == "__main__":
    classes = [
        "1mg/50mL",
        "5mg/50mL",
        "15mg_50mL",
        # "1mg/1L",
        # "5mg/1L",
        # "15mg/1L",
    ]
    folder_path = "./output_inceptionnet/inceptionnet33_50ml/confusion_matrix"
    save_folder_path = "./output_inceptionnet/inceptionnet33_50ml/confusion_matrix_remake"

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # file_list = os.listdir(folder_path)
    file_list = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".pth")
    ]

    for file in tqdm(file_list, desc="draw:"):
        cm = torch.load(f"{folder_path}/{file}", map_location="cpu")

        dpi_val = 68.84
        plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)

        # Seaborn 설정
        sn.set_context(font_scale=1)
        cm_numpy = cm.cpu().numpy()  # Torch tensor를 Numpy 배열로 변환
        df_cm = pd.DataFrame(
            cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
            index=classes,
            columns=classes,
        )
        # Heatmap 그리기
        cax = sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 20}, cbar=True)  # heatmap의 숫자 글자 크기 조절

        # 컬러바 객체 얻기
        cbar = cax.collections[0].colorbar

        # 컬러바 눈금 라벨의 글자 크기 설정
        cbar.ax.tick_params(labelsize=18)

        plt.xticks(fontsize=18)   # x축 틱 레이블의 글자 크기를 14로 설정
        plt.yticks(fontsize=18)   # y축 틱 레이블의 글자 크기를 14로 설정


        plt.xlabel("Predicted labels", labelpad=20, fontsize=25)
        plt.ylabel("True labels", labelpad=20, fontsize=25)
        # plt.title("Confusion Matrix", fontsize=16)   # 제목 글자 크기 조절
        plt.tight_layout()


        save_path = os.path.join(save_folder_path, file.replace(".pth", ".png"))
        plt.savefig(save_path)

        # 그림을 닫아 리소스를 해제
        plt.close()

    #
    # cm = plot_confusion_matrix(c_mat.cpu().numpy())
    #
    # if not os.path.exists("./confusion_matrix_remake"):
    #     os.makedirs("./confusion_matrix_remake")
    #
    # plt.savefig(
    #     "./confusion_matrix/epoch{}_{}_fold{}_batch{}_lr{}_acc{:.2f}.png".format(
    #         epoch,
    #         config["model"],
    #         i + 1,
    #         config["batch_size"],
    #         config["lr"],
    #         eval_result["acc"],
    #     )
    # )
