import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from tqdm import tqdm

def plot_confusion_matrix(cf_matrix):
    classes = [
        "0_output",
        "50_output",
        "100_output",
        "150_output",
        "200_output",
        "250_output",
    ]

    # classes = [
    #     "Control Group",
    #     "Hypoglycemia",
    #     "Normal",
    #     "Impaired fasting glucose",
    #     "Diabetes mellitus",
    # ]
    # classes=[]
    # data_path = "./data/data_specto/JL_230829_ML6"
    # clss = os.listdir(data_path)
    # for cls in clss:
    #     classes.append(cls)

    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)
    sn.set_context(font_scale=1)
    cm_numpy = cf_matrix
    df_cm = pd.DataFrame(
        cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
        index=classes,
        columns=classes,
    )
    
    return sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 10}, cbar=True)


def get_data_list(data_path):
    data_name_list = []
    y = []
    data_path_list = []
    # data_path = "./data/data_specto/JL_230829_ML6"
    clss = os.listdir(data_path)
    for cls in clss:
        print(f"class {clss.index(cls)} : {cls}")
        labels = os.listdir(f"{data_path}/{cls}")
        # print(cls)
        for data_name in os.listdir(f"{data_path}/{cls}"):
            data_name_list.append(data_name)
            data_path_list.append(f"{data_path}/{cls}/{data_name}")

            y.append(clss.index(cls))
    print(f"find {len(data_name_list)} files")
    return [data_name_list, data_path_list, y]


if __name__ == "__main__":
    classes = [
        "0mg/dL",
        "50mg/dL",
        "60mg/dL",
        "70mg/dL",
        "75mg/dL",
        "80mg/dL",
        "85mg/dL",
        "90mg/dL",
        "95mg/dL",
        "100mg/dL",
        "105mg/dL",
        "110mg/dL",
        "115mg/dL",
        "120mg/dL",
        "125mg/dL",
        "130mg/dL",
        "135mg/dL",
        "140mg/dL",
        "145mg/dL",
        "150mg/dL",
        "160mg/dL",
        "170mg/dL",
        "180mg/dL",
        "190mg/dL",
        "200mg/dL",
        "210mg/dL",
        "220mg/dL",
        "230mg/dL",
        "240mg/dL",
        "250mg/dL",
    ]
    folder_path = ".\\fail\\alexnet66b64s6(10.24)\\confusion_matrix"
    save_folder_path = ".\\fail\\alexnet66b64s6(10.24)\\confusion_matrix_remake"

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
        cax = sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 10},
                         cbar=True)  # heatmap의 숫자 글자 크기 조절

        # 컬러바 객체 얻기
        cbar = cax.collections[0].colorbar

        # 컬러바 눈금 라벨의 글자 크기 설정
        cbar.ax.tick_params(labelsize=18)

        plt.xticks(fontsize=18)  # x축 틱 레이블의 글자 크기를 14로 설정
        plt.yticks(fontsize=18)  # y축 틱 레이블의 글자 크기를 14로 설정

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
