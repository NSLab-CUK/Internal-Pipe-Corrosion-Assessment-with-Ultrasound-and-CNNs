import torch
import numpy as np
import os
from tqdm import tqdm

from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y):
        super(TensorDataset, self).__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])

        return {"input": x, "label": y}


if __name__ == "__main__":
    DATA_ROOT = "C:/minseo/Water_qaulity_measurement/specto"
    file_path = [
        "1mg_50mL",
        "1mg_1L",
        "5mg_50mL",
        "5mg_1L",
        "15mg_15mL",
        "15mg_1L",
    ]  # not include '15mg_15mL','15mg_1L' file

    tensor3d = torch.Tensor()  # 빈 텐서 생성
    cnt = 1
    class_np = np.array([])
    for i in range(len(file_path)):  # 'i' use class number
        # print(f'class:{i}')
        absolute_folder_path = os.path.join(DATA_ROOT, file_path[i])
        for root, directories, files in os.walk(absolute_folder_path):
            # print(tensor3d.shape)
            if not files:  # files==[] 일때
                continue
            for file in tqdm(files, desc=f"{cnt}/80, DIR:{root}, {tensor3d.shape}"):
                if file == "Unnamed":  # 안에 필요없는 Unnamed파일을 제외함
                    continue
                # print(os.path.join(root,file))
                np_loaded = np.load(os.path.join(root, file))
                ts_loaded = torch.Tensor(np_loaded)
                ts_loaded = ts_loaded.unsqueeze(0)  # dim 추가
                # print(tensor3d.shape)
                tensor3d = torch.concat([tensor3d, ts_loaded], dim=0)

                class_np = np.append(class_np, [i])
            print(f"class_np shape: {class_np.shape}")
            cnt += 1

        # x=np.load(os.path.join(DATA_ROOT, file_path[i]))

    # x1 = torch.rand(513, 1357)
    # x2 = torch.rand(513, 1357)
    # x1 = x1.unsqueeze(0)
    # x2 = x2.unsqueeze(0)
    #
    # x = torch.concat([x1, x2], dim=0)
    #
    # print(x.shape)
    #
    # y = torch.tensor([0, 2])
    #
    # print(y)
    # print(y.shape)
