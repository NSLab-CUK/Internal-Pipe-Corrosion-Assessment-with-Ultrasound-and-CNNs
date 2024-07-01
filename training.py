import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics.classification import (
    ConfusionMatrix,
    Recall,
    Precision,
    Accuracy,
    F1Score,
)

from model.ResNet import ResNet
from model.DenseNet import DenseNet
from model.EfficientNet import EfficientNet
from model.InceptionNet import Inception3, InceptionOutputs
from model.VGG import VGG
from model.AlexNet import AlexNet


class Trainer:
    def __init__(self, config, model, device):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.model = model

        self.device = device

        self.optim = Adam(lr=config["lr"], params=self.model.parameters())

        self.acc = Accuracy(
            num_classes=self.config["num_classes"], average="macro", task="multiclass"
        ).to(device)
        self.precision = Precision(
            num_classes=self.config["num_classes"], average="macro", task="multiclass"
        ).to(device)
        self.recall = Recall(
            num_classes=self.config["num_classes"], average="macro", task="multiclass"
        ).to(device)
        self.f1 = F1Score(
            num_classes=self.config["num_classes"], average="macro", task="multiclass"
        ).to(device)

        self.c_mat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["num_classes"]
        ).to(device)

        # add gpt
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        #--

    def train(self, epoch, train_dataset: Dataset):
        self.model.train()
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        data_iter = tqdm(
            train_loader,
            desc=f"EP:{epoch}_train",
            total=len(train_loader),
            bar_format="{l_bar}{r_bar}",
        )

        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        avg_f1 = []

        for idx, batch in enumerate(data_iter):
            if isinstance(self.model, nn.DataParallel):
                primary_device = f"cuda:{self.model.device_ids[0]}"
            else:
                primary_device = self.device

            batch = {k: v.to(primary_device) for k, v in batch.items()}

            # batch = {k: v.to(self.device) for k, v in batch.items()}

            # batch = {
            #     k: v.to(f"cuda:{self.model.device_ids[0]}") for k, v in batch.items()
            # }

            self.optim.zero_grad()
            batch["input"] = batch["input"].unsqueeze(1)

            output = self.model(batch["input"])

            if isinstance(output, InceptionOutputs):
                loss1 = self.criterion(output.logits, batch["label"].long())
                loss2 = self.criterion(output.aux_logits, batch["label"].long())
                loss = loss1 + 0.4 * loss2
                preds = torch.argmax(output.logits, dim=1)
            else:
                loss = self.criterion(output, batch["label"].long())
                preds = torch.argmax(output, dim=1)


            acc = self.acc(preds, batch["label"])
            recall = self.recall(preds, batch["label"])
            precision = self.precision(preds, batch["label"])
            f1 = self.f1(preds, batch["label"])

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            loss.backward()
            self.optim.step()

            post_fix = {
                "loss": loss.item(),
                "acc": acc.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "f1-score": f1.item(),
            }

            data_iter.set_postfix(post_fix)

        avg_loss = sum(avg_loss) / len(avg_loss)
        avg_acc = sum(avg_acc) / len(avg_acc)
        avg_precision = sum(avg_precision) / len(avg_precision)
        avg_recall = sum(avg_recall) / len(avg_recall)
        avg_f1 = sum(avg_f1) / len(avg_f1)

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }

    def eval(self, epoch, val_dataset):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"])
        data_iter = tqdm(val_loader, desc=f"EP:{epoch}_valid", total=len(val_loader))

        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        avg_f1 = []

        c_mat = None
        for idx, batch in enumerate(data_iter):
            if isinstance(self.model, nn.DataParallel):
                primary_device = f"cuda:{self.model.device_ids[0]}"
            else:
                primary_device = self.device

            batch = {k: v.to(primary_device) for k, v in batch.items()}

            # batch = {k: v.to(self.device) for k, v in batch.items()}

            # batch = {
            #     k: v.to(f"cuda:{self.model.device_ids[0]}") for k, v in batch.items()
            # }

            with torch.no_grad():
                batch["input"] = batch["input"].unsqueeze(1)
                output = self.model(batch["input"])

                if isinstance(output, InceptionOutputs):
                    loss = self.criterion(output.logits, batch["label"].long())
                    preds = torch.argmax(output.logits, dim=1)
                else:
                    loss = self.criterion(output, batch["label"].long())
                    preds = torch.argmax(output, dim=1)



            loss = self.criterion(output, batch["label"].long())

            acc = self.acc(output, batch["label"])
            recall = self.recall(output, batch["label"])
            precision = self.precision(output, batch["label"])
            f1 = self.f1(output, batch["label"])

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            if c_mat is None:
                c_mat = self.c_mat(output, batch["label"])
            else:
                c_mat += self.c_mat(output, batch["label"])

        avg_loss = sum(avg_loss) / len(avg_loss)
        avg_acc = sum(avg_acc) / len(avg_acc)
        avg_precision = sum(avg_precision) / len(avg_precision)
        avg_recall = sum(avg_recall) / len(avg_recall)
        avg_f1 = sum(avg_f1) / len(avg_f1)

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }, c_mat


if __name__ == "__main__":
    print(torch.cuda.is_available())
