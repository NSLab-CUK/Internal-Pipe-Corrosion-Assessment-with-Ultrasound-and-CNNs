import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(16) ,
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.drop = nn.Dropout(0.5)

        self.linear1 = nn.Linear(53760, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            for i in layer:
                self.init_weight(i)

        for linear in [self.linear1, self.linear2, self.linear3, self.linear4]:
            self.init_weight(linear)

    def init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight.data)
            torch.nn.init.zeros_(module.bias.data)

        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight.data)
            torch.nn.init.zeros_(module.bias.data)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = torch.relu(x)
        # x = self.drop(x)
        x = self.linear2(x)
        x = torch.relu(x)
        # x = self.drop(x)
        x = self.linear3(x)
        x = torch.relu(x)
        # x = self.drop(x)
        x = self.linear4(x)

        return x


##########################################################################################


if __name__ == "__main__":
    from pprint import PrettyPrinter

    pp = PrettyPrinter()

    model = ResNet(ResidualBlock, [3, 4, 6, 9])

    model = nn.DataParallel(model, device_ids=[0])

    model.load_state_dict(
        torch.load("./output/resnet_fold1_batch64_lr5e-06_sampleall4_acc0.89.ckpt")
    )
