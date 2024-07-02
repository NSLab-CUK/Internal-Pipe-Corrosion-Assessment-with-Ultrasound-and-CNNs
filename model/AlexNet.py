import pprint

import torch
import torch.nn as nn #Layer, Active function, loss function
import numpy as np


class AlexNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.feature = nn.Sequential(
            #input channel: 1, output channel: 96, stride=4, kernel_size=11
            nn.Conv2d(1, 96, stride=4, kernel_size=11),
            # kernel_size=3 max pooling layer
            nn.MaxPool2d(3, stride=2),
            #
            nn.Conv2d(96, 256, padding=2, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, padding=1, kernel_size=3),
            nn.Conv2d(384, 384, padding=1, kernel_size=3),
            nn.Conv2d(384, 256, padding=1, kernel_size=3),
            nn.MaxPool2d(3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(157440, 3840),
            nn.Linear(3840, 3840),
            nn.Linear(3840, num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = AlexNet(num_class=6)
    inputs = torch.rand(513, 1357).unsqueeze(0).unsqueeze(0)
    print(inputs.shape)

    output = model(inputs)

    print(output.shape)

