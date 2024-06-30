import torch
import torch.nn as nn  # Layer, Active function, loss function
import numpy as np


class DenseNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.feature = nn.Sequential()

        self.classifier = nn.Sequential()

    def forward(self, x):
        return x
