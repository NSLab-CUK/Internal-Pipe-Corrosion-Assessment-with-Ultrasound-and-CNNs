import torch
import torch.nn as nn

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-excitation module
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=4):
        super(SEBlock, self).__init__()
        self.se_reduce = nn.Conv2d(in_channels, in_channels // se_ratio, kernel_size=1, stride=1, padding=0, bias=True)
        self.se_expand = nn.Conv2d(in_channels // se_ratio, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        se_tensor = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        se_tensor = self.se_expand(torch.relu(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x

# Mobile inverted bottleneck block (MBConv)
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand = in_channels != out_channels
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ]
        # Depthwise convolution phase
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ]
        # Squeeze-and-excitation phase
        layers += [SEBlock(hidden_dim, se_ratio)]
        # Output phase
        layers += [
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.block(x)
        else:
            out = self.block(x)
        print(f"After MBConv block : ", 'x'.join(map(str, out.shape[1:])))
        return out

# EfficientNet model
class EfficientNet(nn.Module):
    def __init__(self, num_class=30):
        super(EfficientNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        # Define MBConv blocks
        self.blocks = nn.Sequential(
            MBConv(32, 16, 3, 1, 1, 4),
            MBConv(16, 24, 3, 2, 6, 4),
            MBConv(24, 24, 3, 1, 6, 4),
            MBConv(24, 40, 5, 2, 6, 4),
            MBConv(40, 40, 5, 1, 6, 4),
            MBConv(40, 80, 3, 2, 6, 4),
            MBConv(80, 80, 3, 1, 6, 4),
            MBConv(80, 80, 3, 1, 6, 4),
            MBConv(80, 112, 5, 1, 6, 4),
            MBConv(112, 112, 5, 1, 6, 4),
            MBConv(112, 192, 5, 2, 6, 4),
            MBConv(192, 192, 5, 1, 6, 4),
            MBConv(192, 192, 5, 1, 6, 4),
            MBConv(192, 320, 3, 1, 6, 4)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_class)
        )

        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(1280)
        self.swish_head = Swish()
        self.avgpool_head = nn.AdaptiveAvgPool2d(1)
        self.flatten_head = nn.Flatten()
        self.dropout_head = nn.Dropout(0.2)
        self.linear_head = nn.Linear(1280, num_class)

    def forward(self, x):
        x = self.stem(x)
        print("After stem: ", 'x'.join(map(str, x.shape[1:])))
        x = self.blocks(x)
        # x = self.head(x)
        x = self.conv_head(x)
        print("After conv_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.bn_head(x)
        print("After bn_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.swish_head(x)
        print("After swish_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.avgpool_head(x)
        print("After avgpool_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.flatten_head(x)
        print("After flatten_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.dropout_head(x)
        print("After dropout_head: ", 'x'.join(map(str, x.shape[1:])))
        x = self.linear_head(x)
        print("After linear_head: ", 'x'.join(map(str, x.shape[1:])))
        return x

# Create EfficientNet model
# model = EfficientNet()
