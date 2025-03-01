import torch
from torch import nn
import torchvision.models as models


class ResNetUNet(nn.Module):
    def __init__(self, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        # Используем ResNet34 в качестве энкодера
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.base_layers = list(resnet.children())

        self.in_layer = nn.Sequential(*self.base_layers[:3])  # conv1 + bn1 + relu
        self.layer1 = self.base_layers[4]  # resnet layer1
        self.layer2 = self.base_layers[5]  # resnet layer2
        self.layer3 = self.base_layers[6]  # resnet layer3
        self.layer4 = self.base_layers[7]  # resnet layer4

        # Декодер (Up-блоки)
        self.up1 = _Up(512, 256, bilinear)
        self.up2 = _Up(256, 128, bilinear)
        self.up3 = _Up(128, 64, bilinear)
        self.up4 = _Up(64, 64, bilinear)

        # Выходной слой
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
