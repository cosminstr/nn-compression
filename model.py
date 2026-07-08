import torch
import torch.nn as nn
# Based on a residual U-NET


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_c, out_channels=out_c, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_c, out_channels=out_c, kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x1))
        x = x + x1

        return x


class DownConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(in_c, out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.downsample(x)
        x = self.doubleconv(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv = nn.Conv2d(
            in_channels=in_c, out_channels=int(in_c / 2), kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv(x))

        return x


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.doubleconv1 = DoubleConv(3, 64)
        self.downconv1 = DownConv(64, 128)
        self.downconv2 = DownConv(128, 256)
        self.downconv3 = DownConv(256, 512)
        self.downconv4 = DownConv(512, 1024)

        # Decoder
        self.upconv1 = UpConv(1024)
        self.doubleconv2 = DoubleConv(1024, 512)

        self.upconv2 = UpConv(512)
        self.doubleconv3 = DoubleConv(512, 256)

        self.upconv3 = UpConv(256)
        self.doubleconv4 = DoubleConv(256, 128)

        self.upconv4 = UpConv(128)
        self.doubleconv5 = DoubleConv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.doubleconv1(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)

        x = self.upconv1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.doubleconv2(x)

        x = self.upconv2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.doubleconv3(x)

        x = self.upconv3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.doubleconv4(x)

        x = self.upconv4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.doubleconv5(x)

        x = self.relu(self.out(x))

        return x
