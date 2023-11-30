import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.model(inputs)


class ConvDown(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.model(inputs)


class ConvUp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        )

    def forward(self, inputs):
        return self.model(inputs)


class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk0 = ConvBlock(3, 64)
        self.down0 = ConvDown(64)
        self.blk1 = ConvBlock(64, 128)
        self.down1 = ConvDown(128)
        self.blk2 = ConvBlock(128, 256)
        self.down2 = ConvDown(256)
        self.blk3 = ConvBlock(256, 512)
        self.down3 = ConvDown(512)
        self.blk4 = ConvBlock(512, 1024)

    def forward(self, inputs):
        f0 = self.blk0(inputs)
        d0 = self.down0(f0)
        f1 = self.blk1(d0)
        d1 = self.down1(f1)
        f2 = self.blk2(d1)
        d2 = self.down2(f2)
        f3 = self.blk3(d2)
        d3 = self.down3(f3)
        f4 = self.blk4(d3)
        return f0, f1, f2, f3, f4


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = ConvUp(1024)
        self.blk3 = ConvBlock(1024, 512)
        self.up2 = ConvUp(512)
        self.blk2 = ConvBlock(512, 256)
        self.up1 = ConvUp(256)
        self.blk1 = ConvBlock(256, 128)
        self.up0 = ConvUp(128)
        self.blk0 = ConvBlock(128, 64)
        self.last_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, inputs):
        f0, f1, f2, f3, f4 = inputs
        u3 = self.up3(f4)
        df2 = self.blk3(torch.concat((f3, u3), dim=1))
        u2 = self.up2(df2)
        df1 = self.blk2(torch.concat((f2, u2), dim=1))
        u1 = self.up1(df1)
        df0 = self.blk1(torch.concat((f1, u1), dim=1))
        u0 = self.up0(df0)
        f = self.blk0(torch.concat((f0, u0), dim=1))
        return torch.tanh(self.last_conv(f))


class ReConstructionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, inputs):
        fs = self.encoder(inputs)
        return self.decoder(fs)
