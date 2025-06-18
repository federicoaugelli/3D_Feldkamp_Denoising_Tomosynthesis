from torch import nn
from torch.nn import functional as F
import torch

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.residual_conv:
            residual = self.residual_conv(residual)

        out += residual
        return F.relu(out)

class ResUnet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            ResidualBlock3D(64, 64)
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            ResidualBlock3D(128, 128)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            ResidualBlock3D(256, 256)
        )

        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, output_padding=1)

        self.dec1 = ResidualBlock3D(256, 128)


        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)

        self.dec2 = ResidualBlock3D(128, 64)

        self.out = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.up1(b)

        depth_e2, height_e2, width_e2 = e2.shape[-3:]
        d1 = d1[:, :, :depth_e2, :height_e2, :width_e2]

        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)

        depth_e1, height_e1, width_e1 = e1.shape[-3:]
        d2 = d2[:, :, :depth_e1, :height_e1, :width_e1]

        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return torch.sigmoid(self.out(d2))
