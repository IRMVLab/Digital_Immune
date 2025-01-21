import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)
        return x

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBnSiLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            SiLU()
        )

    def forward(self, x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0)
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels // 2, in_channels // 2, 1, 1, 0),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0)
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        x = self.channel_shuffle(x)
        return x

class ResidualDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0)
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 2, 1, groups=out_channels // 2),
            nn.BatchNorm2d(out_channels // 2),
            ConvBnSiLu(out_channels // 2, out_channels // 2, 1, 1, 0)
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.channel_shuffle(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for _ in range(3)],
            ResidualBottleneck(in_channels, out_channels // 2)
        )
        self.conv1 = ResidualDownsample(out_channels // 2, out_channels)

    def forward(self, x):
        x_shortcut = self.conv0(x)
        x = self.conv1(x)
        return [x, x_shortcut]

class Unet_t(nn.Module):
    def __init__(self, in_channels=3, base_dim=32, dim_mults=[2, 4, 8, 16]):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)
        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1]) for c in channels])
        self.mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2)
        )
        self.final_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1][1] // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.init_conv(x)
        for encoder_block in self.encoder_blocks:
            x, _ = encoder_block(x)
        x = self.mid_block(x)
        x = self.final_fc(x)
        return x.squeeze(1)

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        return channels

if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = Unet_t()
    y = model(x)
    print(y.shape)
