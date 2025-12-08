import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, n_feat, use_act=False):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if use_act else nn.Identity()
            )
    
    def forward(self, x):

        x = self.body(x)

        return x

class Upsample(nn.Module):
    def __init__(self, n_feat, use_act=False):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, stride=1, padding=0, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if use_act else nn.Identity(),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.body(x)
        return x