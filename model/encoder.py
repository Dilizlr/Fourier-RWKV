import torch
import torch.nn as nn

from .modules.frwkv import FRWKV
from .modules.BaseUnit import Downsample

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # self.patch_size = 4
        self.head = nn.Conv2d(3, self.dim, kernel_size=3, stride=1, padding=1, bias=False)

        # stage 1 
        self.enhanceBlock1 = FRWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=2)
        self.dw1 = Downsample(self.dim)


        # stage 2
        self.enhanceBlock2 = FRWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*2, depth=3)
        self.dw2 = Downsample(self.dim*2)

        # stage 3
        self.enhanceBlock3 = FRWKV(patch_size=3, in_channels=self.dim*4, embed_dims=self.dim*4, depth=3)
        self.dw3 = Downsample(self.dim*4)

        # latent stage
        self.enhanceBlock4 = FRWKV(patch_size=3, in_channels=self.dim*8, embed_dims=self.dim*8, depth=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.head(x) 

        # stage 1
        x1 = self.enhanceBlock1(x)
        x1_out = self.dw1(x1)  # C, H, W  
        
        # stage 2
        x2 = self.enhanceBlock2(x1_out) 
        x2_out = self.dw2(x2)  # 2C, H/2, W/2
     
        # stage3
        x3 = self.enhanceBlock3(x2_out)
        x3_out = self.dw3(x3)  # 4C, H/4, W/4

        # latent stage
        x4 = self.enhanceBlock4(x3_out) # 8C, H/8, W/8

        feat_list = [x1, x2, x3, x4]
        return feat_list