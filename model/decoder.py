# troch imports
import torch
from torch import nn


from .modules.SBM import SBM
from .modules.frwkv import FRWKV
from .modules.BaseUnit import Upsample


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # stage 3
        self.up3 = Upsample(self.dim*8)
        self.enhanceBlock3 = FRWKV(patch_size=3, in_channels=self.dim*4, embed_dims=self.dim*4, depth=3)
        
        # stage 2
        self.up2 = Upsample(self.dim*4)
        self.enhanceBlock2 = FRWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*2, depth=3)
        
        # stage 1
        self.up1 = Upsample(self.dim*2)
        self.enhanceBlock1 = FRWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=2)

    
        self.tail = nn.Conv2d(self.dim, 3, kernel_size=3, stride=1, padding=1, bias=False)
        

        self.fuse3 = SBM(self.dim*4)
        self.fuse2 = SBM(self.dim*2)
        self.fuse1 = SBM(self.dim)
        


    def forward(self, x, encode_list):
        feat_1s, feat_2s, feat_4s = encode_list[0], encode_list[1], encode_list[2]
        
        # stage 3 
        x3 = self.fuse3(self.up3(x), feat_4s)
        x3 = self.enhanceBlock3(x3)

        # stage 2
        x2 = self.fuse2(self.up2(x3), feat_2s)
        x2 = self.enhanceBlock2(x2)

        # stage 1
        x1 = self.fuse1(self.up1(x2), feat_1s)
        x1 = self.enhanceBlock1(x1)

        out = self.tail(x1)

        return out

class Basefuse(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.zip = nn.Conv2d(dim*2, dim, 1, 1, 0)

    def forward(self, x_dec, x_enc):

        out = self.zip(torch.cat([x_dec, x_enc], dim=1))

        return out