# troch imports
import torch
from torch import nn
from timm.layers import trunc_normal_
import torch.nn.functional as F

# own files import
from .encoder import Encoder
from .decoder import Decoder

# recursive network based on residual units
class DehazeNet(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.dim = dim

        self.encoder = Encoder(dim=self.dim) 
        self.decoder = Decoder(dim=self.dim)

        self.apply(self._init_weights)  # Correctly apply init_weights to all submodules

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outer_shortcut = x
        encode_list = self.encoder(x)
        x = encode_list[-1]
        x = self.decoder(x, encode_list)
        x=torch.add(x, outer_shortcut)

        return x