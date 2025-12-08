import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torchvision.ops as ops


## Calculate Similarity
def compute_similarity(identity_tensor, k=3, dilation=1, sim='cos'):

    B, C, H, W = identity_tensor.shape
    unfold_tensor = F.unfold(identity_tensor, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)

    # compute similarity
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    similarity = similarity.view(B, k * k - 1, H, W)

    return similarity

## The Conv2d version 
@torch.no_grad()
def _build_extract_kernels(k, C, device, dtype):

    eye = torch.eye(k * k, device=device, dtype=dtype).view(k * k, 1, 1, k * k)
    eye = eye.view(k * k, 1, k, k)  # (k*k, 1, k, k)

    weight = eye.unsqueeze(0).expand(C, -1, -1, -1, -1)
    weight = weight.reshape(C * k * k, 1, k, k).contiguous()
    
    return weight

def compute_similarity_conv2d(x, k=3, dilation=1, sim='cos'):
    B, C, H, W = x.shape
    if k % 2 == 0:
        raise ValueError(f"Only odd kernel sizes are supported, got even k={k}")
    
    weight = _build_extract_kernels(k, C, x.device, x.dtype)
    patches = F.conv2d(
        x, weight, bias=None,
        padding=(k // 2) * dilation,
        dilation=dilation,
        groups=C
    )  # Shape: (B, C*k*k, H, W)
    patches = patches.view(B, C, k * k, H, W)
    center = patches[:, :, k * k // 2:k * k // 2 + 1, ...]
    
    if sim == 'cos':
        similarity = F.cosine_similarity(center, patches, dim=1)
    elif sim == 'dot':
        similarity = (center * patches).sum(dim=1)
    else:
        raise NotImplementedError(f"Similarity metric '{sim}' is not implemented. Use 'cos' or 'dot' instead.")
    
    sim_left = similarity[:, :k * k // 2, ...]
    sim_right = similarity[:, k * k // 2 + 1:, ...]
    similarity = torch.cat((sim_left, sim_right), dim=1)

    return similarity


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DQShift(nn.Module):
    def __init__(self, dim, layer_id, shift_pixel=1, groups=4, kernel_size=3, dilation=1, local_window=3, use_sim=False):
        super(DQShift, self).__init__()

        self.layer_id = layer_id
        self.groups = groups
        self.local_window = local_window
        self.use_sim = use_sim

        if self.layer_id==0 and self.use_sim:
            in_ch = dim + local_window**2 - 1
        else:
            in_ch = dim

        self.offset = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, groups*2, 1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_ch, groups*2, 1),
            nn.Sigmoid(),
        )
        normal_init(self.offset[-1], std=0.001)
        constant_init(self.weight[0], val=0.)

        shifts = torch.tensor([
            [0., -shift_pixel],
            [0., +shift_pixel],
            [-shift_pixel, 0.],
            [+shift_pixel, 0.],
        ]).transpose(0, 1).view(1, 2, 4, 1, 1)
        self.register_buffer("shifts", shifts, persistent=False)

        # coords 缓存
        self._coords = None
        self._coords_key = None

    def get_coords(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if self._coords is None or self._coords_key != key:
            coords_h = torch.arange(H, device=device, dtype=dtype) + 0.5
            coords_w = torch.arange(W, device=device, dtype=dtype) + 0.5
            coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0) # [1,2,H,W] -> [1,2,1,H,W]
            self._coords = coords
            self._coords_key = key
        return self._coords

    def forward(self, x, offset_list, patch_resolution=None):
        B, N, C = x.shape
        H, W = patch_resolution[0], patch_resolution[1]
        x = x.transpose(1, 2).reshape(B, C, H, W)
  

        # offset prediction
        if self.layer_id==0 and self.use_sim:
            x_sim = compute_similarity(x, self.local_window, dilation=2, sim='cos')
            x = torch.cat([x, x_sim], dim=1)

        offset = self.offset(x)
        weight = self.weight(x)
        offset = (offset * weight).view(B, 2, -1, H, W)
        offset = offset + self.shifts.to(dtype=x.dtype, device=x.device)

        # coordinate
        coords = self.get_coords(H, W, x.device, x.dtype)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        # feature sample
        x_sample = F.grid_sample(x.reshape(B*self.groups, -1, H, W), 
                                 coords, 
                                 mode='bilinear',
                                 align_corners=False, 
                                 padding_mode="border").view(B, -1, H, W)
        x_sample = x_sample.flatten(2).transpose(1, 2)

        return x_sample, offset_list