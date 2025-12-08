import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init



class SBM(nn.Module):
    def __init__(self, dim, kernel_sizes=[3, 5, 7], dilations=[1, 1, 1], bias=False):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.num_branches = len(kernel_sizes)

        out_dim = sum(k ** 2 for k in kernel_sizes)
        self.dim = dim
        self.out_dim = out_dim

        # filter generator
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weights_proj = nn.Linear(dim, out_dim, bias=bias)
        # self.bn = nn.BatchNorm1d(dim * out_dim)
        self.act = nn.Softmax(dim=-1)    # nn.Sigmoid()
        nn.init.kaiming_normal_(self.weights_proj.weight, mode='fan_out', nonlinearity='relu')

        self.fuse = nn.Sequential(
            nn.Conv2d(dim*self.num_branches, dim*self.num_branches, kernel_size=3, stride=1, padding=1, groups=dim*self.num_branches, bias=bias),
            nn.Conv2d(dim*self.num_branches, self.num_branches, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Softmax(dim=1)
        )

        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.final_conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        
    
    def forward(self, x_dec, x_enc):
        B, C, H, W = x_enc.shape
        enc_vec = self.gap(x_enc).squeeze(-1).squeeze(-1)
        dec_vec = self.gap(x_dec).squeeze(-1).squeeze(-1)

        sim = torch.bmm(enc_vec.unsqueeze(2), dec_vec.unsqueeze(1))
        sim_flat = sim.view(B*C, C) 
        conv_weights = self.weights_proj(sim_flat)
        # conv_weights = self.bn(conv_weights.reshape(B, self.dim * self.out_dim))
        # conv_weights = conv_weights.reshape(B * self.dim, self.out_dim)

        branch_outs = []
        param_idx = 0
        for i in range(self.num_branches):
            kernel_size = self.kernel_sizes[i]
            dilation = self.dilations[i]
            param_size = kernel_size ** 2

            params = conv_weights[:, param_idx:param_idx+param_size]
            params = self.act(params)
            param_idx += param_size
            params = params.view(B*C, 1, kernel_size, kernel_size)
            x_enc_res = x_enc.reshape(1, B*C, H, W)
            branch_out = F.conv2d(
                x_enc_res,
                weight=params,
                stride=1,
                padding=((kernel_size - 1) // 2) * dilation,
                dilation=dilation,
                groups=B*C
            )
            branch_out = branch_out.view(B, C, H, W)
            branch_outs.append(branch_out)

        fused = torch.cat(branch_outs, dim=1)
        masks = self.fuse(fused)
        stacked = torch.stack(branch_outs, dim=1) 

        low_part = (stacked * masks.unsqueeze(2)).sum(dim=1)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(x_enc)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = (x_enc) * (self.lamb_h[None,:,None,None] + 1.) 

        x_enc_new = out_low + out_high
        
        final_out = self.final_conv(torch.cat([x_enc_new, x_dec], dim=1))

        return final_out