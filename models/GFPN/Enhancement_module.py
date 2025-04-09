import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.LeakyReLU(0.01,inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class SCAM(nn.Module):
    def __init__(self,ch_in):
        super(SCAM, self).__init__()
        self.ca = ChannelAttention(ch_in)
        self.sa = SpatialAttention()
    def forward(self,x):
        channel_weight = self.ca(x)
        xc = x*channel_weight
        spatial_weight = self.sa(xc)
        xs = x*spatial_weight
        x = xs+xc
        return x

class FBSIE(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,hidden_dim=128,kernel_size = 3):
        """
        Fore -Background Contrast Module
        """
        super(FBSIE,self).__init__()
        # self.Enhance = CorlorCorrection(in_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1)
        self.enhance_module = SpatialAttention(kernel_size)
        self.k_oper = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.01,inplace=True)
        )


        self.s_oper = nn.Sigmoid()
        self.l_oper_F=nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        self.l_oper_B=nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        # print(f"1:{x.shape}")
        # x = self.Enhance(x)
        # print(f"2:{x.shape}")
        # residual spaptial attention
        # x = self.enhance_module(x)*x
        B,C,H,W = x.shape
        F_k = self.k_oper(x)
        F_branch = self.s_oper(F_k)
        B_branch = 1-F_branch
        F_branch = F_branch.view(B,1,H*W)
        B_branch = B_branch.view(B,1,H*W)
        x_eq = x.view(B,H*W,C)
        ## [B,1,H*W]*[B,H*W,C]==>[B,1,C]
        F_branch = torch.bmm(F_branch,x_eq).view(B,C)
        B_branch = torch.bmm(B_branch,x_eq).view(B,C)
        F_branch_weight = self.l_oper_F(F_branch)
        B_branch_weight = self.l_oper_B(B_branch)
        channel_weight = F_branch_weight-B_branch_weight
        channel_weight = channel_weight.view(B,C,1,1)
        x = x*channel_weight
        return x
class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = swish()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class RepConv(nn.Module):
    default_act = swish()  # default activation
 
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
 
    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))
 
    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)
 
class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
 
    def forward(self, x):
        identity = x
 
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
 
        return out

class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut
 
    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y

class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)
 
    def forward(self, x):
        outs = [x]
 
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
 
        y = self.conv(y)
        return y

    
class CSPStageEM(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,
                 block_fn='BasicBlock_3x3_Reverse',
                 ch_hidden_ratio=1.0,
                 act='silu',
                 spp=False):
        super(CSPStageEM, self).__init__()
 
        split_ratio = 2
        self.scam = SCAM(ch_in)
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()
        self.em = FBSIE(ch_first,ch_first)
        # self.em = CorlorCorrection(ch_in, ch_in*2,kernel_size=3,stride=1, padding=1)
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)
        
 
    def forward(self, x):
        x = self.scam(x)
        y1 = self.conv1(x)
        y1 = self.em(y1)+y1
        # y1 = self.em(y1)
        y2 = self.conv2(x)
 
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
