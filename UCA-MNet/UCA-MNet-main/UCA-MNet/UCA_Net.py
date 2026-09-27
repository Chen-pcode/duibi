import random

import torch
from torch import nn
import torch.nn.functional as F

from UCA.frcm import FRCM
from UCA.EEAtten import EEA
from UCA.CCAtten import CrissCrossAttention


def Channel_Shuffle(x):
    n, c, h, w = x.size()
    groups = random.choice([2, 4, 8, 16])
    
    channels_per_group = c // groups
    x = x.view(n, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(n, -1, h, w)
    return x


# Squeeze and Excitation Attention
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, reduction, kernel_size=1,bias=False),
            nn.PReLU(num_parameters=reduction,init=0.02),
            nn.Conv2d(reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)
    
    
# Convolution layer
class iConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, dilation=1, groups=1, bias=False, act='identity'):
        super().__init__()
        self.iconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding="same", dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(ch_out//2, ch_out)
            )
        if act == 'identity':
            self.act = nn.Identity()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=ch_out,init=0.02)
        elif act == 'relu':
            self.act = nn.ReLU(True)
            
    def forward(self, x):
        return self.act(self.iconv(x))
    
    
class ASPP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.c1 = iConv(ch_in, ch_out, 3, 1, 4)
        self.c2 = iConv(ch_in, ch_out, 3, 2, 2)
        self.c3 = iConv(ch_in, ch_out, 3, 4, 4)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.c4 = iConv(ch_out*4, ch_out, 1, 1, 4, act='prelu')
        
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        xa = F.interpolate(self.aap(x), size=x.shape[2:], mode='bilinear')
        
        return self.c4(torch.cat([x1, x2, x3, xa], 1))
    
    
# Feature Extraction Module
class FEM(nn.Module):
    def __init__(self, ch_in):
        super(FEM, self).__init__()
        r = 3
        # maxpool & 1x1 branch
        self.c1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            iConv(ch_in, ch_in, 1, 1, 4)
            )
        
        # dilated 3x3 conv branch
        self.c2 = iConv(ch_in, ch_in, 3, 2, 2)
        
        # 3x3 conv branch 
        self.c3 = iConv(ch_in, int(ch_in*r), 3, 1, 2, act='prelu')
                
    def forward(self, x):
        x0 = self.c1(x) + self.c2(x)
        x = self.c3(x0)
        
        return x, x0
    
    
# Multiscale Self-attention Module   
class MSM(nn.Module):
    def __init__(self, ch_in, ps):
        super(MSM, self).__init__()
        self.ch_mid = ch_in//4
        self.ps = ps
        self.dim = self.ps**2
        self.softmax = nn.Softmax(-1)
        
        self.nt = nn.Sequential(nn.GroupNorm(ch_in//2, ch_in),
                                nn.PReLU(num_parameters=ch_in, init=0.02)
                                )
        
        self.q1 = iConv(self.ch_mid, self.ch_mid, 1, 2, 4)
        self.k1 = iConv(self.ch_mid, self.ch_mid, 1, 1, 4)
        self.q2 = nn.Linear(self.dim, self.dim)
        self.k2 = nn.Linear(self.dim, self.dim)
        self.q3 = nn.Linear(self.ch_mid, self.ch_mid)
        self.k3 = nn.Linear(self.ch_mid, self.ch_mid)
        
        self.prelu_f1 = nn.PReLU(num_parameters=self.ch_mid, init=0.01)
        self.prelu_b1 = nn.PReLU(num_parameters=self.ch_mid, init=0.9)
        self.prelu_f2 = nn.PReLU(num_parameters=self.dim, init=0.01)
        self.prelu_b2 = nn.PReLU(num_parameters=self.dim, init=0.9)
        
        self.EEA = EEA(self.ch_mid)
        
    # non-linear V projection
    def v1(self, q, k):
        return -self.prelu_b1(-self.prelu_f1(q + k))
    def v2(self, q, k):
        return -self.prelu_b2(-self.prelu_f2(q + k))
    
    # Channel Self-attention
    def CSA(self, x):
        b, c, h, w = x.size()
        
        q = self.q1(x).view(b, c, h*w)
        k = self.k1(x).view(b, c, h*w)
        v = self.v1(q, k)
        
        _, c, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().view(b, c, h, w)
        
        return v
    
    # Patch Self-attention
    def PSA(self, x):
        b, c, h, w = x.shape
        p = (h//self.ps)**2
        
        x = x.view(b, c, p, self.ps**2)
        q = self.q2(x)
        k = self.k2(x)
        v = self.v1(q, k)
        
        _, _, p, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (p ** -0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().view(b, c, h, w)
        
        return v
    
    # Pixel Self-attention
    def PiSA(self, x):
        b, c, h, w = x.size()
        p = (h//self.ps)**2
        
        x = x.view(b, c, self.ps**2, p).permute(0, 2, 3, 1)
        q = self.q3(x)
        k = self.k3(x)
        v = self.v2(q, k)
        
        _, _, p, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (p ** 0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().permute(0, 3, 1, 2).view(b, c, h, w)
        
        return v
    
    def forward(self, x):
        x1, x2, x3, x4 = torch.split(self.nt(x), [self.ch_mid, self.ch_mid, self.ch_mid, self.ch_mid], 1)
        csa = self.CSA(x1)
        psa = self.PSA(x2)
        pisa = self.PiSA(x3)
        eea = self.EEA(x4)
        
        return x + torch.cat([csa, psa, pisa, eea], 1)
    
    
# Bottleneck Block  
class Block(nn.Module):
    def __init__(self, ch_in, ch_out, ps=0):
        super(Block, self).__init__()
        self.ps = ps
        ch_mid = int(ch_in*4)
        
        self.block = nn.ModuleList([])
        self.block.append(nn.ModuleList([
            MSM(ch_in, self.ps) if self.ps > 0 else nn.Identity(),
            FEM(ch_in)
                ]))
        self.squeeze = iConv(ch_mid, ch_out, 1, 1, 4, act='prelu')
        self.shortcut = iConv(ch_in, ch_out, 1, 1, 8, act='prelu') if ch_in != ch_out else nn.Identity()
        self.sem1 = SEM(ch_out, reduction=8)
        self.sem2 = SEM(ch_out, reduction=8)
        
    def forward(self, x):
        
        for (MSM, FEM) in self.block:
            v = MSM(x)
            x1, x0 = FEM(v)
            v = v if self.ps > 0 else x0
        
        x1 = torch.cat([x+v, x1], 1)
        x1 = self.squeeze(x1)
        x1 = x1 + self.sem1(x1)
        
        x = self.shortcut(x)
        x = x + self.sem2(x)
        
        return Channel_Shuffle(x + x1)
    
    
class Encoder(nn.Module):
    def __init__(self, img_in, ch_in):
        super(Encoder, self).__init__()
        self.E1 = nn.ModuleList([])
        self.E1.append(nn.ModuleList([
            nn.Conv2d(img_in, ch_in[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, ch_in[0]),
            Block(ch_in[0], ch_in[0]),
            Block(ch_in[0], ch_in[0])
            ]))
        self.E2 = nn.ModuleList([])
        self.E2.append(nn.ModuleList([
            Block(ch_in[0], ch_in[1], 8),
            Block(ch_in[1], ch_in[1], 8),
            Block(ch_in[1], ch_in[1])
            ]))
        self.E3 = nn.ModuleList([])   
        self.E3.append(nn.ModuleList([
            Block(ch_in[1], ch_in[2], 4),
            Block(ch_in[2], ch_in[2], 4),
            Block(ch_in[2], ch_in[2]),
            Block(ch_in[2], ch_in[2])
            ]))
        self.E4 = nn.ModuleList([])
        self.E4.append(nn.ModuleList([
            Block(ch_in[2], ch_in[3], 2),
            Block(ch_in[3], ch_in[3], 2),
            ASPP(ch_in[3], ch_in[3])
            ]))
        
        self.P = nn.MaxPool2d(2,2)
        self.d1 = nn.Dropout(0.3)
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Dropout(0.1)
            
    def forward(self, x):
        x1 = []
        for (conv, gn, e1b1, e1b2) in self.E1:
            x1.append(e1b2(e1b1(gn(conv(x)))))
        x2 = []
        for (e2b1, e2b2, e2b3) in self.E2:
            x2.append(self.d1(self.P(x1[-1])))
            x2.append(e2b3(e2b2(e2b1(x2[-1]))))
        x3 = []
        for (e3b1, e3b2, e3b3, e3b4) in self.E3:
            x3.append(self.d2(self.P(x2[-1])))
            x3.append(e3b4(e3b3(e3b2(e3b1(x3[-1])))))
        x4 = []
        for (e4b1, e4b2, aspp) in self.E4:
            x4.append(self.d3(self.P(x3[-1])))
            x4.append(aspp(e4b2(e4b1(x4[-1]))))
        
        return x1, x2, x3, x4
    
    
class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder, self).__init__()
        self.D1 = nn.ModuleList([])
        self.D1.append(nn.ModuleList([
            iConv(ch_in[1], ch_in[0]//2, 1, 1, 1),
            iConv(ch_in[1], ch_in[0]//2, 1, 2, 2),
            iConv(ch_in[0], ch_in[0], 1, 1, 2, act='relu')
            ]))
        self.D2 = nn.ModuleList([])
        self.D2.append(nn.ModuleList([
            iConv(ch_in[2], ch_in[1]//2, 1, 1, 1),
            iConv(ch_in[2], ch_in[1]//2, 1, 2, 4),
            nn.MaxPool2d(3, stride=1, padding=1),
            iConv(ch_in[1], ch_in[1], 3, 1, 2, act='relu')
            ]))
        self.D3 = nn.ModuleList([])
        self.D3.append(nn.ModuleList([
            iConv(ch_in[3], ch_in[2]//2, 1, 1, 1),
            iConv(ch_in[3], ch_in[2]//2, 1, 2, 4),
            iConv(ch_in[2], ch_in[2], 1, 1, 4, act='relu')
            ]))
        
        self.FRCM = FRCM(ch_ins=[ch_in[0], ch_in[0], ch_in[1], ch_in[2], ch_in[3]], ch_out=4)
        
        ch_in = ch_in[0]+24
        self.sem = SEM(ch_in, reduction=8)
        self.conv = iConv(ch_in, 64, 3, 1, 64//16, True, act='relu')
        self.out = nn.Sequential(
            iConv(64, 64, 1, 1, 1, False, act='relu'),
            nn.Conv2d(64, ch_out, kernel_size=1, bias=False)
            )
        
        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Dropout(0.3)
        
    def forward(self, x, img_shape):
        x1, x2, x3, x4 = x
        
        for (d3c1a, d3c1b, d3c2) in self.D3:
            x = torch.cat([d3c1a(x4[-1]), d3c1b(x4[-1])], 1)
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            x = d3c2(self.d1(x) + x3[-1])
            
        for (d2c1a, d2c1b, mp, d2c2) in self.D2:
            x = torch.cat([d2c1a(x), d2c1b(x)], 1)
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            x = d2c2(mp(self.d2(x) + x2[-1]))
            
        for (d1c1a, d1c1b, d1c2) in self.D1:
            x = torch.cat([d1c1a(x), d1c1b(x)], 1)
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            x = d1c2(self.d3(x) + x1[-1])
            
        sides = self.FRCM([x, x1[-1], x2[-1], x3[-1], x4[-1]], img_shape)
        x = torch.cat([x, sides],1)
        
        return self.out(self.conv(x + self.sem(x)))
   

class UCA_Net(nn.Module):
    def __init__(self, img_in, segout):
        super(UCA_Net, self).__init__()
        ch_in = [16, 32, 32, 64]
            
        self.E = Encoder(img_in, ch_in)
        self.D = Decoder(ch_in, segout)
            
    def forward(self, x):
        img_shape = x.shape[2:]
        
        x = self.E(x)
        SR = self.D(x, img_shape)
        
        return SR 
    