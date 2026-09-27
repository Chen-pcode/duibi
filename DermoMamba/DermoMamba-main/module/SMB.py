from module.VSSBlock import SS2D
import torch.nn as nn
import torch 

class Sweep_Mamba(nn.Module):
    def __init__(self, dim, ratio = 8):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim//ratio,1)
        self.mamba1 = SS2D(d_model=dim//ratio, dropout=0, d_state=16)
        self.mamba2 = SS2D(d_model=6, dropout=0, d_state=16)
        self.mamba3 = SS2D(d_model=8, dropout=0, d_state=16)
        self.act = nn.SiLU()
        self.relu = nn.ReLU()
        self.proj_out = nn.Linear(dim//ratio, dim, 1)
        self.scale = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        skip = x
        x = self.proj_in(self.ln(x))
        x1 = self.mamba1(x)
        x2 = self.mamba2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x3 = self.mamba3(x.permute(0,3,1,2)).permute(0,2,3,1)
        w = self.act(x)
        out = w*x1 + w*x2 + w*x3
        out = self.proj_out(out) + skip*self.scale
        out = out.permute(0,3,1,2)
        out = self.bn(out)
        out = self.relu(out)
        return out