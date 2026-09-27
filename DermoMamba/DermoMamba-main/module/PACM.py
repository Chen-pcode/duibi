import torch.nn as nn
from einops import reduce, rearrange
import torch

class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        c = reduce(x, 'b c h w -> b c', 'mean')
        x = self.dw(x)
        c_ = reduce(x, 'b c h w -> b c', 'mean')
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        return torch.einsum('bchw, bc -> bchw', x, att_score)