import torch.nn as nn
import torch 
from module.CSMB import Cross_Scale_Mamba_Block
class ResMambaBlock(nn.Module):
    def __init__(self, in_c):
      super().__init__()
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.block = Cross_Scale_Mamba_Block(in_c)
      self.conv = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 'same')
      self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
      x = self.block(x)
      x = self.act(self.ins_norm(self.conv(x))) + x*self.scale
      return x
class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pw= nn.Conv2d(in_c, out_c, kernel_size=3, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.resmamba = ResMambaBlock(in_c)
        self.down = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.resmamba(x)
        skip = self.act(self.bn(self.pw(x)))
        x = self.down(skip)

        return x, skip