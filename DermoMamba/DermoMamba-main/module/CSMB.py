import torch.nn as nn
import torch
from module.VSSBlock import VSSBlock

class Axial_Spatial_DW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.mixer_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.mixer_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)
        self.conv = nn.Conv2d(dim, dim, kernel_size = 3, padding = 'same', groups = dim,dilation = dilation)
    def forward(self, x):
        skip = x
        x = self.mixer_w(x)
        x = self.mixer_h(x)
        x = self.conv(x)
        return x + skip
class Cross_Scale_Mamba_Block(nn.Module):
  def __init__(self,dim, norm_layer=nn.LayerNorm):
    super().__init__()
    self.dw1 = Axial_Spatial_DW(dim//4, (7,7), dilation = 1)
    self.dw2 = Axial_Spatial_DW(dim//4, (7,7), dilation = 2)
    self.dw3 = Axial_Spatial_DW(dim//4, (7,7), dilation = 3)
    # self.dw4 = Axial_Spatial_DW(dim//4, (7,7), dilation = 4)
    self.vss = VSSBlock(dim//4)
    self.bn = nn.BatchNorm2d(dim)
    self.act = nn.ReLU()

  def forward(self,x):
    x1, x2, x3, x4 = torch.chunk(x, 4, dim = 1)
    x1 = self.vss(self.dw1(x1).permute(0,2,3,1)).permute(0,3,1,2)
    x2 = self.vss(self.dw2(x2).permute(0,2,3,1)).permute(0,3,1,2)
    x3 = self.vss(self.dw3(x3).permute(0,2,3,1)).permute(0,3,1,2)
    # x4 = self.vss(self.dw4(x4).permute(0,2,3,1)).permute(0,3,1,2)
    x = torch.cat([x1,x2,x3,x4], dim = 1)
    x = self.act(self.bn(x))

    return x