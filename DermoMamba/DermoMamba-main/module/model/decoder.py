import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c*2, in_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        # self.dw = AxialMixer(in_c, (7,7), dilation = 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding = 'same')

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        x = self.bn(self.act(self.pw2(x)))

        return x