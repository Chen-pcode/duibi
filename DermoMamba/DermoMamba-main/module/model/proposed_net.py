import torch.nn as nn
from model.encoder import EncoderBlock
from model.decoder import DecoderBlock
from module.CBAM import CBAM
from module.PACM import PCA
from module.SMB import Sweep_Mamba
class DermoMamba(nn.Module):
    def __init__(self):
        super().__init__()

        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)
        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Skip connection"""
        self.s1 = CBAM(32)
        self.s2 = CBAM(64)
        self.s3 = CBAM(128)
        self.s4 = CBAM(256)
        self.s5 = CBAM(512)

        """Bottle Neck"""
        self.b1 = Sweep_Mamba(512)
        self.b2 = PCA(512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
         # Final layer
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x):
        """Encoder"""
        x = self.pw_in(x)
        x, skip1 = self.e1(x)

        x, skip2 = self.e2(x)

        x, skip3 = self.e3(x)

        x, skip4 = self.e4(x)

        x, skip5 = self.e5(x)

        # """Skip connection"""
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)

        """BottleNeck"""
        x = self.b1(self.b2(x) + x)

        # """Decoder"""
        x1 = self.d5(x, skip5)
        x2 = self.d4(x1, skip4)
        x3 = self.d3(x2, skip3)
        x4 = self.d2(x3, skip2)
        x5 = self.d1(x4, skip1)

        x = self.conv_out(x5)
        return x

