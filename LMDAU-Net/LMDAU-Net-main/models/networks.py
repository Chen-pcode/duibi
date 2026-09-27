import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from models.Encoder import Encoder
from models.Decoder import Decoder


class LMDAUNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, dcag_ks=3, activation='relu', encoder='Encoder', pretrain=True):
        super(LMDAUNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone = Encoder()
        channels=[768, 384, 192, 96]
    
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = Decoder(channels=channels, activation=activation)

        
        print('Model %s created, param count: %d' %
                     ('Decoder decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head1 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head4 = nn.Conv2d(channels[3], num_classes, 1)
        
    def forward(self, x, mode='test'):
        
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])
        p1 = self.out_head1(dec_outs[0])
        p2 = self.out_head2(dec_outs[1])
        p3 = self.out_head3(dec_outs[2])
        p4 = self.out_head4(dec_outs[3])

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')


        if mode == 'test':
            return [p1, p2, p3, p4]
        
        print(datetime.datetime.now())
        return [p1, p2, p3, p4]
               

        
