import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from torchvision.ops import DeformConv2d


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class PMSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(PMSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs

class CSFU(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(CSFU, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.pmsdc = PMSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        pmsdc_outs = self.pmsdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in pmsdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(pmsdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
#   Cross-Scale Feature Unifier (CSFU)
def CSFULayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
        convs = []
        csfu = CSFU(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        convs.append(csfu)
        if n > 1:
            for i in range(1, n):
                csfu = CSFU(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
                convs.append(csfu)
        conv = nn.Sequential(*convs)
        return conv

#   Upscaling Convolution Layer (UCL)
class UCL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(UCL,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

#   Large-kernel grouped attention gate (DCAG)
class DCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCB, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.deform_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.offset = nn.Conv2d(out_channels, 2 * 3 * 3, kernel_size=3, padding=1)
        init.constant_(self.offset.weight, 0)
        init.constant_(self.offset.bias, 0)

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        offset = self.offset(out)
        out = self.deform_conv(out, offset)
        out = self.conv1x1_2(out)
        out = self.bn2(out)
        return out

class DCAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, activation='relu'):
        super(DCAG,self).__init__()

        self.W_g = DCB(F_g, F_int)
        self.W_x = DCB(F_l, F_int)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(inplace=True)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    def forward(self, g, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CSAP(nn.Module):
    """
    CSAP: A module combining CAB, SAB, and CSFU.
    """
    def __init__(self, in_channels, out_channels=None, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6', ratio=16):
        super(CSAP, self).__init__()

        # Initialize CAB
        self.cab = CAB(in_channels, out_channels, ratio, activation)

        # Initialize SAB
        self.sab = SAB(kernel_size=7)

        # Initialize CSFU
        self.csfu = CSFULayer(in_channels if out_channels is None else out_channels, 
                              out_channels if out_channels is not None else in_channels, 
                              n=1, stride=1, kernel_sizes=kernel_sizes, 
                              expansion_factor=expansion_factor, dw_parallel=dw_parallel, 
                              add=add, activation=activation)

    def forward(self, x):
        # Apply CAB
        x = self.cab(x) * x

        # Apply SAB
        x = self.sab(x) * x

        # Apply CSFU
        x = self.csfu(x)

        return x

#   Efficient multi-scale convolutional attention decoding (Decoder)
class Decoder(nn.Module):
    def __init__(self, channels=[768, 384, 192, 96], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True, add=True, dcag_ks=3, activation='relu'):
        super(Decoder, self).__init__()
        ucl_ks = 3  # kernel size for ucl

        self.csap4 = CSAP(channels[0], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.ucl3 = UCL(in_channels=channels[0], out_channels=channels[1], kernel_size=ucl_ks, stride=ucl_ks // 2, activation=activation)
        self.dcag3 = DCAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, activation=activation)
        self.csap3 = CSAP(channels[1], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.ucl2 = UCL(in_channels=channels[1], out_channels=channels[2], kernel_size=ucl_ks, stride=ucl_ks // 2, activation=activation)
        self.dcag2 = DCAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, activation=activation)
        self.csap2 = CSAP(channels[2], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.ucl1 = UCL(in_channels=channels[2], out_channels=channels[3], kernel_size=ucl_ks, stride=ucl_ks // 2, activation=activation)
        self.dcag1 = DCAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), activation=activation)
        self.csap1 = CSAP(channels[3], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

    def forward(self, x, skips):
        x = x.permute(0, 3, 1, 2).contiguous()

        # Adaptive Interactive Fusion Module
        d4 = self.csap4(x)
        d3 = self.ucl3(d4)
        x3 = self.dcag3(g=d3, x=skips[0])
        d3 = d3 + x3

        # Adaptive Interactive Fusion Module
        d3 = self.csap3(d3)
        d2 = self.ucl2(d3)
        x2 = self.dcag2(g=d2, x=skips[1])
        d2 = d2 + x2

        # Adaptive Interactive Fusion Module
        d2 = self.csap2(d2)
        d1 = self.ucl1(d2)
        x1 = self.dcag1(g=d1, x=skips[2])
        d1 = d1 + x1

        # CSAP
        d1 = self.csap1(d1)

        return [d4, d3, d2, d1]
