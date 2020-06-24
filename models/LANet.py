# model codes for 'LANet: Local Attention Embedding to Improve the Semantic Segmentation of Remote Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2020.'
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
from models.FCN_32s import FCN_res50 as FCN
#from models.FCN_16s import FCN_res50 as FCN

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# ASP module from: 'Multi-scale context aggregation for semantic segmentation of remote sensing images[J]. Remote Sensing.'
class ASP(nn.Module):
    def __init__(self, in_channels, in_stride, reduction=4, RF=(320, 160, 80, 40)):
        super(ASP, self).__init__()
        self.strides = [R // in_stride for R in RF]
        out_channels = in_channels // reduction
        
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels) for i in range(4)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+4*out_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU()
            )

    def _make_stage(self, in_channels, out_channels):
        #prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size()[2:]
                
        priors = [feats]
        for idx, stage in enumerate(self.stages):
            h_out = h // self.strides[idx]
            w_out = w // self.strides[idx]
            feats_avg = F.adaptive_avg_pool2d(feats, [h_out, w_out])
            feats_avg = stage(feats_avg)
            priors.append(F.upsample(input=feats_avg, size=(h, w), mode='bilinear', align_corners=True))
        
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

# Patch attention module. Parameters: reduction is the rate of channel reduction. pool_window should be set according to the scaling rate.
class Patch_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(Patch_Attention, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        A = self.SA(A)
        
        A = F.upsample(A, (h,w), mode='bilinear')        
        output = x*A
        if self.add_input:
            output += x
        
        return output

# Calculate pixel-wise local attention. Costs more computations.
class Patch_AttentionV2(nn.Module):
    def __init__(self, in_channels, reduction=16, pool_window=10, add_input=False):
        super(Patch_AttentionV2, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.AvgPool2d(kernel_size=pool_window+1, stride=1, padding = pool_window//2),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        A = self.SA(x)
        
        A = F.upsample(A, (h,w), mode='bilinear')        
        output = x*A
        if self.add_input:
            output += x
            
        return output

# Attention embedding module. Parameters: reduction is the rate of channel reduction. pool_window should be set according to the scaling rate.
class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_window=6, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.SE = nn.Sequential( 
            nn.AvgPool2d(kernel_size=pool_window+1, stride=1, padding = pool_window//2),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.BatchNorm2d(in_channels//reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels//reduction, out_channels, 1),
            nn.Sigmoid())
            
    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        A = self.SE(high_feat)
        A = F.upsample(A, (h,w), mode='bilinear')
        
        output = low_feat*A
        if self.add_input:
            output += low_feat
        
        return output

class LANet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(LANet, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        
        self.PA0 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                nn.BatchNorm2d(64, momentum=0.95), nn.ReLU(inplace=False),
                                Patch_Attention(64, reduction=8, pool_window=20, add_input=True))
                
        self.PA2 = Patch_Attention(128, reduction=16, pool_window=4, add_input=True)
        self.AE = Attention_Embedding(128, 64)
        
        self.classifier0 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x0 = self.FCN.layer1(x) #size:1/4, C256
        x = self.FCN.layer2(x0) #size:1/8, C512
        x = self.FCN.layer3(x) #size:1/16, C1024
        x = self.FCN.layer4(x) #size:1/16 or 1/32, C2048
        x2 = self.FCN.head(x)  #size:1/16 or 1/32, C128
                
        x2 = self.PA2(x2)
        x0 = self.PA0(x0)
        x0 = self.AE(x2.detach(), x0)
        
        low = self.classifier0(x0)
        low = F.upsample(low, x_size[2:], mode='bilinear')
        
        high = self.classifier1(x2)
        high = F.upsample(high, x_size[2:], mode='bilinear')
                
        return high+low #, high , low
