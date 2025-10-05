import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import math

import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import os.path
import warnings

from .pvtv2 import *
from .vmamba import VSSM
from .mamba_vision import mamba_vision_T2
# from .ConvNeXt import ConvNeXt
# from .ResNet import resnet50
from .Res2Net import *
from torchvision import models
from .FFTformer import *

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        self.cbr = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.cbr(x)
        return x

class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x 
    

 # 定义插值函数
def align_feature(m, c, mode='bilinear'):
    return F.interpolate(
        m, 
        size=(c.shape[2], c.shape[3]), 
        mode=mode, 
        align_corners=False
    )

class combine_cm(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 96,   # 96
                 dim = 48,
                 # depths=[2, 2, 9, 2], 
                 depths=[1, 3, 11, 4], 
                 drop_path_rate=0.1,
                 load_ckpt_path = None,
                 # load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
                 pretrained = True,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                ):
        super().__init__()
        
        self.num_classes = num_classes
        # self.load_ckpt_path = load_ckpt_path
        
        #mamba vision
        self.backbone = mamba_vision_T2()  # [64, 128, 320, 512]
        self.vmencoder = mamba_vision_T2()
        # self.cencoder = res2net50_v1b_26w_4s(pretrained=True)
        
        #mamba vision
        # self.Translayer_1 = BasicConv2d(2*mid_channel, 2*mid_channel, 1)
        # self.Translayer_2 = BasicConv2d(4*mid_channel, 4*mid_channel, 1)
        # self.Translayer_3 = BasicConv2d(8*mid_channel, 8*mid_channel, 1)
        # self.Translayer_4 = BasicConv2d(8*mid_channel, 8*mid_channel, 1)
        self.Translayer_1 = BasicConv2d(1*mid_channel, 1*mid_channel, 1)
        self.Translayer_2 = BasicConv2d(2*mid_channel, 2*mid_channel, 1)
        self.Translayer_3 = BasicConv2d(4*mid_channel, 4*mid_channel, 1)
        self.Translayer_4 = BasicConv2d(8*mid_channel, 8*mid_channel, 1) 
        
        #resnet50-res2net50
        # self.Translayer_5 = BasicConv2d(256, 1*mid_channel, 1,stride=1)
        # self.Translayer_6 = BasicConv2d(512, 2*mid_channel, 1,stride=1)
        # self.Translayer_7 = BasicConv2d(1024, 4*mid_channel, 1,stride=1)
        # self.Translayer_8 = BasicConv2d(2048, 8*mid_channel, 1,stride=1)  
        
        self.Translayer_11 = BasicConv2d(3*mid_channel, mid_channel, 1)
        self.Translayer_12 = BasicConv2d(6*mid_channel, mid_channel, 1)
        self.Translayer_13 = BasicConv2d(12*mid_channel, mid_channel, 1)
        self.Translayer_14 = BasicConv2d(16*mid_channel, mid_channel, 1)

        self.deconv = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)
        
        self.Up4x = LRDU(mid_channel,4)      
        self.convout = nn.Conv2d(80, num_classes, kernel_size=1, stride=1, padding=0)  # 96
        self.apply(self._init_weights)
        
        # add fftformer decoder
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[3])])
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])
        
        self.up4_2 = Upsample(int(dim * 2 ** 3))
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmencoder.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数 
            # print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmencoder.load_state_dict(model_dict)
            # not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]

    
    def forward(self, x):
        # VMamba
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        m1, m2, m3, m4 = self.vmencoder(x) #  m1 [2, 128, 128, 96]  m3  [2, 16, 16, 768]  [b h w c]
        print('1打印m1-m4:, ',m1.shape,m2.shape,m3.shape,m4.shape)
        
        m4 = self.Translayer_4(m4)
        m3 = self.Translayer_3(m3)
        m2 = self.Translayer_2(m2)
        m1 = self.Translayer_1(m1)
        print('2打印m1-m4:, ',m1.shape,m2.shape,m3.shape,m4.shape)
        
        #res2net50
#         size = x.size()[2:]
#         x = self.cencoder.conv1(x)
#         x = self.cencoder.bn1(x)
#         layer1 = self.cencoder.relu(x)
#         x = self.cencoder.maxpool(layer1)  
#         layer2 = self.cencoder.layer1(x)  
#         layer3 = self.cencoder.layer2(layer2)  
#         layer4 = self.cencoder.layer3(layer3)  
#         layer5 = self.cencoder.layer4(layer4)
        
#         c1 = self.Translayer_5(layer2)
#         c2 = self.Translayer_6(layer3)
#         c3 = self.Translayer_7(layer4)
#         c4 = self.Translayer_8(layer5)
        # print('打印c1-c4:, ',c1.shape,c2.shape,c3.shape,c4.shape)
       
        # y31 = self.deconv4(c4) + c3
        # y21 = self.deconv3(y31) + c2
        # y11 = self.deconv2(y21) + c1
    
#         fuse1 = torch.cat([align_feature(m1, c1),c1], 1)
#         fuse2 = torch.cat([align_feature(m2, c2),c2], 1)
#         fuse3 = torch.cat([align_feature(m3, c3),c3], 1)   # fuse3 = torch.cat([m3, c3], 1)
#         fuse4 = torch.cat([align_feature(m4, c4),c4], 1)
#         # print('打印fuse:, ',fuse1.shape,fuse2.shape,fuse3.shape,fuse4.shape)
        
        # fuse1 = self.Translayer_11(m2)
        # fuse2 = self.Translayer_12(m2)
        # fuse3 = self.Translayer_13(m3)
        # fuse4 = self.Translayer_14(m4)
        
        
        out4 = self.decoder_level4(m4)
        print('3打印out4',out.shape)
        out4 = self.up4_2(out4)
        out4 = self.fuse3(out4, m3)
        
        out3 = self.decoder_level3(out4)
        out3 = self.up3_2(out3)
        out3 = self.fuse2(out3, m2)

        out2 = self.decoder_level2(out3)
        out2 = self.up2_1(out2)
        out2 = self.fuse1(out2, m1)
        
        out1 = self.decoder_level1(out2)
        out1 = self.refinement(out1)
        outs = self.output(out1) + x
        
#         y31 = self.deconv(fuse4) + fuse3
#         y21 = self.deconv(y31) + fuse2
#         y11 = self.deconv(y21) + fuse1
        
#         y31 = self.deconv4(fuse4) + fuse3
#         y21 = self.deconv3(y31) + fuse2
#         y11 = self.deconv2(y21) + fuse1
        
#         y = self.seg_outs(y11)
#         d2 = self.Up4x(y)
#         d1 = self.convout(d2)
        
        return outs
                   
          
        


if __name__ == "__main__":
    
    model = combine_cm().cuda()
    
    img = torch.randn(2, 3, 512, 512).cuda()
    output = model(img)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 
    
        
        