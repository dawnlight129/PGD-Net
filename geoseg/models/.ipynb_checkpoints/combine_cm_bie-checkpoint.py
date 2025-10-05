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
# from .mobilemamba import MobileMamba
# from .ConvNeXt import ConvNeXt
# from .ResNet import resnet50
from .Res2Net import *
from torchvision import models
# from .feature_fusion.EFC import EFC


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


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = None
        if norm_cfg == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)
        self.act = None
        if act_cfg == 'ReLU':
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
# 上采样工具函数
def resize(input: torch.Tensor, size, mode='bilinear', align_corners=False):
    return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)

class combine_cm(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 96,
                 depths=[2, 2, 9, 2], 
                 drop_path_rate=0.1,
                 interpolate_mode = 'bilinear',
                 ignore_index = 255,
                 norm_cfg = 'BN',
                 act_cfg = 'ReLU',
                 align_corners=False,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
                 # load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/pspnet.pth',
                 pretrained = True
                ):
        super().__init__()
        
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners  # 定义属性
        # Pvtv2
        # self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # path = '/root/EB-TDFNet/pre_trained_weights/pspnet.pth'
        # vmamba
        
        self.backbone = VSSM()  # [64, 128, 320, 512]
        ## 不加载预训练权重，就注释掉以下加载权重的代码
        path = '/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
         
        
        #v-mamba
        self.Translayer_1 = BasicConv2d(1*mid_channel, 1*mid_channel, 1)
        self.Translayer_2 = BasicConv2d(2*mid_channel, 2*mid_channel, 1)
        self.Translayer_3 = BasicConv2d(4*mid_channel, 4*mid_channel, 1)
        self.Translayer_4 = BasicConv2d(8*mid_channel, 8*mid_channel, 1)
        
        # resnet50-res2net50
        self.Translayer_5 = BasicConv2d(256, 1*mid_channel, 1,stride=1)
        self.Translayer_6 = BasicConv2d(512, 2*mid_channel, 1,stride=1)
        self.Translayer_7 = BasicConv2d(1024, 4*mid_channel, 1,stride=1)
        self.Translayer_8 = BasicConv2d(2048, 8*mid_channel, 1,stride=1)  
        
        
        self.Translayer_11 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_12 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_13 = BasicConv2d(8*mid_channel, mid_channel, 1)
        self.Translayer_14 = BasicConv2d(16*mid_channel, mid_channel, 1)

        self.deconv = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)
        
        self.Up4x = LRDU(mid_channel,4)      
        self.convout = nn.Conv2d(96, num_classes, kernel_size=1, stride=1, padding=0)
            
        self.cencoder = res2net50_v1b_26w_4s(pretrained=True)
        
                
        self.vmencoder = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           drop_path_rate=drop_path_rate
                        )
        
        self.apply(self._init_weights)

        # 分割分支卷积
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(
                ConvModule(
                    in_channels=mid_channel * 2 ** i + mid_channel,
                    out_channels=mid_channel,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # 重建分支卷积
        self.convs_second = nn.ModuleList()
        for i in range(4):
            self.convs_second.append(
                ConvModule(
                    in_channels = mid_channel * 2 ** i,
                    out_channels = mid_channel,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # 多尺度特征融合（分割）
        self.fusion_conv = ConvModule(
            in_channels=mid_channel * 4,
            out_channels=mid_channel,
            kernel_size=1,
            norm_cfg=norm_cfg
        )

        # 多尺度特征融合（重建）
        self.fusion_conv_second = ConvModule(
            in_channels=mid_channel * 4,
            out_channels=mid_channel,
            kernel_size=1,
            norm_cfg=norm_cfg
        )

        # 输出层
        self.cls_seg = nn.Conv2d(mid_channel, num_classes, kernel_size=1)
        self.conv_rec = nn.Conv2d(mid_channel, num_classes + 1, kernel_size=1)
        self.conv_mlp = ConvModule(
            in_channels=num_classes + 1,
            out_channels=num_classes + 1,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.sigmoid = nn.Sigmoid()




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

    
    def forward(self, x):
        # VMamba
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        m1, m2, m3, m4 = self.vmencoder(x) #  m1 [2, 128, 128, 96]  m3  [2, 16, 16, 768]  [b h w c]
        # b h w c --> b c h w
        m1 = m1.permute(0, 3, 1, 2) # m1 [2, 96, 128, 128]
        m2 = m2.permute(0, 3, 1, 2)
        m3 = m3.permute(0, 3, 1, 2)
        m4 = m4.permute(0, 3, 1, 2)
        
        m4 = self.Translayer_4(m4)
        m3 = self.Translayer_3(m3)
        m2 = self.Translayer_2(m2)
        m1 = self.Translayer_1(m1)

        inputs = []
        inputs.append(m1)
        inputs.append(m2)
        inputs.append(m3)
        inputs.append(m4)

        # 提取重建特征
        outs_rec = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            middle_out = self.convs_second[idx](x)
            outs_rec.append(middle_out)

        # 分割分支计算
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x_fuse = torch.cat((x, outs_rec[idx].detach()), dim=1)
            x_conv = self.convs[idx](x_fuse)
            x_resize = resize(
                x_conv,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )
            outs.append(x_resize)

        seg_logits = self.fusion_conv(torch.cat(outs, dim=1))
        seg_logits = self.cls_seg(seg_logits)

        # 重建分支计算
        outs_rec_resize = []
        for idx in range(len(inputs)):
            rec_resize = resize(
                outs_rec[idx],
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )
            outs_rec_resize.append(rec_resize)

        rec_logits = self.fusion_conv_second(torch.cat(outs_rec_resize, dim=1))
        rec_logits = self.conv_rec(rec_logits)
        rec_logits = self.sigmoid(rec_logits)

        return seg_logits, rec_logits
        
#         #pvtv2
#         size = x.size()[2:]
#         outputs = self.backbone(x)

#         layer2,layer3,layer4,layer5 = self.backbone(x)

#         m1 = self.Translayer_1(layer2)
#         m2 = self.Translayer_2(layer3)
#         m3 = self.Translayer_3(layer4)
#         m4 = self.Translayer_4(layer5)

        
        # #vgg16/vgg19
        # size = x.size()[2:]
        # layer1 = self.inc(x)
        # layer2 = self.down1(layer1)
        # layer3 = self.down2(layer2)
        # layer4 = self.down3(layer3)
        # layer5 = self.down4(layer4)



#         #res2net50
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
       

#         # y31 = self.deconv4(c4) + c3
#         # y21 = self.deconv3(y31) + c2
#         # y11 = self.deconv2(y21) + c1
        
#         fuse1 = torch.cat([m1, c1], 1)
#         fuse2 = torch.cat([m2, c2], 1)
#         fuse3 = torch.cat([m3, c3], 1)
#         fuse4 = torch.cat([m4, c4], 1)
        
#         # fuse1 = self.EFC_Fuse1([m1, c1])
#         # fuse2 = self.EFC_Fuse2([m2, c2])
#         # fuse3 = self.EFC_Fuse3([m3, c3])
#         # fuse4 = self.EFC_Fuse4([m4, c4])
        
#         fuse1 = self.Translayer_11(fuse1)
#         fuse2 = self.Translayer_12(fuse2)
#         fuse3 = self.Translayer_13(fuse3)
#         fuse4 = self.Translayer_14(fuse4)
#         y31 = self.deconv(fuse4) + fuse3
#         y21 = self.deconv(y31) + fuse2
#         y11 = self.deconv(y21) + fuse1
        
# #         y31 = self.deconv4(fuse4) + fuse3
# #         y21 = self.deconv3(y31) + fuse2
# #         y11 = self.deconv2(y21) + fuse1
        
#         y = self.seg_outs(y11)
#         d2 = self.Up4x(y)
#         d1 = self.convout(d2)
        
#         return d1
                   
          
        


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
    
        
        