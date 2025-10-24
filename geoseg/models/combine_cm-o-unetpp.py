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

# U-Net++ 核心模块：密集跨尺度连接块
class UNetPlusPlusBlock(nn.Module):
    """
    融合多尺度编码器特征的密集连接块
    输入：当前解码器特征 + 多个尺度的编码器特征
    输出：融合后的特征
    """
    def __init__(self, decoder_ch, encoder_chs, out_ch):
        """
        decoder_ch: 当前解码器输入通道数
        encoder_chs: 多个编码器特征的通道数列表（如 [c1, c2, c3]）
        out_ch: 输出通道数
        """
        super().__init__()
        # 1. 调整编码器特征通道数并上采样到当前解码器尺寸
        self.encoder_convs = nn.ModuleList()
        for ch in encoder_chs:
            # 1x1卷积调整通道 + 上采样
            self.encoder_convs.append(nn.Sequential(
                BasicConv2d(ch, out_ch, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 逐步上采样到当前尺度
            ))
        
        # 2. 调整解码器输入通道数
        self.decoder_conv = BasicConv2d(decoder_ch, out_ch, kernel_size=1)
        
        # 3. 密集融合卷积块（嵌套卷积）
        self.fuse_conv1 = BasicConv2d(out_ch * (len(encoder_chs) + 1), out_ch, kernel_size=3, padding=1)
        self.fuse_conv2 = BasicConv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, decoder_feat, encoder_feats):
        """
        decoder_feat: 当前解码器特征（来自上一层解码器）
        encoder_feats: 多个尺度的编码器特征列表（需与encoder_chs对应）
        """
        # 调整解码器特征
        decoder_feat = self.decoder_conv(decoder_feat)
        
        # 调整所有编码器特征（通道 + 尺寸）
        encoder_feats_processed = []
        for i, feat in enumerate(encoder_feats):
            # 上采样到与解码器特征相同尺寸
            processed = self.encoder_convs[i](feat)
            # 若尺寸不匹配，强制调整（应对多尺度差异）
            if processed.shape[2:] != decoder_feat.shape[2:]:
                processed = F.interpolate(processed, size=decoder_feat.shape[2:], mode='bilinear', align_corners=True)
            encoder_feats_processed.append(processed)
        
        # 密集拼接：解码器特征 + 所有处理后的编码器特征
        fused = torch.cat([decoder_feat] + encoder_feats_processed, dim=1)
        
        # 融合卷积
        fused = self.fuse_conv1(fused)
        fused = self.fuse_conv2(fused)
        return fused



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

class combine_cm(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 96,
                 depths=[2, 2, 9, 2], 
                 drop_path_rate=0.1,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
                 # load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/pspnet.pth',
                 pretrained = True
                ):
        super().__init__()
        
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        
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
        
        
        self.Translayer_11 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_12 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_13 = BasicConv2d(8*mid_channel, mid_channel, 1)
        self.Translayer_14 = BasicConv2d(16*mid_channel, mid_channel, 1)
       
        self.deconv = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)
        

        # 3. U-Net++ 解码器密集跨尺度连接块
        # 解码器第4层（最深层）：融合编码器第4层特征
        self.decoder4 = UNetPlusPlusBlock(
            decoder_ch=mid_channel,  # 输入来自编码器最深层
            encoder_chs=[mid_channel],  # 仅连接编码器第4层
            out_ch=mid_channel
        )
        # 解码器第3层：融合编码器第3+4层特征
        self.decoder3 = UNetPlusPlusBlock(
            decoder_ch=mid_channel,
            encoder_chs=[mid_channel, mid_channel],  # 连接编码器第3、4层
            out_ch=mid_channel
        )
        # 解码器第2层：融合编码器第2+3+4层特征
        self.decoder2 = UNetPlusPlusBlock(
            decoder_ch=mid_channel,
            encoder_chs=[mid_channel, mid_channel, mid_channel],  # 连接编码器第2、3、4层
            out_ch=mid_channel
        )
        # 解码器第1层：融合编码器第1+2+3+4层特征（密集跨尺度连接核心）
        self.decoder1 = UNetPlusPlusBlock(
            decoder_ch=mid_channel,
            encoder_chs=[mid_channel, mid_channel, mid_channel, mid_channel],  # 连接所有编码器层
            out_ch=mid_channel
        )


        self.Up4x = LRDU(mid_channel,4)      
        self.convout = nn.Conv2d(96, num_classes, kernel_size=1, stride=1, padding=0)
            

        self.vmencoder = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           drop_path_rate=drop_path_rate
                        )
        
        self.apply(self._init_weights)

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
            # print('Not loaded keys:', not_loaded_keys)
            # print("vmunet loaded finished!")

    
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
        
        
        e1 = self.Translayer_11(m1)
        e2 = self.Translayer_12(m2)
        e3 = self.Translayer_13(m3)
        e4 = self.Translayer_14(m4)

        d4 = self.decoder4(e4, [e4])  # 仅连接同尺度编码器特征
        
        # 解码器第3层：输入d4（上采样），融合编码器第3+4层
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(d4_up, [e3, e4])  # 连接编码器3和4层
        
        # 解码器第2层：输入d3（上采样），融合编码器第2+3+4层
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.decoder2(d3_up, [e2, e3, e4])  # 连接编码器2、3、4层
        
        # 解码器第1层：输入d2（上采样），融合编码器所有层（1+2+3+4）
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(d2_up, [e1, e2, e3, e4])  # 密集跨尺度连接核心


        
        out = self.Up4x(d1)
        out = self.convout(out)
        
        return out
                   
          
        


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
    
        
        