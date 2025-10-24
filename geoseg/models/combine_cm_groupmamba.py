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

# from .pvtv2 import *
try:
    from .groupmamba import groupmamba_tiny
except:
    from groupmamba import groupmamba_tiny

# from .Res2Net import *
from torchvision import models




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
    def __init__(self, in_c, factor):
        super(LRDU, self).__init__()

        self.up_factor = factor
        self.factor1 = factor * factor // 2
        self.factor2 = factor * factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1 * in_c, (1, 7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1 * in_c, self.factor2 * in_c, (7, 1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class combine_cm(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=2,
                 mid_channel=64,
                 depths=[2, 2, 9, 2],
                 drop_path_rate=0.1,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/groupmamba_tiny_ema.pth',
                 pretrained=True
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path


        # vmamba

        self.backbone = groupmamba_tiny()  # [64, 128, 348, 512]
        path = '/root/EB-TDFNet/pre_trained_weights/groupmamba_tiny_ema.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # v-mamba
        # self.Translayer_1 = BasicConv2d(1 * mid_channel, 1 * mid_channel, 1)
        # self.Translayer_2 = BasicConv2d(2 * mid_channel, 2 * mid_channel, 1)
        # self.Translayer_3 = BasicConv2d(  348          , 4 * mid_channel, 1)
        # self.Translayer_4 = BasicConv2d(  448          , 8 * mid_channel, 1)


        # 单
        # self.deconv4 = nn.ConvTranspose2d(8*mid_channel, 4*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.deconv3 = nn.ConvTranspose2d(4*mid_channel, 2*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.deconv2 = nn.ConvTranspose2d(2*mid_channel, 1*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)

        # 组合
        # self.deconv4 = nn.ConvTranspose2d(16*mid_channel, 8*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.deconv3 = nn.ConvTranspose2d(8*mid_channel, 4*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.deconv2 = nn.ConvTranspose2d(4*mid_channel, 2*mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        # self.seg_outs = nn.Conv2d(2*mid_channel, mid_channel, 1, 1)

        self.Translayer_11 = BasicConv2d(1 * mid_channel, mid_channel, 1)
        self.Translayer_12 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_13 = BasicConv2d(348, mid_channel, 1)
        self.Translayer_14 = BasicConv2d(448, mid_channel, 1)

        self.deconv = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)

        self.Up4x = LRDU(mid_channel, 4)
        self.convout = nn.Conv2d(mid_channel, num_classes, kernel_size=1, stride=1, padding=0)

        # self.cencoder = res2net50_v1b_26w_4s(pretrained=True)

        # self.backbone = groupmamba_small()

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
            model_dict = self.backbone.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.backbone.load_state_dict(model_dict)


    def forward(self, x):
        # VMamba
        if x.size()[1] == 1:  # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1, 3, 1, 1)
        output = self.backbone(x) 
        # for feat in output:
        #     print(feat.shape)           
        #  torch.Size([2, 64, 128, 128]) torch.Size([2, 128, 64, 64]) torch.Size([2, 348,32, 32]) torch.Size([2, 512, 16, 16])
        

        fuse4 = self.Translayer_14(output[3])
        fuse3 = self.Translayer_13(output[2])
        fuse2 = self.Translayer_12(output[1])
        fuse1 = self.Translayer_11(output[0])


        # fuse1 = self.Translayer_11(m1)
        # fuse2 = self.Translayer_12(m2)
        # fuse3 = self.Translayer_13(m3)
        # fuse4 = self.Translayer_14(m4)
        y31 = self.deconv(fuse4) + fuse3
        y21 = self.deconv(y31) + fuse2
        y11 = self.deconv(y21) + fuse1


        y = self.seg_outs(y11)          # ([2, 64, 128, 128])
        # print('y.shape', y.shape)
        d2 = self.Up4x(y)
        d1 = self.convout(d2)

        return d1


if __name__ == "__main__":

    model = combine_cm().cuda()

    img = torch.randn(2, 3, 512, 512).cuda()
    output = model(img)

    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total() / 1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6))


