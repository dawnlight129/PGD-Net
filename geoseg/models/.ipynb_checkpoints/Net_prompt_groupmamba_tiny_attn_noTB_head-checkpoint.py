import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

import numbers

from einops import rearrange
from thop import profile
from thop import clever_format

# from .vmamba import VSSM
# from .AFSIM import AFSIModule
# from .Res2Net import *
try:
    from .groupmamba import groupmamba_tiny
except:
    from groupmamba import groupmamba_tiny
# from .MambaBlock import VSSBlock
# geoseg.models

"""
    在groupmamba-t-prompt-process-att-noTB的基础上
    替换seg_head部分
    以及，替换Attention部分
"""



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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
    

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # Branch 1: 3x1 convolution with BN and ReLU
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 51 convolution with BN and ReLU
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Depthwise 3x3 convolution
        self.dwconv3x3 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, groups=out_channels)

    def forward(self, x):
        # Pass input through each branch
        x = self.conv1(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = out1 + out2
        # Apply depthwise convolution
        out = self.dwconv3x3(out)
        return out + x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


# Spatial Conv Module
class SCModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        return attn1, attn2





class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x




# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.shape

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1,
                return_attnmap=False):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.return_attnmap = return_attnmap

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape               # [1, 256, 64]   dim=64, num_heads=8, sr_ratio=2
        # 1、生成Q，拆分注意力头
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()   # [1, 8, 256, 8]     
        # 2、生成K,V，多分支计算（核心创新逻辑）
        if self.sr_ratio > 1:
                # step1：将序列特征重塑为空间特征图,便于卷积下采样
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()    # [1, 64, 16, 16]
                #step2: 两个不同尺度的空间下采样分支
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 64, 64])  [B,L,C]
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 256, 64])
                # step3: 生成K,V，拆分注意力头
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
                # print(f"打印kv1的形状{kv1.shape},kv2的形状{kv2.shape}")        # [2, 1, 4, 64, 8],[2, 1, 4, 256, 8]
                k1, v1 = kv1[0], kv1[1] 
                k2, v2 = kv2[0], kv2[1] 
                # print(f"打印k1的形状{k1.shape},k2的形状{k2.shape}")            # [1, 4, 64, 8],[1, 4, 256, 8]
                # print(f"打印v1的形状{v1.shape},v2的形状{v2.shape}")            # [1, 4, 64, 8],[1, 4, 256, 8] 
                
                # step4: 计算注意力(attn1全局依赖，attn2中尺度依赖)
                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale      # [1, 4, 256, 64]
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)                                               # [1, 4, 256, 64]
                # 局部卷积补充v1的空间细节，避免全局注意力丢失局部信息
                # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                #                         transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                #     view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
                v1.transpose(1,2).reshape(B,-1,C//2)                                # [B,L1,C//2] 
                v1.transpose(1,2).view(B,C//2,H/self.sr_ratio, W//self.sr_ratio)    # [B,C//2,H1,W1]
                v1 = v1 + self.local_conv1(v1)                                      # 残差连接局部卷积结果
                v1 = v1.view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous()  # 转回原维度

                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2).contiguous()  
                
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)
                # step5: 拼接双分支结果
                x = torch.cat([x1,x2], dim=-1)

                if self.return_attnmap:         # 返回注意力图，便于可视化
                    attn = torch.cat([attn1,attn2],dim=3)
        #  3. sr_ratio=1  （带局部卷积的普通Attention,退化逻辑）
        else:   
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2).contiguous() 
        # 4. 最终投影与Dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.return_attnmap:
            return x,attn.mean(1)
        else:
            return x


## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.attn = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,return_attnmap=self.return_attnmap)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

    
    

##---------- Prompt Gen Module -----------------------      PGM 
# 用于生成与输入特征相关的 Prompt（提示向量）
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        # 生成 Prompt 权重
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        # 生成 Prompt
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # 对所有 Prompt 进行加权求和，生成最终的 Prompt
        prompt = torch.sum(prompt,dim=1)
        # 调整 Prompt 的空间大小
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        # 进一步处理 Prompt
        prompt = self.conv3x3(prompt)

        return prompt


#  在模型中加入空间注意力（CBAM的空间分支），让模型主动关注小目标区域的特征。
class SpatialAttention(nn.Module):
    """空间注意力模块：关注小目标的空间位置"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)   # 用一个大卷积核来提取空间位置的重要性
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # 通道维度平均
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 通道维度最大
        out = torch.cat([avg_out, max_out], dim=1)     # 拼接两个特征
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out  # 特征加权

class Decoder(nn.Module):
    """
    Decoder module that reconstructs the feature maps from the encoded features.
    It uses AFEBlocks to enhance features at different levels and TransformerBlocks
    for further feature refinement.
    """
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 decoder=True,
                 heads=[1, 2, 4, 8],
                 num_blocks=[4, 6, 6, 8],
                 LayerNorm_type='WithBias',
                 num_classes=6,
                 mlp_ratio= 2.0,
                 ):
        super(Decoder, self).__init__()
        self.decoder = decoder
        # res1: torch.Size([2, 96, 128, 128]) res2: torch.Size([2, 192, 64, 64]) res3: torch.Size([2, 384, 32, 32]) res4: torch.Size([2, 768, 16, 16])
        if self.decoder:
            self.conv1 = BasicConv2d(decode_channels * 2 ** 3, decode_channels * 2 ** 3, 1)
            self.conv2 = BasicConv2d(decode_channels * 2 ** 2, decode_channels * 2 ** 2, 1)
            self.conv3 = BasicConv2d(decode_channels * 2 ** 1, decode_channels * 2 ** 1, 1)
            self.conv4 = BasicConv2d(decode_channels         , decode_channels         , 1)
            # self.EGA1 = EGA(dim=decode_channels * 2 ** 3, depth=2, stage=3, mlp_ratio=mlp_ratio, drop_path=0., act_layer=nn.ReLU)
            # self.EGA2 = EGA(dim=decode_channels * 2 ** 2, depth=4, stage=2, mlp_ratio=mlp_ratio, drop_path=0., act_layer=nn.ReLU)
            # self.EGA3 = EGA(dim=decode_channels * 2 ** 1, depth=4, stage=1, mlp_ratio=mlp_ratio, drop_path=0., act_layer=nn.ReLU)
            # self.EGA4 = EGA(dim=decode_channels         , depth=1, stage=0, mlp_ratio=mlp_ratio, drop_path=0., act_layer=nn.ReLU)
            # self.AFEB1 = AFEBlock(decode_channels * 2 ** 3, num_heads=heads[2], bias=bias)
            # self.AFEB2 = AFEBlock(decode_channels * 2 ** 2, num_heads=heads[2], bias=bias)
            # self.AFEB3 = AFEBlock(decode_channels * 2 ** 1, num_heads=heads[2], bias=bias)

            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 128,lin_dim = 64)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 64,lin_dim = 128)
            self.prompt3 = PromptGenBlock(prompt_dim=256,prompt_len=5,prompt_size = 32,lin_dim = 256)


        self.noise_level3 = TransformerBlock(dim=int(decode_channels * 2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(decode_channels * 2**3), int(decode_channels * 2**2), kernel_size=1, bias=bias)
        
        self.noise_level2 = TransformerBlock(dim=int(decode_channels * 2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(decode_channels * 2**2), int(decode_channels * 2**1), kernel_size=1, bias=bias)
        
        self.noise_level1 = TransformerBlock(dim=int(decode_channels * 2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(decode_channels * 2**1), int(decode_channels * 2**0), kernel_size=1, bias=bias)


        self.up4 = nn.Conv2d(decode_channels * 2 ** 3, decode_channels * 2 ** 2, 1)
        self.up3 = nn.Conv2d(decode_channels * 2 ** 2, decode_channels * 2 ** 1, 1)
        self.up2 = nn.Conv2d(decode_channels * 2 ** 1, decode_channels * 2 ** 0, 1)

        self.reduce_level3 = nn.Conv2d(int(decode_channels * 2 ** 3), int(decode_channels * 2 ** 2), kernel_size=1,
                                       bias=bias)
        self.reduce_level2 = nn.Conv2d(int(decode_channels * 2 ** 2), int(decode_channels * 2 ** 1), kernel_size=1,
                                       bias=bias)
        self.reduce_level1 = nn.Conv2d(int(decode_channels * 2 ** 1), int(decode_channels * 2 ** 0), kernel_size=1,
                                       bias=bias)

        # self.TB1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(decode_channels * 2 ** 2), num_heads=heads[2],
        #                      ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.TB2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(decode_channels * 2 ** 1), num_heads=heads[1],
        #                      ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # self.TB3 = nn.Sequential(*[
        #     TransformerBlock(dim=int(decode_channels * 1), num_heads=heads[0],
        #                      ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.tb_conv1 = BasicConv2d(decode_channels * 2 ** 2, decode_channels * 2 ** 2, 1)
        self.tb_conv2 = BasicConv2d(decode_channels * 2 ** 1, decode_channels * 2 ** 1, 1)
        self.tb_conv3 = BasicConv2d(decode_channels      * 1, decode_channels *      1, 1)


        self.spatial_attn = SpatialAttention(decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, x, res1, res2, res3, res4, h, w):
        # EGA1
        # print('res4.shape',res4.shape)  # res4: torch.Size([2, 768, 16, 16])
        # print('x.shape',x.shape)      # torch.Size([1, 64, 14, 14])
        inp_dec_level4 = self.conv1(res4)
        inp_dec_level4 = self.up4(inp_dec_level4)
        inp_dec_level3 = F.interpolate(inp_dec_level4, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level3 = torch.cat([inp_dec_level3, res3], 1)
        inp_dec_level3 = self.reduce_level3(inp_dec_level3)
        # print('inp解码3.shape',inp_dec_level3.shape)  # torch.Size([2, 256, 32, 32])
        # Prompt to inp_dec_level3
        dec3_param = self.prompt3(inp_dec_level3)      # PGM 模块
        inp_dec_level3 = torch.cat([inp_dec_level3, dec3_param], 1)  # cat
        inp_dec_level3 = self.noise_level3(inp_dec_level3)   # transformer block
        inp_dec_level3 = self.reduce_noise_level3(inp_dec_level3)  # Conv2d
        # TB1
        inp_dec_level3 = self.tb_conv1(inp_dec_level3)



        # EGA2
        out_dec_level3 = self.conv2(inp_dec_level3)
        out_dec_level3 = self.up3(out_dec_level3)
        inp_dec_level2 = F.interpolate(out_dec_level3, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level2 = torch.cat([inp_dec_level2, res2], 1)
        inp_dec_level2 = self.reduce_level2(inp_dec_level2)
        # print('inp解码2.shape',inp_dec_level2.shape) 
        # Prompt to inp_dec_level2
        dec2_param = self.prompt2(inp_dec_level2)
        inp_dec_level2 = torch.cat([inp_dec_level2, dec2_param], 1)
        inp_dec_level2 = self.noise_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_noise_level2(inp_dec_level2)
        # TB2
        inp_dec_level2 = self.tb_conv2(inp_dec_level2)


        # EGA3
        out_dec_level2 = self.conv3(inp_dec_level2)
        out_dec_level2 = self.up2(out_dec_level2)
        inp_dec_level1 = F.interpolate(out_dec_level2, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level1 = torch.cat([inp_dec_level1, res1], 1)
        inp_dec_level1 = self.reduce_level1(inp_dec_level1)
        # print('inp解码1.shape',inp_dec_level1.shape) 
        # Prompt to inp_dec_level1
        dec1_param = self.prompt1(inp_dec_level1)
        inp_dec_level1 = torch.cat([inp_dec_level1, dec1_param], 1)
        inp_dec_level1 = self.noise_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_noise_level1(inp_dec_level1)
        # TB3
        inp_dec_level1 = self.tb_conv3(inp_dec_level1)
        
        # AFEB4     
        # inp_dec_level1 = self.EGA4(inp_dec_level1)

        # out = self.decoder_level1(out)

        ###   
        inp_dec_level1 = self.spatial_attn(inp_dec_level1)  # 对融合后的特征应用空间注意力模块
        out = self.segmentation_head(inp_dec_level1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        # print('输出解码shape',out.shape)  # torch.Size([1, 6, 224, 224])
        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class AFENet(nn.Module):
    def __init__(self,
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 64,
                 depths=[2, 2, 9, 2], 
                 decode_channels=64,
                 drop_path_rate=0.1,
                 dropout=0.1,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/groupmamba_tiny_ema.pth',
                 pretrained=True,
                 ):
        super().__init__()
        self.backbone = groupmamba_tiny()  # [64, 128, 348, 448]
        ## 不加载预训练权重，就注释掉以下加载权重的代码
        path = '/root/EB-TDFNet/pre_trained_weights/groupmamba_tiny_ema.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.cencoder = res2net50_v1b_26w_4s(pretrained=True)

        encoder_channels = [64, 128, 256, 512]  # backbone feature channels

        # self.vmencoder = VSSM(in_chans=input_channels,
        #                    num_classes=num_classes,
        #                    depths=depths,
        #                    drop_path_rate=drop_path_rate
        #                 )

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes=num_classes)

        self.Translayer_1 = BasicConv2d(1*mid_channel, 1*decode_channels, 1)
        self.Translayer_2 = BasicConv2d(2*mid_channel, 2*decode_channels, 1)
        self.Translayer_3 = BasicConv2d( 348         , 4*decode_channels, 1)
        self.Translayer_4 = BasicConv2d( 448         , 8*decode_channels, 1)

        #resnet50-res2net50
        # self.Translayer_5 = BasicConv2d(256, 1*decode_channels, 1,stride=1)
        # self.Translayer_6 = BasicConv2d(512, 2*decode_channels, 1,stride=1)
        # self.Translayer_7 = BasicConv2d(1024, 4*decode_channels, 1,stride=1)
        # self.Translayer_8 = BasicConv2d(2048, 8*decode_channels, 1,stride=1)


    def forward(self, x):
        # vmamba骨干
        h, w = x.size()[-2:]
        output = self.backbone(x) 

        res4 = self.Translayer_4(output[3])
        res3 = self.Translayer_3(output[2])
        res2 = self.Translayer_2(output[1])
        res1 = self.Translayer_1(output[0])
       
        # print('打印形状：res1:', res1.shape, 'res2:', res2.shape, 'res3:', res3.shape, 'res4:', res4.shape)
        
        if self.training:
            x = self.decoder(x, res1, res2, res3, res4, h, w)
            return x
        else:
            x = self.decoder(x, res1, res2, res3, res4, h, w)
            return x


if __name__ == '__main__':
    input = torch.rand(2, 3, 512, 512).cuda()
    model = AFENet(decode_channels=64, num_classes=2, pretrained=True).cuda()
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops)
    print('params:', params)

# if __name__ == "__main__":
    
#     model = AFENet().cuda()
    
#     img = torch.randn(2, 3, 512, 512).cuda()
#     output = model(img)
    
#     if 1:
#         from fvcore.nn import FlopCountAnalysis, parameter_count_table
#         flops = FlopCountAnalysis(model, img)
#         print("FLOPs: %.4f G" % (flops.total()/1e9))

#         total_paramters = 0
#         for parameter in model.parameters():
#             i = len(parameter.size())
#             p = 1
#             for j in range(i):
#                 p *= parameter.size(j)
#             total_paramters += p
#         print("Params: %.4f M" % (total_paramters / 1e6)) 
