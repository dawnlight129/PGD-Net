# 用DSSA_MambaBlock替换原有的MambaBlock
# 该模块结合了Mamba和DeformMixFFN，适用于特征处理和增强

import torch
import torch.nn as nn
from torch.nn import ModuleList

from .ossm import OSSM
from .utils import nchw_to_nlc, nlc_to_nchw
from .module import DropPath, Stem
from .module import PatchEmbed
from DCNv4 import DCNv4


class MixFFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims    # 嵌入（输入特征）维度
        self.feedforward_channels = feedforward_channels    # 前馈网络的隐藏层通道数
        self.activate = nn.GELU()

        in_channels = embed_dims

        fc1 = nn.Conv2d(        # 1*1卷积，用于线性变换，将输入通道emb_dims映射到feedforward_channels
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        pe_conv = nn.Conv2d(    # 3*3卷积，分组卷积，用于捕获局部空间信息
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = nn.Conv2d(       # 1*1卷积，用于将feedforward_channels映射回嵌入维度
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        # out = nlc_to_nchw(x, hw_shape)  # 将输入从NLC格式转换为NCHW格式
        x = x.permute(0,3,1,2).contiguous()
        out = x
        out = self.layers(out)
        out = nchw_to_nlc(out)  # 将输出从NCHW格式转换回NLC格式
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
#  MixFFN是一个典型的前馈网络，结合了卷积操作和残差连接，适用于特征处理，通过卷积捕获局部空间信息，
#  同时利用Dropout 和 DropPath防止过拟合。


# 改进版前馈网络，与MixFFN类似，但使用了DCNv4卷积层和LayerNormer来增强特征提取能力
class DeformMixFFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dims)  # 多的
        self.norm2 = nn.LayerNorm(embed_dims)

        in_channels = embed_dims

        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1)

        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1)

        self.dcnv4 = DCNv4(
                    channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=(3 - 1) // 2,
                    group=2)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        # x 由nhwc转换成 nchw
        N, H, W, C = x.shape
        x = x.permute(0,3,1,2).contiguous()
        out = x
        # out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        out = self.dropout_layer(self.norm2(out))
        out = out.view(N, H, W, C).permute(0, 3, 1, 2)
        print("打印1-x",x.shape)
        print("打印2-out",out.shape)
        out = x + out    # 残差连接
        
        return out

# 与MixFFN类似，但使用了DCNv4卷积层捕获复杂空间特征，使用LayerNormer稳定训练
# 适用于更复杂的特征处理，适用于动态场景或高维数据


# 编码器层，结合Mamba和DeformMixFFN
class DeformMambaEncoderLayer(nn.Module):
    """
    Implements one encoder layer compose of mamba and DeformMixFFN in UVMamba.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,  # 前馈网络的隐藏层通道数
                 drop_rate=0.,
                 drop_path_rate=0.,    # 随机深度丢弃
                 proj_drop=0.,      # 投影层的dropout率
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2,
                 bias=False):   # 编码器层的深度，未使用？？
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(    # 调用 OSSM 模块
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:   # identity是残差连接的输入
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        print("打印调用mamba模块前,x",x.shape)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x

