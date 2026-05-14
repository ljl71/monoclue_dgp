import torch
import torch.nn as nn


class FEM(nn.Module):
    """
    特征增强模块 (Feature Enhancement Module)
    用于增强输入特征图的表达能力，通过多分支卷积捕获不同尺度的特征
    特别是结合了普通卷积和空洞卷积来获取更丰富的感受野
    """

    def __init__(self, in_channels, out_channels=None, dilation=[3, 5]):
        """
        初始化特征增强模块

        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数，默认与输入通道数相同
            dilation: 空洞卷积的 dilation 系数列表，默认使用 [3, 5]
        """
        super(FEM, self).__init__()
        # 如果未指定输出通道数，则默认与输入通道数相同
        out_channels = out_channels or in_channels

        # 1x1 卷积用于特征降维或升维，调整通道数同时减少计算量
        # 不改变特征图的空间尺寸
        self.pre_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 分支1: 3x3 普通卷积，padding=1保持空间尺寸不变
        # 用于捕获较近范围的局部特征
        self.branch1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # 分支2: 两个空洞卷积串联，用于捕获更大范围的上下文信息
        # 第一层: 3x3 空洞卷积，dilation=dilation[0]，padding与dilation相同以保持尺寸
        # 第二层: 3x3 空洞卷积，dilation=dilation[1]，进一步扩大感受野
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation[0], dilation=dilation[0]),
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation[1], dilation=dilation[1]),
        )

        # 融合层: 1x1 卷积将两个分支的特征(通道数翻倍)融合为目标通道数
        self.fusion = nn.Conv2d(in_channels * 2, out_channels, 1)
        # 批归一化层: 加速训练收敛，减轻过拟合
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU激活函数: 引入非线性，增强特征表达能力
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入特征图，形状为 (batch_size, in_channels, height, width)

        返回:
            增强后的特征图，形状为 (batch_size, out_channels, height, width)
        """
        # 保存输入特征作为残差连接的身份标识
        identity = x
        # 预处理卷积，调整特征分布
        x = self.pre_conv(x)

        # 分支1特征提取
        b1 = self.branch1(x)
        # 分支2特征提取(大感受野)
        b2 = self.branch2(x)

        # 沿通道维度拼接两个分支的特征
        out = torch.cat([b1, b2], dim=1)
        # 融合特征并调整通道数
        out = self.fusion(out)
        # 批归一化
        out = self.bn(out)

        # 残差连接: 当输出与输入尺寸相同时，添加原始输入特征
        # 有助于缓解深层网络中的梯度消失问题
        if out.shape == identity.shape:
            out += identity

        # 激活函数输出
        return self.relu(out)