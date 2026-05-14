import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import math
from sklearn.cluster import KMeans


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ECAModule(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAModule, self).__init__()
        # 计算卷积核大小
        k_size = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: 输入特征图，形状为 [B, C, H, W]
        b, c, h, w = x.size()

        # 全局平均池化
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]

        # 1D卷积
        y = self.conv(y)  # [B, 1, C]

        # Sigmoid激活
        y = self.sigmoid(y)  # [B, 1, C]

        # 调整形状并加权
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


# 可选：添加空间注意力机制的增强版ECAModule
class EnhancedECAModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(EnhancedECAModule, self).__init__()
        self.eca = ECAModule(channel)
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        channel_att = self.eca(x)

        # 空间注意力
        spatial_att = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.conv(spatial_att)
        spatial_att = self.sigmoid(spatial_att)

        return x * channel_att * spatial_att





class RegionSegHead(nn.Module):
    def __init__(self, n_levels=4, d_model=256, num_classes=1, attention_type="eca"):
        super(RegionSegHead, self).__init__()
        input_proj_list = []
        pred_list = []
        for _ in range(n_levels):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            ))
            pred_list.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, num_classes, 3, 1, 1)))
        self.input_proj = nn.ModuleList(input_proj_list)
        self.pred = nn.ModuleList(pred_list)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv = nn.ModuleList([
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2) for _ in range(n_levels - 1)
        ])
        if attention_type == "se":
            self.attention = nn.ModuleList([SEBlock(d_model) for _ in range(n_levels)])
        elif attention_type == "eca":
            self.attention = nn.ModuleList([ECAModule(d_model) for _ in range(n_levels)])
        elif attention_type in ["none", None]:
            self.attention = nn.ModuleList([nn.Identity() for _ in range(n_levels)])
        else:
            raise ValueError(f"Unsupported region attention type: {attention_type}")
    def forward(self, features):

        p = [None] * len(features)
        p[-1] = self.input_proj[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            p[i] = self.input_proj[i](features[i]) + self.upsample(p[i + 1])

        region_probs = []
        for i in range(len(features)):
            p[i] = self.attention[i](p[i])
            prob = torch.sigmoid(self.pred[i](p[i]))
            region_probs.append(torch.clamp(prob, 1e-5, 1 - 1e-5))

        # enhance multi-scale features
        enhanced_features = [feat * prob for feat, prob in zip(features, region_probs)]
        # region_masks = [self.cluster_regions(prob) for prob in region_probs]

        # generate segment embeddings
        seg_embed = [
            torch.where(prob > 0.5, torch.ones_like(prob), torch.zeros_like(prob)) for prob in region_probs]

        return enhanced_features, region_probs, seg_embed

    def cluster_regions(self, region_prob, num_clusters=8):
        # flatten low prob region
        flattened_prob = region_prob.view(-1)
        indices = torch.nonzero(flattened_prob > 0.5).squeeze()
        high_prob_features = flattened_prob[indices].unsqueeze(1).cpu().numpy()

        # cluster
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(high_prob_features)
        clusters = kmeans.labels_

        # generate cluster mask
        mask = torch.zeros_like(flattened_prob)
        mask[indices] = torch.tensor(clusters + 1, dtype=torch.float32)
        mask = mask.view(region_prob.shape)

        return mask
