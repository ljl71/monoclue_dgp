import torch
import torch.nn as nn


class FEM(nn.Module):
    """Feature Enhancement Module with local and dilated-context branches."""

    def __init__(self, in_channels, out_channels=None, dilation=(3, 5)):
        super().__init__()
        out_channels = out_channels or in_channels

        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.branch_local = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.branch_context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation[0],
                dilation=dilation[0],
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation[1],
                dilation=dilation[1],
            ),
        )
        self.fusion = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.pre_conv(x)

        local_feat = self.branch_local(x)
        context_feat = self.branch_context(x)
        out = self.fusion(torch.cat([local_feat, context_feat], dim=1))
        out = self.bn(out)

        if out.shape == identity.shape:
            out = out + identity

        return self.relu(out)
