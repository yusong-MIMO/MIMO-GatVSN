import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from basicsr.archs.arch_util import flow_warp, ResidualBlockNoBN
from models.modules.module_util import initialize_weights_xavier

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 4, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


class EfficientChannelAttention(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()

        # 1×1 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))   # ← 修复：真正的最大池化

        # 可学习权重
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
        self.beta = nn.Parameter(torch.FloatTensor([0.5]))

        # 用卷积替代全连接 (ratio=4)
        hidden = max(dim // ratio, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_weight=False):

        B, C, H, W = x.shape

        # 全局池化
        avg_map = self.avg_pool(x)  # [B, C, 1, 1]
        max_map = self.max_pool(x)  # [B, C, 1, 1]

        # 融合池化结果
        fused = self.alpha * avg_map + self.beta * max_map

        # 1×1 conv MLP
        fused = self.fc(fused)  # [B, C, 1, 1]

        weight = self.sigmoid(fused)  # [B, C, 1, 1]

        if return_weight:
            return weight
        else:
            return x * (1 + weight)



class StaticalAttention(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.dw_dconv = nn.Conv2d(
            dim, dim, kernel_size,
            padding=((kernel_size - 1) // 2) * dilation,
            groups=dim, dilation=dilation
        )
        self.conv1x1 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_weight=False):
        attn = self.dwconv(x)
        attn = self.dw_dconv(attn)
        gate = self.sigmoid(self.conv1x1(attn))
        if return_weight:
            return gate
        else:
            return x * (1 + gate)


class MSFE(nn.Module):
    def __init__(self, dim, expand=1.5):
        super().__init__()

        hidden = int(dim * expand)

        self.conv_in = nn.Conv2d(dim, hidden, 1)

        self.dw_s = nn.Conv2d(
            hidden, hidden, 3,
            padding=1,
            groups=hidden
        )

        self.dw_m = nn.Conv2d(
            hidden, hidden, 5,
            padding=6,
            dilation=3,
            groups=hidden
        )

        self.dw_l = nn.Conv2d(
            hidden, hidden, 7,
            padding=9,
            dilation=3,
            groups=hidden
        )

        self.fusion = nn.Conv2d(
            hidden * 3, dim, 1
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        identity = x

        y = self.conv_in(x)

        y_s = self.dw_s(y)
        y_m = self.dw_m(y)
        y_l = self.dw_l(y)

        y = torch.cat(
            [y_s, y_m, y_l],
            dim=1
        )

        y = self.fusion(y)
        y = self.act(y)

        return identity + y



class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()

        self.MSFE_en = True
        if self.MSFE_en:
            self.MSFE = MSFE(channel_in)


        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ca = EfficientChannelAttention(channel_out)
        
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = self.MSFE(x)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.ca(x5)

        return x6

class DenseBlock_v2(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock_v2, self).__init__()

        self.MSFE_en = True
        if self.MSFE_en:
            self.MSFE = MSFE(channel_in)

        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sa = StaticalAttention(channel_in)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = self.sa(x)
        x = self.MSFE(x)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class DenseBlock_v3(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock_v3, self).__init__()

        self.MSFE_en = True
        if self.MSFE_en:
            self.MSFE = MSFE(channel_in)

        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ca = EfficientChannelAttention(channel_in)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        x = self.ca(x)
        x = self.MSFE(x)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out, groups=None, block_version='v1'):
        if net_structure != 'DBNet':
            return None

        if block_version == 'v1':
            return DenseBlock(channel_in, channel_out, init)
        elif block_version == 'v2':
            return DenseBlock_v2(channel_in, channel_out, init)
        elif block_version == 'v3':
            return DenseBlock_v3(channel_in, channel_out, init)
        else:
            raise ValueError(f"Unsupported DenseBlock version: {block_version}")

    return constructor
