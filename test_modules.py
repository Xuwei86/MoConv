import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # 定义 g、theta、phi、out 四个卷积层
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

        # 定义 softmax 层，用于将 f_ij 进行归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.size(0)

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 计算 theta(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # 计算 phi(x)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算 f_ij
        f = torch.matmul(theta_x, phi_x)

        # 对 f_ij 进行归一化
        f_div_C = self.softmax(f)

        # 计算 y_i
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        # 计算 z_i
        y = self.out(y)
        z = y + x

        return z

class SE(nn.Module):
    def __init__(self,channel, ratio=1):
        super(SE,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * ratio, bias=False),
            h_swish(),
            nn.Linear(channel * ratio, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y  = self.fc(y).view(b, c, 1, 1)
        return  x * y

# from cvnets.modules.transformer import LinearAttnFFN, TransformerEncoder
# from cvnets.modules import MobileViTBlockv2 as Block
# from cvnets.models.classification.config.mobilevit_v2 import get_configuration
# from cvnets.models.classification.base_image_encoder import BaseImageEncoder
# import argparse
class MLP(nn.Module):
    def __init__(self,in_featrues, ratio=4, act=nn.GELU):
        super().__init__()
        hidden_featrues = in_featrues * ratio
        self.fc1 = nn.Conv2d(in_featrues,hidden_featrues, 1, bias=False)
        self.act = act()
        self.fc2 = nn.Conv2d(hidden_featrues, in_featrues, 1, bias=False)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CA_atten(nn.Module):
    def __init__(self, inp, ks=7, ratio=2):
        super(CA_atten, self).__init__()

        # mip = max(8, inp // ratio)
        p = ks // 2


        self.conv0 = nn.Conv1d(inp, inp, kernel_size=ks, padding=p, groups=inp, bias=False)
        # self.conv1 = nn.Conv1d(inp, inp, kernel_size=ks, padding=p, groups=1, bias=False)
        self.bn0 = nn.BatchNorm1d(inp)
        # self.bn1 = nn.BatchNorm1d(mip)
        self.sig = nn.Sigmoid()

        self.relu = h_swish()

    def forward(self, x):

        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        identity = x

        # x_h = self.sig(self.bn0(self.conv0(x_h))).view(b, c, h, 1)
        # x_w = self.sig(self.bn0(self.conv0(x_w))).view(b, c, 1, w)

        x_h = self.relu(self.bn0(self.conv0(x_h))).view(b, c, h, 1)
        x_w = self.relu(self.bn0(self.conv0(x_w))).view(b, c, 1, w)

        x_h = self.sig(self.conv0(x_h.view(b, c, h))).view(b, c, h, 1)
        x_w = self.sig(self.conv0(x_w.view(b, c, w))).view(b, c, 1, w)

        y = identity * x_w * x_h

        return y

class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        ks = int(abs((math.log(channel,2) + b) / gamma)) +2
        ks = ks if ks % 2 else ks + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=ks //2, bias = False)
        self.sg = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sg(y).view([b, c, 1, 1])
        return x * y

class CA(nn.Module):
    def __init__(self, inp, ratio=8):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // ratio)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, groups=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, groups=1, padding=0)
        self.conv3 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, groups=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class h_swish(nn.Module):
    def __init__(self,inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self,inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3)/6