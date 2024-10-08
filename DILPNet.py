import torch
import torch.nn as nn
# from mmcv.cnn import ConvAWS2d, constant_init
# from mmcv.ops.deform_conv import deform_conv2d
import torchvision.models as models
import scipy.stats as st
from torch.nn import functional as F
import numpy as np
import cv2
from torch.nn.parameter import Parameter
from backbone.mix_transformerdep import dmit_b0
from backbone.mix_transformer1 import mit_b0
from toolbox.models.DCTMO0.manba import SS2D
from torch.nn.modules.batchnorm import _BatchNorm
# from toolbox.models.DCTMO0.lv import GaborLayer
import time
from einops import rearrange
# from toolbox.models.DCTMO0/bo.py
# from toolbox.model.cai.修layer import MultiSpectralAttentionLayer
import math
# from toolbox.models.DCTMO0.bo import WaveAttention, SAN
from mmcv.cnn import ConvAWS2d
from mmcv.ops.deform_conv import deform_conv2d
from mmengine.model import constant_init
class SAConv2d(ConvAWS2d):
    """SAC (Switchable Atrous Convolution)
    This is an implementation of `DetectoRS: Detecting Objects with Recursive
    Feature Pyramid and Switchable Atrous Convolution
    <https://arxiv.org/abs/2006.02334>`_.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # self.weight_c = nn.Parameter(torch.Tensor(self.weight.size()))  # 权重变化的部分
        self.offset_r = nn.Conv2d(self.in_channels, 18, kernel_size=3, padding=1, stride=stride, bias=True)
        # self.offset_g = nn.Conv2d(self.in_channels, 18, kernel_size=3, padding=1, stride=stride, bias=True)
        # 它们通过学习像素位置的偏移量来对输入进行变形，以适应不同的目标形状
        self.init_weights()
        self.unfold = torch.nn.Unfold(kernel_size=(2, 2), stride=2)
        # self.Fold = torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
    def init_weights(self):
        constant_init(self.offset_r, 0)
        # constant_init(self.offset_g, 0)

    def forward(self, x):
        # weight = self._get_weight(self.weight)  # 用于获取权重参数 self.weight全部零偏置
        # x = x.contiguous()
        offset = self.offset_r(x)
        t = x.contiguous()
        out_s = deform_conv2d(t, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, 1)
        return out_s
# class SAConv2d(ConvAWS2d):
#     """SAC (Switchable Atrous Convolution)
#     This is an implementation of `DetectoRS: Detecting Objects with Recursive
#     Feature Pyramid and Switchable Atrous Convolution
#     <https://arxiv.org/abs/2006.02334>`_.
#     ""
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  use_deform=True):
#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias)
#         self.weight_c = nn.Parameter(torch.Tensor(self.weight.size()))  # 权重变化的部分
#         self.offset_r = nn.Conv2d(self.in_channels, 18, kernel_size=3, padding=1, stride=stride, bias=True)
#         self.offset_g = nn.Conv2d(self.in_channels, 18, kernel_size=3, padding=1, stride=stride, bias=True)
#         # 它们通过学习像素位置的偏移量来对输入进行变形，以适应不同的目标形状
#         self.init_weights()
#
#     def init_weights(self):
#         constant_init(self.offset_r, 0)
#         constant_init(self.offset_g, 0)
#
#     def forward(self, x):
#         weight = self._get_weight(self.weight)  # 用于获取权重参数 self.weight全部零偏置
#         offset = self.offset_r(x)
#         t = x.contiguous()
#         out_s = deform_conv2d(t, offset, weight, self.stride, self.padding,
#                               self.dilation, self.groups, 1)
#         weight = weight + self.weight_c
#         offset = self.offset_g(t)
#         out_l = deform_conv2d(t, offset, weight, self.stride, self.padding, self.dilation, self.groups, 1)
#
#         out = out_s + out_l
#         return out

class MWI(nn.Module):
    '''T
    '''

    def __init__(self, c, k, stage_num=3):
        super(MWI, self).__init__()
        self.stage_num = stage_num
        # nn.Parameter(torch.Tensor(self.weight.size()))

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.w = nn.Parameter(torch.tensor(1).float())
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False))
        self.unfold_sizes = [1,3,5]
        self.convcat = nn.Conv2d(3*c, c, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x

        b, c, h, w = x.size()

        multi_scale_features = []
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        for size in self.unfold_sizes:
            # 使用不同大小的窗口进行 unfold 操作
            # print(x.shape)
            unfolded_x = F.unfold(x, kernel_size=size, padding=(size-1) // 2,stride=1)

            unfolded_x = unfolded_x.view(b, c, -1)

            # mu = self.mu.repeat(b, 1, 1)  # b * c * k
            with torch.no_grad():
                for i in range(self.stage_num):
                    unfolded_xt = unfolded_x.permute(0, 2, 1)  # b * n * c
                    z = torch.bmm(unfolded_xt, mu)  # b * n * k
                    z = F.softmax(z, dim=2)  # b * n * k
                    z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                    mu = torch.bmm(unfolded_x, z_)  # b * c * k
                    mu = self._l2norm(mu, dim=1)

            z_t = z.permute(0, 2, 1)
            z = mu.matmul(z_t).view(b,-1, h * w)
            # print(x.shape)#torch.Size([2, 256, 169]) torch.Size([2, 256, 16])

            folded_x = F.fold(z, output_size=(h, w), kernel_size=size, padding=(size-1) // 2,stride=1)
            # print(folded_x.shape)
            multi_scale_features.append(folded_x)

        # !!! The moving averaging operation is writtern in train.py, which is significant.
        u = self.convcat(torch.cat(multi_scale_features, dim=1))
        # print(x.shape)
        # z_t = u.permute(0, 2, 1)  # b * k * n
        # x = mu.matmul(z_t)  # b * c * n
        # x = x.view(b, c, h, w)  # b * c * h * w
        # x = F.relu(x, inplace=True)

        # The second 1x1 conv
        # x = self.conv2(x)
        x = u * self.w + idn
        x = F.relu(x, inplace=True)

        return x, mu

# class EMAU(nn.Module):
#     '''The Expectation-Maximization Attention Unit (EMAU).
#
#     Arguments:
#         c (int): The input and output channel number.
#         k (int): The number of the bases.
#         stage_num (int): The iteration number for EM.
#     '''
#
#     def __init__(self, c, k, stage_num=3):
#         super(EMAU, self).__init__()
#         self.stage_num = stage_num
#         # nn.Parameter(torch.Tensor(self.weight.size()))
#
#         mu = torch.Tensor(1, c, k)
#         mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
#         mu = self._l2norm(mu, dim=1)
#         self.register_buffer('mu', mu)
#         self.w = nn.Parameter(torch.tensor(1).float())
#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         idn = x
#         # The first 1x1 conv
#         # x = self.conv1(x)
#         # The EM Attention
#         b, c, h, w = x.size()
#
#         x = x.view(b, c, h * w)  # b * c * n
#         # print("x", x.shape)
#         mu = self.mu.repeat(b, 1, 1)  # b * c * k
#         with torch.no_grad():
#             for i in range(self.stage_num):
#                 x_t = x.permute(0, 2, 1)  # b * n * c
#                 z = torch.bmm(x_t, mu)  # b * n * k
#                 z = F.softmax(z, dim=2)  # b * n * k
#                 z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
#                 mu = torch.bmm(x, z_)  # b * c * k
#                 mu = self._l2norm(mu, dim=1)
#
#         # !!! The moving averaging operation is writtern in train.py, which is significant.
#
#         z_t = z.permute(0, 2, 1)  # b * k * n
#         x = mu.matmul(z_t)  # b * c * n
#         x = x.view(b, c, h, w)  # b * c * h * w
#         x = F.relu(x, inplace=True)
#
#         # The second 1x1 conv
#         # x = self.conv2(x)
#         x = x*self.w + idn
#         x = F.relu(x, inplace=True)
#
#         return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))



# 卷积
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        # self.conv = Dynamic_conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=dilation, bias=False)  ##改了动态卷积
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x

class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size

        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)

def conv_bn(inp, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        RepConv(inp, inp, kernel_size=3, stride=1, padding=1, groups=1, map_k=3),
        #conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(inp),
        nlin_layer(inplace=True)
    )

class LSFE(nn.Module):
    def __init__(self, inp, oup, ratio=2, dw_size=3):
        super(LSFE, self).__init__()
        self.oup = oup
        hidden_channels = oup // ratio
        new_channels = hidden_channels * (ratio - 1)
        c = new_channels *(dw_size ** 2)
        self.primary_conv = nn.Conv2d(inp, hidden_channels, 1)
        self.dw_size = dw_size
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(hidden_channels, c, dw_size, 1, dw_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        # self.conv_bn = conv_bn(hidden_channels)in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
        #                                                   groups=in_channel, bias=False)
        self.conv = nn.Sequential(nn.Conv2d(new_channels, new_channels, kernel_size=dw_size, stride=dw_size),
                                  nn.BatchNorm2d(new_channels),
                                  nn.ReLU())
        self.SAConv2d = nn.Sequential(nn.Conv2d(inp, hidden_channels, 1),SAConv2d(hidden_channels,c,3, 1, 1,groups=hidden_channels))
        c2wh = dict([(32, 104), (64, 52), (160, 26), (256, 13)])
        h = c2wh[inp*2]
        self.w = nn.Parameter(torch.randn(1, new_channels,dw_size ** 2, h, h),requires_grad=True)
    def forward(self, x):

        b, C, h, w = x.size()
        x1 = self.primary_conv(x)
        N = C//2
        # x2 = self.cheap_operation(x1)
        x2_weight = self.SAConv2d(x).view(b, N, self.dw_size ** 2, h, w).softmax(2)
        x2 = self.cheap_operation(x1).view(b, N, self.dw_size ** 2, h, w)
        x2 = rearrange(x2*(x2_weight+self.w), 'b N (n1 n2) h w -> b N (h n1) (w n2)', n1=self.dw_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.dw_size)
        x2 = self.conv(x2)+x1

        out = torch.cat([x1, x2], dim=1)
        return out
# BatchNorm2d = nn.BatchNorm2d
# BatchNorm1d = nn.BatchNorm1d
class RFAConv(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


class MWA(nn.Module):
    def __init__(self, img1channel, img2channel):
        super(MWA, self).__init__()
        #b0 32, 64, 160, 256    64, 128, 320, 512

        # self.w = nn.Parameter(torch.tensor(0.2))
        self.layer_cat1 = BasicConv2d(img1channel, img1channel,3,1,1)
        self.LSFE = LSFE(img1channel // 2, img1channel // 2)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.SS2D = SS2D(d_model=img1channel//2, dropout=0.0, d_state=16)
        self.conv = BasicConv2d(img1channel, img2channel,3,1,1)

    def forward(self, ful, img1):
        ################################[2, 32, 28, 28]
        """
        :param ful: 2, 64, 52
        :param img1: 2, 32, 104
        :param dep1:
        :param img: 2,64,52
        :param dep:
        :return:
        """

        # print(dep2.shape)
        out1 = img1+ful
        # out = self.upsample_2(self.conv(out1))
        # bqep = out1.permute(0, 2, 3, 1)
        input_left, input_right = out1.chunk(2, dim=1)
        input_right = self.SS2D(input_right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input_left = self.LSFE(input_left)
        weighting = self.layer_cat1(torch.cat((input_left, input_right),dim=1))
        out = self.upsample_2(self.conv(weighting))
        # out = self.upsample_2(self.conv(out1))
        return out


"""
rgb和d分别与融合的做乘法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
rgb和d分别与融合的做加法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
输出就是融合
"""


####################################################自适应1,2,3,6###########################

class LiSPNetx22(nn.Module):
    def __init__(self, channel=32):
        super(LiSPNetx22, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # Backbone model   In
        # Backbone model32, 64, 160, 256
        # self.layer_dep0 = nn.Conv2d(3, 3, kernel_size=1)
        # res_channels = [32, 64, 160, 256, 256]
        # channels = [64, 128, 256, 512, 512]
        self.resnet = mit_b0()
        self.d = dmit_b0()
        # self.layer1_depth = d.layer1
        # self.layer2_depth = d.layer2
        self.resnet.init_weights("/home/noone/桌面/李超/backbone/mit_b0.pth")

        self.d.load_state_dict(torch.load("/home/noone/桌面/李超/backbone/mit_b0.pth"), strict=False)
        # self.d.init_weights("/media/user/sdb1/LC/lichao/backbone/backbone/mit_b0.pth")
        ###############################################
        channels = [32, 64, 160, 256]
        # channels = [64, 128, 320, 512]
        self.ful_3 = MWA(channels[3], channels[2])
        self.ful_2 = MWA(channels[2], channels[1])
        self.ful_1 = MWA(channels[1], channels[0])
        self.ful_0 = MWA(channels[0], 16)
        # self.conv_img_03 = nn.Sequential(nn.Conv2d(channels[3], channels[2], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        # self.conv_img_02 = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1),
        #                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        # self.conv_img_01 = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1),
        #                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv_out1 = nn.Conv2d(16, 1, 1)
        self.conv_out2 = nn.Conv2d(channels[0], 1, 1)
        self.conv_out3 = nn.Conv2d(channels[1], 1, 1)
        # self.dualgcn = DualGCN(256)
        # self.CoordAtt = CoordAtt(channels[3])

        self.MWI = MWI(channels[3], 8, 7)
        # funsion encoders #
        ## rgb64, 128, 320, 512
        # channels = [32, 64, 160, 256]
        # channels = [64, 128, 320, 512]


        # self.UnetD = UnetD(channelout)

    def forward(self, imgs,depths):
        # depths = imgs
        img_1 = self.resnet.layer1(imgs)
        img_2 = self.resnet.layer2(img_1)

        dep_1 = self.d.layer1(depths)
        dep_2 = self.d.layer2(dep_1)
        imgd_3 = self.resnet.layer3(img_2+dep_2)
        imgd_4 = self.resnet.layer4(imgd_3)

        ful_03,mu = self.MWI(imgd_4)
        # ful_03 = imgd_4
        ful_3 = self.ful_3(ful_03, imgd_4)
        # img_2 = img_2 + self.d3to2r(img_3)
        # dep_2 = dep_2 + self.d3to2d(dep_3)
        ful_2 = self.ful_2(ful_3, imgd_3)
        # img_1 = img_1 + self.d2to1r(img_2)
        # dep_1 = dep_1 + self.d2to1d(dep_2)
        ful_1 = self.ful_1(ful_2, img_2+dep_2)
        # img_0 = img_0 + self.d1to0r(img_1)
        # dep_0 = dep_0 + self.d1to0d(dep_1)
        ful_0 = self.ful_0(ful_1, img_1+dep_1)

        ful1 = self.conv_out1(self.upsample_2(ful_0))
        ful2 = self.conv_out2(self.upsample_4(ful_1))
        ful3 = self.conv_out3(self.upsample_8(ful_2))
        predict = ful1,ful2,ful3

        # return predict
        if self.training:
        #     loss = self.crit(pred, lbl)

            return predict, mu
        else:
            return predict

        # # ful4 = self.conv_out4(self.upsample_16(ful_3))
        # return ful1, ful2, ful3  # ,ful4,OUT_3,OUT_2,OUT_1,OUT_0,img_3,dep_3,img_2, dep_2, img_1, dep_1, img_0, dep_0,ful_3,ful_2,ful_1,ful_0
        # return img_0,dep_0,#imgd_4

if __name__ == "__main__":
    rgb = torch.randn(2, 3, 416, 416).cuda()
    t = torch.randn(2, 3, 416, 416).cuda()
    model = LiSPNetx22().cuda()
    model.eval()
    out = model(rgb,t)
    for i in range(len(out)):
        print(out[i].shape)  #
