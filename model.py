import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from Utils import *
import torch.fft
import einops
from ConvGRU_net import Conv2dGRU


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1.
        output[input < 0] = 0.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply


class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y


class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.CAlayer = CALayer(channel)

    def forward(self, x):
        # shortcut = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        shortcut2 = x
        x = self.CAlayer(x)
        x = x * shortcut2
        # x = x + shortcut
        return x



class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(n_feat, 28, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(28, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)

    def forward(self, x, x_img):
        img = self.conv2(x) + x_img

        return img


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, gh, gw, C = x.shape

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        weight_1 = weight.data.cpu().numpy()
        x = x * weight
        x = torch.fft.irfft2(x, s=(gh, gw), dim=(2, 3), norm='ortho')

        return x, weight_1


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (fh fw) gh gw c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (fh fw) gh gw c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


## Res-Encoder

class GFBlock(nn.Module):

    def __init__(self, dim, patch_size=16, mlp_ratio=4., drop=0., h=14, w=8):
        super(GFBlock, self).__init__()

        self.fh, self.fw = patch_size, patch_size
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        shortcut = x
        n, h, w, num_channels = x.shape
        gh, gw = h // self.fh, w // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))
        # print('x.size=',x.size())
        x = self.norm1(x)
        # x = self.filter(x)
        x, weight = self.filter(x)
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        x = x + shortcut
        x = self.norm2(x)
        x = self.mlp(x)
        return x, weight


def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)

    mean = var_mean[1]

    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature


class GetSpatialGatingWeights(nn.Module):
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, in_channels, use_bias=True):
        super().__init__()

        self.features = in_channels
        self.bias = use_bias
        self.conv = nn.Conv2d(self.features, self.features, kernel_size=1, stride=1, bias=self.bias)
        self.dwconv = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1, groups=self.features,
                                bias=self.bias)
        self.gelu = nn.GELU()

    def forward(self, x):
        # input projection
        x = layer_norm_process(x)
        x = self.conv(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        return x


class CrossGatingBlock(nn.Module):  # 缺dim     num_channels  n, h, w, num_channels = x.shape
    """Cross-gating MLP block."""

    def __init__(self, features, use_bias=True):
        super().__init__()
        self.features = features
        self.bias = use_bias

        self.conv6 = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1, bias=self.bias)
        self.conv5 = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

        self.getspatialgatingweights1 = GetSpatialGatingWeights(in_channels=self.features, use_bias=self.bias)
        self.getspatialgatingweights2 = GetSpatialGatingWeights(in_channels=self.features, use_bias=self.bias)

    def forward(self, x, y):
        shortcut_x = x
        shortcut_y = y

        gx = self.getspatialgatingweights1(x)
        # gy = self.getspatialgatingweights2(y)
        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.conv5(y)
        y = y + shortcut_y
        # x = x * gy  # gating x using y
        # x = self.conv6(x)
        # x = x + y + shortcut_x  # get all aggregated signals
        return y




class MCBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.features, self.features, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(self.features * 2, self.features * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.confuse = nn.Conv2d(self.features * 2, self.features, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_1 = self.relu(self.conv1(x))
        x_2 = self.relu(self.conv2(x))
        out_1 = torch.cat((x_1, x_2), 1)
        out_2 = self.relu(self.conv3(out_1))
        out = self.confuse(out_2)
        return out


class Encoding(nn.Module):
    def __init__(self, mlp_ratio=4., drop_rate=0.):
        super(Encoding, self).__init__()

        self.GFBlock1 = GFBlock(dim=32, patch_size=16, mlp_ratio=mlp_ratio, drop=drop_rate, h=16, w=9)
        self.CAB1 = CAB(channel=32)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.GFBlock2 = GFBlock(dim=64, patch_size=16, mlp_ratio=mlp_ratio, drop=drop_rate, h=8, w=5)
        self.CAB2 = CAB(channel=64)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.GFBlock3 = GFBlock(dim=128, patch_size=8, mlp_ratio=mlp_ratio, drop=drop_rate, h=8, w=5)
        self.CAB3 = CAB(channel=128)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.MC1 = MCBlock(features=64)
        self.MC2 = MCBlock(features=128)
        self.MC3 = MCBlock(features=128)

    def forward(self, x):
        ## encoding blocks

        gfb1 = x.permute(0, 2, 3, 1)
        gfb1 = self.GFBlock1(gfb1)
        gfb1 = gfb1.permute(0, 3, 1, 2)
        gfb1_1 = gfb1.data.cpu().numpy()

        cab1 = self.CAB1(x)
        cab1_1 = cab1.data.cpu().numpy()

        E1 = torch.cat([gfb1, cab1], dim=1)
        E1 = self.conv1(E1)
        # E1 = self.MC1(E1)

        E3 = F.avg_pool2d(E1, kernel_size=2, stride=2)
        gfb2 = E3.permute(0, 2, 3, 1)
        gfb2 = self.GFBlock2(gfb2)
        gfb2 = gfb2.permute(0, 3, 1, 2)
        cab2 = self.CAB2(E3)
        E3 = torch.cat([gfb2, cab2], dim=1)
        # E3 = self.CrossGatingBlock2(gfb2, cab2)
        E3 = self.conv2(E3)
        # E3 = self.MC2(E3)

        E5 = F.avg_pool2d(E3, kernel_size=2, stride=2)
        gfb3 = E5.permute(0, 2, 3, 1)
        gfb3 = self.GFBlock3(gfb3)
        gfb3 = gfb3.permute(0, 3, 1, 2)
        cab3 = self.CAB3(E5)
        E5 = torch.cat([gfb3, cab3], dim=1)
        # E5 = self.CrossGatingBlock3(gfb3, cab3)
        E5 = self.conv3(E5)
        E5 = self.MC3(E5)

        return E1,  E3,  E5


class Encoding_C0(nn.Module):
    def __init__(self):
        super(Encoding_C0, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                )

        self.E3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

        self.E5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                )

    def forward(self, x):
        ## encoding blocks
        E1 = self.E1(x)

        E3 = self.E3(F.avg_pool2d(E1, kernel_size=2, stride=2))

        E5 = self.E5(F.avg_pool2d(E3, kernel_size=2, stride=2))

        return E1, E3, E5


class Decoding(nn.Module):
    def __init__(self, mlp_ratio=4., drop_rate=0.):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'

        # self.GFBlock4 = GFBlock(dim=64, patch_size = 16, mlp_ratio=mlp_ratio,drop=drop_rate, h=8, w=5)
        # self.CAB4 = CAB(channel=64)
        # self.GFBlock5 = GFBlock(dim=32, patch_size = 16, mlp_ratio=mlp_ratio,drop=drop_rate, h=16, w=9)
        # self.CAB5 = CAB(channel=32)
        # self.D1 = nn.Conv2d(in_channels=128+128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=128 + 128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        # self.D3 = nn.Conv2d(in_channels=64+64,  out_channels=32, kernel_size=3, stride=1, padding=1)
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        # self.D3 = nn.Conv2d(in_channels=64+64,  out_channels=32, kernel_size=3, stride=1, padding=1)

        # self.conv = nn.Conv2d(in_channels=32+32,  out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, E1, E3, E5):
        ## decoding blocks
        # D1 = self.D1(torch.cat([E3, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))  #(b,64,128,128)
        # gfb4 = D1.permute(0, 2, 3, 1)
        # gfb4 = self.GFBlock4(gfb4)
        # gfb4 = gfb4.permute(0, 3, 1, 2)
        # cab4 = self.CAB4(D1)
        # D1 = torch.cat([gfb4, cab4], dim=1)

        #
        # D3 = self.D3(torch.cat([E1, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1)) #(b,32,256,256)
        # gfb5 = D3.permute(0, 2, 3, 1)
        # gfb5 = self.GFBlock5(gfb5)
        # gfb5 = gfb5.permute(0, 3, 1, 2)
        # cab5 = self.CAB5(D3)
        # D3 = torch.cat([gfb5, cab5], dim=1)
        # D3 = self.conv(D3)

        D1 = self.D1(torch.cat([E3, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))  # (b,64,128,128)
        # D1 = D1.permute(0, 2, 3, 1)
        # D1 = self.GFBlock4(D1)
        # D1 = D1.permute(0, 3, 1, 2)
        # D1 = self.CAB4(D1)

        D3 = self.D3(torch.cat([E1, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1))  # (b,32,256,256)
        # D3 = D3.permute(0, 2, 3, 1)
        # D3 = self.GFBlock5(D3)
        # D3 = D3.permute(0, 3, 1, 2)
        # D3 = self.CAB5(D3)

        return D3


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.
    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock of the RecurrentVarNet.
    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int = 4,
    ):

        super().__init__()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            block = [nn.Conv2d(in_channels, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class DFFMM(nn.Module):
    def __init__(self, Ch, stages, size, in_channels: int = 128, recurrent_hidden_channels: int = 128,recurrent_num_layers: int = 1
                 ):
        super(DFFMM, self).__init__()
        self.Ch = Ch
        self.s = stages
        self.size = size

        self.initializer1 = RecurrentInit(64, 64, depth=1)
        self.initializer2 = RecurrentInit(128, 128, depth=1)
        self.initializer3 = RecurrentInit(in_channels, recurrent_hidden_channels, depth=recurrent_num_layers)

        regularizer_params1 = {"in_channels": 64, "hidden_channels": 64, "num_layers": 1, "replication_padding": True}
        regularizer_params2 = {"in_channels": 128, "hidden_channels": 128, "num_layers": 1, "replication_padding": True}
        regularizer_params3 = {"in_channels": 128, "hidden_channels": 128, "num_layers": 1, "replication_padding": True}

        self.Conv2dGRU1 = Conv2dGRU(**regularizer_params1)
        self.Conv2dGRU2 = Conv2dGRU(**regularizer_params2)
        self.Conv2dGRU3 = Conv2dGRU(**regularizer_params3)


        self.Phi = Parameter(torch.ones(self.size, self.size), requires_grad=True)
        torch.nn.init.normal_(self.Phi, mean=0, std=0.1)

        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                               Resblock(64), Resblock(64),
                               nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())


        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_5 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_6 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_7 = Parameter(torch.ones(1), requires_grad=True)

        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_7, mean=0.1, std=0.01)


        self.Encoding = Encoding()
        self.Encoding_C0 = Encoding_C0()
        self.Decoding = Decoding()

        self.conv = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)

        self.SAM0 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM1 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM2 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM3 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM4 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM5 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM6 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM7 = SAM(n_feat=32, kernel_size=3, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def recon(self, res1, Xt, i):
        if i == 0:
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        elif i == 4:
            delta = self.delta_4
        elif i == 5:
            delta = self.delta_5
        elif i == 6:
            delta = self.delta_6
        elif i == 7:
            delta = self.delta_7

        Xt = Xt - delta * res1
        return Xt

    def forward(self, training_label):

        ## Sampling Subnet ##
        batch, _, _, _ = training_label.shape
        Phi_ = MyBinarize(self.Phi)

        PhiWeight = Phi_.contiguous().view(1, 1, self.size, self.size)
        PhiWeight = PhiWeight.repeat(batch, 28, 1, 1)

        temp = training_label.mul(PhiWeight)
        temp_shift = torch.Tensor(np.zeros((batch, 28, self.size, self.size + (28 - 1) * 2))).cuda()
        temp_shift[:, :, :, 0:self.size] = temp
        for t in range(28):
            temp_shift[:, t, :, :] = torch.roll(temp_shift[:, t, :, :], 2 * t, dims=2)
        meas = torch.sum(temp_shift, dim=1).cuda()

        y = meas / 28 * 2
        y = y.unsqueeze(1).cuda()

        Xt = y2x(y)
        Xt_ori = Xt

        OUT = []

        for i in range(0, self.s):
            AXt = x2y(self.A(Xt))
            Res1 = self.AT(y2x(AXt - y))
            Xt = self.recon(Res1, Xt, i)
            fea = self.conv(Xt)
            if i == 0:
                fea_init_C1, fea_init_C2, fea_init_C3 = self.Encoding_C0(fea)
                previous_state1 = self.initializer1(fea_init_C1)
                previous_state2 = self.initializer2(fea_init_C2)
                previous_state3 = self.initializer3(fea_init_C3)

            E1,  E3,  E5 = self.Encoding(fea)
            E1, previous_state1 = self.Conv2dGRU1(E1, previous_state1)
            E3, previous_state2 = self.Conv2dGRU2(E3, previous_state2)
            E5, previous_state3 = self.Conv2dGRU3(E5, previous_state3)
            Xt = self.Decoding(E1, E3, E5)

            if i == 0:
                Xt = self.SAM0(Xt, Xt_ori)
            elif i == 1:
                Xt = self.SAM1(Xt, Xt_ori)
            elif i == 2:
                Xt = self.SAM2(Xt, Xt_ori)
            elif i == 3:
                Xt = self.SAM3(Xt, Xt_ori)
            elif i == 4:
                Xt = self.SAM4(Xt, Xt_ori)
            elif i == 5:
                Xt = self.SAM5(Xt, Xt_ori)
            elif i == 6:
                Xt = self.SAM6(Xt, Xt_ori)
            elif i == 7:
                Xt = self.SAM7(Xt, Xt_ori)
            OUT.append(Xt)

        return OUT, Phi_,y
