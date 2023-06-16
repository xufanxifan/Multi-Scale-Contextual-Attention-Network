import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
# from network.utils import BNReLU
# from network.mynn import Norm2d
# from config import cfg

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
#
# def BNReLU(ch):
#     return nn.Sequential(
#         Norm2d(ch),
#         nn.ReLU())
#
# def Norm2d(in_channels, **kwargs):
#     """
#     Custom Norm Function to allow flexible switching
#     """
#     # layer = getattr(cfg.MODEL, 'BNFUNC')
#     # normalization_layer = layer(in_channels, **kwargs)
#     return nn.Sequential(
#         nn.BatchNorm2d(in_channels, **kwargs))

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PSA_m(nn.Module):
    """
        mask_cha: channel of mask (mask can also be x)
        x_cha: channel of x

    """

    def __init__(self, mask_cha, x_chan):
        super(PSA_m, self).__init__()

        self.opt_cm_down_1 = nn.Conv2d(mask_cha, x_chan, kernel_size=1, stride=1)
        self.opt_cm_down_2 = nn.Conv2d(x_chan, x_chan, kernel_size=3, stride=1, padding=1, bias=True)
        self.opt_sm_down_1 = nn.Conv2d(mask_cha, 1, kernel_size=1, stride=1)
        self.opt_sm_down_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.reset_parameters()
        self.bn_c = nn.BatchNorm2d(x_chan)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_s = nn.BatchNorm2d(1)

    def reset_parameters(self):
        kaiming_init(self.opt_cm_down_1, mode='fan_in')
        kaiming_init(self.opt_sm_down_1, mode='fan_in')
        kaiming_init(self.opt_cm_down_2, mode='fan_in')
        kaiming_init(self.opt_sm_down_2, mode='fan_in')

        self.opt_cm_down_1.inited = True
        self.opt_sm_down_1.inited = True
        self.opt_cm_down_2.inited = True
        self.opt_sm_down_2.inited = True

    def spatial_pool(self, x):
        # (batchasize, classes, 1, 1)
        feats_opt_c_1 = self.opt_cm_down_1(x)
        feats_opt_c_1 = self.opt_cm_down_2(feats_opt_c_1)
        feats_opt_c_1 = self.bn_c(feats_opt_c_1)
        feats_opt_c_1 = self.relu(feats_opt_c_1)
        feats_opt_c_1 = self.pool(feats_opt_c_1)
        feats_opt_cm = nn.functional.softmax(feats_opt_c_1, dim=1)

        return feats_opt_cm

    def channel_pool(self, x):
        # (batchasize, 1, h, w)
        feats_opt_sm = self.opt_sm_down_1(x)
        feats_opt_sm = self.opt_sm_down_2(feats_opt_sm)
        feats_opt_sm = self.bn_s(feats_opt_sm)
        feats_opt_sm = self.relu(feats_opt_sm)

        return feats_opt_sm

    def forward(self, mask, x):
        mask_s = self.spatial_pool(mask)
        mask_c = self.channel_pool(mask)
        out = x * mask_s * mask_c

        return out
