"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
# from torch import nn
from torch import nn, tensor, empty, Tensor, matmul, functional
import pandas as pd
from network.mynn import initialize_weights, Upsample, scale_as, Norm2d
from network.mynn import ResizeX
from network.utils import get_trunk
from network.utils import BNReLU, get_aspp
from network.utils import make_attn_head, make_multi_attn_head
from network.ocr_utils import SpatialGather_Module, SpatialOCR_Module, SpatialOCR_Module_v2
from config import cfg
from utils.misc import fmt_scale
from collections import OrderedDict
from torch import cat
from torch import stack
import numpy as np
# from .PSA import PSA_p, PSA_s
from .PSAv2 import PSA_m


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, high_level_ch):
        super(OCR_block, self).__init__()

        ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats



class OCR_block_v7(nn.Module):
    """
    Achieve polarized self-attention function
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, high_level_ch):
        super(OCR_block_v7, self).__init__()

        ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module_v2(in_channels=ocr_mid_channels,
                                                    key_channels=ocr_key_channels,
                                                    out_channels=ocr_mid_channels,
                                                    scale=1,
                                                    dropout=0.05,
                                                    )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.extra = nn.Sequential(
            nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=1),
            BNReLU(1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            BNReLU(1),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # self.extra_1 = nn.Sequential(
        #     nn.Conv2d(high_level_ch, num_classes, kernel_size=3, stride=1, padding=1),
        #     BNReLU(num_classes),
        #     nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=1),
        #     BNReLU(1),
        #     nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )
        self.drop = nn.Dropout(0.5)

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, lo_high_level_features, hi_high_level_features):
        feats = self.conv3x3_ocr(hi_high_level_features)
        lo_aux_out = self.aux_head(lo_high_level_features)
        lo_aux_out = scale_as(lo_aux_out, feats)
        hi_aux_out = self.aux_head(hi_high_level_features)
        hi_aux_out = scale_as(hi_aux_out, feats)



        lo_context = self.ocr_gather_head(feats, lo_aux_out)
        hi_context = self.ocr_gather_head(feats, hi_aux_out)

        ocr_feats = self.ocr_distri_head(feats, lo_context, hi_context)

        ini_cls_out = self.cls_head(ocr_feats)
        # # test22
        # cls_out = self.extra(
        #     ini_cls_out ) + ini_cls_out
        # test23
        # cls_out = self.extra(
        #     ini_cls_out ) + ini_cls_out
        # test24
        cls_out = self.drop(self.extra(
            ini_cls_out ) )+ ini_cls_out


        return cls_out, lo_aux_out, hi_aux_out

class OCRNet(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(OCRNet, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            aux_loss = self.criterion(aux_out, gts,
                                      do_rmi=cfg.LOSS.OCR_AUX_RMI)
            main_loss = self.criterion(cls_out, gts)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict


class MHAModule(nn.Module):
    """
    MHA module
    """

    def __init__(self, input_ch):
        super(MHAModule, self).__init__()

        self.heads = 8
        # test1
        # self.scale_attn = make_attn_head(
        #     in_ch=input_ch, out_ch=self.heads)
        # od = OrderedDict([('conv0', nn.Conv2d(input_ch * self.heads, input_ch, kernel_size=1, bias=False)),
        #                   ('sig', nn.Sigmoid())])
        # test2&test4
        self.scale_attn = make_multi_attn_head(
            in_ch=input_ch, out_ch=self.heads)
        od = OrderedDict([('conv0', nn.Conv2d(input_ch * self.heads, input_ch, kernel_size=1, bias=False)),
                          ('bn0', Norm2d(input_ch)),
                          ('re0', nn.ReLU(inplace=True))])
        self.final_head = nn.Sequential(od)
        # test3
        # self.scale_attn = make_attn_head(
        #     in_ch=input_ch, out_ch=self.heads)
        # od = OrderedDict([('conv0', nn.Conv2d(input_ch * self.heads, input_ch, kernel_size=1, bias=False)),
        #                   ('bn0', Norm2d(input_ch)),
        #                   ('re0', nn.ReLU(inplace=True))])
        # self.final_head = nn.Sequential(od)

    def forward(self, inputs):
        multi_head_kq = self.scale_attn(inputs)
        heads_att = []
        combine_each_pic = []
        heads_per_pic = []
        for i in range(self.heads):
            head_att_i = []
            for j in range(multi_head_kq.shape[0]):
                head_att_i.append(multi_head_kq[j, i, :, :] * inputs[j, :, :, :])
            heads_att.append(head_att_i)

        for i in range(multi_head_kq.shape[0]):
            for j in range(self.heads):
                heads_per_pic.append(heads_att[j][i])

        for i in range(multi_head_kq.shape[0]):
            combine_each_pic.append(cat(heads_per_pic[i * self.heads:(i + 1) * self.heads], 0))
        combine_each_pic = stack(combine_each_pic)
        output = self.final_head(combine_each_pic)
        return output


class SAModule(nn.Module):
    """
    MHA module
    """

    def __init__(self):
        super(SAModule, self).__init__()
        input_ch = 8
        qod = OrderedDict([('conv0', nn.Conv2d(input_ch, 1, kernel_size=3, padding=1, bias=False)),
                           ('bn0', Norm2d(1)),
                           ('re0', nn.ReLU(inplace=True))])
        testod = OrderedDict([('conv0', nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)),
                              ('bn0', Norm2d(1)),
                              ('re0', nn.ReLU(inplace=True)),
                              ('conv1', nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)),
                              ('bn1', Norm2d(1)),
                              ('re1', nn.ReLU(inplace=True)),
                              ('conv2', nn.Conv2d(1, 1, kernel_size=1, bias=False)),
                              ('sig', nn.Sigmoid())])
        kod = OrderedDict(
            [('conv0', nn.Conv2d(cfg.MODEL.OCR.MID_CHANNELS, 1, kernel_size=1, bias=False)),
             ('bn0', Norm2d(1)),
             ('re0', nn.ReLU(inplace=True))])
        # self.scale_attn = make_multi_attn_head(in_ch=1, out_ch=1)
        self.wq_x = nn.Sequential(qod)
        # self.wq_y = nn.Sequential(qod)
        # self.wk_x = nn.Sequential(kod)
        self.wk_y = nn.Sequential(qod)
        self.w_x = nn.Sequential(testod)
        # od1 = OrderedDict([('conv0', nn.Conv2d(input_ch, input_ch, kernel_size=1, bias=False)),
        #                    ('bn0', Norm2d(input_ch)),
        #                    ('re0', nn.ReLU(inplace=True))])
        # self.final = nn.Sequential(od1)

    def forward(self, input_x, att_x, input_y, att_y):
        q_x = self.wq_x(input_x)
        # k_x = self.wk_x(att_x)
        # q_y = self.wq_y(input_y)
        # k_y = self.wk_y(att_y)
        k_y = self.wk_y(input_y)
        # q_x = nn.functional.softmax(input_x, dim=1)
        # q_y = nn.functional.softmax(input_y, dim=1)

        q_x = scale_as(q_x, input_y)
        # k_x = scale_as(k_x, input_y)
        k_y = scale_as(k_y, input_y)
        input_x = scale_as(input_x, input_y)

        # test1-7
        # output_x = q_x * k_y * input_x + q_x * k_x * input_x
        # output_y = q_y * k_x * input_y + q_y * k_y * input_y
        # # test4
        # # output = q_x * k_y * input_x + q_y * k_x * input_y
        # # test5&test6
        # output = self.final(output_x + output_y)
        # test8
        # output=[]
        # for i in range(input_x.shape[0]):
        #     # test8
        #     # w= q_y[i, :, :, :] * k_x[i, :, :, :]
        #     # output.append((1 - w) * input_x[i, :, :, :] + w * input_y[i, :, :, :])
        #     #test9
        #     w = q_x[i, :, :, :] * k_y[i, :, :, :]
        #     output.append(w * input_x[i, :, :, :] + input_y[i, :, :, :])
        # output=stack(output,0)
        # #test10
        # output=input_y
        # test11
        output_x = self.w_x(q_x * k_y) * input_x
        output = output_x + input_y
        # # test12
        # output_x = q_x * k_y * input_x + q_x*k_x*input_x
        # output = output_x + input_y
        return output


class OCRNetASPP(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(OCRNetASPP, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8)
        self.ocr = OCR_block(aspp_out_ch)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        aspp = self.aspp(high_level_features)
        cls_out, aux_out, _ = self.ocr(aspp)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            loss = cfg.LOSS.OCR_ALPHA * self.criterion(aux_out, gts) + \
                   self.criterion(cls_out, gts)
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict


class MscaleOCR(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)
        self.multihead = MHAModule()
        self.selfhead = SAModule()

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        multi_head_features = self.multihead(high_level_features)
        cls_out, aux_out, ocr_mid_feats = self.ocr(multi_head_features)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)

        return {'cls_out': cls_out,
                'aux_out': aux_out,
                'mid_feats': ocr_mid_feats}

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        feat_lo = lo_outs['mid_feats']

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']
        feat_hi = hi_outs['mid_feats']

        joint_pred = self.selfhead(pred_05x, feat_lo, pred_10x, feat_hi)
        joint_aux = self.selfhead(aux_lo, feat_lo, aux_1x, feat_hi)

        if self.training:
            gts = inputs['gts']
            do_rmi = cfg.LOSS.OCR_AUX_RMI
            aux_loss = self.criterion(joint_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_lo
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
            }
            return output_dict

    def forward(self, inputs):

        if cfg.MODEL.N_SCALES and not self.training:
            return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)

        return self.two_scale_forward(inputs)


def HRNet(num_classes, criterion):
    return OCRNet(num_classes, trunk='hrnetv2', criterion=criterion)


def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)


class MscaleOCR_v2(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(MscaleOCR_v2, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block_v7(high_level_ch)
        input_ch = 720
        self.multihead = MHAModule(input_ch)
        # input_ch = 48
        # self.multihead_single_h = MHAModule(input_ch)
        # input_ch = 96
        # self.multihead_single_mh = MHAModule(input_ch)
        # input_ch = 192
        # self.multihead_single_ml = MHAModule(input_ch)
        # input_ch = 384
        # self.multihead_single_l = MHAModule(input_ch)
        # self.selfhead = SAModule()

    def _fwd(self, x):

        _, _, high_level_features = self.backbone(x)

        multi_head_features = self.multihead(high_level_features)
        # multi_head_features = high_level_features

        # multi_head_single_h_features = self.multihead_single_h(high_resolution_features)
        # multi_head_single_mh_features = self.multihead_single_mh(midh_resolution_features)
        # multi_head_single_ml_features = self.multihead_single_ml(midl_resolution_features)
        # multi_head_single_l_features = self.multihead_single_l(low_resolution_features)

        # cls_out, aux_out, ocr_mid_feats = self.ocr(multi_head_features)

        return {'hr_feats': multi_head_features}

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        feat_05x = lo_outs['hr_feats']
        hi_outs = self._fwd(x_1x)
        feat_10x = hi_outs['hr_feats']
        # feat_hi_10x = hi_outs['ext_h_feats']
        # feat_mh_10x = hi_outs['ext_mh_feats']
        # feat_ml_10x = hi_outs['ext_ml_feats']
        # feat_lo_10x = hi_outs['ext_l_feats']
        # joint_pred, lo_aux, _, hi_s_aux, mh_s_aux, ml_s_aux, lo_s_aux = self.ocr(feat_05x, feat_10x, feat_hi_10x, feat_mh_10x, feat_ml_10x, feat_lo_10x)
        joint_pred, lo_aux, hi_aux, = self.ocr(feat_05x, feat_10x)
        x_1x_size = x_1x.size()[2:]
        joint_pred = Upsample(joint_pred, x_1x_size)
        lo_aux = Upsample(lo_aux, x_1x_size)
        hi_aux = Upsample(hi_aux, x_1x_size)
        # hi_aux = Upsample(hi_s_aux, x_1x_size)
        # mh_aux = Upsample(mh_s_aux, x_1x_size)
        # ml_aux = Upsample(ml_s_aux, x_1x_size)
        # ls_aux = Upsample(lo_s_aux, x_1x_size)

        if self.training:
            gts = inputs['gts']
            do_rmi = cfg.LOSS.OCR_AUX_RMI

            lo_aux_loss = self.criterion(lo_aux, gts, do_rmi=do_rmi)
            hi_aux_loss = self.criterion(hi_aux, gts, do_rmi=do_rmi)
            # mh_aux_loss = self.criterion(mh_aux, gts, do_rmi=do_rmi)
            # ml_aux_loss = self.criterion(ml_aux, gts, do_rmi=do_rmi)
            # ls_aux_loss = self.criterion(ls_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = cfg.LOSS.OCR_ALPHA * 0.5 * lo_aux_loss \
                   + cfg.LOSS.OCR_ALPHA * 0.5 * hi_aux_loss + main_loss

            return loss
        else:
            output_dict = {
                'pred': joint_pred,
            }
            return output_dict

    def forward(self, inputs):

        if cfg.MODEL.N_SCALES and not self.training:
            return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)

        return self.two_scale_forward(inputs)


def HRNet(num_classes, criterion):
    return OCRNet(num_classes, trunk='hrnetv2', criterion=criterion)


def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)
