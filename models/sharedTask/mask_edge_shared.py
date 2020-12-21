import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

import pickle
affine_par = True
import functools

from libs import InPlaceABN, InPlaceABNSync
from utils.ops.dcn.modules import DeformConv2dPack
from utils.utils import get_logger

from models.sharedTask.sub_models import GCN
from models.sharedTask.sub_models import EdgeMaskModule, MaskModule, Decoder_Module, EdgeMaskGridsModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1, deform_conv=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        if deform_conv:
            self.conv2 = DeformConv2dPack(planes, planes, kernel_size=3, stride=stride, padding=dilation*multi_grid,
                                          dilation=dilation*multi_grid, bias=False, deformable_groups=1)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=False)
        self.relu_downsample = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


def save_intermediate_feature(save_feature, batch_num, save_file=None):
    assert save_file is not None
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    save_file += '/parsing_feats_{}'.format(batch_num)
    print(save_file)
    with open(save_file, 'wb') as f:
        xfg_sig_max = torch.max(save_feature, dim=1)[0]
        pickle.dump(xfg_sig_max, f)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, 
            with_dcn=[1,1,1], 
            with_edge=True,
            with_mask_atten=True,
            with_mask_edge=True,
            with_mask_pars=True,
            with_rescore=True, 
            rescore_k_size=31):
        self.with_edge = with_edge
        self.with_mask_atten = with_mask_atten
        self.with_mask_edge = with_mask_edge
        self.with_mask_pars = with_mask_pars

        self.inplanes = 128
        self.inplanes_init = self.inplanes
        self.with_rescore = with_rescore
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, deform_conv=with_dcn[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, deform_conv=with_dcn[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1), deform_conv=with_dcn[2])
        self.layer5 = PSPModule(2048, 512)

        if with_edge:
            self.edge_layer = EdgeMaskModule(num_classes=num_classes, 
                                        with_mask_edge=with_mask_edge, 
                                        with_mask_pars=with_mask_pars)
        else:
            self.edge_layer = MaskModule(num_classes=num_classes, 
                                        with_mask_pars=with_mask_pars)
        self.layer6 = Decoder_Module(num_classes)

        fusions = []
        if with_edge and with_mask_edge:#fuse with edge features, mask features also fused with edge features
            fusions.append(nn.Conv2d(5*256, 256, kernel_size=1, padding=0, dilation=1, bias=False))
        elif with_edge:#only fuse with edge features
            fusions.append(nn.Conv2d(4*256, 256, kernel_size=1, padding=0, dilation=1, bias=False))
        else:#only fuse with mask branch
            fusions.append(nn.Conv2d(256+512, 256, kernel_size=1, padding=0, dilation=1, bias=False))
        fusions.append(InPlaceABNSync(256))
        fusions.append(nn.Dropout2d(0.1))
        fusions.append(nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        self.layer7 = nn.Sequential(*fusions)

            
        if with_rescore:
            assert rescore_k_size >= 1
            if rescore_k_size > 3:
                self.seg_rescore_conv = GCN(num_classes*3, num_classes, [rescore_k_size, rescore_k_size])
            elif rescore_k_size == 1:
                self.seg_rescore_conv = nn.Conv2d(num_classes*3, num_classes, 1, stride=1)
            elif rescore_k_size == 3:
                self.seg_rescore_conv = nn.Conv2d(num_classes*3, num_classes, 3, padding=1, stride=1)
        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, deform_conv=False):
        if deform_conv:
            print("###conv layer with {} blocks run in deformable convolution.".format(blocks))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), deform_conv=deform_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid), deform_conv=deform_conv))

        return nn.Sequential(*layers)

    def forward(self, x, batch_num=1):
        x_conv1 = self.relu1(self.bn1(self.conv1(x)))
        x_conv2 = self.relu2(self.bn2(self.conv2(x_conv1)))
        x_conv3 = self.relu3(self.bn3(self.conv3(x_conv2)))
        x_max = self.maxpool(x_conv3)

        x2 = self.layer1(x_max)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x_psp = self.layer5(x5) # PSPModule

        if self.with_edge:
            preds, x_fg, x_edges = self.edge_layer(x2, x3, x4, x_psp)
            edge_pred, fg_pred, seg_pred = preds
        else:
            preds, x_fg, x_seg = self.edge_layer(x2, x3, x4, x_psp, batch_num)
            fg_pred, seg_pred = preds


        if not self.with_mask_atten:
            seg1, x_decod = self.layer6(x_psp, x2, xfg=None)  
        else:
            seg1, x_decod = self.layer6(x_psp, x2, xfg=x_fg)

        if self.with_edge:
            x_cat = torch.cat([x_decod, x_edges], dim=1)
        else:
            x_cat = torch.cat([x_decod, x_seg], dim=1)

        seg2 = self.layer7(x_cat)

        if not self.with_edge:
            edge_pred = None

        if self.with_rescore:
            seg_resore = self.seg_rescore_conv(torch.cat([seg1, seg2, seg_pred], dim=1))
            return [[seg1, seg2, seg_pred, seg_resore], edge_pred, fg_pred]
        else:
            if seg_pred is not None:
                return [[seg1, seg2, seg_pred], edge_pred, fg_pred]
            else:
                return [[seg1, seg2], edge_pred, fg_pred]


def Res_Deeplab(num_classes=7, with_dcn=[1,1,1], with_rescore=True, rescore_k_size=31):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, with_dcn=with_dcn, 
                    with_rescore=with_rescore, rescore_k_size=rescore_k_size)
    return model

def build_ResDeeplab(cfg):
    layers = [3, 4, 23, 3] if int(cfg.MODEL.RES_TYPE) == 101 else [3, 4, 6, 3]
    model = ResNet(Bottleneck, 
                layers,
                cfg.TRAIN.NUM_CLASSES,
                with_dcn=cfg.MODEL.WITH_DCN, 
                with_edge=cfg.MODEL.WITH_EDGE,
                with_mask_atten=cfg.MODEL.WITH_MASK_ATTEN,
                with_mask_edge=cfg.MODEL.WITH_MASK_EDGE,
                with_mask_pars=cfg.MODEL.WITH_MASK_PARS,
                with_rescore=cfg.MODEL.WITH_RESCORE, 
                rescore_k_size=cfg.MODEL.RESCORE_K)
    return model