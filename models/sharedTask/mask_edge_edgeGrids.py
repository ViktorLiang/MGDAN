import torch.nn as nn
from torch.nn import functional as F
import torch

import pickle
affine_par = True
import functools

from libs import InPlaceABN, InPlaceABNSync
from utils.ops.dcn.modules import DeformConv2d, _DeformConv2d, DeformConv2dPack, DeformConv2dPackMore
# from models.simplified.sub_models import EdgeMaskGridsModule, EdgeGridsMaskModule
from models.sharedTask.sub_models import GCN, EdgeGridsMaskModule

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


class EdgeMaskModule(nn.Module):
    def __init__(self,in_fea=[256, 512, 1024, 512], mid_fea=256, out_fea=2, num_classes=7):
        super(EdgeMaskModule, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_fea[3], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv_seg_down = nn.Sequential(
            nn.Conv2d(in_fea[3], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv_layer_edge = GCN(mid_fea, out_fea, k=(5, 5))
        self.conv_final_edge = GCN(out_fea*4, out_fea, k=(7, 7))

        self.cat_fg_conv = nn.Sequential(nn.Conv2d(4 * 256, 256, kernel_size=3, padding=1, bias=False),
                                      InPlaceABNSync(256),
                                      nn.ReLU(inplace=False))

        self.cat_seg_conv = nn.Sequential(nn.Conv2d(4 * 256, 512, kernel_size=3, padding=1, bias=False),
                                      InPlaceABNSync(512),
                                      nn.ReLU(inplace=False))

        self.down_fg_conv = nn.Sequential(
                                nn.Dropout2d(0.1),
                                nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, bias=True))
        self.down_seg_conv = nn.Sequential(
                                nn.Dropout2d(0.1),
                                nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

        self._init_weight()
            
    def forward(self, x1, x2, x3, x_psp):
        _, _, h, w = x1.size()
        
        layer1_fea = self.conv1(x1)
        layer2_fea = self.conv2(x2)
        layer3_fea = self.conv3(x3)
        layer4_fea = self.conv4(x_psp)##
        layer2_fea = F.interpolate(layer2_fea, size=(h, w), mode='bilinear', align_corners=True)
        layer3_fea = F.interpolate(layer3_fea, size=(h, w), mode='bilinear', align_corners=True)
        layer4_fea = F.interpolate(layer4_fea, size=(h, w), mode='bilinear', align_corners=True)
        layers_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, layer4_fea], dim=1)

        fg_fea = self.cat_fg_conv(layers_fea)
        fg_pred = self.down_fg_conv(fg_fea)
        seg_fea = self.cat_seg_conv(layers_fea)
        seg_pred = self.down_seg_conv(seg_fea)

        edge1 = self.conv_layer_edge(layer1_fea)
        edge2 = self.conv_layer_edge(layer2_fea)
        edge3 = self.conv_layer_edge(layer3_fea)
        edge4_fea = self.conv_seg_down(seg_fea) ## 512->256
        edge4 = self.conv_layer_edge(edge4_fea) ##
        # edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=True)
        # edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=True)
        edge = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        edge_pred = self.conv_final_edge(edge)

        edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, edge4_fea], dim=1)
        
        return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')

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


class Decoder_Module(nn.Module):

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # self.save_file_before = './output/parsing_mask_atten_max/before_atten/snapshots_mskDCN_edgLK7/'
        # self.save_file_after = './output/parsing_mask_atten_max/after_atten/snapshots_mskDCN_edgLK7/'

    def forward(self, xt, xl, xfg, batch_num):
        _, _, h, w = xl.size()

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x) # torch.Size([bsz, 256, 96, 96])
        xfg_sig = torch.sigmoid(xfg)
        # save_intermediate_feature(x, batch_num, save_file=self.save_file_before)
        x = x*xfg_sig
        # save_intermediate_feature(x, batch_num, save_file=self.save_file_after)
        seg = self.conv4(x)
        return seg, x


def save_intermediate_feature(save_feature, batch_num, save_file=None):
    assert save_file is not None
    save_file += '/parsing_x_{}'.format(batch_num)
    print(save_file)
    with open(save_file, 'wb') as f:
        xfg_sig_max = torch.max(save_feature, dim=1)[0]
        pickle.dump(xfg_sig_max, f)



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, with_dcn=[1,1,1]):
        self.inplanes = 128
        self.inplanes_init = self.inplanes
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

        # self.edge_layer = EdgeMaskModule(num_classes=num_classes)
        # self.edge_layer = EdgeMaskGridsModule(num_classes=num_classes)
        self.edge_layer = EdgeGridsMaskModule(num_classes=num_classes)
        self.layer6 = Decoder_Module(num_classes)
        self.layer7 = nn.Sequential(
            nn.Conv2d(5*256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )
        #self.seg_rescore_conv = nn.Conv2d(num_classes*3, num_classes, 1, stride=1)
        self.seg_rescore_conv = GCN(num_classes*3, num_classes, [31, 31])


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, deform_conv=False):
        if deform_conv:
            print("###layer run in deformable convolution.")
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
        x5 = self.layer4(x4) # torch.Size([3, 2048, 24, 24])

        x_psp = self.layer5(x5) # PSPModule
        # return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea
        preds, x_fg, x_edges = self.edge_layer(x2, x3, x4, x_psp, unfold_step_ratio=(2,2))
        edge_pred, fg_pred, seg_pred = preds

        seg1, x_decod = self.layer6(x_psp, x2, x_fg, batch_num) # Decoder, x_decode [bsz, 256, 96, 96]
        x_cat = torch.cat([x_decod, x_edges], dim=1) # torch.Size([bsz, 1024, 96, 96])
        seg2 = self.layer7(x_cat) # Fuse
        seg_resore = self.seg_rescore_conv(torch.cat([seg1, seg2, seg_pred], dim=1))
        return [[seg1, seg2, seg_pred, seg_resore], edge_pred, fg_pred]


def Res_Deeplab(num_classes=7, with_dcn=[1,1,1]):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, with_dcn=with_dcn)
    return model

def build_ResDeeplab(cfg):
    layers = [3, 4, 23, 3] if int(cfg.MODEL.RES_TYPE) == 101 else [3, 4, 6, 3]
    model = ResNet(Bottleneck, 
            layers, 
            cfg.TRAIN.NUM_CLASSES, 
            with_dcn=cfg.MODEL.WITH_DCN,)
    return model