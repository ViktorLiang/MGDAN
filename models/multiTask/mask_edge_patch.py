import torch.nn as nn
from torch.nn import functional as F
import torch

import pickle
affine_par = True
import functools

from libs import InPlaceABN, InPlaceABNSync
from utils.ops.dcn.modules import DeformConv2dPack
from models.sharedTask.sub_models import GCN

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


class Edge_Module(nn.Module):

    def __init__(self,in_fea=[256,512,1024,512], mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_fea[3], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv_edge = GCN(mid_fea, out_fea, k=(5, 5))
        self.conv_cat_edge = GCN(out_fea*4, out_fea, k=(7, 7))
        self._init_weight()
    
    def unfold_horizon_vertical(self, x, step_r):
        """
        This function is used to unfold x into patches.
        Number of patch is step_r[0] along wide side, step_r[1] along long side.
        All patches are concatenated along the batch dimension.
        """
        b, c, h, w = x.shape
        stp_w = w//step_r[0]
        stp_h = h//step_r[1]
        gnum = step_r[0]*step_r[1]
        x_unf = x.unfold(2, stp_w, stp_w).unfold(3, stp_h, stp_h) #2:row, 3:column
        return x_unf.reshape(b, c, gnum, stp_w, stp_h).permute(0, 2, 1, 3, 4).reshape(b*gnum, c, stp_w, stp_h)
    
    def fold_horizon_vertical(self, x, step_r):
        """
        Folding the x which is unfolded by 'unfold_horizon_vertical' function. 
        Patches that are unfolded into batches are recoverd into original position in spatial(along width and length dimensions).
        """
        b, c, h, w = x.shape
        gnum = step_r[0]*step_r[1]
        b_org = b//gnum
        return x.reshape(b_org, step_r[0], step_r[1], c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b_org, c, h*step_r[0], w*step_r[1])

    def forward(self, x1, x2, x3, x_fg, unfold_step_ratio=(2,2)):
        _, _, h, w = x1.size()
        # dividing input features into patches
        x1 = self.unfold_horizon_vertical(x1, unfold_step_ratio)
        x2 = self.unfold_horizon_vertical(x2, unfold_step_ratio)
        x3 = self.unfold_horizon_vertical(x3, unfold_step_ratio)
        x_fg = self.unfold_horizon_vertical(x_fg, unfold_step_ratio)
        
        edge1_fea = self.conv1(x1)
        edge1 = self.conv_edge(edge1_fea)

        edge2_fea = self.conv2(x2)
        edge2 = self.conv_edge(edge2_fea)

        edge3_fea = self.conv3(x3)
        edge3 = self.conv_edge(edge3_fea)

        edge4_fea = self.conv4(x_fg)
        edge4 = self.conv_edge(edge4_fea)

        
        _,_,h,w = x1.shape
        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear',align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear',align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=True)

        edge = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea, edge4_fea], dim=1)
        edge = self.conv_cat_edge(edge)

        # recovering divided patches into original position
        edge = self.fold_horizon_vertical(edge, unfold_step_ratio)
        edge_fea = self.fold_horizon_vertical(edge_fea, unfold_step_ratio)

        return edge, edge_fea

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
        self.save_file_before = './output/parsing_mask_atten_max/before_atten/snapshots_mskDCN_edgLK7/'
        self.save_file_after = './output/parsing_mask_atten_max/after_atten/snapshots_mskDCN_edgLK7/'

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


class Foreground_Module(nn.Module):
    def __init__(self, num_classes=7, seg_conv_dcn=False):
        super(Foreground_Module, self).__init__()
        self.res1_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                       BatchNorm2d(256), nn.ReLU(inplace=False))

        self.res2_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                       BatchNorm2d(256), nn.ReLU(inplace=False))

        self.res3_conv1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(512),nn.ReLU(inplace=False))
        self.res3_conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(256), nn.ReLU(inplace=False))

        self.res4_conv1 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(1024), nn.ReLU(inplace=False))
        self.res4_conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(512), nn.ReLU(inplace=False))
        self.res4_conv3 = nn.Sequential(nn.Conv2d(512,256, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(256), nn.ReLU(inplace=False))

        self.cat_fg_conv = nn.Sequential(nn.Conv2d(4 * 256, 256, kernel_size=3, padding=1, bias=False),
                                      InPlaceABNSync(256),
                                      nn.ReLU(inplace=False))

        if seg_conv_dcn:
            self.cat_seg_conv = nn.Sequential(DeformConv2dPack(4*256, 512, kernel_size=3, padding=1, stride=1, bias=False, deformable_groups=1),
                                              InPlaceABNSync(512),
                                              nn.ReLU(inplace=False))
            print("###Foreground branch segmentation running with Deformable Convoluation.")
        else:
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


    def forward(self, x1, x2, x3, x4):
        x1 = self.res1_conv(x1)
        h,w = x1.shape[2], x1.shape[3]

        x2 = self.res2_conv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)

        x3 = self.res3_conv1(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.res3_conv2(x3)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)

        x4 = self.res4_conv1(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = self.res4_conv2(x4)
        x4 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)
        x4 = self.res4_conv3(x4)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_fg = self.cat_fg_conv(x_cat)
        x_seg = self.cat_seg_conv(x_cat)
        fg = self.down_fg_conv(x_fg)
        seg = self.down_seg_conv(x_seg)
        return [[x_fg, x_seg], [fg, seg]]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Foreground_Module inited.')


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, deform_conv=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, deform_conv=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1), deform_conv=True)
        self.fg_layer = Foreground_Module(num_classes=num_classes)
        self.layer5 = PSPModule(2048, 512)
            
        self.edge_layer = Edge_Module()
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
        x_fg_res = self.fg_layer(x2, x3, x4, x5)
        x_fg, x_seg = x_fg_res[0]
        fg_pred, seg_pred = x_fg_res[1]


        x_psp = self.layer5(x5) # PSPModule

        edge, edge_fea = self.edge_layer(x2, x3, x4, x_seg, unfold_step_ratio=(2,2))
        seg1, x_decod = self.layer6(x_psp, x2, x_fg, batch_num) # Decoder, x_decode [bsz, 256, 96, 96]
        x_cat = torch.cat([x_decod, edge_fea], dim=1) # torch.Size([bsz, 1024, 96, 96])
        seg2 = self.layer7(x_cat) # Fuse
        seg_resore = self.seg_rescore_conv(torch.cat([seg1, seg2, seg_pred], dim=1))
        return [[seg1, seg2, seg_pred, seg_resore], edge, fg_pred]


def Res_Deeplab(num_classes=7):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

def build_ResDeeplab(cfg):
    layers = [3, 4, 23, 3] if int(cfg.MODEL.RES_TYPE) == 101 else [3, 4, 6, 3]
    model = ResNet(Bottleneck,
                layers,
                cfg.TRAIN.NUM_CLASSES)
    return model

