import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

import functools
from libs import InPlaceABN, InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class GCN(nn.Module):
    def __init__(self, c, out_c, k=(7, 7)):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=(int((k[0] - 1) / 2), 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, int((k[0] - 1) / 2)))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, int((k[1] - 1) / 2)))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(int((k[1] - 1) / 2), 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x

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

    def forward(self, xt, xl, xfg, batch_num=0):
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


class EdgeMaskModule(nn.Module):
    def __init__(self, in_fea=[256, 512, 1024, 512], mid_fea=256, out_fea=2, num_classes=7):
        super(EdgeMaskModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
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
        self.conv_final_edge = GCN(out_fea * 4, out_fea, k=(7, 7))

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
        layer4_fea = self.conv4(x_psp)  ##
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
        edge4_fea = self.conv_seg_down(seg_fea)  ## 512->256
        edge4 = self.conv_layer_edge(edge4_fea)  ##
        edge = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        edge_pred = self.conv_final_edge(edge)

        edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, edge4_fea], dim=1)

        return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')



class EdgeMaskGridsModule(nn.Module):
    def __init__(self,
                in_fea=[256, 512, 1024, 512], 
                mid_fea=256, 
                edge_out_fea=2, 
                num_classes=7,
                with_mask_edge=True,
                with_mask_pars=True):
        super(EdgeMaskGridsModule, self).__init__()
        self.with_mask_edge = with_mask_edge
        self.with_mask_pars = with_mask_pars
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_fea[3], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )

        self.cat_fg_conv = nn.Sequential(nn.Conv2d(4 * 256, 256, kernel_size=3, padding=1, bias=False),
                                    InPlaceABNSync(256),
                                    nn.ReLU(inplace=False))
        self.down_fg_conv = nn.Sequential(nn.Dropout2d(0.1),
                                    nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, bias=True))

        self.conv_layer_edge = GCN(mid_fea, edge_out_fea, k=(5, 5))
        if with_mask_edge:
            if with_mask_pars:
                self.conv_seg_down = nn.Sequential(nn.Conv2d(512, mid_fea, kernel_size=1, bias=False),
                                            InPlaceABNSync(mid_fea))
            else:
                self.conv_fg_down = nn.Sequential(nn.Conv2d(256, mid_fea, kernel_size=1, bias=False),
                                            InPlaceABNSync(mid_fea))

            self.conv_final_edge = GCN(edge_out_fea*4, edge_out_fea, k=(7, 7))
        else:
            self.conv_final_edge = GCN(edge_out_fea*3, edge_out_fea, k=(7, 7))
            
        if with_mask_pars:
            self.cat_seg_conv = nn.Sequential(nn.Conv2d(4 * 256, 512, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(512),
                                        nn.ReLU(inplace=False))
            self.down_seg_conv = nn.Sequential(nn.Dropout2d(0.1),
                                        nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

        self._init_weight()
            
    def forward(self, x1, x2, x3, x_psp, unfold_step_ratio=(4,4), batch_num=0):
        _, _, h, w = x1.size()
        # padding featuers if it cant't be divisible by unfold_step_ratio
        def padding_last_column(x, step_r):
            pnum_h = x.shape[-2] % step_r[0]
            pnum_w = x.shape[-1] % step_r[1]
            if pnum_h == 0 and pnum_w == 0:
                return x
            else:
                return F.pad(x, (0, pnum_w, pnum_h, 0), "constant", 0)

        def unfold_horizon(x, step_r):
            b, c, h, w = x.shape
            stp = w//step_r
            x_unf = x.transpose(3, 2).unfold(2, stp, stp)
            # print(x_unf.shape, step_r, x_unf.permute(0,2,1,3,4).shape)
            return x_unf.permute(0,2,1,3,4).reshape(b*step_r, c, h, stp)
        
        def fold_horizon(x, step_r):
            b, c, h, w = x.shape
            b_org = b//step_r
            return x.reshape(b_org, step_r, c, h, w).permute(0,2,3,1,4).reshape(b_org, c, h, w*step_r)
        
        # def unfold(self, x, grid_size,bc):
        #     b,c = bc
        #     return x.unfold(2, grid_size, grid_size).unfold(3, grid_size, grid_size).reshape(b,c,-1,grid_size,grid_size)

        # def fold(self, x, src_size):
        #     b,c,h = src_size
        #     return x.transpose(4,5).permute(0,1,2,5,3,4).reshape(b,c,h,h)

        # t1_unf = t1.unfold(2,6//stps[0],6//stps[0]).unfold(3,6//stps[1],6//stps[1])
        # t1_unf.reshape(1,1,stps[0]*stps[1],6//stps[0],6//stps[1]).permute(0,2,1,3,4).reshape(1*stps[1]*stps[0], 1, 6//stps[0],6//stps[1])
        def unfold_horizon_vertical(x, step_r):
            b, c, h, w = x.shape
            stp_w = w//step_r[0]
            stp_h = h//step_r[1]
            gnum = step_r[0]*step_r[1]
            x_unf = x.unfold(2, stp_w, stp_w).unfold(3, stp_h, stp_h) #2:row, 3:column
            return x_unf.reshape(b, c, gnum, stp_w, stp_h).permute(0, 2, 1, 3, 4).reshape(b*gnum, c, stp_w, stp_h)
        
        def fold_horizon_vertical(x, step_r):
            b, c, h, w = x.shape
            gnum = step_r[0]*step_r[1]
            b_org = b//gnum
            return x.reshape(b_org, step_r[0], step_r[1], c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b_org, c, h*step_r[0], w*step_r[1])

        x1p = padding_last_column(x1, unfold_step_ratio)
        x2p = padding_last_column(x2, unfold_step_ratio)
        x3p = padding_last_column(x3, unfold_step_ratio)
        xpspp = padding_last_column(x_psp, unfold_step_ratio)

        # shape: (b*stp, c, h, w//unfold_step_ratio). i.e (2,256,96,96)->(2*4,256,96,24)
        x1g = unfold_horizon_vertical(x1p, unfold_step_ratio)
        x2g = unfold_horizon_vertical(x2p, unfold_step_ratio)
        x3g = unfold_horizon_vertical(x3p, unfold_step_ratio)
        xpspg = unfold_horizon_vertical(xpspp, unfold_step_ratio)

        # print(x1g.shape, x2g.shape, x3g.shape, xpspg.shape)

        layer1_fea = self.conv1(x1g)
        layer2_fea = self.conv2(x2g)
        layer3_fea = self.conv3(x3g)
        layer4_fea = self.conv4(xpspg)##

        hg, wg = x1g.shape[-2:]
        layer2_fea = F.interpolate(layer2_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        layer3_fea = F.interpolate(layer3_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        layer4_fea = F.interpolate(layer4_fea, size=(hg, wg), mode='bilinear', align_corners=True)

        layers_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, layer4_fea], dim=1)

        fg_fea = self.cat_fg_conv(layers_fea)
        ##save mask features for vis
        # save_intermediate_feature(fg_fea, batch_num, save_file="/home/ly/data/save_features/cihp/ce2pMask_maskFeats")

        fg_pred = self.down_fg_conv(fg_fea)

        if self.with_mask_pars:
            seg_fea = self.cat_seg_conv(layers_fea)
            seg_pred = self.down_seg_conv(seg_fea)
        else:
            seg_pred = None

        edge1 = self.conv_layer_edge(layer1_fea)
        edge2 = self.conv_layer_edge(layer2_fea)
        edge3 = self.conv_layer_edge(layer3_fea)
        if self.with_mask_edge:
            if self.with_mask_pars:
                edge4_fea = self.conv_seg_down(seg_fea) ## 512->256
            else:
                edge4_fea = self.conv_fg_down(fg_fea) ## 256->256
            edge4 = self.conv_layer_edge(edge4_fea) ##
            edge_feats = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        else:
            edge_feats = torch.cat([edge1, edge2, edge3], dim=1)
        
        #save feature for vis
        # save_intermediate_feature(edge_feats, batch_num, save_file="/home/ly/data/save_features/cihp/ce2pMask_edgeFeats")

        edge_pred = self.conv_final_edge(edge_feats)

        if self.with_mask_edge:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, edge4_fea], dim=1)
        else:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea], dim=1)
        
        # print("preds:", edge_pred.shape, fg_pred.shape, seg_pred.shape)
        # print("features:", fg_fea.shape, edges_fea.shape)
        
        edge_pred = fold_horizon_vertical(edge_pred, unfold_step_ratio)
        fg_pred = fold_horizon_vertical(fg_pred, unfold_step_ratio)
        seg_pred = fold_horizon_vertical(seg_pred, unfold_step_ratio)
        fg_fea = fold_horizon_vertical(fg_fea, unfold_step_ratio)
        edges_fea = fold_horizon_vertical(edges_fea, unfold_step_ratio)

        # print("fold back:")
        # print("preds:", edge_pred.shape, fg_pred.shape, seg_pred.shape)
        # print("features:", fg_fea.shape, edges_fea.shape)
        # exit()
                
        
        return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')


class EdgeGridsMaskModule(nn.Module):
    def __init__(self,
                in_fea=[256, 512, 1024, 512], 
                mid_fea=256, 
                edge_out_fea=2, 
                num_classes=7,
                with_mask_edge=True,
                with_mask_pars=True):
        super(EdgeGridsMaskModule, self).__init__()
        self.with_mask_edge = with_mask_edge
        self.with_mask_pars = with_mask_pars
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_fea[3], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea)
        )

        self.cat_fg_conv = nn.Sequential(nn.Conv2d(4 * 256, 256, kernel_size=3, padding=1, bias=False),
                                    InPlaceABNSync(256),
                                    nn.ReLU(inplace=False))
        self.down_fg_conv = nn.Sequential(nn.Dropout2d(0.1),
                                    nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, bias=True))

        self.conv_layer_edge = GCN(mid_fea, edge_out_fea, k=(5, 5))
        if with_mask_edge:
            if with_mask_pars:
                self.conv_seg_down = nn.Sequential(nn.Conv2d(512, mid_fea, kernel_size=1, bias=False),
                                            InPlaceABNSync(mid_fea))
            else:
                self.conv_fg_down = nn.Sequential(nn.Conv2d(256, mid_fea, kernel_size=1, bias=False),
                                            InPlaceABNSync(mid_fea))

            self.conv_final_edge = GCN(edge_out_fea*4, edge_out_fea, k=(7, 7))
        else:
            self.conv_final_edge = GCN(edge_out_fea*3, edge_out_fea, k=(7, 7))
            
        if with_mask_pars:
            self.cat_seg_conv = nn.Sequential(nn.Conv2d(4 * 256, 512, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(512),
                                        nn.ReLU(inplace=False))
            self.down_seg_conv = nn.Sequential(nn.Dropout2d(0.1),
                                        nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

        self._init_weight()
            
    def forward(self, x1, x2, x3, x_psp, unfold_step_ratio=(4,4), batch_num=0):
        _, _, h, w = x1.size()
        # padding featuers if it cant't be divisible by unfold_step_ratio
        def padding_last_column(x, step_r):
            pnum_h = x.shape[-2] % step_r[0]
            pnum_w = x.shape[-1] % step_r[1]
            if pnum_h == 0 and pnum_w == 0:
                return x
            else:
                return F.pad(x, (0, pnum_w, pnum_h, 0), "constant", 0)

        # t1_unf = t1.unfold(2,6//stps[0],6//stps[0]).unfold(3,6//stps[1],6//stps[1])
        # t1_unf.reshape(1,1,stps[0]*stps[1],6//stps[0],6//stps[1]).permute(0,2,1,3,4).reshape(1*stps[1]*stps[0], 1, 6//stps[0],6//stps[1])
        def unfold_horizon_vertical(x, step_r):
            b, c, h, w = x.shape
            stp_w = w//step_r[0]
            stp_h = h//step_r[1]
            gnum = step_r[0]*step_r[1]
            x_unf = x.unfold(2, stp_w, stp_w).unfold(3, stp_h, stp_h) #2:row, 3:column
            return x_unf.reshape(b, c, gnum, stp_w, stp_h).permute(0, 2, 1, 3, 4).reshape(b*gnum, c, stp_w, stp_h)
        
        def fold_horizon_vertical(x, step_r):
            b, c, h, w = x.shape
            gnum = step_r[0]*step_r[1]
            b_org = b//gnum
            return x.reshape(b_org, step_r[0], step_r[1], c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b_org, c, h*step_r[0], w*step_r[1])
        
        ### Mask branch
        layer1_fea = self.conv1(x1)
        layer2_fea = self.conv2(x2)
        layer3_fea = self.conv3(x3)
        layer4_fea = self.conv4(x_psp)##
        h, w = x1.shape[-2:]
        layer2_fea = F.interpolate(layer2_fea, size=(h, w), mode='bilinear', align_corners=True)
        layer3_fea = F.interpolate(layer3_fea, size=(h, w), mode='bilinear', align_corners=True)
        layer4_fea = F.interpolate(layer4_fea, size=(h, w), mode='bilinear', align_corners=True)
        layers_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, layer4_fea], dim=1)
        fg_fea = self.cat_fg_conv(layers_fea)
        fg_pred = self.down_fg_conv(fg_fea)
        if self.with_mask_pars:
            seg_fea = self.cat_seg_conv(layers_fea)
            seg_pred = self.down_seg_conv(seg_fea)
        else:
            seg_pred = None


        ### Edge branch with divided input
        x1p = padding_last_column(x1, unfold_step_ratio)
        x2p = padding_last_column(x2, unfold_step_ratio)
        x3p = padding_last_column(x3, unfold_step_ratio)
        xpspp = padding_last_column(x_psp, unfold_step_ratio)

        # shape: (b*stp, c, h, w//unfold_step_ratio). i.e (2,256,96,96)->(2*4,256,96,24)
        x1g = unfold_horizon_vertical(x1p, unfold_step_ratio)
        x2g = unfold_horizon_vertical(x2p, unfold_step_ratio)
        x3g = unfold_horizon_vertical(x3p, unfold_step_ratio)
        xpspg = unfold_horizon_vertical(xpspp, unfold_step_ratio)

        layer1_edge_fea = self.conv1(x1g)
        layer2_edge_fea = self.conv2(x2g)
        layer3_edge_fea = self.conv3(x3g)
        # layer4_edge_fea = self.conv4(xpspg)##
        hg, wg = x1g.shape[-2:]
        layer2_edge_fea = F.interpolate(layer2_edge_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        layer3_edge_fea = F.interpolate(layer3_edge_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        # layer4_edge_fea = F.interpolate(layer4_edge_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        # layers_edge_fea = torch.cat([layer1_edge_fea, layer2_edge_fea, layer3_edge_fea, layer4_edge_fea], dim=1)


        edge1 = self.conv_layer_edge(layer1_edge_fea)
        edge2 = self.conv_layer_edge(layer2_edge_fea)
        edge3 = self.conv_layer_edge(layer3_edge_fea)
        if self.with_mask_edge:
            if self.with_mask_pars:
                seg_fea = unfold_horizon_vertical(seg_fea, unfold_step_ratio)
                edge4_fea = self.conv_seg_down(seg_fea) ## 512->256
            else:
                fg_fea = unfold_horizon_vertical(fg_fea, unfold_step_ratio)
                edge4_fea = self.conv_fg_down(fg_fea) ## 256->256
            edge4 = self.conv_layer_edge(edge4_fea) ##
            edge_feats = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        else:
            edge_feats = torch.cat([edge1, edge2, edge3], dim=1)
        
        edge_pred = self.conv_final_edge(edge_feats)
        edge_pred = fold_horizon_vertical(edge_pred, unfold_step_ratio)

        layer1_edge_fea = fold_horizon_vertical(layer1_edge_fea, unfold_step_ratio)
        layer2_edge_fea = fold_horizon_vertical(layer2_edge_fea, unfold_step_ratio)
        layer3_edge_fea = fold_horizon_vertical(layer3_edge_fea, unfold_step_ratio)
        if self.with_mask_edge:
            edge4_fea = fold_horizon_vertical(edge4_fea, unfold_step_ratio)
            edges_fea = torch.cat([layer1_edge_fea, layer2_edge_fea, layer3_edge_fea, edge4_fea], dim=1)
        else:
            edges_fea = torch.cat([layer1_edge_fea, layer2_edge_fea, layer3_edge_fea], dim=1)
        
        return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def seperate_conv(in_channels, out_channels, k=1,p=0,d=1,g=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=p, dilation=d, groups=g),
        InPlaceABNSync(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, 1),
        InPlaceABNSync(out_channels),
        nn.ReLU()
    )




class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        # self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
        #                                      bias=False),
        #                            InPlaceABNSync(inner_features))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
        #     InPlaceABNSync(inner_features))
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
        #     InPlaceABNSync(inner_features))
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
        #     InPlaceABNSync(inner_features))
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
        #     InPlaceABNSync(inner_features))

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = seperate_conv(features, inner_features, k=3, p=dilations[0], d=dilations[0], g=features)
        self.conv4 = seperate_conv(features, inner_features, k=3, p=dilations[1], d=dilations[1], g=features)
        self.conv5 = seperate_conv(features, inner_features, k=3, p=dilations[2], d=dilations[2], g=features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


if __name__ == '__main__':
    # class EdgeGridsMaskModule(nn.Module):
    #     def __init__(self,
    #             in_fea=[256, 512, 1024, 512], 
    #             mid_fea=256, 
    #             edge_out_fea=2, 
    #             num_classes=7,
    #             with_mask_edge=True,
    #             with_mask_pars=True):

    egm = EdgeGridsMaskModule(in_fea=[256, 512, 1024, 512], mid_fea=256, )
    l1x = torch.randn(2, 256, 112, 112)
    l2x = torch.randn(2, 512, 56, 56)
    l3x = torch.randn(2, 1024, 28, 28)
    l4x = torch.randn(2, 512, 28, 28)
    # return [edge_pred, fg_pred, seg_pred], fg_fea, fuse_fea
    preds, fg_fea, fuse_fea = egm(l1x, l2x, l3x, l4x)
    for i in preds:
        print(i.shape)
    print(fg_fea.shape, fuse_fea.shape)
