import torch
import torch.nn as nn
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


class EdgeMaskModule(nn.Module):
    def __init__(self,
                in_fea=[256, 512, 1024, 512], 
                mid_fea=256, 
                edge_out_fea=2, 
                num_classes=7,
                with_mask_edge=True,
                with_mask_pars=True):
        super(EdgeMaskModule, self).__init__()
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
            
    def forward(self, x1, x2, x3, x_psp, batch_num=0):
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
        
        edge_pred = self.conv_final_edge(edge_feats)

        if self.with_mask_edge:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, edge4_fea], dim=1)
        else:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea], dim=1)
        
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
        # padding featuers if it is not divisible by unfold_step_ratio
        def padding_last_column(x, step_r):
            pnum_h = x.shape[-2] % step_r[0]
            pnum_w = x.shape[-1] % step_r[1]
            if pnum_h == 0 and pnum_w == 0:
                return x
            else:
                return F.pad(x, (0, pnum_w, pnum_h, 0), "constant", 0)

       
        def unfold_horizon_vertical(x, step_r):
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
        
       
        def fold_horizon_vertical(x, step_r):
            """
            Folding input which is unfolded by 'unfold_horizon_vertical' function. 
            Patches that are unfolded into batches are recoverd into original position in spatial(along width and length dimensions).
            """
            b, c, h, w = x.shape
            gnum = step_r[0]*step_r[1]
            b_org = b//gnum
            return x.reshape(b_org, step_r[0], step_r[1], c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b_org, c, h*step_r[0], w*step_r[1])

        x1p = padding_last_column(x1, unfold_step_ratio)
        x2p = padding_last_column(x2, unfold_step_ratio)
        x3p = padding_last_column(x3, unfold_step_ratio)
        xpspp = padding_last_column(x_psp, unfold_step_ratio)

        x1g = unfold_horizon_vertical(x1p, unfold_step_ratio)
        x2g = unfold_horizon_vertical(x2p, unfold_step_ratio)
        x3g = unfold_horizon_vertical(x3p, unfold_step_ratio)
        xpspg = unfold_horizon_vertical(xpspp, unfold_step_ratio)


        layer1_fea = self.conv1(x1g)
        layer2_fea = self.conv2(x2g)
        layer3_fea = self.conv3(x3g)
        layer4_fea = self.conv4(xpspg)

        hg, wg = x1g.shape[-2:]
        layer2_fea = F.interpolate(layer2_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        layer3_fea = F.interpolate(layer3_fea, size=(hg, wg), mode='bilinear', align_corners=True)
        layer4_fea = F.interpolate(layer4_fea, size=(hg, wg), mode='bilinear', align_corners=True)

        layers_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, layer4_fea], dim=1)

        fg_fea = self.cat_fg_conv(layers_fea)

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
            edge4 = self.conv_layer_edge(edge4_fea)
            edge_feats = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        else:
            edge_feats = torch.cat([edge1, edge2, edge3], dim=1)
        
        edge_pred = self.conv_final_edge(edge_feats)

        if self.with_mask_edge:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea, edge4_fea], dim=1)
        else:
            edges_fea = torch.cat([layer1_fea, layer2_fea, layer3_fea], dim=1)
        
        edge_pred = fold_horizon_vertical(edge_pred, unfold_step_ratio)
        fg_pred = fold_horizon_vertical(fg_pred, unfold_step_ratio)
        seg_pred = fold_horizon_vertical(seg_pred, unfold_step_ratio)
        fg_fea = fold_horizon_vertical(fg_fea, unfold_step_ratio)
        edges_fea = fold_horizon_vertical(edges_fea, unfold_step_ratio)

        return [edge_pred, fg_pred, seg_pred], fg_fea, edges_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')




class MaskModule(nn.Module):
    def __init__(self,
                in_fea=[256, 512, 1024, 512], 
                mid_fea=256, 
                num_classes=20,
                with_mask_pars=True):
        super(MaskModule, self).__init__()
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

        if with_mask_pars:
            self.cat_seg_conv = nn.Sequential(nn.Conv2d(4 * 256, 512, kernel_size=3, padding=1, bias=False),
                                        InPlaceABNSync(512),
                                        nn.ReLU(inplace=False))
            self.down_seg_conv = nn.Sequential(nn.Dropout2d(0.1),
                                        nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

        self._init_weight()
            
    def forward(self, x1, x2, x3, x_psp, batch_num=0):
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

        if self.with_mask_pars:
            seg_fea = self.cat_seg_conv(layers_fea)
            seg_pred = self.down_seg_conv(seg_fea)
        else:
            seg_pred = None

        return [fg_pred, seg_pred], fg_fea, seg_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        print('Edge_Module inited.')

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

    def forward(self, xt, xl, xfg=None, batch_num=0):
        _, _, h, w = xl.size()

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        if xfg is not None:
            xfg_sig = torch.sigmoid(xfg)
            x = x*xfg_sig
        seg = self.conv4(x)
        return seg, x



if __name__ == '__main__':
    # class EdgeMaskGridsModule(nn.Module):
    #     def __init__(self,
    #             in_fea=[256, 512, 1024, 512], 
    #             mid_fea=256, 
    #             edge_out_fea=2, 
    #             num_classes=7,
    #             with_mask_edge=True,
    #             with_mask_pars=True):
    # def forward(self, x1, x2, x3, x_psp, unfold_step_ratio=4, batch_num=0):
    

    l1x = torch.randn(2, 256, 96, 96)
    l2x = torch.randn(2, 512, 48, 48)
    l3x = torch.randn(2, 1024, 24, 24)
    l4x = torch.randn(2, 512, 24, 24)


    EMG = EdgeMaskGridsModule()
    output = EMG(l1x, l2x, l3x, l4x, unfold_step_ratio=4)
