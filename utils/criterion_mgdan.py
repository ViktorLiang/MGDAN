import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from utils.lovasz_softmax import LovaszSoftmax

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255, vis_loss=True, num_classes=20, 
                    with_edge=True,
                    with_lovasz_loss=True):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.with_edge = with_edge
        self.with_lovasz_loss = with_lovasz_loss
        self.num_classes = num_classes
        self.IGNORE_INDEX = 255
        self.vis_detail_loss = vis_loss

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if self.with_lovasz_loss:
            self.lovasz_loss = LovaszSoftmax()

    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        loss = 0
        # loss for parsing
        preds_parsing = preds[0]
        parsing_loss_details = []

        if isinstance(preds_parsing, list):
            for i, pred_parsing in enumerate(preds_parsing):
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w), mode='bilinear', align_corners=True)
                if self.with_lovasz_loss:
                    parsing_loss_i = self.criterion(scale_pred, target[0])*0.5 + self.lovasz_loss(scale_pred, target[0])*0.5
                else:
                    parsing_loss_i = self.criterion(scale_pred, target[0])
                loss += parsing_loss_i
                parsing_loss_details.append(parsing_loss_i)
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w), mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        # loss for edge
        if self.with_edge:
            pos_num = torch.sum(target[1] == 1, dtype=torch.float)
            neg_num = torch.sum(target[1] == 0, dtype=torch.float)
            weight_pos = neg_num / (pos_num + neg_num)
            weight_neg = pos_num / (pos_num + neg_num)
            weights = torch.tensor([weight_neg, weight_pos])

            preds_edge = preds[1]
            if isinstance(preds_edge, list):
                edge_loss = 0
                for i, pred_edge in enumerate(preds_edge):
                    up_pred_edge = F.interpolate(input=pred_edge, size=(h, w), mode='bilinear', align_corners=True)
                    edge_loss += F.cross_entropy(up_pred_edge, target[1], weights.cuda(), ignore_index=self.ignore_index)
            else:
                up_pred_edge = F.interpolate(input=preds_edge, size=(h, w), mode='bilinear', align_corners=True)
                edge_loss = F.cross_entropy(up_pred_edge, target[1], weights.cuda(), ignore_index=self.ignore_index)
            loss += edge_loss

        # loss for foreground
        preds_fg = preds[2]
        fg_scale_pred = F.interpolate(input=preds_fg, size=(h, w), mode='bilinear', align_corners=True)
        target_saliency = torch.zeros_like(target[0])
        # target_saliency.copy_(target[0].detach())
        target_saliency.copy_(target[0])

        bg_idx = target_saliency == 0
        fg_idx = (target_saliency > 0) & (target_saliency != self.ignore_index)
        target_saliency[bg_idx] = 0
        target_saliency[fg_idx] = 1

        fg_loss = F.cross_entropy(fg_scale_pred, target_saliency, ignore_index=self.ignore_index)
        loss += fg_loss
        if self.vis_detail_loss:
            vis_fg_loss = torch.zeros(fg_loss.shape)
            vis_fg_loss.copy_(fg_loss.cpu())

            if self.with_edge:
                vis_edge_loss = torch.zeros(edge_loss.shape)
                vis_edge_loss.copy_(edge_loss.cpu())
                vis_pars_details = []
                for plss in parsing_loss_details:
                    vis_pars = torch.zeros(plss.shape)
                    vis_pars.copy_(plss.cpu())
                    vis_pars_details.append(vis_pars)

                return loss, vis_edge_loss, vis_fg_loss, vis_pars_details[0], vis_pars_details[1],vis_pars_details[2],vis_pars_details[3]
            else:
                return loss, vis_fg_loss

        else:
            return loss


    def forward(self, preds, target):
        loss = self.parsing_loss(preds, target)
        return loss


if __name__ == '__main__':
    d_src = np.array([31716, 0, 1290, 0,  0,  326,    0,   0,  0,   0,  0,    0, 0, 1159, 342,   0,  0,   0,  0,   0], dtype=np.float)

    d_pos_idx = np.where(d_src > 0)[0]
    d_pos_value = d_src[d_pos_idx]
    d_pos_inv = np.sum(d_pos_value) - d_pos_value

    # d_pos_normed = (d_pos_inv - np.min(d_pos_inv))/(np.max(d_pos_inv) - np.min(d_pos_inv))
    d_pos_normed = d_pos_inv/np.sum(d_pos_inv)
    print(d_src)
    d_src[d_pos_idx] = d_pos_normed
    print(d_src)


    # d_pos_argsort = np.argsort(d_pos_normed)
    # idx_end = d_pos_normed.shape[0] - 1
    # mid_swap_idx = np.array(list(
    #     map(lambda x: d_pos_argsort[idx_end - np.where(d_pos_argsort == x)[0]][0], range(d_pos_normed.shape[0]))
    # ))

    # print(d_pos_normed)
    # print(d_pos_normed[mid_swap_idx])
    #
    # print("----")
    # print(d_src)
    # print(d_src[d_pos_idx])
    # d_src[d_pos_idx] = d_pos_normed[mid_swap_idx]
    # print(d_src)
