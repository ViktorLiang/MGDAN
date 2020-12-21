import torch.nn as nn
import torch
from torch.nn import functional as F

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255, vis_loss=False, num_classes=20, 
                    with_edge=True):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.with_edge = with_edge
        self.num_classes = num_classes
        self.IGNORE_INDEX = 255
        self.vis_detail_loss = vis_loss

    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        loss = 0
        # loss for parsing
        preds_parsing = preds[0]
        parsing_loss = 0
        up_preds_parsing = []
        if isinstance(preds_parsing, list):
            for i, pred_parsing in enumerate(preds_parsing):
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w), mode='bilinear', align_corners=True)
                up_preds_parsing.append(scale_pred)
                parsing_loss_i = self.criterion(scale_pred, target[0])
                parsing_loss += parsing_loss_i
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w), mode='bilinear', align_corners=True)
            parsing_loss = self.criterion(scale_pred, target[0])
        loss += parsing_loss

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
            vis_parsing_loss = torch.zeros(parsing_loss.shape)
            vis_fg_loss = torch.zeros(fg_loss.shape)
            vis_parsing_loss.copy_(parsing_loss.cpu())
            vis_fg_loss.copy_(fg_loss.cpu())

            if self.with_edge:
                vis_edge_loss = torch.zeros(edge_loss.shape)
                vis_edge_loss.copy_(edge_loss.cpu())
                return loss, vis_parsing_loss, vis_edge_loss, vis_fg_loss
            else:
                return loss, vis_parsing_loss, vis_fg_loss

        else:
            return loss

    def forward(self, preds, target):
        loss = self.parsing_loss(preds, target)
        return loss

