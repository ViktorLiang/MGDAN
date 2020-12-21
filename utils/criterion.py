import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
WITH_SSIM_LOSS = False

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.mse_loss = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()

        self.CLASS_NUM = 7
        self.IGNORE_INDEX = 255
        self.PARS_LOSS_W = 1
        self.AREA_LOSS_W = 0.001
        self.IS_FOCAL_LOSS = False
        self.WITH_LABEL_WEIGHT = False

        self.SIDE_EDGE_WEIGHT = 1
        self.FUSE_EDGE_WEIGHT = 1

        if self.IS_FOCAL_LOSS:
            print("###run with focal loss")
        if self.WITH_LABEL_WEIGHT:
            print("###run with label weights")

        # self.ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
   
    def parsing_loss(self, preds, target, label_weights=None):
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                if self.WITH_LABEL_WEIGHT:
                    # loss += self.criterion(scale_pred, target[0])
                    loss += F.cross_entropy(scale_pred, target[0], weight=label_weights, ignore_index=self.ignore_index)
                else:
                    loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            if self.WITH_LABEL_WEIGHT:
                # loss += self.criterion(scale_pred, target[0])
                loss += F.cross_entropy(scale_pred, target[0], weight=label_weights, ignore_index=self.ignore_index)
            else:
                loss += self.criterion(scale_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            total_edge_pred = len(preds_edge)
            for i, pred_edge in enumerate(preds_edge):
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                if i != total_edge_pred - 1:
                    edge_loss_weight = self.SIDE_EDGE_WEIGHT
                else:
                    edge_loss_weight = self.FUSE_EDGE_WEIGHT
                loss += edge_loss_weight*F.cross_entropy(scale_pred, target[1], weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)
        # loss for edge, ssim loss
        if WITH_SSIM_LOSS:
            edge_target = torch.unsqueeze(target[1], dim=1).type(torch.float)
            for i, pred_edge in enumerate(preds_edge):
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                # loss += scale_pred[:,1],target[1][1]
                edge_pred = torch.unsqueeze(scale_pred[:, 1], dim=1)
                ssim_out_edge = 1 - self.ssim_loss(edge_pred, edge_target)
                loss += ssim_out_edge

        # loss for foreground
        preds_fg = preds[2]
        fg_scale_pred = F.interpolate(input=preds_fg, size=(h, w),
                                   mode='bilinear', align_corners=True)

        target_saliency = torch.zeros_like(target[0])
        target_saliency.copy_(target[0].detach())

        bg_idx = target_saliency == 0
        fg_idx = (target_saliency > 0) & (target_saliency != self.ignore_index)
        target_saliency[bg_idx] = 0
        target_saliency[fg_idx] = 1

        loss += F.cross_entropy(fg_scale_pred, target_saliency, ignore_index=self.ignore_index)

        if WITH_SSIM_LOSS:
            saliency_target = torch.unsqueeze(target_saliency, dim=1).type(torch.float)
            saliency_pred = torch.unsqueeze(fg_scale_pred[:,1], dim=1)
            ssim_out_fg = 1 - self.ssim_loss(saliency_pred, saliency_target)
            loss += ssim_out_fg


        return loss

    def area_sense_loss(self, pred_score, target):
        pred_label = torch.argmax(pred_score, dim=1)
        bsz, _, _ = pred_label.shape
        pred_label = pred_label.reshape(bsz, -1)
        # target[target == self.IGNORE_INDEX] = 0
        gt_label = target.reshape(bsz, -1)

        loss = 0
        for bs_id, bs_pred_label in enumerate(pred_label):
            pred_counts = self.bincout_with_label(bs_pred_label)
            gt_counts = self.bincout_with_label(gt_label[bs_id])
            area_loss = self.l1loss(pred_counts.type(torch.float), gt_counts.type(torch.float))
            loss += area_loss.mean()
        return area_loss

    def bincout_with_label(self, label_to_count):
        counts = torch.bincount(label_to_count)
        if counts.shape[0] < self.CLASS_NUM:
            m = nn.ConstantPad1d((0, self.CLASS_NUM - counts.shape[0]), 0)
            counts = m(counts)
            assert counts.shape[0] == self.CLASS_NUM

        if counts.shape[0] > self.CLASS_NUM:
            counts = counts[:self.CLASS_NUM]

        return counts

    def multi_class_focal_loss(self, model_output, target, class_weights=None, gamma=2.0, alpha=0.25, reduction='mean'):
        outputs = F.softmax(model_output, dim=1)
        bsz, ch, _, _ = outputs.shape
        label_one_hot = torch.zeros_like(outputs)
        for bs_id in range(bsz):
            for label_id in range(ch):
                label_one_hot[bs_id, label_id] = (target[bs_id] == label_id)

        ce = label_one_hot * (-1 * torch.log(outputs))
        weight = label_one_hot * torch.pow((1 - outputs), gamma)
        fl = ce * weight * alpha
        fl = fl.reshape(bsz, ch, -1)
        if class_weights is not None:
            fl_p = fl.permute(1, 0, 2)
            fl_p = fl_p.reshape(ch, -1)
            wts_col = class_weights.reshape(ch, -1)
            fl_p_w = fl_p*wts_col
            fl_p_w = fl_p_w.reshape(ch, bsz, -1)
            fl = fl_p_w.permute(1, 0, 2)

        fl_max = fl.max(dim=1)[0]
        loss = fl_max.mean(dim=1).sum()
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