import os
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
from utils.utils import decode_parsing, inv_preprocess

CLASS_NUM = 20
IGNORE_INDEX = 255

def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge


  
def _box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #    scale = scale * 1.25

    return center, scale


# def count_label_areas_weights(label_parsing):
#     AREA_WEIGHT_ALPHA = 10
#     counts = np.bincount(label_parsing[np.where(label_parsing != IGNORE_INDEX)]).astype(np.uint32)
#     if counts.shape[0] < CLASS_NUM:
#         counts = np.pad(counts, (0, CLASS_NUM - counts.shape[0]), mode='constant', constant_values=0)
#         assert counts.shape[0] == CLASS_NUM
#     # area_wt = counts/np.sum(counts)
#     area_wt = counts
#     area_wt = (area_wt - np.min(area_wt))/(np.max(area_wt) - np.min(area_wt))
#
#     w_pos_idx = np.where(area_wt > 0)[0]
#     w_pos = area_wt[w_pos_idx]
#     w_pos_argsort = np.argsort(w_pos)
#     idx_end = w_pos_idx.shape[0] - 1
#     mid_swap_idx = np.array(list(
#         map(lambda x: w_pos_argsort[idx_end - np.where(w_pos_argsort == x)[0]][0], range(w_pos.shape[0]))
#     ))
#     area_wt[w_pos_idx] = w_pos[mid_swap_idx]
#     return counts, area_wt

def count_label_areas_weights(label_parsing, num_classes=CLASS_NUM, inverse=True):
    AREA_WEIGHT_ALPHA = 10
    POWER_ALPHA = 2

    counts = np.bincount(label_parsing[np.where(label_parsing != IGNORE_INDEX)]).astype(np.uint32)
    if counts.shape[0] < num_classes:
        counts = np.pad(counts, (0, num_classes - counts.shape[0]), mode='constant', constant_values=0)

    area_wt = counts.copy().astype(np.float)
    pos_idx = np.where(counts > 0)[0]
    neg_idx = np.where(counts <= 0)[0]
    pos_counts = counts[pos_idx]
    if inverse:
        pos_area_wt = np.array(np.sum(pos_counts) - pos_counts, dtype=np.float)
        sum_wt = np.sum(pos_area_wt)
    else:
        pos_area_wt = np.array(pos_counts, dtype=np.float)
        sum_wt = np.sum(pos_area_wt)

    # in case only one label counts greater than zero
    if sum_wt <= 0:
        sum_wt = 1
        pos_area_wt = 1
    # pos_area_wt = np.exp(pos_area_wt / np.sum(pos_area_wt))
    # pos_area_wt = (pos_area_wt / np.sum(pos_area_wt))*AREA_WEIGHT_ALPHA
    # pos_area_wt = (pos_area_wt - np.mean(pos_area_wt))/(np.max(pos_area_wt) - np.min(pos_area_wt))
    area_wt[pos_idx] = np.power((pos_area_wt / sum_wt)*AREA_WEIGHT_ALPHA, POWER_ALPHA)
    area_wt[neg_idx] = 1.0

    return area_wt

def generate_canny_edge(label_parsing):
    label_0 = (label_parsing == 0) | (label_parsing == 255)
    # label_1 = (label_parsing > 0) & (label_parsing != 255)
    edge_from_parse = label_parsing.copy()
    # edge_from_parse[label_0] = 0
    # edge_from_parse[label_1] = 100

    # edge_parse = np.zeros((h, w), dtype=np.uint8)
    # edge_parse = cv2.Canny(edge_from_parse, 50, 150)
    edge_parse = cv2.Canny(edge_from_parse, 0, 20)

    edge_parse = cv2.GaussianBlur(edge_parse, (3, 3), cv2.BORDER_DEFAULT)
    edge_parse[edge_parse > 0] = 1

    # cv2.imshow('pars', label_parsing*100)
    # cv2.imshow('edg', edge_parse*100)
    # cv2.waitKeyEx(0)
    # cv2.destroyAllWindows()
    # exit()
    return edge_parse


def generate_onehot_edge(label_parsing, num_classes):
    label_parsing[label_parsing == 255] = 0

    # label_t = torch.from_numpy(label_parsing[np.newaxis])
    # I_t_lab = decode_parsing(label_t)
    # I_n_lab = np.array(I_t_lab[0].permute(1, 2, 0))
    # cv2.imshow('a', I_n_lab)
    h, w = label_parsing.shape
    label_edge = torch.zeros(num_classes, h, w).scatter_(0, torch.from_numpy(label_parsing[np.newaxis]).long(), 255).type(torch.uint8)
    label_edge = np.array(label_edge)
    cat_edges = np.zeros((num_classes, h, w), dtype=np.uint8)
    for i in range(num_classes):
        j = i
        cat_edges[j] = cv2.Canny(label_edge[i], 50, 150)
        cat_edges[j] = cv2.GaussianBlur(cat_edges[j], (3,3), cv2.BORDER_DEFAULT)
        # cat_edges[j] = generate_edge(label_edge[i], edge_width=6)
        cat_edges[j][cat_edges[j] > 0] = 1
        # cv2.imshow('edge_'+str(j), cat_edges[j]*255)
    # cv2.waitKeyEx(0)
    # cv2.destroyAllWindows()
    cat_edges = cat_edges.astype(np.int64)
    return cat_edges


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def get_edge_width(anno, parsing_path=None, base_width=5, min_width=4):
    if parsing_path is not None:
        I = cv2.imread(parsing_path, cv2.IMREAD_UNCHANGED)
    else:
        I = anno

    I[I == 255] = 0
    im2, contours, hierarchy = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    peri_total = 0
    for coors in contours:
        peri = perimeter(coors)
        peri_total += peri
    area = np.sum((I > 0))
    r_width = (1.0 / base_width)
    ap_r = (area / (peri_total + 0.001) )
    # d = int(ap_r * (1 - r_width) * r_width + min_width)
    d = int(ap_r*(1-r_width*r_width)*0.5 + min_width)
    # d = max(int(ap_r*(1-r_width*r_width)*(1/2.0)), min_width)
    return d


def generate_danamic_kernel_edge(anno, base_width=6, min_width=3):
    d = get_edge_width(anno, base_width=base_width, min_width=min_width)
    gap_binary = generate_edge(anno, edge_width=d)
    gap = gap_binary * anno
    # kernel = exchange_binary(gap_binary) * anno
    return gap


def generate_fg_mask(label_parsing, ignore_idx = 255):
    target_saliency = label_parsing.copy()
    bg_idx = target_saliency == 0
    fg_idx = (target_saliency > 0) & (target_saliency != ignore_idx)
    target_saliency[bg_idx] = 0
    target_saliency[fg_idx] = 1
    return target_saliency


# return type 0:float, 1:integer32
def label_upsample(label, new_size, return_type=0):
    label_in = torch.unsqueeze(label.unsqueeze(dim=0), dim=0).type(torch.float32)
    label_resized = F.interpolate(input=label_in, size=new_size, mode='bilinear', align_corners=True)
    label_resized = torch.squeeze(label_resized.squeeze())
    if return_type == 0:
        return label_resized
    else:
        return label_resized.type(torch.int32)

def generate_dataset_label_weight(dataset_dir):
    labels_list = os.listdir(dataset_dir)
    labels_pathes = [os.path.join(dataset_dir, labels_list[i]) for i in range(len(labels_list))]
    area_w = np.zeros(7)
    for i, pth in enumerate(labels_pathes):
        print("{}/{}, {}".format(i, len(labels_pathes), pth))
        I = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        area_w += count_label_areas_weights(I, num_classes=7, inverse=False)

    print(area_w)
    print(area_w/len(labels_pathes))


if __name__ == "__main__":
    label_dir \
        = '/home/ly/data/datasets/human_parsing/LIP/val_segmentations'
    edge_save_dir = '/home/ly/data/datasets/human_parsing/LIP/vis_annotations/vis_edge_val'
    mask_save_dir = '/home/ly/data/datasets/human_parsing/LIP/vis_annotations/vis_mask_val'
    list_files = os.listdir(label_dir)
    list_files.sort()
    list_images = [os.path.join(label_dir, list_files[i]) for i in range(len(list_files))]
    for i, img_pth in enumerate(list_images):
        I = cv2.imread(img_pth, cv2.IMREAD_UNCHANGED)
        edg_save_name = list_files[i].split('.')[0] + '.png'
        # edg_save_pth = edge_save_dir + "/" + edg_save_name
        msk_save_pth = mask_save_dir + "/" + edg_save_name
    
        edg = generate_edge(I, edge_width=1) * 255
        cv2.imshow("", edg)
        cv2.waitKeyEx()
        cv2.destroyAllWindows()
        exit()
    #     # cv2.imwrite(edg_save_pth, edg)
    #
    #     I[I == 255] = 0
    #     pos = (I > 0)
    #     I[pos] = 255
    #     cv2.imwrite(msk_save_pth, I)
    #     print(i)
    # label_path = '/home/ly/data/datasets/human_parsing/LIP/TrainVal_parsing_annotations/val_segmentations/445128_190472.png'
    # label_pars = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    # dataset_parsing = '/home/ly/workspace/git/segmentation/human_parsing/MDCE_Pascal/dataset/pascal_person_part/train/parsing_anno'
    # generate_dataset_label_weight(dataset_parsing)
    # catEdg = generate_onehot_edge(label_pars, 20)
    # print(catEdg.shape)