import os
import numpy as np
import random
import torch
import cv2
from torch.utils import data
from dataset.target_generation import generate_edge
from utils.transforms import get_affine_transform
from dataset.datasets_utils import get_pathes, dataset_pathes


class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None, num_classes=20,
                 dataset_name=None, label_edge=False):
        """
        :rtype:
        """
        assert dataset_name in ['PASCAL_PERSON', 'LIP', 'CIHP', 'ATR']
        self.DATASET_PATH, self.DATASET_PRE = dataset_pathes()
        self.flip_label_dataset = ['CIHP', 'LIP', 'ATR'] # those datasets have horizontal sysmetry labels
        self.label_edge = label_edge

        self.dataset_name = dataset_name
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        # self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.gray_prob = 0.1

        self.transform = transform
        self.dataset = dataset # 'train' or 'val'

        # list_path = os.path.join(self.root, self.DATASET_PRE[self.dataset_name + "_" + self.dataset],
        #                          self.dataset + '_id.txt')
        list_path = "./dataset/list/{}/{}_id.txt".format(dataset_name.lower(), dataset.lower())
                                 
        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)
        self.num_classes = num_classes
        
        img_dir = os.path.join(root, self.DATASET_PATH[dataset_name+"_"+dataset]['image'])
        parsing_dir = os.path.join(root, self.DATASET_PATH[dataset_name+"_"+dataset]['parsing_anno'])
        rev_parsing_dir = os.path.join(root, self.DATASET_PATH[dataset_name+"_"+dataset]['reverse_anno'])

        self.img_gt_list = []
        for im_name in self.im_list:
            if dataset_name in self.flip_label_dataset:
                self.img_gt_list.append(
                    [os.path.join(img_dir, str(im_name)+'.jpg'), 
                    os.path.join(parsing_dir, str(im_name)+'.png'), 
                    os.path.join(rev_parsing_dir, str(im_name)+'.png')]
                )
            else:
                self.img_gt_list.append(
                    [os.path.join(img_dir, str(im_name)+'.jpg'), 
                    os.path.join(parsing_dir, str(im_name)+'.png'),'']
                )

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]
        # im_path = os.path.join(self.root, self.DATASET_PATH[self.dataset_name + "_" + self.dataset]['image'], im_name + ".jpg")
        # parsing_anno_path = os.path.join(self.root, self.DATASET_PATH[self.dataset_name + "_" + self.dataset]['parsing_anno'], im_name + ".png")
        im_path = self.img_gt_list[index][0]
        parsing_anno_path = self.img_gt_list[index][1]
        rev_parsing_anno_path = self.img_gt_list[index][2]

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        # parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'val':
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train' or self.dataset == 'trainval':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    if self.dataset_name in self.flip_label_dataset:
                        parsing_anno = cv2.imread(rev_parsing_anno_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
                        parsing_anno = parsing_anno[:, ::-1]
                else:
                    parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

                if random.random() <= self.gray_prob:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    im = np.expand_dims(im, axis=0)
                    im = np.repeat(im, 3, axis=0).transpose(1, 2, 0)

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'val':
            return input, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))
            if self.label_edge:
                label_edge = generate_edge(label_parsing, edge_width=2)
                label_edge = torch.from_numpy(label_edge)
                label_parsing = torch.from_numpy(label_parsing)
                return input, label_parsing, label_edge, meta
            else:
                label_parsing = torch.from_numpy(label_parsing)
                return input, label_parsing, meta


