import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing
from dataset.datasets_utils import dataset_pathes
#from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

PSACAL_LABELS = ['Background', 'Head', 'Torso', 'Lower-arm', 'Upper-arm', 'Lower-leg', 'Upper-leg']
LIP_LABELS = ['Background', 'Hat','Hair','Glove','Sunglasses','UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits',
              'Scarf', 'Skirt', 'Face', 'Left-arm','Right-arm','Left-leg','Right-leg', 'Left-shoe', 'Right-shoe']
CIHP_LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Torso-skin',
               'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
ATR_LABELS = ['Background','hat', 'hair', 'sunglasses', 'coat', 'skirt', 'pants', 'dress', 'belt', 'left shoe', 'right shoe', 'face', 'left leg', 'right leg', 'left hand', 'right hand', 'bags', 'scarf']
#PASCAL_PERSON', 'LIP', 'CIHP'
labels_name_dicts = {"LIP":LIP_LABELS, "PASCAL_PERSON":PSACAL_LABELS, 'CIHP':CIHP_LABELS, 'ATR':ATR_LABELS}

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val', dataset_name='CIHP'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)

    name_value = []

    for i, (label, iou) in enumerate(zip(labels_name_dicts[dataset_name], IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value


def compute_edge_F1_score(edge_preds, scales, centers, data_dir, input_size=[473, 473], dataset='val', edge_thresh = 0.5):
    # pidx = (edge_preds > edge_thresh)
    # nids = (edge_preds <= edge_thresh)
    edge_preds[edge_preds > edge_thresh] = 1
    edge_preds[edge_preds <= edge_thresh] = 0

    list_path = os.path.join(data_dir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    confusion_matrix = np.zeros((2, 2))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(data_dir, dataset + '_edges', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = edge_preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255
        gt = gt[ignore_index]
        pred = pred[ignore_index]
        confusion_matrix += get_confusion_matrix(gt, pred, 2)
    gt_edg = confusion_matrix[1,:]
    pred_edg = confusion_matrix[:,1]
    pred_true_edge = confusion_matrix[1,1]

    recall = pred_true_edge/gt_edg.sum()
    precision = pred_true_edge/pred_edg.sum()
    f1_score = 2*recall*precision/(recall+precision)
    return [f1_score, recall, precision]

def get_perImage_IOU(confusion_matrix, img_no, im_name):
    pos = confusion_matrix.sum(1)  # true
    res = confusion_matrix.sum(0)  # pred
    tp = np.diag(confusion_matrix)  # true positive

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    # print(img_no, im_name)
    # print("pixel_accuracy, mean_accuracy, mean_IoU", pixel_accuracy, mean_accuracy, mean_IoU)
    return pixel_accuracy, mean_accuracy, mean_IoU

def compute_mean_ioU_top(preds, scales, centers, num_classes, datadir, input_size=[384, 384],
                         split='val', dataset_pre='', dataset_name='LIP', restore_from = None):
    assert dataset_name in ['PASCAL_PERSON', 'LIP', 'CIHP', 'ATR']
    base_path, base_pre = dataset_pathes()
    LABELS = labels_name_dicts[dataset_name]

    #min56_save = '/home/ly/workspace/git/segmentation/human_parsing/MDCE_Pascal/dataset/output/val_1817_newMeanStd'

    list_path = os.path.join(datadir, dataset_pre, split + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    # pred_count = np.zeros((CLASS_NUM, CLASS_NUM), dtype=np.int64)
    top_pred = np.zeros((num_classes, num_classes), dtype=np.float64)

    single_accuray = np.zeros((len(val_id), 3), dtype=np.float)
    #im_names = []
    if len(dataset_pre) == 0:
        dataset_pre = split

    #if dataset_name == 'PASCAL_PERSON':
    #    anno_base =  dataset_pre + '/parsing_anno'
    #elif dataset_name == 'LIP':
    #    anno_base = dataset_pre + '_segmentations'
    #elif dataset_name == 'CIHP':
    #    anno_base = dataset_pre + '/Category_ids'
    anno_base = base_path[dataset_name+'_'+split]['parsing_anno']

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, anno_base, im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]

        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255
        gt = gt[ignore_index]
        pred = pred[ignore_index]
        # confusion_matrix += get_confusion_matrix(gt, pred, num_classes)
        confusion_matrix_per = get_confusion_matrix(gt, pred, num_classes)

        #single_pixel_accuracy, single_mean_accuracy, single_mean_IoU = get_perImage_IOU(confusion_matrix_per, i, im_name)
        #single_accuray[i] = (single_pixel_accuracy, single_mean_accuracy, single_mean_IoU)
        #im_names.append(im_name)

        confusion_matrix += confusion_matrix_per
        pred_count = np.zeros((num_classes, num_classes), dtype=np.float64)
        pred_count = count_pred_cat_num(pred_count, gt, pred, num_classes)
        softed = np_hard_max(pred_count)
        top_pred += softed

    pos = confusion_matrix.sum(1) # true
    res = confusion_matrix.sum(0) # pred
    tp = np.diag(confusion_matrix) # true positive

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    #count_softmax = np_hard_max(top_pred)
    #top_count_idx = np.argsort(-count_softmax, axis=1)
    #top_count_ratio = np.sort(count_softmax, axis=1)

    #top_cats = top_count_idx[:, :3]
    # flip row wise and column wise
    #top_cats_score = np.flip(np.flip(top_count_ratio[:, -3:], axis=0), axis=1)
    #top_list = []
    #for i in range(top_cats.shape[0]):
    #    cat_names = list(map(lambda cat_id: LABELS[cat_id], top_cats[i]))
    #    top_list.append(dict(zip(cat_names, top_cats_score[i])))

    #pred_top_value = []
    #for i, (label, pred_top) in enumerate(zip(LABELS, top_list)):
    #    pred_top_value.append((label+"_top_pred", pred_top))
    #pred_top_value = OrderedDict(pred_top_value)

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []
    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))
    name_value = OrderedDict(name_value)

    return name_value


def compute_mean_ioU_human(preds, scales, centers, num_classes, datadir, input_size=[384, 384],
                         split='val', dataset_pre='', dataset_name='LIP', restore_from = None):
    assert dataset_name in ['PASCAL_PERSON', 'LIP', 'CIHP', 'ATR']
    base_path, base_pre = dataset_pathes()

    #min56_save = '/home/ly/workspace/git/segmentation/human_parsing/MDCE_Pascal/dataset/output/val_1817_newMeanStd'

    list_path = os.path.join(datadir, dataset_pre, split + '_id_human.txt')
    val_id_human = [i_id.strip() for i_id in open(list_path)]
    val_id = []
    person_num = []
    for i in val_id_human:
        name, pnum = i.split(" ")
        val_id.append(name)
        person_num.append(pnum)
    p_num_unique = np.unique(np.array(person_num))
    pnum_confusion_dict = {}
    pnum_num = {}
    for pn in p_num_unique:
        pnum_confusion_dict[str(pn)] = np.zeros((num_classes, num_classes))
        pnum_num[str(pn)] = 0

    confusion_matrix = np.zeros((num_classes, num_classes))

    #im_names = []
    if len(dataset_pre) == 0:
        dataset_pre = split
    anno_base = base_path[dataset_name+'_'+split]['parsing_anno']

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, anno_base, im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]

        psn_num = str(person_num[i])

        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255
        gt = gt[ignore_index]
        pred = pred[ignore_index]
        # confusion_matrix += get_confusion_matrix(gt, pred, num_classes)
        confusion_matrix_per = get_confusion_matrix(gt, pred, num_classes)
        confusion_matrix += confusion_matrix_per
        pnum_confusion_dict[psn_num] += confusion_matrix_per # conf matrix for groups with psn_num instance in image
        pnum_num[psn_num] += 1 # counts the image number

    def gen_miou(conf_matrix):        
        pos = conf_matrix.sum(1) # true
        res = conf_matrix.sum(0) # pred
        tp = np.diag(conf_matrix) # true positive

        pixel_accuracy = (tp.sum() / pos.sum()) * 100
        mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        IoU_array = IoU_array * 100
        mean_IoU = IoU_array.mean()
        return pixel_accuracy, mean_accuracy, IoU_array, mean_IoU


    pixel_accuracy, mean_accuracy, IoU_array, mean_IoU = gen_miou(confusion_matrix)

    LABELS = labels_name_dicts[dataset_name]
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []
    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value = OrderedDict(name_value)

    for pn, pn_conf in pnum_confusion_dict.items():
        pixel_accuracy, mean_accuracy, IoU_array, mean_IoU  = gen_miou(pn_conf)
        print("pnum:{}, image_counts:{}, pixel_accuracy:{}, mean_accuracy:{}, mean_IoU:{}".format(
           pn, pnum_num[pn],  pixel_accuracy, mean_accuracy, mean_IoU, 
        ))


    return name_value



def np_hard_max(np_array):
    for cat_id in range(np_array.shape[0]):
        cat_sum = np.sum(np_array[cat_id], dtype=np.float64)
        if cat_sum <= 0:
            cat_sum = 1
        cat_norm = np_array[cat_id]/cat_sum
        np_array[cat_id] = cat_norm
    return np_array


def count_pred_cat_num(top_box, gt, pred, num_classes):
    """
    calculate the top predication category
    :param pred_top_box:total top category
    :param gt:
    :param pred:
    :return:
    """
    for cat in range(top_box.shape[0]):
        gt_cat_idx = np.where(gt == cat)
        pred_cats = pred[gt_cat_idx]
        cat_count = np.bincount(pred_cats)
        cat_count = np.pad(cat_count, (0, num_classes - len(cat_count)), 'constant', constant_values=0)
        cat_count = cat_count.astype(np.float64)
        top_box[cat] += cat_count
    return top_box

def compute_mean_ioU_file(preds_dir, num_classes, datadir, dataset='val', dataset_name='CIHP'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_path = os.path.join(preds_dir, im_name + '.png')
        pred = np.asarray(PILImage.open(pred_path))

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum())*100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean())*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array*100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    for i, (label, iou) in enumerate(zip(labels_name_dicts[dataset_name], IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473]):
    palette = get_palette(20)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json_file = os.path.join(datadir, 'annotations', dataset + '.json')
    with open(json_file) as data_file:
        data_list = json.load(data_file)
        data_list = data_list['root']
    for item, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = item['im_name']
        w = item['img_width']
        h = item['img_height']
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        #pred = pred_out
        save_path = os.path.join(result_dir, im_name[:-4]+'.png')

        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    palette = get_palette(20)
    # im_path = '/ssd1/liuting14/Dataset/LIP/val_segmentations/100034_483681.png'
    # #compute_mean_ioU_file(args.pred_path, 20, args.gt_path, 'val')
    # im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    # test = np.asarray( PILImage.open(im_path))
    # print(test.shape)
    # if im.all()!=test.all():
    #     print('different')
    # output_im = PILImage.fromarray(np.zeros((100,100), dtype=np.uint8))
    # output_im.putpalette(palette)
    # output_im.save('test.png')
    pred_dir = '/ssd1/liuting14/exps/lip/snapshots/results/epoch4/'
    num_classes = 20
    datadir = '/ssd1/liuting14/Dataset/LIP/'
    compute_mean_ioU_file(pred_dir, num_classes, datadir, dataset='val')
