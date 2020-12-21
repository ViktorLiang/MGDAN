import sys
import os
from PIL import Image
import numpy as np
import torchvision
import torch
import cv2
import logging

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                
LABEL_COLOURS_CIHP = [(0,0,0)
				, (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0)
				, (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0)
				, (52,86,128), (0,128,0) , (0,0,255), (51,170,221), (0,255,255)
				, (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

LABEL_MAPS_LIP = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes', 'Dress', 'Coat',
            'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
            'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

def decode_parsing(labels, num_images=1, num_classes=20, is_pred=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    pred_labels = labels[:num_images].clone().cpu().data
    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    n, h, w = pred_labels.size()

    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    for i, c in enumerate(LABEL_COLOURS_CIHP):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]

    return labels_color

def decode_parsing_numpy(labels, is_pred=False):
    """Decode batch of segmentation masks.

    Args:
      labels: input matrix to show

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    pred_labels = labels.copy()
    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    assert pred_labels.size >= 2
    if pred_labels.size == 3:
        n, h, w = pred_labels.shape
    else:
        h, w = pred_labels.shape
        n = 1
        pred_labels = pred_labels[None]

    labels_color = np.zeros([n, 3, h, w], dtype=np.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]
    if n == 1:
        labels_color = np.transpose(labels_color[0], (1,2,0))

    return labels_color

def inv_preprocess(imgs, num_images, toRGB=True):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    if toRGB:
        rev_imgs = rev_imgs[:, [2,1,0]]
    return rev_imgs


def merge_tensors_to_one(src_tensor):
    n,c,h,w = src_tensor.shape
    t = torch.zeros(n, h*c, w*c)
    for i in range(c):
        t[:, i*h:(i*h)+h, i*w:(i*w)+w] = src_tensor[:, i]
    return t

def cv2_list_show(mask_list, cat_axis=1, show=False):
    c_m = np.concatenate(mask_list, axis=cat_axis)
    if show:
        cv2.imshow('cv2_list_show', c_m)
        cv2.waitKeyEx(0)
        cv2.destroyAllWindows()
    return c_m


def cv2_image_show(I):
    cv2.imshow('cv2_image_show', I)
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0,3,1,2)

def get_logger(log_file):
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    log_writer = logging.getLogger()
    log_writer.addHandler(ch)
    return log_writer

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    Args:
        mask: result of inference after taking argmax.
        num_images: number of images to decode from the batch.
        num_classes: number of classes to predict (including background).

    Returns:
        A batch with num_images RGB images of the same size as the input.
    """
    
    label_colours = [(0,0,0)
				, (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0)
				, (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0)
				, (52,86,128), (0,128,0) , (0,0,255), (51,170,221), (0,255,255)
				, (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def vis_gt(gt_dir=None, save_dir=None):
    assert gt_dir is not None and save_dir is not None
    gt_files = [gt_dir+"/"+i for i in os.listdir(gt_dir)]
    totno = len(gt_files)
    for i, gtf in enumerate(gt_files):
        imname = os.path.basename(gtf)
        vis_save = os.path.join(save_dir, imname)
        im = cv2.imread(gtf, cv2.IMREAD_UNCHANGED)
        dec_gt = decode_labels(im[None])
        # cv2.imshow(imname, dec_gt[0])
        # cv2.waitKey(0)
        # if i == 10:
        #     cv2.destroyAllWindows()
        #     exit()
        # continue
        im_vis = Image.fromarray(dec_gt[0])
        im_vis.save(vis_save)
        print("{}/{}".format(i, totno), vis_save)

def draw_color_map():
    from matplotlib import pyplot as plt
    w=10
    h=5
    fig=plt.figure(figsize=(8, 8))

    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        # img = np.random.randint(10, size=(h,w))
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for a in range(w):
            for b in range(h):
                pixels[a, b] = LABEL_COLOURS_CIHP[i-1]
        np_img = np.array(img)
        # for j_, j in enumerate(mask[i, :, :]):
            # for k_, k in enumerate(j):
                # if k < num_classes:
                    # pixels[k_,j_] = label_colours[k]
        fig.add_subplot(rows, columns, i)
        plt.title("{}:{}".format(str(i-1), LABEL_MAPS_LIP[i-1]))
        plt.imshow(np_img)
    # plt.axis('off')
    plt.show()
    plt.axis('off')



class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)



if __name__ == '__main__':
    # val_parsing_gt_dir = '/home/ly/data/datasets/human_parsing/LIP/val_segmentations'
    # vis_save_dir = '/home/ly/data/datasets/human_parsing/LIP/vis_annotations/vis_parsing_val'
    # val_pred_parsing_dir = '/home/ly/data/save_features/lip/predictions/maskDCN_conv1Rescore/parsing'
    # vis_pred_save_dir = '/home/ly/data/save_features/lip/predictions/maskDCN_conv1Rescore/parsing_vis'
    # vis_gt(val_parsing_gt_dir, vis_save_dir)
    draw_color_map()