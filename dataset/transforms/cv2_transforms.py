import random

import numpy as np
import cv2

def to_gray(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.expand_dims(im, axis=0)
    im = np.repeat(im, 3, axis=0).transpose(1, 2, 0)
    return im

def channel_shuffle(im_src):
    rgb_idx = [0,1,2]
    random.shuffle(rgb_idx)
    im_shuffled = np.zeros_like(im_src)
    for to_ch, from_ch in enumerate(rgb_idx):
        im_shuffled[:,:,to_ch] = im_src[:,:,from_ch]
    return im_shuffled

def median_blur(im_src, ksize=5):
    median_blur = cv2.medianBlur(im_src, ksize)
    return median_blur


def randomSized(sample, small_scale=0.5, big_scale=2, ):
    img = sample['image']
    mask = sample['label']
    assert img.size == mask.size

    w = int(random.uniform(small_scale, big_scale) * img.size[1])
    h = int(random.uniform(small_scale, big_scale) * img.size[0])
    im = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
    sample = {'image': img, 'label': mask}
    return sample

def randomBrightnessContrast(im, brightness=(-0.5, 0.5), contrast=(-0.5, 0.5)):
    f = random.uniform(1 + contrast[0], 1 + contrast[1])
    im = np.clip(im * f, 0, 255)
    f = random.uniform(brightness[0], brightness[1])*255
    im = np.clip(im + f, 0, 255).astype(np.uint8)
    return im 