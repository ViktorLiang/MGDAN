import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import gc
import cv2
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torchvision.transforms as transforms
from utils.utils import decode_labels,get_logger
from dataset.datasets_utils import get_pathes

from configs.defaults import _C as cfg
from models.multiTask.mask_edge_patch import Res_Deeplab as Res_Deeplab_edgLK

# from dataset.datasets import LIPDataSet
from dataset.datasets_parsing_all import LIPDataSet
from utils.miou_old import compute_mean_ioU, compute_mean_ioU_top, compute_edge_F1_score
from copy import deepcopy


DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = ''
INPUT_SIZE = (390,390)
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
MODEL_TYPE='res101'

PARSING_PRED_LEVEL=1
print('###evaluating the {}th parsing result'.format(PARSING_PRED_LEVEL)) 
EVAL_IS_MIRROR=True


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MGDAN human parsing")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.", default=RESTORE_FROM)
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--eval-ckpts", type=str, help="eval checkpoints No", default='-1')
    parser.add_argument("--eval-ckpts-range", type=str, help="eval checkpoints range", default='-1')
    parser.add_argument("--ckpt-dir", type=str, help="eval checkpoints No")
    parser.add_argument("--with-fg", type=int,
                        help="Whether to add foreground branch into net work.")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--config-file", type=str, default='', help="config file for evaluation.")

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

    # return parser.parse_args()


def valid(model, valloader, input_size, num_samples, gpus, is_mirror=False):
    model.eval()

    print("num_samples:",num_samples, " input_size:",input_size)
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 300 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            if is_mirror:
                image_rev = torch.flip(image, [3])
                image = torch.cat((image, image_rev), 0)

            outputs = model(image.cuda(), index)
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][PARSING_PRED_LEVEL]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                if is_mirror:
                    bsz = outputs[0][PARSING_PRED_LEVEL].shape[0]
                    prediction_1 = outputs[0][PARSING_PRED_LEVEL][:int(bsz/2)]
                    prediction_2 = outputs[0][PARSING_PRED_LEVEL][int(bsz/2):]
                    prediction_rev = torch.flip(prediction_2, [3])
                    parsing = (prediction_1 + prediction_rev) / 2
                else:
                    parsing = outputs[0][PARSING_PRED_LEVEL]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                pars_pred = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = pars_pred

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers


def run_eval(args, save_file=None):
    """Create the model and start the evaluation process."""

    gpus_str = ",".join(list(map(str, args.TEST.GPU)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    gpus = args.TEST.GPU

    h, w = args.TEST.INPUT_SIZE
    input_size = (h, w)
    print("###evaluating with indendent multi-task.")
    model = Res_Deeplab_edgLK(num_classes=args.TRAIN.NUM_CLASSES)

    normalize = transforms.Normalize(mean=args.INPUT.INPUT_MEAN,
                                     std=args.INPUT.INPUT_STD)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = LIPDataSet(args.INPUT.DATA_ROOT, 
                        args.TEST.SPLIT, 
                        crop_size=input_size, 
                        transform=transform,
                        dataset_name=args.INPUT.DATASET_NAME)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, 
                                batch_size=args.TEST.BATCH_SIZE * len(gpus),
                                shuffle=False, 
                                pin_memory=True)

    restore_from = args.TEST.RESTORE_FROM

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    print("###Evaluating in mirror ? {}!".format(str(EVAL_IS_MIRROR)))
    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus), is_mirror=EVAL_IS_MIRROR)

    # mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    print("####################check point "+restore_from+":")


    mIoU, top_pred = compute_mean_ioU_top(parsing_preds, scales, centers, 
                                        args.TRAIN.NUM_CLASSES, args.INPUT.DATA_ROOT, 
                                        input_size,
                                        dataset=args.TEST.SPLIT, 
                                        restore_from=restore_from)
    print(mIoU)
    print("-----------------------------")
    print(top_pred)
    print()
    if save_file is not None:
        with open(save_file, 'a+') as f:
            f.write("####################check point "+restore_from+'\n')
            f.write(str(mIoU)+'\n')
            f.write(str(top_pred)+'\n')

    gc.collect()
    del parsing_preds

def main():
    args = get_arguments()
    eval_save_dir = "./eval_results" 
    save_file = "{}/eval_results_{}.txt".format(eval_save_dir, args.INPUT.DATASET_NAME)
    
    eval_config_file = "eval_cfg.txt"
    if not os.path.isdir(eval_save_dir):
        os.mkdir(eval_save_dir)
    eval_results_save_file = eval_save_dir+"/"+eval_config_file 
    print("###Evaluation results are saving to:{}".format(eval_results_save_file))
    logger = get_logger(eval_results_save_file)
    logger.info("Evaluating with configs:\n{}".format(cfg))
    run_eval(args, save_file)


if __name__ == '__main__':
    main()
