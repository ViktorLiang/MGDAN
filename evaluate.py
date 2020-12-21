import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import gc

import cv2
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from PIL import Image
from torch.utils import data
from utils.utils import decode_labels,get_logger
from dataset.datasets_utils import get_pathes
from configs.defaults import _C as cfg

from models.build_model import *
from models import build_model

from dataset.datasets_parsing_all import LIPDataSet
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU, compute_mean_ioU_top, compute_edge_F1_score
from copy import deepcopy
import timeit

DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = ''
INPUT_SIZE = (390,390)
RESTORE_FROM = ''
MODEL_TYPE='res101'

FINAL_PRED_INDEX=1
EVAL_IS_MIRROR=False


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
    parser.add_argument("--dataset-name", type=str, default='LIP', help="One of LIP/PASCAL/CIHP")
    parser.add_argument("--pred-save-dir", type=str, default='None', help="Directory to save prediction results.")
    parser.add_argument("--config-file", type=str, default='None', help="config file for evaluation.")

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg



def valid(model, valloader, input_size, num_samples, gpus, 
        is_mirror=False, 
        pred_save_dir=None, 
        only_need_meta=False,
        with_edge=True):
    model.eval()

    print("###num_samples:", num_samples, " input_size:", input_size)
    # print('###evaluating the {}th parsing result'.format(FINAL_PRED_INDEX)) 
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    
    if isinstance(pred_save_dir, str) and (len(pred_save_dir) == 0):
        pred_save_dir = None
    if pred_save_dir is not None:
        mask_save_dir =  pred_save_dir+"/masks"
        parsing_save_dir =  pred_save_dir+"/parsing"
        parsing_vis_save_dir =  pred_save_dir+"/parsing_vis"
        mkdir_list = [mask_save_dir, parsing_save_dir, parsing_vis_save_dir]
        if with_edge:
            edge_save_dir =  pred_save_dir+"/edges"
            mkdir_list.append(edge_save_dir)

        for d in mkdir_list:
            if not os.path.isdir(d):
                os.makedirs(d)
            print(d)

    with torch.no_grad():
        # start_inf = timeit.default_timer()
        seconds_total = 0
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if pred_save_dir is not None:
                assert num_images == 1, 'batch size must be 1 when have eval resutls saved.'
            if index % 200 == 0:
                print('%d' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]
            if only_need_meta:
                continue

            if is_mirror:
                image_rev = torch.flip(image, [3])
                image = torch.cat((image, image_rev), 0)

            st_time = timeit.default_timer()
            outputs = model(image.cuda(), index)
            end_time = timeit.default_timer()
            seconds_total += (end_time - st_time)

            if gpus > 1:
                for output in outputs:
                    parsing = output[0][FINAL_PRED_INDEX]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    # parsing = parsing.data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                if is_mirror:
                    # bsz = outputs[0][FINAL_PRED_INDEX].shape[0]
                    # prediction_1 = outputs[0][FINAL_PRED_INDEX][:int(bsz/2)]
                    # prediction_2 = outputs[0][FINAL_PRED_INDEX][int(bsz/2):]
                    # prediction_rev = torch.flip(prediction_2, [3])
                    # parsing = (prediction_1 + prediction_rev) / 2
                    bsz = outputs[0][FINAL_PRED_INDEX].shape[0]
                    prediction_1 = outputs[0][FINAL_PRED_INDEX][:int(bsz / 2)]
                    prediction_2 = outputs[0][FINAL_PRED_INDEX][int(bsz / 2):]
                    prediction_rev = prediction_2
                    prediction_rev[:, 14] = prediction_2[:, 15]
                    prediction_rev[:, 15] = prediction_2[:, 14]
                    prediction_rev[:, 16] = prediction_2[:, 17]
                    prediction_rev[:, 17] = prediction_2[:, 16]
                    prediction_rev[:, 18] = prediction_2[:, 19]
                    prediction_rev[:, 19] = prediction_2[:, 18]
                    prediction_rev = torch.flip(prediction_rev, [3])
                    parsing = (prediction_1 + prediction_rev) / 2
                else:
                    parsing = outputs[0][FINAL_PRED_INDEX]
                parsing = interp(parsing).data.cpu().numpy()
                # parsing = parsing.data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                pars_pred = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = pars_pred

                idx += num_images
            
            
            if pred_save_dir is not None:
                save_name = meta["name"][0] + ".png"
                cv2.imwrite(os.path.join(parsing_save_dir, save_name), pars_pred[0])
                pars_pred_vis = decode_labels(pars_pred)
                pars_pred_vis = Image.fromarray(pars_pred_vis[0])
                pars_pred_vis.save(os.path.join(parsing_vis_save_dir, save_name))

                if with_edge:
                    edge_pred = interp(outputs[1]).data.cpu().numpy()
                    edge_pred = np.asarray(np.argmax(edge_pred, axis=1), dtype=np.uint8)*255
                    cv2.imwrite(os.path.join(edge_save_dir, save_name), edge_pred[0])

                mask_pred = interp(outputs[2]).data.cpu().numpy()
                mask_pred = np.asarray(np.argmax(mask_pred, axis=1), dtype=np.uint8)*255
                cv2.imwrite(os.path.join(mask_save_dir, save_name), mask_pred[0])
                print(save_name)

                

    # end_inf = timeit.default_timer()
    # print(end_inf, start_inf, num_samples)
    print("###FPS", num_samples/seconds_total)

    # parsing_preds = parsing_preds[:num_samples, :, :]
    if not only_need_meta:
        return parsing_preds, scales, centers
    else:
        return None, scales, centers


def run_eval(args, eval_ckpt, save_file=None):
    """Create the model and start the evaluation process."""

    gpus_str = ",".join(list(map(str, args.TEST.GPU)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    # gpus = [int(i) for i in args.TEST.GPU.split(',')]
    gpus = args.TEST.GPU

    h, w = args.TEST.INPUT_SIZE
    # h, w = map(int, args.input_size)

    input_size = (h, w)
    # model = build_interpre_model(args, is_train=False)
    model = getattr(build_model, args.TEST.BUILD_MODEL)(args, is_train=False)
    #print(model)

    normalize = transforms.Normalize(mean=args.INPUT.INPUT_MEAN, std=args.INPUT.INPUT_STD)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print(args.INPUT.DATA_ROOT, args.TEST.SPLIT, input_size)
    lip_dataset = LIPDataSet(args.INPUT.DATA_ROOT, args.TEST.SPLIT, 
                            crop_size=input_size, 
                            transform=transform, 
                            dataset_name=args.INPUT.DATASET_NAME)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, 
                                batch_size=args.TEST.BATCH_SIZE * len(gpus),
                                shuffle=False, 
                                pin_memory=True)

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(eval_ckpt)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])
        # print(key, nkey)
        # keys = nkey.split('.')
        # if keys[0] == 'seg_rescore_conv':
        #     del state_dict[nkey]
        #     print(nkey, ' deleted!')

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus), 
                        is_mirror=args.TEST.AUG_MIRROR, 
                        pred_save_dir=args.TEST.PRED_SAVE_DIR, 
                        only_need_meta=False,
                        with_edge=args.MODEL.WITH_EDGE)
    

    # mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    print("####################check point "+eval_ckpt+":")

    dpath, dpre = get_pathes(args.INPUT.DATASET_NAME, args.TEST.SPLIT)
    mIoU = compute_mean_ioU_top(parsing_preds, scales, centers, args.INPUT.NUM_CLASSES, args.INPUT.DATA_ROOT, input_size,
                                          split=args.TEST.SPLIT, 
                                          dataset_pre=dpre,
                                          dataset_name=args.INPUT.DATASET_NAME,
                                          restore_from=eval_ckpt)
    print(mIoU)
    print("-----------------------------")
    if save_file is not None:
        with open(save_file, 'a+') as f:
            f.write("####################check point {} , prediction index:{} \n".format(eval_ckpt, FINAL_PRED_INDEX))
            f.write(str(mIoU)+'\n')

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

    run_eval(args, args.TEST.RESTORE_FROM, save_file)



if __name__ == '__main__':
    main()
