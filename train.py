import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import timeit
import argparse

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils

from utils.utils import get_logger


import numpy as np

from configs.defaults import _C as cfg
from models.build_model import *
from models import build_model


from utils.encoding import DataParallelModel, DataParallelCriterion
from dataset.datasets_parsing_new import LIPDataSet
from utils.criterion_mgdan import CriterionAll
from utils.utils import decode_parsing, inv_preprocess

def lr_poly(base_lr, iter, max_iter, power, base_lr_ratio=1):
    return base_lr * base_lr_ratio * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters, args):
    lr = lr_poly(args.SOLVER.LEARNING_RATE, i_iter, total_iters, args.SOLVER.LR_POWER)
    for key, v in enumerate(optimizer.param_groups):
        v['lr'] = lr
    return lr


def count_parameters(model, print_model=False):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    import locale
    locale.setlocale(locale.LC_ALL, '')
    if print_model:
        print(model)
    print("###parameters total number: ",  locale.format("%d", params, grouping=True))


def define_writer(args):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(args.TRAIN.SNAPSHOT_DIR)
    if not args.TRAIN.GPU_IDS == 'None':
        gpus = [str(i) for i in args.TRAIN.GPU_IDS]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    return writer


def main():
    args = get_arguments()
    train_cfg = args.TRAIN.SNAPSHOT_DIR+"/train_cfg.txt"
    train_logger = get_logger(train_cfg)
    train_logger.info(args)
    writer = define_writer(args)
    gpus = args.TRAIN.GPU_IDS

    h, w = map(int, args.INPUT.INPUT_SIZE)
    input_size = [h, w]
    model = getattr(build_model, args.MODEL.BUILD_MODEL)(args)

    count_parameters(model, print_model=True)

    restore_from_state_dict = torch.load(args.TRAIN.RESTORE_FROM)
    training_state_dict = model.state_dict().copy()
    if args.TRAIN.RESTORE_FROM.find('epoch') == -1:  # restore from imageNet pre-trained model
        for i in restore_from_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                training_state_dict['.'.join(i_parts[0:])] = restore_from_state_dict[i]
        print("###params restored from ResNet {}.".format(args.TRAIN.RESTORE_FROM))
    else:  # resume from checkpoint
        for i in restore_from_state_dict:
            i_parts = i.split('.')
            new_key = '.'.join(i_parts[1:])
            assert new_key in training_state_dict, new_key
            training_state_dict['.'.join(i_parts[1:])] = restore_from_state_dict[i]
        print("###params resumed from trained {}.".format(args.TRAIN.RESTORE_FROM))
    model.load_state_dict(training_state_dict)

    # for param in model.named_parameters():
    #     p_parts = param[0].split('.')
    #     if not p_parts[0] in ['seg_rescore_conv']:
    #         param[1].requires_grad = False
    # print("###only seg_rescore_conv trainable.")

    model = DataParallelModel(model).cuda()
    model.cuda()

    normalize = transforms.Normalize(mean=args.INPUT.INPUT_MEAN,  # pascal
                                     std=args.INPUT.INPUT_STD)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    params_dict = dict(model.named_parameters())
    params = []
    for key, v in params_dict.items():
        params += [{'params': v, 'lr': args.SOLVER.LEARNING_RATE,
                    'weight_decay': args.SOLVER.WEIGHT_DECAY, 'name': key}]

    optimizer = optim.SGD(
        # model.parameters(),
        params,
        lr=args.SOLVER.LEARNING_RATE,
        momentum=args.SOLVER.MOMENTUM,
        weight_decay=args.SOLVER.WEIGHT_DECAY
    )
    optimizer.zero_grad()

    trainloader = data.DataLoader(LIPDataSet(args.INPUT.DATA_ROOT,
                                             'train',
                                             crop_size=input_size,
                                             transform=transform,
                                             dataset_name=args.INPUT.DATASET_NAME,
                                             label_edge=args.MODEL.WITH_EDGE),
                                  batch_size=args.TRAIN.BATCH_SIZE * len(args.TRAIN.GPU_IDS),
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)
    total_iters = args.TRAIN.EPOCHES * len(trainloader)

    criterion = CriterionAll(ignore_index=255, 
                            vis_loss=True, 
                            num_classes=args.TRAIN.NUM_CLASSES, 
                            with_edge=args.MODEL.WITH_EDGE,
                            with_lovasz_loss=True)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    tensor_board_image_size = [int(h//2.2), int(w//2.2)]
    start = timeit.default_timer()
    for epoch in range(args.TRAIN.START_EPOCH, args.TRAIN.EPOCHES):
            start_epoch_time = timeit.default_timer()
            model.train()

            for i_iter, batch in enumerate(trainloader):
                i_iter += len(trainloader) * epoch
                images, labels, edges, meta = batch
                edges = edges.long().cuda(non_blocking=True)
                labels = labels.long().cuda(non_blocking=True)
                gt = [labels, edges]

                preds = model(images, i_iter)

                loss, edge_loss, fg_loss, parsing_loss_1, parsing_loss_2, parsing_loss_3, parsing_loss_4 = criterion(preds, gt) 

                loss_tasks = [edge_loss, fg_loss]
                vis_tasks_name = ['aux/edge', 'aux/mask']
                vis_tasks_loss = {}
                for i, lss in enumerate(loss_tasks):
                    vis_tasks_loss[vis_tasks_name[i]] = torch.zeros_like(lss).copy_(lss.cpu())
                
                loss_details = [parsing_loss_1, parsing_loss_2, parsing_loss_3, parsing_loss_4]
                vis_details_name = ['parsing/pars_1','parsing/pars_2','parsing/pars_3','parsing/pars_4']
                vis_details_loss = {}
                for i, lss in enumerate(loss_details):
                    vis_details_loss[vis_details_name[i]] = torch.zeros_like(lss).copy_(lss.cpu())
                
                lr = adjust_learning_rate(optimizer, i_iter, total_iters, args)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i_iter % 100 == 0:
                    writer.add_scalar('learning_rate', lr, i_iter)
                    writer.add_scalar('loss_sum', loss.data.cpu().numpy(), i_iter)
                    for vis_name, lss_value in vis_tasks_loss.items():
                        writer.add_scalar(vis_name, lss_value, i_iter)
                    for vis_name, lss_value in vis_details_loss.items():
                        writer.add_scalar(vis_name, lss_value, i_iter)

                if i_iter % 500 == 0:
                    images_inv = inv_preprocess(images, args.VISUALIZE.SAVE_IMAGES_NUM)
                    labels_colors = decode_parsing(labels, args.VISUALIZE.SAVE_IMAGES_NUM, args.TRAIN.NUM_CLASSES, is_pred=False)

                    if len(gpus) > 1 and isinstance(preds, list):
                        preds = preds[0]

                    preds_colors_catted = []
                    for i in range(len(preds[0])):
                        preds_seg = decode_parsing(preds[0][i], args.VISUALIZE.SAVE_IMAGES_NUM, args.TRAIN.NUM_CLASSES,
                                                   is_pred=True)
                        preds_seg = F.interpolate(input=preds_seg.type(torch.float),
                                                  size=(tensor_board_image_size[0], tensor_board_image_size[1]),
                                                  mode='bilinear', align_corners=True)
                        preds_colors_catted.append(preds_seg.type(torch.uint8))
                    preds_colors = torch.cat(preds_colors_catted, dim=3)

                    img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                    lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                    pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                    writer.add_image('Images/', img, i_iter)
                    writer.add_image('LabelParsing/', lab, i_iter)
                    writer.add_image('PredParsing/', pred, i_iter)

                    #writting edge logs
                    edges_colors = decode_parsing(edges, args.VISUALIZE.SAVE_IMAGES_NUM, 2, is_pred=False)
                    edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                    pred_edges = decode_parsing(preds[1], args.VISUALIZE.SAVE_IMAGES_NUM, 2, is_pred=True)
                    pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)
                    writer.add_image('LabelEdges/', edge, i_iter)
                    writer.add_image('PredEdges/', pred_edge, i_iter)

                    #writting mask logs
                    if isinstance(preds[2], list):
                        pred_fg_ = preds[2][0]
                    else:
                        pred_fg_ = preds[2]
                    pred_fg = decode_parsing(pred_fg_, args.VISUALIZE.SAVE_IMAGES_NUM, 2, is_pred=True)
                    pred_fg = F.interpolate(input=pred_fg.type(torch.float),
                                            size=(tensor_board_image_size[0], tensor_board_image_size[1]),
                                            mode='bilinear', align_corners=True)
                    pred_fg = vutils.make_grid(pred_fg.type(torch.uint8), normalize=False, scale_each=True)
                    writer.add_image('PredMasks/', pred_fg, i_iter)
                if i_iter % len(trainloader) == 0 or i_iter % 50 == 0:
                    print('iter = {} of {} completed, loss = {}'.format(i_iter, total_iters, loss.data.cpu().numpy()))

            torch.save(model.state_dict(), os.path.join(args.TRAIN.SNAPSHOT_DIR, '{}_epoch_{}.pth'.format(args.INPUT.DATASET_NAME, epoch)) )
            end_epoch_time = timeit.default_timer()
            print("epoch {}, {} seconds".format(epoch, end_epoch_time - start_epoch_time))

    end = timeit.default_timer()
    print(end - start, 'seconds')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MGDAN parsing network")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not os.path.isdir(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
    return cfg


if __name__ == '__main__':
    m = main()
