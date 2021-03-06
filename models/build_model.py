from utils.utils import get_logger

def build_shared_model(cfg, is_train=True):
    from models.sharedTask.mask_edge_shared import build_ResDeeplab
    if is_train:
        save_dir = cfg.TRAIN.SNAPSHOT_DIR
        logger = get_logger(save_dir)
        logger.info("Training with configs:\n{}".format(cfg))
    res_model = build_ResDeeplab(cfg)
    return res_model

def build_shared_edgeGrids_model(cfg, is_train=True):
    from models.sharedTask.mask_edge_edgeGrids import build_ResDeeplab

    if is_train:
        save_dir = cfg.TRAIN.SNAPSHOT_DIR
        logger = get_logger(save_dir)
        logger.info("Training with configs:\n{}".format(cfg))
    res_model = build_ResDeeplab(cfg)
    return res_model

def build_shared_edgeMaskGrids_model(cfg, is_train=True):
    from models.sharedTask.mask_edge_sharedGrids import build_ResDeeplab

    if is_train:
        save_dir = cfg.TRAIN.SNAPSHOT_DIR
        logger = get_logger(save_dir)
        logger.info("Training with configs:\n{}".format(cfg))
    res_model = build_ResDeeplab(cfg)
    return res_model
