def dataset_pathes():
    DATASET_PATH = {
            "PASCAL_PERSON_train": {"image": "train/JPEGImages", 
                                    "parsing_anno": "train/parsing_anno",
                                    "instance_anno": "train/instance_anno",
                                    "reverse_anno":""},
            "PASCAL_PERSON_val": {"image": "val/JPEGImages", 
                                "parsing_anno": "val/parsing_anno",
                                "instance_anno": "val/instance_anno",
                                "reverse_anno":""},

            "CIHP_train": {"image": "Training/Images", 
                        "parsing_anno": "Training/Category_ids",
                        "instance_anno": "Training/Human_ids",
                        "reverse_anno":"Training/Processed/Category_rev_ids/"},
            "CIHP_val": {"image": "Validation/Images", 
                        "parsing_anno": "Validation/Category_ids",
                        "instance_anno": "Validation/Human_ids",
                        "reverse_anno":""},

            "LIP_train": {"image": "train_images", 
                        "parsing_anno": "train_segmentations", 
                        "instance_anno": None,
                        "reverse_anno":"train_segmentations_reversed"},
            "LIP_val": {"image": "val_images", 
                    "parsing_anno": "val_segmentations", 
                    "instance_anno": None,
                    "reverse_anno":""},
            "ATR_train":{'image':'JPEGImages', 
                        'parsing_anno':'SegmentationClassAug', 
                        'reverse_anno':'Category_rev_ids'},
            "ATR_val":{'image':'JPEGImages', 
                        'parsing_anno':'SegmentationClassAug', 
                        'reverse_anno':'Category_rev_ids'}
        }
    DATASET_PRE = {
            "CIHP_train": "Training", "CIHP_val": "Validation",
            "PASCAL_PERSON_train": "", "PASCAL_PERSON_val": "",
            "LIP_train": "", "LIP_val": "",
            'ATR_train': '', 'ATR_val': ''
        }
    return DATASET_PATH, DATASET_PRE

def get_pathes(dataset_name, split):
    assert dataset_name in ['PASCAL_PERSON', 'LIP', 'CIHP', 'ATR']
    pathes, pres = dataset_pathes()
    dkey = dataset_name+'_'+split
    return pathes[dkey], pres[dkey]