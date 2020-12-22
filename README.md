# MGDAN

A multi-task human parsing network.

## Dependencies

* Pytorch-1.0.1
* cuda-10.1
* Deformable convolution from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0

## Dataset

Download LIP and CIHP from http://sysu-hcp.net/lip/overview.php

Pascal Person-Part from http://roozbehm.info/pascal-parts/pascal-parts.html

Editing configuration for dataset in function 'dataset_pathes' in dataset/dataset_utils.py

## Training:
Train on LIP:
  python train.py --config-file configs/cfg_lip.yaml

Train on CIHP:
  python train.py --config-file configs/cfg_cihp.yaml

Train on Pascal-Person-Part:
  python train.py --config-file configs/cfg_pascal.yaml

## Testing:
Download trained from https://drive.google.com/drive/folders/1A7rSUC_B78Nbt7rRbXiODbOb5zLMTDZS?usp=sharing
Save trained model in directory 'trained_models'.

Evaluation of LIP:
  python evaluate.py --config-file configs/cfg_lip.yaml


Evaluation of CIHP:
  python evaluate.py --config-file configs/cfg_cihp.yaml


Evaluation of Pascal-Person-Part:
  python evaluate_pascal.py --config-file configs/cfg_pascal.yaml


## Trained Model

Trained Model can be found here:https://drive.google.com/drive/folders/1A7rSUC_B78Nbt7rRbXiODbOb5zLMTDZS?usp=sharing

Some codes borrowed from https://github.com/liutinglt/CE2P

