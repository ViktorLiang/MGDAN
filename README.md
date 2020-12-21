# MGDAN
# MaskHumanParsing

A multi-task human parsing network. We archieve competitive performance on both single and multiple human parsing datasets.

## Dependencies

* Pytorch-1.0.1
* cuda-10.1
* Deformable convolution from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0

## Dataset

Download LIP and CIHP from http://sysu-hcp.net/lip/overview.php

Pascal Person-Part from http://roozbehm.info/pascal-parts/pascal-parts.html

Editing configuration for dataset in function 'dataset_pathes' in dataset/dataset_utils.py

## Training:

python train.py --config-file configs/cfg_lip.yaml
python train.py --config-file configs/cfg_cihp.yaml
python train.py --config-file configs/cfg_pascal.yaml

## Testing:

python evaluate.py --config-file configs/cfg_lip.yaml
python evaluate.py --config-file configs/cfg_cihp.yaml
python evaluate_pascal.py --config-file configs/cfg_pascal.yaml

## Trained Model

Trained Model can be found here:https://drive.google.com/drive/folders/1SuU--ycSWOEYxdygmojkAY95vfrRtk9J?usp=sharing

Some codes borrowed from https://github.com/liutinglt/CE2P

