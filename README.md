# MGDAN

Mask Guided Deformation Adaptive Network for Human Parsing.

An efficient strong baseline for human parsing.

## Dependencies

* Pytorch-1.0.1
* cuda-10.1
* Deformable convolution from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0

A deformable convolution models in zip file from above link has been provided in 'utils/ops/Deformable-Convolution-V2-PyTorch-master.zip'. 
Uncompress it and build with 'sh make.sh', then rename the folder name to 'dcn'.

## Dataset

Download LIP and CIHP from http://sysu-hcp.net/lip/overview.php

Pascal Person-Part from http://roozbehm.info/pascal-parts/pascal-parts.html

Editing 'INPUT:DATA_ROOT' in 'configs/cfg_datasetName.yaml' to your dataset directory.

Editing configuration for different split(e.g.'train' or 'val') in function 'dataset_pathes' in 'dataset/dataset_utils.py'

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

