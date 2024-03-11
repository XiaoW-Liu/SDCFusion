# SDCFusion
The project is being continuously updated

## Recommended Environment
 - [ ] torch  1.10.0
 - [ ] cudatoolkit 11.3.1
 - [ ] torchvision 0.11.0
 - [ ] kornia 0.6.5
 - [ ] pillow  8.3.2
 
## Test
`python test_Fusion.py --dataroot=XXX --dataset_name=XXX --resume=./results/SDCFusion/XXX.pth`

## Train
`python train.py --dataroot=XXX --name=SDCFusion`

The code references the following article:

@article{TANG202228SeAFusion,
title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
journal = {Information Fusion},
volume = {82},
pages = {28-42},
year = {2022},
issn = {1566-2535}
}

@article{TANG2023PSFusion,
  title={Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity},
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},
  journal={Information Fusion},
  volume = {99},
  pages = {101870},
  year={2023},
}

