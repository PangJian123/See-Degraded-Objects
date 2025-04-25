# See Degraded Objects: A Physics-guided Approach for Object Detection in Adverse Environments
This repo is the official implementation of our paper “See Degraded Objects: A Physics‑Guided Approach for Object Detection in Adverse Environments,” published in IEEE Transactions on Image Processing (TIP), 2025.

## Requirements:
    CUDA  10.2
    Python  3.9
    Pytorch  1.10.0
    torchvision  0.11.0
    numpy  1.26.3
    detectron2  0.6

## Get Started
### 1.Install:
    download the code
    git clone https://github.com/PangJian123/See-Degraded-Objects.git
    cd See-Degraded-Objects
    
### 2.Datasets
Download datasets: 
    [*Foggy_Cityscapes_beta_0.02*](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/Foggy_Cityscapes_beta_0.02_VOC.zip) (synthetic fog);
    [*RTTS*](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/RTTS_VOC.zip) (real fog);
    [*DUO*](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/DUO_VOC.zip) (real underwater)
Each dataset is in VOC format, consists of images, annotations, frequency prior and environment prior.

### 3.Run the training file:
    training on RTTS dataset.
    python train_res50_fpn.py --epochs 20 --name RTTS_weightfusion_fuzzya_15_mem_tri_ce1 --prior 'weight_fusion' --fuzzy_a 1.5 --member_f 'tri' --p_ce 1 --num-classes 5 --json-name 'voc_fog_classes.json' --data-path 'path/to/RTTS_VOC' --resume ''
    training on Foggy_Cityscapes_beta_0.02 dataset.
    python train_res50_fpn.py --epochs 20 --name FOC_weightfusion_fuzzya_15_mem_tri_ce1  --prior 'weight_fusion' --fuzzy_a 1.5 --member_f 'tri' --p_ce 1 --num-classes 5 --json-name 'voc_fog_classes.json' --data-path 'path/to/Foggy_Cityscapes_beta_0.02_VOC/' --resume ''
    training on DUO dataset.
    python train_res50_fpn.py --epochs 20 --name DUO_ams_fuzzya_1_mem_tri_ce15 --prior 'ams' --fuzzy_a 1 --member_f 'tri' --p_ce 1.5 --num-classes 4 --json-name 'DUO_classes.json' --data-path 'path/to/DUO_VOC' --resume ''
    

## Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Jian Pang
    China University of Petroleum (East China)                                                           
    Email: jianpang@s.upc.edu.cn

## Acknowledgement
* Our code is based on https://github.com/WZMIAOMIAO/deep-learning-for-image-processing.

## Citation
Please consider citing our work as follows if it is helpful.
```
@ARTICLE{see-degraded-objects,
  author={Liu, Weifeng and Pang, Jian and Zhang, Bingfeng and Wang, Jin and Liu, Baodi and Tao, Dapeng},
  journal={IEEE Transactions on Image Processing}, 
  title={See Degraded Objects: A Physics-Guided Approach for Object Detection in Adverse Environments}, 
  year={2025},
  volume={34},
  number={},
  pages={2198-2212},
  doi={10.1109/TIP.2025.3551533}}
```
