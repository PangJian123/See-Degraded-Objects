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
- Download datasets through the links below (datasets are coming up soon): 
*Foggy_Cityscapes_beta_0.02*:? 
*RTTS*:?
*DUO*:?

### 3.Run the training file:
python train_res50_fpn.py --epochs 20 --name RTTS_weightfusion_fuzzya_15_mem_tri_ce1 --prior 'weight_fusion' --fuzzy_a 1.5 --member_f 'tri' --p_ce 1 --fuzzy_c 3 --num-classes 5 --json-name 'voc_fog_classes.json' --data-path 'path/to/RTTS_VOC' --resume ''


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
