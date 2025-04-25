# See Degraded Objects: A Physics‑Guided Approach for Object Detection in Adverse Environments

This repository contains the official PyTorch implementation of our paper **“See Degraded Objects: A Physics‑Guided Approach for Object Detection in Adverse Environments,”** published in *IEEE Transactions on Image Processing (TIP)*, 2025.

## Requirements

```text
CUDA        10.2
Python      3.9
PyTorch     1.10.0
torchvision 0.11.0
numpy       1.26.3
detectron2  0.6
```

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/PangJian123/See-Degraded-Objects.git
cd See-Degraded-Objects
```

### 2. Datasets

All datasets are provided in Pascal VOC format and include images, annotations, frequency priors, and environment priors.

| Dataset                           | Description     | Link                                                                                                                              |
| --------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Foggy\_Cityscapes\_beta\_0.02** | Synthetic fog   | [Download](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/Foggy_Cityscapes_beta_0.02_VOC.zip) |
| **RTTS**                          | Real fog        | [Download](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/RTTS_VOC.zip)                       |
| **DUO**                           | Real underwater | [Download](https://huggingface.co/datasets/PJ-UPC/datasets-for-see-degraded-objects/blob/main/DUO_VOC.zip)                        |

After downloading, extract each archive and adjust the `--data-path` argument in the commands below.

### 3. Training

Example commands for training **ResNet‑50‑FPN** detectors:

```bash
# Train on RTTS
python train_res50_fpn.py --epochs 20 --name RTTS_weightfusion_fuzzya_15_mem_tri_ce1 --prior "weight_fusion" --fuzzy_a 1.5 --member_f "tri" --p_ce 1 --num-classes 5 --json-name "voc_fog_classes.json" --data-path "/path/to/RTTS_VOC"

# Train on Foggy_Cityscapes_beta_0.02
python train_res50_fpn.py --epochs 20 --name FOC_weightfusion_fuzzya_15_mem_tri_ce1 --prior "weight_fusion" --fuzzy_a 1.5 --member_f "tri" --p_ce 1 --num-classes 5 --json-name "voc_fog_classes.json" --data-path "/path/to/Foggy_Cityscapes_beta_0.02_VOC"

# Train on DUO
python train_res50_fpn.py --epochs 20 --name DUO_ams_fuzzya_1_mem_tri_ce15 --prior "ams" --fuzzy_a 1 --member_f "tri" --p_ce 1.5 --num-classes 4 --json-name "DUO_classes.json" --data-path "/path/to/DUO_VOC"
```

Feel free to tweak the hyper‑parameters to suit your use‑case.

## Contact

If you have any questions or encounter any issues, please contact:

```
Jian Pang  
China University of Petroleum (East China)  
jianpang@s.upc.edu.cn
```

## Acknowledgement

This implementation builds upon the excellent [deep‑learning‑for‑image‑processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing) project—many thanks to the authors.

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{see-degraded-objects,
  author  = {Liu, Weifeng and Pang, Jian and Zhang, Bingfeng and Wang, Jin and Liu, Baodi and Tao, Dapeng},
  title   = {See Degraded Objects: A Physics-Guided Approach for Object Detection in Adverse Environments},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025},
  volume  = {34},
  pages   = {2198--2212},
  doi     = {10.1109/TIP.2025.3551533}
}
```
