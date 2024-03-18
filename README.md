# Domain-Aware Triplet Loss in Domain Generalization

Official PyTorch implementation of Domain-Aware Triplet Loss in Domain Generalization.

Kaiyu Guo, Brian C. Lovell.

This Paper has been accepted by the journal of Computer Vision and Image Understanding (CVIU)

Note that this project is built upon [SWAD](https://github.com/khanrc/swad).


### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```
### Environments

Environment details used for our study.

```
Python: 3.9.7
PyTorch: 1.10.2
Torchvision: 0.11.3
CUDA: 11.3
CUDNN: 8200
NumPy: 1.21.2
PIL: 9.5.0
```
## How to Run
Take PACS as an example, you can run the code on Resnet50 with cmd:
```
sh PACStrain.sh
```
if you would like to run the code on RegNet, you can use the cmd:
```
sh regnet-PACS-train.sh
```
If you would like to know the detail of the parameter, pleas check the sh file we provided

## Citation
```
@article{GUO2024103979,
title = {Domain-aware triplet loss in domain generalization},
journal = {Computer Vision and Image Understanding},
pages = {103979},
year = {2024},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2024.103979},
author = {Kaiyu Guo and Brian C. Lovell}
```