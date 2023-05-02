<div align="center">
  <h1>
   <a href="https://www.sciencedirect.com/science/article/pii/S0020025523005972">Hierarchical few-shot learning with feature fusion <br>
   driven by data and knowledge.</a>
  </h1>
</div>



## :heavy_check_mark: Requirements
* Ubuntu 20.04
* Python 3.8.5
* [CUDA 11.1](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.8.0](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name HFFDK --file environment.yml
conda activate HFFDK
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh
bash download_tieredimagenet.sh
```

## :pushpin: Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test HFFDK on the CIFAR-FS dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/cifar_fs_5w1s.sh
```
```
python test.py -dataset cifar_fs -datadir /home/data/cifar_fs -gpu 0 -extra_dir your_run_set -temperature_attn 5.0 
```


## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train HFFDK on the CIFAR-FS dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cifar_fs_5w1s.sh
```
```
python train.py -batch 64 -dataset cifar_fs -datadir /home/data/cifar_fs -gpu 0 -extra_dir your_run_set -temperature_attn 5.0 -lamb 0.5
```

## :scroll: Citing HFFDK
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@article{Wu2023Hierarchical,
title = {Hierarchical Few-Shot Learning with Feature Fusion Driven by Data and Knowledge},
author = {Zhiping Wu and Hong Zhao},
journal = {Information Sciences},
pages = {119012},
year = {2023},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2023.119012},
url = {https://www.sciencedirect.com/science/article/pii/S0020025523005972}
}
```
