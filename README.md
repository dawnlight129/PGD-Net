## PGD-Net


## Introduction

```
```

## Install

Open the folder **PGD-Net** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## DataPreprocess

We follow the [BuildFormer](https://github.com/WangLibo1995/BuildFormer) to pre-process all the datasets.
And then follow the [DSAT-Net](https://github.com/stdcoutzrh/BuildingExtraction) to further process Massachusetts dataset.
More specifically, we use [split_1500_to_512.py](https://github.com/) to resize images from 1500x1500 to 512x512.

```
python PGD-Net/tools/split_1500_to_512.py
```

## Training

```
python train_supervision.py -c ./config/whu/PGDNet.py
```

```
python train_supervision.py -c ./config/inria/PGDNet.py
```

```
python train_supervision.py -c ./config/mass/PGDNet.py
```

## Testing

```
python building_seg_test.py -c ./config/whu/PGDNet.py -o /root/autodl-tmp/whu/result/PGDNet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/inria/PGDNet.py -o /root/autodl-tmp/inria/result/PGDNet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/mass/PGDNet.py -o /root/autodl-tmp/Massa_512/result/PGDNet --rgb -t 'lr'
```

## Citation

```
```

## Acknowledgement

- [BuildFormer](https://github.com/WangLibo1995/BuildFormer)
- [CLCFormer](https://github.com/long123524/CLCFormer)
- [DSAT-Net](https://github.com/stdcoutzrh/BuildingExtraction)
- [UNet](https://github.com/zhixuhao/unet)
- [GroupMamba](https://github.com/Amshaker/GroupMamba)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
