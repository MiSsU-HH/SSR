# SSR [Under Review]

## Overview

Official implementation code for "**A Spatial Semantic Reasoning Flow for Dense Vision-Language Inference** [Under Review]". 



## Dependencies

This repo is built on top of [CLIP](https://github.com/openai/CLIP) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). To run our model, please install the following packages with your Pytorch environment. We recommend using Pytorch==1.10.x for better compatibility to the following MMSeg version.

```
pip install openmim
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex yapf==0.40.1
```





## Datasets
We include the following dataset configurations in this repo: PASCAL VOC, PASCAL Context, Cityscapes, ADE20k, COCO-Stuff10k, and COCO-Stuff164k, with three more variant datasets VOC20, Context59 (i.e., PASCAL VOC and PASCAL Context without the background category), and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

**Remember to modify the dataset paths in the config files in** `config/cfg_DATASET.py`





## Run Our Method
Single-GPU running: 

```
python eval.py --config ./configs/cfg_DATASET.py --workdir YOUR_WORK_DIR
```

Multi-GPU running: 
```
bash ./dist_test.sh ./configs/cfg_DATASET.py
```



## Results

The results are based on the SCLIP base framework.

| Dataset               | mIoU |
| --------------------- | ---- |
| ADE20k                | 17.5 |
| Cityscapes            | 35.3 |
| COCO-Object           | 34.8 |
| COCO-Stuff164k        | 24.6 |
| PASCAL Context59      | 37.1 |
| PASCAL Context60      | 33.4 |
| PASCAL VOC (w/o. bg.) | 84.0 |
| PASCAL VOC (w. bg.)   | 61.5 |


