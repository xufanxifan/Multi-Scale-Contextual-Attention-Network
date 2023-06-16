### [Paper](https://www.sciencedirect.com/science/article/pii/S0924271623001600)  <br>

Our paper [Semantic Segmentation of Urban Building Surface Materials using Multi-Scale Contextual Attention Network](https://www.sciencedirect.com/science/article/pii/S0924271623001600).<br>



## Installation 

* The code is tested with pytorch 1.3 and python 3.6


## Download Weights


* Download pretrained weights from [google drive](https://drive.google.com/drive/folders/1kDoNGVV1SqxJSKvur9vFeWAA9HpWc-yK?usp=drive_link) and put into `logsxf/`

## Download/Prepare Data

Currently the dataset in the paper is not available due to copyright reasons.

## Running the code

The instructions below use `python train.py <args ...>`.

## Train a model

Train hkscapes, using HRNet + multi-head attention + multi-scale OCR 
```bash
> python train.py --arch selfattocrnet.MscaleOCR_v2
```

The first time this command is run, a centroid file has to be built for the dataset. It'll take about 2 minutes. The centroid file is used during training to know how to sample from the dataset in a class-uniform way.


### Run inference and dump images on a folder of images

run:
```bash
> python train.py --arch selfattocrnet.MscaleOCR_v2  --resume logsxf(test26)/best_checkpoint_ep0.pth --eval val --dump_all_images
```



