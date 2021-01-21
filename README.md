# Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation

This repository contains:
1. the implementation of navigation agents for our paper: [Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation](https://arxiv.org/abs/2007.00229);
2. a dataset for pretraining outdoor VLN task.


## Data
In this project, we use the Touchdown dataset and the StreetLearn dataset. More details regarding these two datasets can be found [here](https://arxiv.org/abs/2001.03671).
 
Our pre-training dataset is built upon StreetLearn.
The guiding instructions for the outdoor VLN task are provided in ```touchdown/datasets/```.

To download the panoramas, please refer to [Touchdown Dataset](https://github.com/lil-lab/touchdown#data) and
[StreetLearn Dataset](https://sites.google.com/view/learn-navigate-cities-nips18/dataset).

## Requirements & Setup

- Python 3.6
- PyTorch 1.7.0
- Texar

We conduct experiments on Ubuntu 18.04 and Titan RTX.

Please run the following lines to download the code and install Texar:
```bash
git clone https://github.com/VegB/VLN-Transformer/
cd VLN-Transformer/
pip install [--user] -e .  # install Texar
cd touchdown/
```

## Quick Start

### Train VLN agent from scratch
Training can be performed with the following command:
```bash
python main.py --dataset [DATASET] --img_feat_dir [IMG_DIR] --model [MODEL] --exp_name [EXP_NAME]
```
- ```DATASET``` is the dataset for outdoor navigation. This repo currently support the following three datasets:
    - ```touchdown``` is a dataset for outdoor VLN, the instructions are written by human annotators;
    - ```manh50``` is a subset of StreetLearn, the instructions are generated by Google Map API;
    - ```manh50_mask``` has the same trajectories as ```manh50```, but the instructions are style-modified (which is what we do in this paper).
- ```IMG_DIR``` contains the encoded panoramas for ```DATASET```. After you get access to the panoramas, please encode them accordingly.
Each file in this directory should be a numpy file ```[PANO_ID].npy``` that represent the panorama that has corresponding pano_id. 
The encoding process are described in [Touchdown paper](https://arxiv.org/pdf/1811.12354.pdf), Section D.1.
- ```MODEL``` is the navigation agent, may be ```rconcat``` for RCONCAT or ```vlntrans``` for VLN Transformer.

More parameters and usage are listed [here](https://github.com/VegB/VLN-Transformer/blob/master/touchdown/main.py#L13).

It should be noted here that ```vlntrans``` use BERT (bert-base-uncased) to encode the instruction and it takes a lot of space, 
which means you may need to adjust the batch size accordingly to fit the model into your GPU. 
In our experiments, we use 3 piece of Titan RTX and a batch size of 30. 
This is the command we use to pretrain VLN Transformer on our instruction-style-modified dataset:
```bash
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --dataset 'manh50_mask' --img_feat_dir '/data/manh50_features_mean/' --model 'vlntrans' --batch_size 30 --max_num_epochs 15 --exp_name 'pretrain_mask'
```

### Train VLN agent on top of pre-trained models
We can finetune the VLN agent on pre-trained models.
```bash
python main.py --dataset [DATASET] --img_feat_dir [IMG_DIR] --model [MODEL] --resume_from [PRETRAINED_MODEL] --resume [RESUME_OPTION]
```
- ```PRETRAINED_MODEL``` specified the pre-trained model;
- ```RESUME_OPTION``` specifies the checkpoint
    - ```latest```: the most recent ckpt;
    - ```TC_best```: the ckpt with the best TC score on dev set;
    - ```SPD_best```: the ckpt with the best SPD score on dev set.

### Evaluate outdoor VLN performance
We can evaluate the agent's navigation performance on the test set and dev set with the following command:
```bash
python main.py --test True --dataset [DATASET] --img_feat_dir [IMG_DIR] --model [MODEL] --resume_from [PRETRAINED_MODEL] --resume [RESUME_OPTION] --CLS [True/False] --DTW [True/False]
```

The pre-trained models for VLN Transformer, RCONCAT and GA can be downloaded 
from [here](https://github.com/VegB/VLN-Transformer/blob/master/touchdown/checkpoints/).
Please place them in ```checkpoints/```.

To reproduce the results in our paper, please use the following commands:
```bash
CUDA_VISIBLE_DEVICES="0" python main.py --test True --dataset 'touchdown' --img_feat_dir [IMG_DIR] --model 'rconcat' --resume_from [PRETRAINED_MODEL] --resume 'TC_best' --CLS True --DTW True
CUDA_VISIBLE_DEVICES="1" python main.py --test True --dataset 'touchdown' --img_feat_dir [IMG_DIR] --model 'ga' --resume_from [PRETRAINED_MODEL] --resume 'TC_best' --CLS True --DTW True
CUDA_VISIBLE_DEVICES="2" python main.py --test True --dataset 'touchdown' --img_feat_dir [IMG_DIR] --model 'vlntrans' --batch_size 30 --resume_from [PRETRAINED_MODEL] --resume 'TC_best' --CLS True --DTW True
```
- ```PRETRAINED_MODEL``` specified the pre-trained model
    - ```vanilla```: Navigation agent trained on ```touchdown``` dataset without pre-training on auxiliary datasets.
    - ```finetuned_manh50```: Pre-trained on ```manh50``` dataset, and finetuned on ```touchdown``` dataset.
    - ```finetuned_mask```: Pre-trained on ```manh50_mask``` dataset, and finetuned on ```touchdown``` dataset.

## Citing our work
```
@misc{zhu2020multimodal,
    title={Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation},
    author={Wanrong Zhu and Xin Wang and Tsu-Jui Fu and An Yan and Pradyumna Narayana and Kazoo Sone and Sugato Basu and William Yang Wang},
    year={2020},
    eprint={2007.00229},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgements

The code and data can't be built without [streetlearn](https://github.com/deepmind/streetlearn), 
[speaker_follower](https://github.com/ronghanghu/speaker_follower),
[touchdown](https://github.com/lil-lab/touchdown),
and [Texar](https://github.com/asyml/texar-pytorch).
We also thank [@Jiannan Xiang](https://github.com/szxiangjn) for his contribution in reproducing the Touchdown task.