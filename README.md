# BackgroundMattingV2

---

English | [简体中文](./README_cn.md)

   * [BackgroundMattingV2](#BackgroundMattingV2)
      * [1 Introduction](#1 Introduction)
      * [2 Accuracy](#2 Accuracy)
      * [3 Dataset](#3 Dataset)
      * [4 Environment](#4 Environment)
      * [5 Quick start](#5 Quick start)
         * [step1: clone](#step1-clone)
         * [step2: train](#step2-train)
         * [step3: test](#step3-test)
      * [6 Code structure](#6 Code structure)
         * [6.1 structure](#61-structure)
         * [6.2 Parameter description](#62-Parameter description)
         * [6.3 Training process](#63-Training process)
      * [7 Model information](#7 Model information)

## 1 Introduction

This project reproduces BackgroundMattingV2 based on paddlepaddle framework. BackgroundMattingV2 is divided into two parts: the base and the refine part. The base part generates a rough result output with a low resolution input and is used to provide a coarse regional location.Based on this, the refine network selects a fixed number of PATHS (these areas tend to select hair/hands and other difficult-to-distinguish areas) through path selection for refine. After that, the updated results of path are filled back to the original results to obtain their matting results in high resolution.
![image](./image/4.jpg)

**Paper:**
- [1] Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, and Ira Kemelmacher Shlizerman.
  Real-time high-resolution background matting. 
  In Computer Vision and Pattern Regognition (CVPR), 2021.

**Reference project：**
- [https://github.com/PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)

**The link of aistudio：**
- notebook：[https://aistudio.baidu.com/aistudio/projectdetail/2467759](https://aistudio.baidu.com/aistudio/projectdetail/2467759)

## 2 Accuracy
Accuracy：SAD: 7.58，MSE: 9.49 
>This index is tested in the test set of PhotoMatte85

| |epoch|opt|learning_rate|pretrain|dataset|SAD|MSE|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|stage1|1|Adam|1e-4|none|VideoMatte240K|11.68|12.85|
|stage2|300|Adam|5e-5|stage1.model(step_109999)|Distinctions646_person|7.58|9.49|
|stage3|300|Adam|3e-5|stage2.model(epoch_169)|private|7.61|9.47|

**Model Download**
Address： https://pan.baidu.com/s/1WfpzLcjaDJPXYSrzPWvsyQ 
code：nsfy

## 3 Dataset

[VideoMatte240K & PhotoMatte85 dataset](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)

- Dataset size：
  - train：237,989
  - val：2,720
  - test：85


[Distinctions646_person dataset](https://github.com/cs-chan/Total-Text-Dataset)

- Dataset size：
  - train：362
  - val：11


## 4 Environment

- Hardware: GPU, CPU

- Framework:
  - PaddlePaddle >= 2.1.2

## 5 Quick start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd BackgroundMattingV2
export PYTHONPATH=./
```

### step2: train
```bash
sh ./run.sh
```

Because it is a target segmentation task, we need to pay attention to the gradual decrease of ``loss`` and the gradual decrease of ``SAD``、``MSE``.

### step3: test
```bash
python3 eval.py 
```
According to the test set designed in the original paper, the data will be randomly augmented, so the results will fluctuate.

### Prediction using pre training model

```bash
python3 predict.py
```
save the output image in the ./image

## 6 Code structure

### 6.1 structure

```
├─dataset                        
├─image                        
├─log                          
├─model                       
├─utils                          
│  eval.py                   
│  predict.py                
│  README.md                   
│  README_cn.md                 
│  run.sh                      
│  train.py                             
```

### 6.2 Parameter description

Parameters related to training and evaluation can be set in `train.py`, as follows:

|  Parameters   |  description |
|  ----  | ----  |
| --dataset-name| Name of datasets |
| --learning-rate|Learning rate|
| --log-train-loss-interval | Print the step of loss |
| --epoch_end| Num of epoch |
| --pretrain| Parameter path of pre training model |

### 6.3 Training process

#### Single machine training
``` 
sh ./run.sh
``` 

## 7 Model information

For other information about the model, please refer to the following table:

| information | description |
| --- | --- |
| Author | Jialei Zhao|
| Date | 2021.10 |
| Framework version | Paddle 2.1.2 |
| Application scenarios | High resolution matting |
| Support hardware | GPU、CPU |
| Download link | [Pre training model](https://pan.baidu.com/s/140EWboy_Z3xrQ1TlvEQOgQ)  code：6fnd   |
| Online operation | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/2467759)|
