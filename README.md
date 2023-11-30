# APPN: a Attention-based Pseudo-label Propagation Network for few-shot learning with noisy labels

The aim for this repository is to contain clean, readable and tested
code to produce APPN.

This project is written in python 3.6 and Pytorch and assumes you have
a GPU.

our model in the file model.py 
![APPN](https://github.com/Typistchen/APPN/blob/main/model_picture/zongtu.pdf)


![SETTING](https://github.com/Typistchen/APPN/blob/main/model_picture/jihe.pdf)

# Getting started

## CIFAR-FS

- Change directory to `./filelists/cifar`
- Download [CIFAR-FS](https://drive.google.com/file/d/1i4atwczSI9NormW5SynaHa1iVN1IaOcs/view)
- run `python make.py` in the terminal

## FC100

- Change directory to `./filelists/fc100`
- Download [FC100](https://drive.google.com/file/d/1jWbj03Fo0SXhd_egH52-rVSP9pUU0dBJ/view)
- run `python make.py` in the terminal

## miniImagenet

- Change directory to `./filelists/miniImagenet`
- Download [miniImagenet](https://drive.google.com/file/d/1hQqDL16HTWv9Jz15SwYh3qq1E4F72UDC/view)
- run `python make.py` in the terminal

## tieredImagenet

- Change directory to `./filelists/tieredImagenet`
- Download [tieredImagenet](https://drive.google.com/file/d/1ir7coqTzg_titf3nrH1brahG2PhuCnpJ/view)
- run `python make.py` in the terminal


# Running the scripts
To pre-train the contrastive network in the terminal, use:
```bash
$ python experiment.sh
```

