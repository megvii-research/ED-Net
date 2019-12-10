## A Lightweight Encoder-Decoder Path for Deep Residual Networks.
Xin Jin, Yanping Xie, Xiu-shen Wei*, Borui Zhao, Yongshun Zhang, Xiaoyang Tan

This repository is the official PyTorch implementation of paper `A Lightweight Encoder-Decoder Path for Deep Residual Networks`. The paper is under revision, and will be released soon.

## Requirements

    numpy
    
    torch-1.1.0
    
    torchvision-0.3.0
    

## Usage
    
    1.download your dataset by yourself, such as ImageNet-1k
    
    2.create a list for your dataset,such as 
        imagename label
        xxx.jpg 1
        xxx.jpg 3
        xxx.jpg 999
    
    3.python3 imagenet_train.py --test_data_path your_path --train_data_path  your_path -a ED50 --epochs 100 --schedule 30 -b 256 --lr 0.1

## visualization
![visualization](https://github.com/Megvii-Nanjing/ED-Net/blob/master/2.png)

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: weixs.gm@gmail.com

Xin Jin: x.jin@nuaa.edu.cn

Yanping Xie: nuaaxyp@nuaa.edu.cn
