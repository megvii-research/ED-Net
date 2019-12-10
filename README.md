## A Lightweight Encoder-Decoder Path for Deep Residual Networks.
Xin Jin, Yanping Xie, Xiu-Shen Wei*, Borui Zhao, Yongshun Zhang, Xiaoyang Tan

This repository is the official PyTorch implementation of paper "A Lightweight Encoder-Decoder Path for Deep Residual Networks". The paper is under revision, and will be released after acceptance.

## Introduction
We present a novel lightweight path for deep residual neural networks. The proposed method integrates a simple plug-and-play module, ie, a convolutional Encoder-Decoder (ED), as an augmented path to the original residual building block. Thanks to the abstract design and ability of the encoding stage, the decoder part tends to generate feature maps where highly semantically relevant responses are activated while irrelevant responses are restrained. By a simple element-wise addition operation, the learned representations derived from identity shortcut and original transformation branch are enhanced by our ED path. Furthermore, we exploit lightweight counterparts by removing a portion of channels in the original transformation branch. Fortunately, our lightweight processing will not cause an obvious performance drop, but bring computational economy. By conducting comprehensive experiments on ImageNet, MS-COCO, CUB200-2011 and CIFAR, we prove the consistent accuracy gain obtained by our ED path for various residual architectures, with comparable or even lower model complexity. Concretely, it decreases the top-1 error of ResNet-50 and ResNet-101 by 1.22\% and 0.91\%, respectively, on the task of ImageNet classification, and increases the mmAP of Faster R-CNN with ResNet-101 by 2.5\% on the MS-COCO object detection task.

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

## Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

## Key Results
**Comparisons to baselines on ImageNet-1K classification**

Model | Top-1 err. | Top5 err.
:- | :-: | :-: 
ResNet-50 | 24.34 | 7.32| 
ResNet-50 + ED | **23.12**| **6.54**|
ResNet-101 | 23.12 | 6.52 |
ResNet-101 + ED | **22.21** | **6.23**|
ResNet-152 | 22.44 | 6.37|
ResNet-152 + ED | **21.98** | **6.09**|
ResNeXt-50 | 22.59 | 6.41 |
ResNeXt-50 + ED | **22.01** | **6.11**|
ResNeXt-101 | 21.34 | 5.66|
ResNeXt-101 + ED| **20.93**| **5.32**|

**Results of efficient ED-Nets for ImageNet-1K classification**

Model | Top-1 err. | Top5 err. | GFLOPs
:- | :-: | :-: | :-: 
ResNet-50 | 24.34 | 7.32| 4.1 |
ED-ResNet-50-A | **23.08**| **6.47**| 4.0 |
ED-ResNet-50-B | 23.94| 6.95| **2.1**|
ResNet-101 | 23.12 | 6.52 | 7.9 |
ED-ResNet-101-A | **22.23**| **6.24**| 7.8 |
ED-ResNet-101-B | 23.14| 6.49| **3.9**|
ResNet-152 | 22.44 | 6.37| 11.7 |
ED-ResNet-152-A | **22.01**| **6.11**| 11.5|
ED-ResNet-152-B | 22.52| 6.41| **5.6**|
ResNeXt-50 | 22.59 | 6.41 | 4.2 |
ED-ResNeXt-50-A | **22.03**| **6.12**| 4.2 |
ED-ResNeXt-50-B | 22.61| 6.43| **2.9**|
ResNeXt-101 | 21.34 | 5.66| 8.0 |
ED-ResNeXt-101-A | **20.97**| **5.33**| 7.9 |
ED-ResNeXt-101-B | 21.57| 5.71| **5.4**|

Here `-A` means that we remove a portion of channels of the transform branches to make ED-Net have the same/comparable FLOPs as that of the baseline network. `-B` means that we remove half of the 3×3 convolution filters of the transformation branches to make ED-Net have much less FLOPs than that of the baseline network.

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: weixs.gm@gmail.com

Xin Jin: x.jin@nuaa.edu.cn

Yanping Xie: nuaaxyp@nuaa.edu.cn
