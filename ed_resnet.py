from __future__ import division

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, layer_idx):

        super(Bottleneck, self).__init__()

        conv1_in = in_channels
        conv1_out = math.floor(out_channels)

        conv2_in = math.floor(out_channels)
        conv2_out = math.floor(out_channels)

        conv3_in = math.floor(out_channels)
        conv3_out = out_channels * 4

        encoder_in = out_channels * 4
        encoder_out = out_channels

        decoder_in = out_channels
        decoder_out = out_channels * 4

        C = 32

        self.conv_conv1 = nn.Conv2d(conv1_in, conv1_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(conv1_out)
        self.conv_conv2 = nn.Conv2d(conv2_in, conv2_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_bn2 = nn.BatchNorm2d(conv2_out)
        self.conv_conv3 = nn.Conv2d(conv3_in, conv3_out, kernel_size=1, stride=1 ,padding=0, bias=False)
        self.bn_bn3 = nn.BatchNorm2d(conv3_out)

        self.shortcut = nn.Sequential()
        if (in_channels != out_channels * 4) or stride != 1:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels*4))

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_conv',
                                nn.Conv2d(encoder_in, encoder_out, kernel_size=3, stride=2, padding=1, groups=C,
                                          bias=False))
        self.encoder.add_module('encoder_bn',
                                nn.BatchNorm2d(encoder_out))

        self.decoder = nn.Sequential()
        if layer_idx != 4:
            self.decoder.add_module('decoder_conv',
                                    nn.ConvTranspose2d(decoder_in, decoder_out, kernel_size=3, stride=2, padding=1,
                                                       output_padding=1, groups=C, bias=False))
            self.decoder.add_module('decoder_bn',
                                    nn.BatchNorm2d(decoder_out))
        else:
            self.decoder.add_module('decoder_conv',
                                    nn.ConvTranspose2d(decoder_in, decoder_out, kernel_size=3, stride=2, padding=1,
                                                       output_padding=0, groups=C, bias=False))
            self.decoder.add_module('decoder_bn',
                                    nn.BatchNorm2d(decoder_out))

    def forward(self, x):
        proj = self.shortcut.forward(x)

        encode = F.relu(self.encoder.forward(proj), inplace=True)
        decode = self.decoder.forward(encode)

        res = self.conv_conv1.forward(x)
        res = F.relu(self.bn_bn1.forward(res), inplace=True)

        res = self.conv_conv2.forward(res)
        res = F.relu(self.bn_bn2.forward(res), inplace=True)

        res = self.conv_conv3.forward(res)
        res = self.bn_bn3.forward(res)

        shtcut = self.shortcut.forward(x)
        return F.relu(res + shtcut + decode, inplace=True)

class ResNet_ED(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        self.inplanes = 64
        super(ResNet_ED, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer('layer1', block, 64,  layers[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer('layer2', block, 128, layers[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer('layer3', block, 256, layers[2], stride=2, layer_idx=3)
        self.layer4 = self._make_layer('layer4', block, 512, layers[3], stride=2, layer_idx=4)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, name, block, planes, blocks, stride, layer_idx):#out_channel = planes

        layers = nn.Sequential()
        for block_idx in range(blocks):
            name_ = '%s_block_%d' % (name, block_idx)
            if block_idx == 0:
                layers.add_module(name_, block(self.inplanes, planes, stride, layer_idx))
                self.inplanes = planes * block.expansion
            else:
                layers.add_module(name_, block(self.inplanes, planes, 1, layer_idx))
        return layers

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.relu(self.bn1.forward(x), inplace=True)
        x = self.maxpool(x)

        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50_ed():
    return ResNet_ED(Bottleneck,[3,4,6,3])

def resnet101_ed():
    return ResNet_ED(Bottleneck,[3,4,23,3])

def resnet152_ed():
    return ResNet_ED(Bottleneck,[3,8,36,3])