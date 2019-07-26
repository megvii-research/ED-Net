from __future__ import division

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride, C, d, layer_idx=0):

        super(Bottleneck, self).__init__()

        conv1_in = in_channels
        conv1_out = out_channels
        conv2_in = conv1_out
        conv2_out = conv2_in
        conv3_in = conv2_out
        conv3_out = out_channels*2

        encoder_in = out_channels*2
        encoder_out = out_channels
        decoder_in = out_channels
        decoder_out = out_channels*2

        self.conv_conv1 = nn.Conv2d(conv1_in, conv1_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(conv1_out)
        self.conv_conv2 = nn.Conv2d(conv2_in, conv2_out, kernel_size=3, stride=stride, padding=1, bias=False, groups=C)
        self.bn_bn2 = nn.BatchNorm2d(conv2_out)
        self.conv_conv3 = nn.Conv2d(conv3_in, conv3_out, kernel_size=1, stride=1 ,padding=0, bias=False)
        self.bn_bn3 = nn.BatchNorm2d(conv3_out)

        if out_channels==128:
            self.GAPool = nn.AvgPool2d(56, stride=1)
        elif out_channels==256:
            self.GAPool = nn.AvgPool2d(28, stride=1)
        elif out_channels==512:
            self.GAPool = nn.AvgPool2d(14, stride=1)
        elif out_channels==1024:
            self.GAPool = nn.AvgPool2d(7, stride=1)
        else:
            print('GAPool Error\n')
            assert 1==0
        self.fc_reduction = nn.Linear(in_features=conv3_out, out_features=conv3_out//16)
        self.fc_extention = nn.Linear(in_features=conv3_out//16, out_features=conv3_out)
        self.sigmoid = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if (in_channels != out_channels * 2) or stride != 1:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels*2, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels*2))

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_conv',
                                nn.Conv2d(encoder_in, encoder_out, kernel_size=3, stride=2, padding=1,groups=C, bias=False))
        self.encoder.add_module('encoder_bn',
                                nn.BatchNorm2d(encoder_out))

        self.decoder = nn.Sequential()
        if layer_idx != 4:
            self.decoder.add_module('decoder_conv',
                                    nn.ConvTranspose2d(decoder_in,decoder_out,kernel_size=3,stride=2,padding=1,output_padding=1,groups=C,bias=False))
            self.decoder.add_module('decoder_bn',
                                    nn.BatchNorm2d(decoder_out))
        else:
            self.decoder.add_module('decoder_conv',
                                      nn.ConvTranspose2d(decoder_in,decoder_out,kernel_size=3,stride=2,padding=1,output_padding=0,groups=C,bias=False))
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

        se_out = self.GAPool(res)
        se_out = se_out.view(se_out.size(0), -1)
        se_out = F.relu(self.fc_reduction(se_out), inplace=True)
        se_out = self.fc_extention(se_out)
        se_out = self.sigmoid(se_out)
        se_out = se_out.view(se_out.size(0), se_out.size(1), 1, 1)
        res = se_out*res

        shtcut = self.shortcut.forward(x)

        return F.relu(res + shtcut + decode, inplace=True)

class ResNeXt_SEED(nn.Module):
    def __init__(self, block, C, d, layers, num_classes = 1000):
        self.inplanes = 64
        super(ResNeXt_SEED, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer('layer1', block, planes=C*d*1, blocks=layers[0], stride=1, C=C, d=d, layer_idx=1)
        self.layer2 = self._make_layer('layer2', block, planes=C*d*2, blocks=layers[1], stride=2, C=C, d=d, layer_idx=2)
        self.layer3 = self._make_layer('layer3', block, planes=C*d*4, blocks=layers[2], stride=2, C=C, d=d, layer_idx=3)
        self.layer4 = self._make_layer('layer4', block, planes=C*d*8, blocks=layers[3], stride=2, C=C, d=d, layer_idx=4)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear( (C*d*8) * block.expansion, num_classes)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, name, block, planes, blocks, stride=1, C=32, d=4, layer_idx=0):#out_channel = planes

        layers = nn.Sequential()
        for block_idx in range(blocks):
            name_ = '%s_block_%d' % (name, block_idx)
            if block_idx == 0:
                layers.add_module(name_, block(self.inplanes, planes, stride, C, d, layer_idx=layer_idx))
                self.inplanes = planes * block.expansion
            else:
                layers.add_module(name_, block(self.inplanes, planes, 1, C, d, layer_idx=layer_idx))
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


def se_resnext50_ed():
    return ResNeXt_SEED(block=Bottleneck,C=32,d=4,layers=[3,4,6,3])

def se_resnext101_ed():
    return ResNeXt_SEED(block=Bottleneck,C=32,d=4,layers=[3,4,23,3])

def se_resnext152_ed():
    return ResNeXt_SEED(block=Bottleneck,C=32,d=4,layers=[3,8,36,3])