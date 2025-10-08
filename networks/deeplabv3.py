import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
# from networks.decoder_old import build_decoder
from networks.decoder import build_decoder
from networks.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


        self.proj = nn.Sequential(
                nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
        )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input,use_corr=False):
        x, low_level_feat = self.backbone(input)

        c4 = x
        x = self.aspp(x)
        x, features = self.decoder(x, low_level_feat)
        # x1, x2, features = self.decoder(x, low_level_feat)

        # x = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        dict_return = {}
        dict_return['out'] = x

        return dict_return

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16,num_classes=5,sync_bn=True)
    model.eval()
    input = torch.rand(7, 3, 256, 256)
    output = model(input,use_corr=True)
    print(output['corr_out'].size())


