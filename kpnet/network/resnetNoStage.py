import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
import attr
from kpnet.network.resnetBackbone import ResnetConfig, ResNetBackbone, resnet_spec
from kpnet.network.deconvHead import DeconvHead

class ResnetNoStage(nn.Module):
    def __init__(self, config=ResnetConfig()):
        super(ResnetNoStage, self).__init__()
        block_type, layers, channels, name = resnet_spec[config.num_layers]
        self.backbone_net = ResNetBackbone(block_type, layers, config.image_channels)
        self.head_net = DeconvHead(
            channels[-1],
            config.num_deconv_layers,
            config.num_deconv_filters,
            config.num_deconv_kernel,
            config.final_conv_kernel,
            config.num_keypoints,
            config.depth_per_keypoint)

    def forward(self, x):
        x = self.backbone_net(x)
        x = self.head_net(x)
        return x