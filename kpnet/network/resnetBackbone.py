import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
import attr


@attr.s
class ResnetConfig(object):
    image_channels = 3
    num_layers = 34
    num_deconv_layers = 3
    num_deconv_filters = 256
    num_deconv_kernel = 4
    final_conv_kernel = 1
    # Need to modifiy the num_keypoints to match the required value
    num_keypoints = -1

    # For 2d keypoint, this value should be one
    # For 2d keypoint + depth, this value should be 2
    # For 3d volumetric keypoint, this value should be the depth resolution
    depth_per_keypoint = 1

# The specification of resnet
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101')}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def initialize_backbone_from_modelzoo(
        backbone,  # type: ResNetBackbone,
        resnet_num_layers,  # type: int
        image_channels,  # type: int
    ):
    assert image_channels == 3 or image_channels == 4
    _, _, _, name = resnet_spec[resnet_num_layers]
    org_resnet = torch.utils.model_zoo.load_url(model_urls[name])
    # Drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    # Load the backbone
    if image_channels is 3:
        backbone.load_state_dict(org_resnet)
    elif image_channels is 4:
        # Modify the first conv
        conv1_weight_old = org_resnet['conv1.weight']
        conv1_weight = torch.zeros((64, 4, 7, 7))
        conv1_weight[:, 0:3, :, :] = conv1_weight_old
        avg_weight = conv1_weight_old.mean(dim=1, keepdim=False)
        conv1_weight[:, 3, :, :] = avg_weight
        org_resnet['conv1.weight'] = conv1_weight
        # Load it
        backbone.load_state_dict(org_resnet)


def init_from_modelzoo(
        network,  # type: ResnetNoStage,
        config,  # type: ResnetNoStageConfig
    ):
    initialize_backbone_from_modelzoo(
        network.backbone_net,
        config.num_layers,
        config.image_channels)
