import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class MyResNet(nn.Module):

    def __init__(self, cfg, backbone_path=None):
        super().__init__()

        self.model = resnet50(pretrained=True)
        # self.out_channels = [256, 512, 512, 256, 256, 256] # resnet34
        self.out_channels = [1024, 512, 512, 256, 256, 256] # resnet50

        self.feature_extractor = nn.Sequential(*list(self.model.children())[:7])
        
        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)
        
        self._build_additional_features(self.out_channels)
    
    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    '''
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = self.feature_extractor
        output_features = []
        feature_output = x 
        for idx, feature in enumerate(out_features):
            feature_output = feature(feature_output)
            print(feature_output.shape)
            output_features.append(feature_output)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            feature_map_size = feature.shape[2]
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        return tuple(output_features)
        '''
    def forward(self, x):
        feature = self.feature_extractor(x)
        features = [feature]
        for i, extra_layer in enumerate(self.additional_blocks):
            feature = extra_layer(feature) 
            features.append(feature)
        return tuple(features)
