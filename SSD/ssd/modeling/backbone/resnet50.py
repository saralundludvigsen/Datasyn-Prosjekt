import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, Bottleneck
#from torchvision.models import resnet

class ResNet50(nn.Module):

    def __init__(self, cfg, backbone_path=None):
        super().__init__()

        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.backbone = resnet50(pretrained=True)

        self.modules = []

        self.bank1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=self.output_channels[0],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        self.bank2 = nn.Sequential(
                Bottleneck(
                    inplanes=1024, 
                    planes=128, 
                    stride=2, 
                    downsample=nn.Sequential(
                        nn.Conv2d(
                            in_channels=1024,
                            out_channels=512,
                            kernel_size=1, 
                            stride=2,
                            padding=0, 
                            bias=False),
                        nn.BatchNorm2d(512)
                    ), 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                Bottleneck(
                    inplanes=512, 
                    planes=128, 
                    stride=1, 
                    downsample=None, 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                )
                
                )
        #self.modules.append(self.bank1)
        #print(self.bank2)
        
        self.bank3 = nn.Sequential(
                Bottleneck(
                    inplanes=512, 
                    planes=128, 
                    stride=2, 
                    downsample=nn.Sequential(
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=512,
                            kernel_size=1, 
                            stride=2,
                            padding=0, 
                            bias=False),
                        nn.BatchNorm2d(512)
                    ), 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                Bottleneck(
                    inplanes=512, 
                    planes=128, 
                    stride=1, 
                    downsample=None, 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                )
                
                )
        #print(self.bank3)

        self.bank4 = nn.Sequential(
                Bottleneck(
                    inplanes=512, 
                    planes=64, 
                    stride=2, 
                    downsample=nn.Sequential(
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=256,
                            kernel_size=1, 
                            stride=2,
                            padding=0, 
                            bias=False),
                        nn.BatchNorm2d(256)
                    ), 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                Bottleneck(
                    inplanes=256, 
                    planes=64, 
                    stride=1, 
                    downsample=None, 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                )
                
                )
        #print(self.bank4)
        
        self.bank5 = nn.Sequential(
                Bottleneck(
                    inplanes=256, 
                    planes=64, 
                    stride=2, 
                    downsample=nn.Sequential(
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=1, 
                            stride=2,
                            padding=0, 
                            bias=False),
                        nn.BatchNorm2d(256)
                    ), 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                Bottleneck(
                    inplanes=256, 
                    planes=64, 
                    stride=1, 
                    downsample=None, 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                )
                
                )
        #print(self.bank5)

        self.bank6 = nn.Sequential(
                Bottleneck(
                    inplanes=256, 
                    planes=64, 
                    stride=2, 
                    downsample=nn.Sequential(
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=1, 
                            stride=2,
                            padding=0, 
                            bias=False),
                        nn.BatchNorm2d(256)
                    ), 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                Bottleneck(
                    inplanes=256, 
                    planes=64, 
                    stride=1, 
                    downsample=None, 
                    groups=1,
                    base_width=64, 
                    dilation=1, 
                    norm_layer=None
                ),
                nn.MaxPool2d(kernel_size=2,stride=2)
                
                )
        #print(self.bank5)
        

        self.feature_extractor = nn.Sequential(*self.modules)
        print(self.feature_extractor)

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
        #out_features = self.modules
        feature_output = self.bank1(x) # remember that each feature output from one bank needs to be passed over to the next bank
        self.modules = [self.bank2, self.bank3, self.bank4, self.bank5, self.bank6]
        out_features = [feature_output]
        for idx, feature in enumerate(self.modules):
            feature_output = feature(feature_output)
            out_features.append(feature_output)

        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = feature.shape[2]
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
