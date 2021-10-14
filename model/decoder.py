#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Decoder(nn.Layer):
    """
    Decoder upsamples the image by combining the feature maps at all resolutions from the encoder.
    
    Input:
        x4: (B, C, H/16, W/16) feature map at 1/16 resolution.
        x3: (B, C, H/8, W/8) feature map at 1/8 resolution.
        x2: (B, C, H/4, W/4) feature map at 1/4 resolution.
        x1: (B, C, H/2, W/2) feature map at 1/2 resolution.
        x0: (B, C, H, W) feature map at full resolution.
        
    Output:
        x: (B, C, H, W) upsampled output at full resolution.
    """
    
    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2D(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels[1])
        self.conv2 = nn.Conv2D(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels[2])
        self.conv3 = nn.Conv2D(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(channels[3])
        self.conv4 = nn.Conv2D(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = paddle.concat([x, x3], axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = paddle.concat([x, x2], axis=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = paddle.concat([x, x1], axis=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = paddle.concat([x, x0], axis=1)
        x = self.conv4(x)
        return x
