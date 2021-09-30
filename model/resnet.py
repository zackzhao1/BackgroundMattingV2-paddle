import paddle
from paddle import nn
from paddle.vision.models.resnet import BottleneckBlock


class ResNetEncoder(nn.Layer):
    # Paddle的自带的resnet无法设置空洞,故自己实现
    # replace_stride_with_dilation=[False, False, True(layer4)]
    def __init__(self, in_channels, variant='resnet101', block=BottleneckBlock):
        super(ResNetEncoder, self).__init__()
        layers = {
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
        }
        layers = layers[variant]
        self._norm_layer = nn.BatchNorm2D
        self.inplanes = 64
        self.dilation = 1

        # Replace first conv layer if in_channels doesn't match.
        self.conv1 = nn.Conv2D(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0


if __name__ == '__main__':
    data = paddle.rand([4, 6, 256, 200])
    model = ResNetEncoder(in_channels=6, variant='resnet50')
    x, *x_short = model(data)
    print(x.shape)