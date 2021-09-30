import paddle
from paddle import nn
from paddle.nn import functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2D(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Layer):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.Sequential(*modules)

        self.project = nn.Sequential(
            nn.Conv2D(len(self.convs) * out_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = paddle.concat(res, axis=1)
        return self.project(res)
