"""
A supernet
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..initialize import kaiming_normal_

DEFAULT_SPACE = [[ 4, 8, 12, 16]] * 7 +\
            [[ 4, 8, 12, 16, 20, 24, 28, 32]] * 6 +\
            [[ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]]*6

class Supernet(nn.Layer):
    def __init__(self, superbn, superconv, superfc, space=DEFAULT_SPACE):
        super().__init__()
        self.bn_class = superbn
        self.conv_class = superconv
        self.fc_class = superfc
        self.space = space
        self.max = [max(x) for x in self.space]
        self.sizes = [len(x) for x in self.space]
        self._build_layers()

        # initialize
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, op='conv', mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2D):
                paddle.assign(paddle.ones(m.weight.shape), m.weight)
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_normal_(m.weight, op='linear', mode='fan_out', nonlinearity='relu')
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)

    def _build_layers(self):
        bn, conv, fc = self.bn_class, self.conv_class, self.fc_class
        # stem conv
        self.stem_conv = conv([3], self.space[0])
        self.stem_bn = bn([3], self.space[0])
        for i in range(18):
            # every layer
            stride = 2 if i in [6, 12] else 1
            setattr(self, f'conv-{i}', conv(self.space[i], self.space[i + 1], stride=stride))
            setattr(self, f'bn-{i}', bn(self.space[i], self.space[i + 1]))
            # for downsample
            if i % 2 == 0:
                setattr(self, f'conv-{i}-down', conv(self.space[i], self.space[i + 2], stride=stride, down=True))
                setattr(self, f'bn-{i}-down', bn(self.space[i], self.space[i + 2], down=True))
        # final fc
        self.fc = fc(self.space[18], 100)

    def forward(self, x, arch, sc=True):
        # stem
        x = self.stem_conv(x, 3, arch[0])
        x = self.stem_bn(x, 3, arch[0])
        x = F.relu(x)
        for block in range(3):
            for layer in [0, 2, 4]:
                base = block * 6 + layer
                r = x
                # first conv
                x = getattr(self, f'conv-{base}')(x, arch[base], arch[base + 1])
                x = getattr(self, f'bn-{base}')(x, arch[base], arch[base + 1])
                x = F.relu(x)
                # second conv
                x = getattr(self, f'conv-{base + 1}')(x, arch[base + 1], arch[base + 2])
                x = getattr(self, f'bn-{base + 1}')(x, arch[base + 1], arch[base + 2])
                if sc or layer == 0 or arch[base] != arch[base + 2]:
                    # possible skip connect
                    r = getattr(self, f'conv-{base}-down')(r, arch[base], arch[base + 2])
                    r = getattr(self, f'bn-{base}-down')(r, arch[base], arch[base + 2])
                x = x + r
                x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).flatten(1)
        return self.fc(x, arch[18])

    def inference(self, x, arch, sc=True):
        x = self.stem_conv.inference(x, 3, arch[0])
        x = self.stem_bn.inference(x, 3, arch[0])
        x = F.relu(x)
        for block in range(3):
            for layer in [0, 2, 4]:
                base = block * 6 + layer
                r = x
                # first conv
                x = getattr(self, f'conv-{base}').inference(x, arch[base], arch[base + 1])
                x = getattr(self, f'bn-{base}').inference(x, arch[base], arch[base + 1])
                x = F.relu(x)
                # second conv
                x = getattr(self, f'conv-{base + 1}').inference(x, arch[base + 1], arch[base + 2])
                x = getattr(self, f'bn-{base + 1}').inference(x, arch[base + 1], arch[base + 2])
                if sc or layer == 0 or arch[base] != arch[base + 2]:
                    # possible skip connect
                    r = getattr(self, f'conv-{base}-down').inference(r, arch[base], arch[base + 2])
                    r = getattr(self, f'bn-{base}-down').inference(r, arch[base], arch[base + 2])
                x = x + r
                x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).flatten(1)
        return self.fc.inference(x, arch[18])
