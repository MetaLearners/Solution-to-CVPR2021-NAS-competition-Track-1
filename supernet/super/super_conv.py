import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..utils import pad
from ..initialize import kaiming_normal_

class BaseConv(nn.Layer):
    def __init__(self, indims, outdims, stride=1, down=False):
        super(BaseConv, self).__init__()
        self.indims = indims
        self.outdims = outdims
        self.max_in = max(self.indims)
        self.max_out = max(self.outdims)
        self.stride = stride
        self.down = down
        self.kernel_size = 1 if down else 3

    def _make_conv(self, indim, outdim):
        return nn.Conv2D(indim, outdim, self.kernel_size, self.stride, self.kernel_size // 2, bias_attr=False)

    def forward(self, x, indim, outdim):
        raise NotImplementedError()
    
    def inference(self, x, indim, outdim):
        return self(x, indim, outdim)

class IndependentConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cin in indims:
            for cout in outdims:
                setattr(self, f'conv-{cin}-{cout}', self._make_conv(cin, cout))
    
    def forward(self, x, indim, outdim):
        x = x[:, :indim]
        return getattr(self, f'conv-{indim}-{outdim}')(x)
    
class FrontShareConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cin in indims:
            setattr(self, f'conv-{cin}', self._make_conv(cin, self.max_out))
    
    def forward(self, x, indim, outdim):
        x = x[:, :indim]
        return getattr(self, f'conv-{indim}')(x)

class EndShareConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cout in outdims:
            for cout in outdims:
                setattr(self, f'conv-{cout}', self._make_conv(self.max_in, cout))
    
    def forward(self, x, indim, outdim):
        return getattr(self, f'conv-{outdim}')(pad(x[:,:indim], self.max_in))

class FullConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        self.conv = nn.Conv2D(self.max_in, self.max_out, self.kernel_size, stride, self.kernel_size // 2, bias_attr=False)
 
    def forward(self, x, indim, outdim):
        return self.conv(pad(x[:,:indim], self.max_in))

BestConv = FullConv
