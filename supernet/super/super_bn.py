import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..utils import pad

class BaseBN(nn.Layer):
    def __init__(self, indims, outdims, down=False):
        super().__init__()
        self.indims = indims
        self.outdims = outdims
        self.down = down
        self.max = max(self.outdims)
    
    def forward(self, x, indim, outdim):
        raise NotImplementedError()
    
    def inference(self, x, indim, outdim):
        return self(x, indim, outdim)

class IndependentBN(BaseBN):
    def __init__(self, indims, outdims, down=False):
        super().__init__(indims, outdims, down=down)
        for cin in indims:
            for cout in outdims:
                setattr(self, f'bn-{cin}-{cout}', nn.BatchNorm2D(cout))
    def forward(self, x, indim, outdim):
        x = x[:, :outdim]
        return getattr(self, f'bn-{indim}-{outdim}')(x)

class FrontShareBN(BaseBN):
    def __init__(self, indims, outdims, down=False):
        super().__init__(indims, outdims, down)
        for cin in indims:
            setattr(self, f'bn-{cin}', nn.BatchNorm2D(self.max))
    
    def forward(self, x, indim, outdim):
        x = getattr(self, f'bn-{indim}')(pad(x, self.max))
        return pad(x[:,:outdim], self.max)

class EndShareBN(BaseBN):
    def __init__(self, indims, outdims, down=False):
        super(EndShareBN, self).__init__(indims, outdims, down)
        for cout in outdims:
            setattr(self, f'bn-{cout}', nn.BatchNorm2D(cout))
    
    def forward(self, x, indim, outdim):
        x = x[:, :outdim]
        return getattr(self, f'bn-{outdim}')(x)

class FullBN(BaseBN):
    def __init__(self, indims, outdims, down=False):
        super(FullBN, self).__init__(indims, outdims, down)
        self.bn = nn.BatchNorm2D(self.max)
    
    def forward(self, x, indim, outdim):
        x = self.bn(pad(x, self.max))
        x = pad(x[:, :outdim], self.max)
        return x
    
    def inference(self, x, indim, outdim):
        # directly use running mean and var as input
        x = F.batch_norm(x, self.bn._mean, self.bn._variance, self.bn.weight, self.bn.bias, training=False, use_global_stats=False)
        x = pad(x[:, :outdim], self.max)
        return x

BestBN = FullBN