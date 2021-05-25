import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..utils import pad
from ..initialize import kaiming_normal_

class BaseFC(nn.Layer):
    def __init__(self, indims, outdim):
        super(BaseFC, self).__init__()
        self.indims = indims
        self.outdim = outdim
        self.max = max(self.indims)
    
    def forward(self, x, indim):
        raise NotImplementedError()
        
    def inference(self, x, indim):
        return self(x, indim)

class IndependentFC(BaseFC):
    def __init__(self, indims, outdim):
        super().__init__(indims, outdim)
        for cin in indims:
            setattr(self, f'fc-{cin}', nn.Linear(cin, self.outdim))
    
    def forward(self, x, indim):
        return getattr(self, f'fc-{indim}')(x[:,:indim])

class FullFC(BaseFC):
    def __init__(self, indims, outdim):
        super().__init__(indims, outdim)
        self.fc = nn.Linear(self.max, self.outdim)
        kaiming_normal_(self.fc.weight, op='linear', mode='fan_out', nonlinearity='relu')
        paddle.assign(paddle.zeros(self.fc.bias.shape), self.fc.bias)
    
    def forward(self, x, indim):
        x = pad(x[:, :indim], self.max)
        return self.fc(x)

BestFC = FullFC
