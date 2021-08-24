# ENet: https://arxiv.org/abs/1606.02147
# based off of https://github.com/bermanmaxim/Enet-PyTorch

from models.sem_seg.fcn3d import SemSegNet
import torch
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func, self.forward_prepare(input))


class Padding(nn.Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at
    # index [index] in that dimension. If pad<0, index counts from the left.
    # If pad>0 index counts from the right.
    # When nInputDim is provided, inputs larger than that value will be considered batches
    # where the actual dim to be padded will be dimension dim + 1.
    def __init__(self, dim, pad, value, index, nInputDim):
        super(Padding, self).__init__()
        self.value = value
        # self.index = index
        self.dim = dim
        self.pad = pad
        self.nInputDim = nInputDim
        if index != 0:
            raise NotImplementedError("Padding: index != 0 not implemented")

    def forward(self, input):
        dim = self.dim
        if self.nInputDim != 0:
            dim += input.dim() - self.nInputDim
        pad_size = list(input.size())
        pad_size[dim] = self.pad
        padder = Variable(input.data.new(*pad_size).fill_(self.value))

        if self.pad < 0:
            padded = torch.cat((padder, input), dim)
        else:
            padded = torch.cat((input, padder), dim)
        return padded


class Dropout(nn.Dropout):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """
    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """
    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout2d, self).forward(input)


class StatefulMaxPool2d(nn.MaxPool2d): # object keeps indices and input sizes

    def __init__(self, *args, **kwargs):
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, x):
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = x.size()
        if return_indices:
            return output, indices
        return output


class StatefulMaxUnpool2d(nn.Module):
    def __init__(self, pooling):
        super(StatefulMaxUnpool2d, self).__init__()
        self.pooling = pooling
        self.unpooling = nn.MaxUnpool2d(pooling.kernel_size, pooling.stride, pooling.padding)

    def forward(self, x):
        return self.unpooling.forward(x, self.pooling.indices, self.pooling.input_size)

class ENet(SemSegNet):
    '''
    Taken from 3DMV on Github
    https://github.com/angeladai/3DMV
    '''
    def __init__(self, num_classes, cfg=None):
        super().__init__(num_classes, cfg)

        pooling_0 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
        pooling_1 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
        pooling_2 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
        
        # kernel, stride, padding, dilation, groups
        self.enet = nn.Sequential( # Sequential, 
            LambdaMap(lambda x: x, # ConcatTable, 
                nn.Conv2d(3, 13, (3, 3), (2, 2), (1, 1), (1, 1), 1), 
                pooling_0, 
            ), 
            LambdaReduce(lambda x, y: torch.cat((x, y), 1)), 
            nn.BatchNorm2d(16, 0.001, 0.1, True), 
            nn.PReLU(16), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(16, 16, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(64, 0.001, 0.1, True), 
                        Dropout2d(0.01), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                        pooling_1, 
                        Padding(0, 48, 0, 0, 3), 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(64), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(64, 0.001, 0.1, True), 
                        Dropout2d(0.01), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(64), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(64, 0.001, 0.1, True), 
                        Dropout2d(0.01), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(64), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(64, 0.001, 0.1, True), 
                        Dropout2d(0.01), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(64), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(16, 0.001, 0.1, True), 
                        nn.PReLU(16), 
                        nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(64, 0.001, 0.1, True), 
                        Dropout2d(0.01), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(64), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(64, 32, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                        pooling_2, 
                        Padding(0, 64, 0, 0, 3), 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False), 
                        nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False), 
                        nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False), 
                        nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False), 
                        nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ), 
            nn.Sequential( # Sequential, 
                LambdaMap(lambda x: x, # ConcatTable, 
                    nn.Sequential( # Sequential, 
                        nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1), 
                        nn.BatchNorm2d(32, 0.001, 0.1, True), 
                        nn.PReLU(32), 
                        nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False), 
                        nn.BatchNorm2d(128, 0.001, 0.1, True), 
                        Dropout2d(0.1), 
                    ), 
                    nn.Sequential( # Sequential, 
                        Lambda(lambda x: x), # Identity, 
                    ), 
                ), 
                LambdaReduce(lambda x,y: x+y), # CAddTable, 
                nn.PReLU(128), 
            ),
            nn.Sequential(
                nn.Conv2d(128, num_classes, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False)
            )
        )

    def forward(self, x):
        '''
        x: (batch_size, h, w, 3)
        '''
        return self.enet(x)

class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.

    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet2(SemSegNet):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, num_classes, cfg=None, encoder_relu=False, decoder_relu=False):
        super().__init__(num_classes, cfg)

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self, x, return_features=False):
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        if return_features:
            return x

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)

        return x        