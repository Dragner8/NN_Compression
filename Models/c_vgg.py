
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.utils import ReLU

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}




class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):

        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class MLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MLinear, self).__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.mask_flag=False


        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data * mask_var.data
        self.mask_flag = True

    def get_mask(self):
        # print(self.mask_flag)
        return self.mask

    def forward(self, input):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return LinearFunction.apply(input, weight, self.bias)

        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )





class VGG(nn.Module):

    def __init__(self, features, num_class):
        super().__init__()

        lin_num=768
        if num_class == 10:
            lin_num = 512
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(len(features[-3].weight), lin_num),
            ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(lin_num, lin_num),
            ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(lin_num, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)




def compress_vgg19_bn(cnfg=None,num_class=100):
    if not cnfg :
        cnfg=cfg['E']

    return VGG(make_layers(cnfg, batch_norm=True),num_class)