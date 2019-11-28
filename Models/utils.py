import torch
import torch.nn as nn

class MyReLU(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        if False:
            from Quantizer.utils import linear_tensor_quant, fixed_point_tensor_quant
            input = input.cpu().numpy()
            input=fixed_point_tensor_quant(input,4)
            #input = linear_tensor_quant(input, 0.5, -0.5, 6)
            input=torch.from_numpy(input).cuda()
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class ReLU(nn.Module):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return MyReLU.apply(input)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
