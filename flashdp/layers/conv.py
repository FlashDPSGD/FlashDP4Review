import torch
import torch.nn as nn
from torch.autograd.function import Function
from transformers.pytorch_utils import Conv1D
try:
    from flashdp import bmtm_clip
    FlashDPAvailable = True
except:
    FlashDPAvailable = False

from .utils import DPParameter as Parameter


class DPConv1D(Conv1D):
    def __init__(self, nf, nx, C: float = 1.0, clamp_value: float = 1.0):
        super().__init__(nf, nx)
        self.dp = True
        self.C = C
        self.clamp_value = clamp_value
        self.nf = nf
        self.weight = Parameter(torch.empty(nx, nf))
        self.bias = Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = ApplyDPConv1DFunc(self.C, self.clamp_value)(self.bias, x, self.weight)
        x = x.view(size_out)
        return x


def ApplyDPConv1DFunc(C, clamp_value):
    """
        Returns a function that computes the conv1d function with differential privacy.
    """
    class DPConv1DFunc(Function):
        @staticmethod
        def forward(ctx, bias, input, weight):
            ctx.save_for_backward(bias, input, weight)
            output = input @ weight + bias
            return output

        @staticmethod
        def backward(ctx, grad_output):
            bias, input, weight = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_bias = _bias_grad_clip(grad_output, C, clamp_value)
                grad_bias.add_(torch.normal(
                    mean=0,
                    std=1.0,
                    size=grad_bias.shape,
                    device=grad_bias.device,
                ))
            else:
                grad_bias = None

            if ctx.needs_input_grad[1]:
                grad_input = grad_output @ weight.t()
            else:
                grad_input = None

            if ctx.needs_input_grad[2]:
                if FlashDPAvailable:
                    grad_weight = _weight_flashdp(input, grad_output, C, clamp_value)
                grad_weight.add_(torch.normal(
                    mean=0,
                    std=1.0,
                    size=grad_weight.shape,
                    device=grad_weight.device,
                ))
            else:
                grad_weight = None

            return grad_bias, grad_input, grad_weight
    
    return DPConv1DFunc.apply


def _weight_flashdp(input: torch.Tensor, grad_output: torch.Tensor, C: float, clamp_value: float):
    clip_args = torch.Tensor([C, clamp_value]).to(input.device)
    clipped_grad_weight = bmtm_clip(input, grad_output, clip_args=clip_args)
    return clipped_grad_weight


# @torch.jit.script
def _bias_grad_clip(grad_output: torch.Tensor, C: float, clamp_value: float):
    grad_bias_flatten = grad_output.view(grad_output.shape[0], -1)
    grad_bias_norm = torch.norm(grad_bias_flatten, p=2, dim=-1, keepdim=True)
    clip_factor = torch.clamp(C / (grad_bias_norm + 1e-6), max=clamp_value)
    grad_bias_flatten.mul_(clip_factor)
    clipped_grad_bias = grad_bias_flatten.view(-1, grad_output.shape[-1]).sum(0)

    return clipped_grad_bias


def _bias_flashdp(grad_output: torch.Tensor, C: float, clamp_value: float):
    raise NotImplementedError

