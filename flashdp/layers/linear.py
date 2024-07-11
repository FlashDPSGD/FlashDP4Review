import torch
import torch.nn as nn
from torch.autograd.function import Function

try:
    from flashdp import bmtm_clip
    FlashDPAvailable = True
except:
    FlashDPAvailable = False
from .utils import DPParameter as Parameter


class DPLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, C: float = 1.0, clamp_value: float = 1.0):
        use_bias = False if (bias is None or bias==False) else True
        super(DPLinear, self).__init__(in_features, out_features, use_bias)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dp = True
        self.C = C
        self.clamp_value = clamp_value
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return ApplyDPLinearFunc(self.C, self.clamp_value)(input, self.weight, self.bias)


def ApplyDPLinearFunc(C, clamp_value):
    """
        Returns a function that computes the linear function with differential privacy.
    """
    class DPLinearFunc(Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            ctx.save_for_backward(input, weight, bias)
            output = input @ weight.t()
            if bias is not None:
                output += bias
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight
            else:
                grad_input = None

            if ctx.needs_input_grad[1]:
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

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = _bias_grad_clip(grad_output, C, clamp_value)
                grad_bias.add_(torch.normal(
                    mean=0,
                    std=1.0,
                    size=grad_bias.shape,
                    device=grad_bias.device,
                ))
            else:
                grad_bias = None
            
            # Check if the grad shape is correct
            if grad_input is not None and grad_input.shape != input.shape:
                raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
            if grad_weight is not None and grad_weight.shape != weight.shape:
                raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
            if grad_bias is not None and grad_bias.shape != bias.shape:
                raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")

            return grad_input, grad_weight, grad_bias
    
    return DPLinearFunc.apply


def _weight_flashdp(input: torch.Tensor, grad_output: torch.Tensor, C: float, clamp_value: float):
    clip_args = torch.Tensor([C, clamp_value]).to(input.device)
    clipped_grad_weight = bmtm_clip(grad_output, input, clip_args=clip_args)
    return clipped_grad_weight


def _bias_grad_clip(grad_output: torch.Tensor, C: float, clamp_value: float):
    grad_bias_flatten = grad_output.view(grad_output.shape[0], -1)
    grad_bias_norm = torch.norm(grad_bias_flatten, p=2, dim=-1, keepdim=True)
    clip_factor = torch.clamp(C / (grad_bias_norm + 1e-6), max=clamp_value)
    grad_bias_flatten.mul_(clip_factor)
    clipped_grad_bias = grad_bias_flatten.view(-1, grad_output.shape[-1]).sum(0)
    return clipped_grad_bias

def _bias_flashdp(grad_output: torch.Tensor, C: float, clamp_value: float):
    raise NotImplementedError

