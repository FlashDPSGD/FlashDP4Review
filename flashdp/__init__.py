import torch.nn as nn
from torch.nn import Linear

# from .triton.mm_clip_loop import mm_clip_triton as mm_clip
from .triton.bmm_clip_loop import bmm_clip_triton as bmm_clip
from .triton.bmtm_clip_loop import bmtm_clip_triton as bmtm_clip

from .layers.linear import DPLinear
from .layers.conv import Conv1D, DPConv1D


def wrap_model(model, target_modules=[nn.Linear], C=1.0, clamp_value=1.0):

    dp_supported_modules = {
        nn.Linear: DPLinear,
        Conv1D: DPConv1D,
    }
    
    def get_init_args(child, target_module: nn.Module):
        if target_module == nn.Linear:
            return {
                "in_features": child.in_features,
                "out_features": child.out_features,
                "bias": child.bias,
            }
        elif target_module == Conv1D:
            return {
                "nf": child.nf,
                "nx": child.weight.shape[0],
            }
        
    supported_modules = dp_supported_modules.keys()
    if not all(m in supported_modules for m in target_modules):
        raise ValueError(f"target_modules must be among {supported_modules}, got {target_modules}")

    def replace_module(model: nn.Module, target_module: nn.Module, dp_module: nn.Module):
        def replace_module_recursive(model: nn.Module):
            for child_name, child in model.named_children():
                if isinstance(child, target_module):
                    child_init_args = get_init_args(child, target_module)
                    new_module = dp_module(**child_init_args, C=C, clamp_value=clamp_value)
                    child_device = next(child.parameters()).device if list(child.parameters()) else None
                    new_module = new_module.to(child_device)
                    new_module.load_state_dict(child.state_dict())
                    new_module.train(child.training)
                    setattr(model, child_name, new_module)
                else:
                    replace_module_recursive(child)
        replace_module_recursive(model)
    
    for target_module in target_modules:
        replace_module(model, target_module, dp_supported_modules[target_module])
    
    return model
