import torch
import triton
import triton.language as tl


@triton.jit
def tl_clamp(x, max):
    return tl.where(x < max, x, max)


@triton.jit
def get_clip_factor(
    norm, 
    clip_args_ptr
):
    return tl_clamp(tl.load(clip_args_ptr) / (norm + 1e-10), tl.load(clip_args_ptr+1))
