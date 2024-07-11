import torch
import triton.language as tl

def to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


supported_acc_dtypes = {
    torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
    torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
}

