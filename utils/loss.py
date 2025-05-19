import torch
from einops import rearrange
import torch.nn as nn
from jaxtyping import Float
from lpips import LPIPS
from functools import cache
from torch import Tensor

def l1_loss(x, y, weight=1.0, mask=None):
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
    if len(x.shape) > 4 and not mask is None:
        mask = rearrange(mask, "b t h w -> b t 1 h w")
    if not mask is None:
        diff = torch.abs(x-y).mean(1, keepdim=True) * weight
        return (diff * mask).mean()
    return torch.abs((x - y) * weight).mean()

def diff_l1_loss(x, y, mask=None):
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"

    diff = torch.abs(x-y)
    diff = torch.where(diff < 1, 0.5 * diff**2, diff - 0.5)
    return diff.mean()

def l2_loss(x, y, weight=1.0, mask=None):
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
    if len(x.shape) > 4 and not mask is None:
        mask = rearrange(mask, "b t h w -> b t 1 h w")
    if not mask is None:
        diff = torch.square(x-y).mean(1, keepdim=True) * weight
        return (diff * mask).mean()
    return (torch.square(x-y) * weight).mean()

def lpips_loss(x, y, lpips_model, mask=None):
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
    if len(x.shape) > 4:
        x = rearrange(x, "b t c h w -> (b t) c h w")
        y = rearrange(y, "b t c h w -> (b t) c h w")
    if not mask is None:
        diff = lpips_model.forward(x, y, normalize=True).mean(1, keepdim=True)
        return (diff * mask).mean()
    return lpips_model.forward(x, y, normalize=True).mean()


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)