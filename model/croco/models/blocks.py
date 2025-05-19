# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import torch
import torch.nn as nn 
import xformers
import os
import warnings

from itertools import repeat
import collections.abc
import loralib as lora
from einops import rearrange
import numpy as np
import torch.nn.functional as F


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class EffiAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., pano_rot=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 
        self.pano_rot = pano_rot

    def forward(self, x, xpos, h=None, w=None, mode='train'):
        B, N, C = x.shape
        q, k, v = [self.qkv(x).reshape(B, N, 3, C)[:,:,i] for i in range(3)]

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
               
        if self.rope is not None:
            q = self.rope(q, xpos, pano_rot=self.pano_rot, h=h, w=w)
            k = self.rope(k, xpos, pano_rot=self.pano_rot, h=h, w=w)
        
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        x = memory_efficient_attention(q, k, v, scale=self.scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., pano_rot=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 
        self.pano_rot = pano_rot

    def forward(self, x, xpos, h=None, w=None, mode='train'):
        B, N, C = x.shape
        
        q, k, v = [self.qkv(x).reshape(B, N, 3, C)[:,:,i] for i in range(3)]

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
               
        if self.rope is not None:
            q = self.rope(q, xpos, pano_rot=self.pano_rot, h=h, w=w)
            k = self.rope(k, xpos, pano_rot=self.pano_rot, h=h, w=w)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mode == 'train':
            attn = attn.softmax(dim=-1, dtype=torch.float16)
        else:
            attn = attn.softmax(dim=-1) ## Warning! dtype=torch.float16 for auto mixed precision
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None, pano_rot=True, effi_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if effi_attention:
            self.attn = EffiAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                pano_rot=pano_rot)
        else:
            self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                pano_rot=pano_rot)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def post_init(self):
        # inflate input conv block to attach plucker coordinates
        conv_block = self.proj_plucker
        conv_params = {
            k: getattr(conv_block, k)
            for k in [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
            ]
        }
        conv_params["in_channels"] += 6
        conv_params["device"] = conv_block.weight.device

        # copy original weights for input conv block
        inflated_proj_in = nn.Conv2d(**conv_params)
        inp_weight = conv_block.weight.data
        feat_shape = inp_weight.shape

        # intialize new weights for plucker coordinates as zeros
        feat_weight = torch.zeros(
            (feat_shape[0], 6, *feat_shape[2:]), device=inp_weight.device
        )

        # assemble new weights and bias
        inflated_proj_in.weight.data.copy_(
            torch.cat([inp_weight, feat_weight], dim=1)
        )
        inflated_proj_in.bias.data.copy_(conv_block.bias.data)
        self.proj_plucker = inflated_proj_in

    def forward(self, x, xpos, h, w, mode):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos, h, w, mode))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class EffiCrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., pano_rot=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        self.pano_rot = pano_rot
        
    def forward(self, query, key, value, qpos, kpos, h=None, w=None, mode='train'):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query)
        k = self.projk(key)
        v = self.projv(value)

        q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            q = self.rope(q, qpos, pano_rot=self.pano_rot, h=h, w=w)
            k = self.rope(k, kpos, pano_rot=self.pano_rot, h=h, w=w)
            
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        x = memory_efficient_attention(q, k, v, scale=self.scale)
        x = x.reshape([B, Nq, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., pano_rot=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        self.pano_rot = pano_rot
        
    def forward(self, query, key, value, qpos, kpos, h=None, w=None, mode='train'):
            
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query)
        k = self.projk(key)
        v = self.projv(value)

        q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            q = self.rope(q, qpos, pano_rot=self.pano_rot, h=h, w=w)
            k = self.rope(k, kpos, pano_rot=self.pano_rot, h=h, w=w)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mode == 'train':
            attn = attn.softmax(dim=-1, dtype=torch.float16) ## Warning! dtype=torch.float16 for auto mixed precision
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None, pano_rot=True, effi_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if effi_attention:
            self.attn = EffiAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                pano_rot=pano_rot)
            self.cross_attn = EffiCrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                pano_rot=pano_rot)
        else:
            self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                pano_rot=pano_rot)
            self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                            pano_rot=pano_rot)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, h, w, mode):

        x = x + self.drop_path(self.attn(self.norm1(x), xpos, h, w, mode))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos, h, w, mode))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y
        
        
# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)

            pixel_coords = torch.cartesian_prod(y, x)
            # y_coords = pixel_coords[:, 0]
            # x_coords = pixel_coords[:, 1]
            # longitude = 2 * torch.pi * (x_coords / w - 0.5)
            # latitude = torch.pi * (0.5 - y_coords / h)
            # lat_lon_pairs = torch.stack([latitude, longitude], dim=1).view(h, w, 2)

            self.cache_positions[h,w] = pixel_coords # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) 

