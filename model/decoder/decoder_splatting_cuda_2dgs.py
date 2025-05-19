from dataclasses import dataclass
from typing import Literal, Union

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from model.types import Gaussians
from .cuda_splatting_2dgs import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        # intrinsics: Union[Float[Tensor, "batch view 3 3"], None] = None,
        render_mode: str = "pinhole",
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            # rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
            repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            render_mode=render_mode,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        # depth = rearrange(depth, "(b v) h w -> b v 1 h w", b=b, v=v)

        return DecoderOutput(
            color,
            rearrange(self.render_depth(
                gaussians, extrinsics,  near, far, image_shape
            ), "(b v) h w -> b v 1 h w", b=b, v=v),
        )
    
    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        device = extrinsics.device

        depths = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
            repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        )            
        return depths
    