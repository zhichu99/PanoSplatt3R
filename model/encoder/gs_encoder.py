from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from model.dust3r.heads import head_factory
from dataset.shims.normalize_shim import apply_normalize_shim
from dataset.types import BatchedExample, DataShim
from model.types import Gaussians
from model.dust3r.model import AsymmetricCroCo3DStereo
from model.encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from model.encoder.encoder import Encoder

inf = float('inf')

@dataclass
class BackboneCrocoCfg:
    name: Literal["croco", "croco_multi"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi"]
    d_feature: int
    backbone: BackboneCrocoCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: UnifiedGaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.backbone = AsymmetricCroCo3DStereo(cfg.croco, cfg.circular_pad)
        self.hooks = 0
        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter, cfg.twodgs)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity, 3 for delta means3D

        self.gs_params_head_type = cfg.gs_params_head_type
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim, circular_pad=self.cfg.circular_pad)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim, circular_pad=self.cfg.circular_pad)  # for view2 3DGS

        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim, circular_pad=self.cfg.circular_pad)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim, circular_pad=self.cfg.circular_pad)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int = 1,
        visualization_dump: Optional[dict] = None,
        mode: Optional[str] = 'train',
    ) -> Gaussians:
        device = context["pano_image"].device
        b, v, _, h, w = context["pano_image"].shape

        # Encode the context images.
        res1, res2, dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True, mode=mode)
        with torch.cuda.amp.autocast(enabled=False):
            # for the 3DGS heads
            if self.gs_params_head_type == 'linear':
                GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
            elif self.gs_params_head_type == 'dpt':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            elif self.gs_params_head_type == 'dpt_gs':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], view1['img'][:, :3], shape1[0].cpu().tolist())
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")

        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        gaussians = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)

        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        # Convert the features and depths into Gaussians.
        gaussians_out = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            # depths,
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
        )

        return Gaussians(
            rearrange(
                gaussians_out.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians_out.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians_out.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians_out.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                gaussians_out.scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians_out.rotations,
                "b v r srf spp xyzw -> b (v r srf spp) xyzw",
            ),
        )
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            )

            return batch

        return data_shim
