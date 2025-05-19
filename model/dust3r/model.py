# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
from dataset.types import BatchedExample, DataShim

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape, make_batch_symmetric
from .heads import head_factory
from .patch_embed import get_patch_embed
from geometry.camera_emb import get_intrinsic_embedding
from dataset.shims.normalize_shim import apply_normalize_shim

from ..croco.models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True, return_args=False):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R").replace('AsymmetricMASt3R','AsymmetricCroCo3DStereo').replace('two_confs=True, ', '').replace(", desc_conf_mode=('exp', 0, inf)", "")

    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    if return_args:
        return net.to(device), ckpt['args']
    else:
        return net.to(device)

croco_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'img_size': (512, 512),
    'pano_rot': True,
    'effi_attention': True,
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}



class AsymmetricCroCo3DStereo (
    CroCoNet
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self, cfg, circular_pad=False):
        self.patch_embed_cls = cfg.patch_embed_cls
        self.cfg = cfg
        for k, v in croco_params.items():
            if k in cfg:
                croco_params[k] = cfg.get(k)

        self.croco_args = fill_default_args(croco_params, super().__init__)
        super().__init__(**croco_params)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.circular_pad = circular_pad
        # set landscape_only True
        self.set_downstream_head(default_dust3r_params['output_mode'], default_dust3r_params['head_type'], True, default_dust3r_params['depth_mode'], default_dust3r_params['conf_mode'], **croco_params)
        self.set_freeze('none')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_args=False, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu', return_args=return_args)
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return
    
    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)
    
    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), circular_pad=self.circular_pad)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), circular_pad=self.circular_pad)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


    def _encode_image(self, image, true_shape, mode):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos, self.token_h, self.token_w, mode)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, mode):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0), mode=mode)
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, mode=mode)
            out2, pos2, _ = self._encode_image(img2, true_shape2, mode=mode)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, force_asym=False, mode='train'):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, mode=mode)
        else:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2], mode=mode)
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, mode):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2, self.token_h, self.token_w, mode)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1, self.token_h, self.token_w, mode)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)
    
    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                mode='train',
                ):
        b, v, _, h, w = context["pano_image"].shape
        self.token_h = h // self.patch_embed.patch_size[0]
        self.token_w = w // self.patch_embed.patch_size[1]

        view1, view2 = ({'img': context["pano_image"][:, 0]},
                        {'img': context["pano_image"][:, 1]})

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = [0 for _ in range(b)], [1 for _ in range(b)]
            view1['instance'] = instance_list_view1
            view2['instance'] = instance_list_view2
            view1['idx'] = instance_list_view1
            view2['idx'] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=False, mode=mode)
        else:
        #     # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=True, mode=mode)

        # else:
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, mode=mode)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        if return_views:
            return res1, res2, dec1, dec2, shape1, shape2, view1, view2
        return res1, res2, dec1, dec2, shape1, shape2
    
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            )

            return batch

        return data_shim
