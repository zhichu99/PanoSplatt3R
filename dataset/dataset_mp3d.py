from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset
import numpy as np
import cv2

from .types import Stage

from einops import rearrange
import torch.nn.functional as F


# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img



@dataclass
class DatasetMP3DCfg():
    name: Literal["mp3d"]
    roots: list[Path]
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_datasets: list[dict]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = False


class DatasetMP3D(Dataset):
    cfg: DatasetMP3DCfg
    stage: Stage

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetMP3DCfg,
        stage: Stage,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        cubemap_Rs = torch.eye(4, dtype=torch.float32)
        self.cubemap_Rs = repeat(cubemap_Rs, "... -> f ...", f=6).clone()
        # 'F', 'R', 'B', 'L', 'U', 'D'
        self.cubemap_Rs[:, :3, :3] = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            ],
            dtype=torch.float32
        ).inverse()
        
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # scan folders in cfg.roots[0]
        if stage == "predict":
            stage = "test"

        height = cfg.image_shape[0]
        height = max(height, 512)
        resolution = (height * 2, height)
        resolution = 'x'.join(map(str, resolution))
        if stage == "test":
            self.roots = []
            for test_dataset in cfg.test_datasets:
                name = test_dataset["name"]
                dis = test_dataset["dis"]
                self.roots.append(
                    os.path.join(cfg.roots[0], f"png_render_{stage}_{resolution}_seq_len_3_{name}_dist_{dis}")
                )
        else:
            self.roots = [os.path.join(r, f"png_render_{stage}_{resolution}_seq_len_3_m3d_dist_0.5") for r in cfg.roots]

        data = []
        for root, test_dataset in zip(self.roots, cfg.test_datasets):
            if not os.path.exists(root):
                continue
            scenes = os.listdir(root)
            scenes.sort()
            for s in scenes:
                data.append({
                    'root': root,
                    'scene_id': s,
                    'name': test_dataset["name"],
                    'dis': test_dataset["dis"],
                    'baseline': test_dataset["dis"] * 2,
                })
        self.data = data

        self.e2c_mono = Equirec2Cube(512, 1024, 256)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        scene = data['scene_id']
        scene_path = os.path.join(data['root'], scene)
        views = os.listdir(scene_path)
        views.sort()

        # Load the images.
        rgbs_path = [os.path.join(scene_path, str(v), 'rgb.png') for v in views]
        context_indices = torch.tensor([0, 2])
        target_indices = torch.tensor([1])
        context_images = [rgbs_path[i] for i in context_indices]
        target_images = [rgbs_path[i] for i in target_indices]
        context_images = self.convert_images(context_images)
        target_images = self.convert_images(target_images)

        # Load the depth.
        if data['name'] == 'm3d':
            depths_path = [os.path.join(scene_path, str(v), 'depth.png') for v in views]
            context_depths = [depths_path[i] for i in context_indices]
            target_depths = [depths_path[i] for i in target_indices]
            context_depths = self.convert_images(context_depths)
            target_depths = self.convert_images(target_depths)
            context_depths = context_depths.float() / 1000.0
            target_depths = target_depths.float() / 1000.0
            context_depths = context_depths.clamp(min=0.)
            target_depths = target_depths.clamp(min=0.)
            context_mask = (context_depths > self.near) & (context_depths < self.far)
            target_mask = (target_depths > self.near) & (target_depths < self.far)

        # load camera
        trans_path = [os.path.join(scene_path, str(v), 'tran.txt') for v in views]
        rots_path = [os.path.join(scene_path, str(v), 'rot.txt') for v in views]
        trans = []
        rots = []
        for tran_path, rot_path in zip(trans_path, rots_path):
            trans.append(np.loadtxt(tran_path))
            rots.append(np.loadtxt(rot_path))
        trans = torch.tensor(trans)
        rots = torch.tensor(rots)
        extrinsics = self.convert_poses(trans, rots)

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics = repeat(intrinsics, "h w -> b h w", b=len(extrinsics)).clone()

        # resize images for mono depth
        mono_images = F.interpolate(context_images, size=(256, 512), mode='bilinear')
        mono_images = F.interpolate(mono_images, size=(512, 1024), mode='bilinear')

        # Project the images to the cube.
        cube_image = []
        for img in mono_images:
            img = img.numpy()
            img = rearrange(img, "c h w -> h w c")
            img = self.e2c_mono.run(img)
            cube_image.append(img)
        cube_image = np.stack(cube_image)
        cube_image = rearrange(cube_image, "v h w c -> v c h w")

        mono_images = F.interpolate(target_images, size=(256, 512), mode='bilinear')
        mono_images = F.interpolate(mono_images, size=(512, 1024), mode='bilinear')

        # Project the images to the cube.
        target_cube_image = []
        for img in mono_images:
            img = img.numpy()
            img = rearrange(img, "c h w -> h w c")
            img = self.e2c_mono.run(img)
            target_cube_image.append(img)
        target_cube_image = np.stack(target_cube_image)
        target_cube_image = rearrange(target_cube_image, "n h (w v) c -> n v c h w", v=6)

        nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
        data.pop('root')
        data.update({
            "context": {
                "pano_extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "pano_image": context_images,
                "mono_image": mono_images,
                "image": cube_image,
                "extrinsics": torch.einsum('nij,mjk -> nmik', extrinsics[context_indices], self.cubemap_Rs),
                "near": self.get_bound("near", len(context_images)) / nf_scale,
                "far": self.get_bound("far", len(context_images)) / nf_scale,
                "index": context_indices,
                # "depth": context_depths,
                # "mask": context_mask,
            },
            "target": {
                "pano_extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "extrinsics": torch.einsum('nij,mjk -> nmik', extrinsics[target_indices], self.cubemap_Rs)[:, [4,2,3,0,1,5], ...],
                "pano_image": target_images,
                "image": target_cube_image,
                "near": repeat(self.get_bound("near", len(target_indices)) / nf_scale, "v -> v c", c=6),
                "far": repeat(self.get_bound("far", len(target_indices)) / nf_scale, "v -> v c", c=6),
                "index": target_indices,
                # "depth": target_depths,
                # "mask": target_mask,
            },
            "scene": scene,
        })
        

        ref_extrinsic_inv = torch.inverse(data['context']['pano_extrinsics'][0])
        data['context']['pano_extrinsics'] = torch.einsum('ij,njk->nik', ref_extrinsic_inv, data['context']['pano_extrinsics'])
        data['context']['extrinsics'] = torch.einsum('ij,mnjk->mnik', ref_extrinsic_inv, data['context']['extrinsics'])
        data['target']['pano_extrinsics'] = torch.einsum('ij,njk->nik', ref_extrinsic_inv, data['target']['pano_extrinsics'])
        data['target']['extrinsics'] = torch.einsum('ij,mnjk->mnik', ref_extrinsic_inv, data['target']['extrinsics'])

        data['target']['extrinsics'][:, :, [0, 2]] = - data['target']['extrinsics'][:, :, [0, 2]]
        data['context']['extrinsics'][:, :, [0, 2]] = - data['context']['extrinsics'][:, :, [0, 2]]

        if data['name'] == 'm3d':
            data["context"]["depth"] = context_depths
            data["context"]["mask"] = context_mask
            data["target"]["depth"] = target_depths
            data["target"]["mask"] = target_mask

        return data

    def convert_poses(
        self,
        trans: Float[Tensor, "batch 3"],
        rots: Float[Tensor, "batch 3 3"],
    ) -> Float[Tensor, "batch 4 4"]:  # extrinsics
        b, _ = trans.shape

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        c2w = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        c2w[:, :3, :3] = rots
        c2w[:, :3, 3] = trans
        w2w = torch.tensor([  # X -> X, -Z -> Y, upY -> Z
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).float()
        c2c = torch.tensor([  # rightx -> rightx, upy -> -downy, backz -> -forwardz
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).float()
        c2w = w2w @ c2w @ c2c
        return c2w

    def convert_images(
        self,
        images: list[str],
    ):
        torch_images = []
        for image in images:
            image = Image.open(image)
            image = image.resize(self.cfg.image_shape[::-1], Image.LANCZOS)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data)
