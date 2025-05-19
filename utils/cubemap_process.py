import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def spherical_to_cubemap(theta, phi, hfov=90/180 * np.pi):
  """Converts spherical coordinates to cubemap coordinates.

  Args:
    theta: Longitude (azimuthal angle) in radians. [0, 2pi]
    phi: Latitude (altitude angle) in radians. [0, pi]

  Returns:
    uv: UVS in channel_last format
    idx: Side of the cubemap

  """
  u = np.zeros(theta.shape, dtype=np.float32)
  v = np.zeros(theta.shape, dtype=np.float32)
  side = np.zeros(theta.shape, dtype=np.float32)
  side[:] = -1
  focal_len = 1 / np.tan(hfov / 2.0)

  for i in range(0, 4):
    indices = np.logical_or(
      np.logical_and(theta >= i * np.pi / 2 - np.pi / 4, theta <=
                     (i + 1) * np.pi / 2 - np.pi / 4),
      np.logical_and(theta >= i * np.pi / 2 - np.pi / 4 + 2 * np.pi, theta <=
                     (i + 1) * np.pi / 2 - np.pi / 4 + 2 * np.pi))
    u[indices] = np.tan(theta[indices] - i * np.pi / 2) 
    v[indices] = 1 / (np.tan(phi[indices]) *
                      np.cos(theta[indices] - i * np.pi / 2))
    u[indices] *= focal_len
    v[indices] *= focal_len
    side[indices] = i + 1
  top_indices = np.logical_or(phi < np.pi / 4, v >= 1)
  u[top_indices] = -np.tan(phi[top_indices]) * np.sin(theta[top_indices] -
                                                      np.pi)
  v[top_indices] = np.tan(phi[top_indices]) * np.cos(theta[top_indices] - np.pi)
  u[top_indices] *= focal_len
  v[top_indices] *= focal_len
  side[top_indices] = 0
  bottom_indices = np.logical_or(phi >= 3 * np.pi / 4, v <= -1)
  u[bottom_indices] = -np.tan(phi[bottom_indices]) * np.sin(
    theta[bottom_indices])
  v[bottom_indices] = -np.tan(phi[bottom_indices]) * np.cos(
    theta[bottom_indices])
  u[bottom_indices] *= focal_len
  v[bottom_indices] *= focal_len
  side[bottom_indices] = 5

  assert not np.any(side < 0), "Side less than 0"

  return np.stack(((u + 1) / 2, (-v + 1) / 2), axis=-1), side


def bilinear_interpolate(image, x, y):
  """Applies bilinear interpolation on numpy encoded images.

  Assumes a channel_last format.
  Based on https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python

  Args:
    image: Input image.
    x: x-coordinates.
    x: y-coordinates.

  Returns:
    Interpolated image.

  """
  x = np.asarray(x)
  y = np.asarray(y)

  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1

  x0 = np.clip(x0, 0, image.shape[1] - 1)
  x1 = np.clip(x1, 0, image.shape[1] - 1)
  y0 = np.clip(y0, 0, image.shape[0] - 1)
  y1 = np.clip(y1, 0, image.shape[0] - 1)

  top_left = image[y0, x0]
  bottom_left = image[y1, x0]
  top_right = image[y0, x1]
  bottom_right = image[y1, x1]

  tl_weight = (x1 - x) * (y1 - y)
  bl_weight = (x1 - x) * (y - y0)
  tr_weight = (x - x0) * (y1 - y)
  br_weight = (x - x0) * (y - y0)

  if len(top_left.shape) > len(tl_weight.shape):
    tl_weight = tl_weight[..., np.newaxis]
    bl_weight = bl_weight[..., np.newaxis]
    tr_weight = tr_weight[..., np.newaxis]
    br_weight = br_weight[..., np.newaxis]

  return tl_weight * top_left + bl_weight * bottom_left + tr_weight * top_right + br_weight * bottom_right


class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), np.bool_) # , np.bool
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)
        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5
        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):
        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi
        lon, lat = np.meshgrid(lon, lat)
        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)
        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])
        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)
        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):
        bs = cube_feat.shape[0]
        cube_feat = cube_feat[:, :, [3, 4, 1, 2, 0, 5], :, :]
        cube_feat[:, :, 4:] = torch.flip(cube_feat[:, :, 4:], [3, 4])
        sample_grid = torch.cat(bs * [self.sample_grid], dim=0)
        equi_feat = F.grid_sample(cube_feat, sample_grid, padding_mode="border", align_corners=True)
        return equi_feat.squeeze(2)
    
def focal_from_fov(fov, width):
  '''fov in radian'''
  focal = width / 2 / math.tan(fov/2)
  return focal

def get_z_to_euclidian_weight(focal, width, height):
    u = np.linspace(-width/2, width/2, width)
    v = np.linspace(-height/2, height/2, height)
    u, v = np.meshgrid(u,v, indexing='ij')
    distance = np.sqrt(u**2 + v**2 + focal**2)
    weight = distance / focal
    return weight
