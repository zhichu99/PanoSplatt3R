import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )

def build_rotation(r):
    b, n, c = r.shape
    r = r.reshape(-1, c)
    
    norm = torch.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 + r[:, 3]**2)
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R.reshape(b, n, 3, 3)

def build_scaling_rotation(s, r):
    b, n, c = s.shape
    s = s.reshape(-1, c)
    R = build_rotation(r)  # Shape (b*n, 3, 3)

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L

    return L.reshape(b, n, 3, 3)

def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    # Flatten b and n dimensions for processing
    b, n, c = center.shape

    # Scaling and modifier
    scaling_combined = torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1)
    RS = build_scaling_rotation(scaling_combined, rotation).permute(0, 1, 3, 2)  # (b, n, 3, 3)

    # Initialize transformation matrix
    trans = torch.zeros((b, n, 4, 4), dtype=torch.float, device=center.device)

    trans[:, :, :3, :3] = RS  # Set the rotation-scaling part
    trans[:, :, 3, :3] = center  # Set the translation part
    trans[:, :, 3, 3] = 1  # Set the homogeneous coordinate

    return trans