import numpy as np
import torch
import cv2

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def panorama_pnp(world_points, panorama_points, img_width, img_height):
    theta = (panorama_points[:, 0] / img_width) * 2 * np.pi - np.pi
    phi = (panorama_points[:, 1] / img_height) * np.pi - np.pi / 2

    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    valid_mask = z > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    valid_world_points = world_points[valid_mask]
    norm_points = np.column_stack((x / z, y / z))

    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        valid_world_points, norm_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("PnP failure")

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers


def get_scale(pts3d, H, W, gt_extrinsic):
    """
    pts3d: N x 3 numpy.array
    """
    pts2d1 = xy_grid(W, H).reshape((-1, 2))
    R, tvec, _ = panorama_pnp(pts3d, pts2d1, W, H)

    eye = np.eye(4)
    eye[:3,:3] = R
    eye[:3,3:]  = tvec
    norm_pred = np.linalg.norm(np.linalg.inv(eye)[:3, 3:])
    norm_gt = np.linalg.norm(gt_extrinsic[:3, 3:])

    scale = norm_gt / norm_pred
    return scale