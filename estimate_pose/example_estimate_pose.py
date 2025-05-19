import cv2
import numpy as np 
import time
from estimate_pose.pose_eval import rotation_angle, translation_angle
import torch  
import pickle 
from einops import rearrange
from scipy.spatial import cKDTree as KDTree

from estimate_pose.lib_est_rel_pos import estimate_relative_pose_from_matches, estimate_relative_pose

def panorama_pnp(world_points, panorama_points, img_width, img_height):
    theta = (panorama_points[:, 0] / img_width) * 2 * np.pi 
    phi = (panorama_points[:, 1] / img_height) * np.pi - np.pi / 2

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    norm_points = np.column_stack((x / z, y / z))
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points, norm_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("PnP failure")
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers

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

    tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_pose_pnp(pts3d, H, W):
    pts2d1 = xy_grid(W, H).reshape((-1, 2))
    R, tvec, _ = panorama_pnp(pts3d, pts2d1, W, H)

    eye = np.eye(4)
    eye[:3,:3] = R
    eye[:3,3:] = tvec
    pred_pose = np.linalg.inv(eye)
    return pred_pose


##### example usage
if __name__ == '__main__':
    gs = torch.load(f'/data/weights/gaussians/gaussian_0.pth')
    pts = rearrange(gs['means'], '(b h w) c -> b h w c', b=2, h=512, w=1024)
    pcd0 = rearrange(pts[0], 'h w c -> (h w) c')
    pcd1 = rearrange(pts[1], 'h w c -> (h w) c')
    gt_extrinsic = (np.linalg.inv(gs['extrinsic_1']) @ gs['extrinsic_2']).astype(np.float32)


    ### Ours-PnP
    est_pose = get_pose_pnp(pcd1.numpy(), 512, 1024)


    ### Ours-8PA
    # find 2D-2D correspondence from nearest 3D points
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(pcd0, pcd1)

    pts2d0 = xy_grid(1024, 512).reshape(-1, 2)
    pts2d1 = xy_grid(1024, 512).reshape(-1, 2)

    matches_1 = pts2d1[reciprocal_in_P2][::100]
    matches_0 = pts2d0[nn2_in_P1][reciprocal_in_P2][::100]

    est_pose = estimate_relative_pose_from_matches(matches_0, matches_1, (1024, 512))


    ### SIFT-8PA
    gt_rgb = ... # b 2 c h w
    img1 = (rearrange(gt_rgb[0,0], 'c h w -> h w c') * 255).astype(np.uint8)
    img2 = (rearrange(gt_rgb[0,1], 'c h w -> h w c') * 255).astype(np.uint8)

    est_pose = estimate_relative_pose(img1, img2)


    # normalize translation
    est_pose[:3, 3] = est_pose[:3, 3] / np.linalg.norm(est_pose[:3, 3]) * np.linalg.norm(gt_extrinsic[:3, 3])

    print(est_pose)
    print(gt_extrinsic)

    print(rotation_angle(est_pose[None, :3, :3], gt_extrinsic[None, :3, :3], batch_size=1))
    print(translation_angle(est_pose[None, :3, 3], gt_extrinsic[None, :3, 3], batch_size=1))