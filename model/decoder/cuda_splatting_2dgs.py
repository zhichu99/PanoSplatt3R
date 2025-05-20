from math import isqrt
from typing import Literal, Union, Optional
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result



def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    # intrinsics: Optional[Float[Tensor, "batch 3 3"]] = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
    render_mode: str = "pinhole",
) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"]]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    assert render_mode in ["pinhole", "panorama"]
    
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        gaussian_scales = gaussian_scales * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    ori_h, ori_w = image_shape
    h, w = int(ori_h), int(ori_w)

    b, _, _ = extrinsics.shape

    fov_x = torch.tensor([90 for _ in range(b)], device=extrinsics.device).deg2rad()
    fov_y = torch.tensor([90 for _ in range(b)], device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    # extrinsics is c2w
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    # view_matrix = extrinsics.inverse()
    full_projection = view_matrix @ projection_matrix

    all_images = []
    # all_radii = []
    all_depths = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        # means3D = gaussian_means[i]
        # sh = shs[i] if use_sh else None
        # opacity = gaussian_opacities[i, ..., None]
        # scales, rotations = gaussian_scales[i], gaussian_rotations[i]

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color,
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )   
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, allmap = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            scales = gaussian_scales[i],
            rotations = gaussian_rotations[i],
            cov3D_precomp=None,
        )

        all_images.append(image)

        # # additional regularizations
        # render_alpha = allmap[1:2]
    
        # get median depth map
        # render_depth_median = allmap[5:6]
        # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # # get expected depth map
        # render_depth_expected = allmap[0:1]
        # render_depth_expected = (render_depth_expected / render_alpha)
        # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        # surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
        # all_depths.append(render_depth_median)

    return torch.stack(all_images)#, torch.stack(all_depths)

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt.to(R.device,dtype=torch.float32)

def decompose_covariances(covariances):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    scales = torch.sqrt(eigenvalues)
    rotations = eigenvectors
    return scales, rotations

def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: Union[dict, None] = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            spherical=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        gaussian_scales,
        gaussian_rotations,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)