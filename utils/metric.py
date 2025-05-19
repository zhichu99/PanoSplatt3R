import torch
import numpy as np
from skimage.metrics import structural_similarity
from torch import Tensor
from jaxtyping import Float
from einops import reduce
from lpips import LPIPS
from functools import cache


class Evaluator:
    def __init__(self):
        self.wspsnr_calculator = WSPSNR()

    def eval_metrics_img(self,gt_img, pr_img):
        psnr = compute_psnr(gt_img, pr_img).mean().item()
        ssim = compute_ssim(gt_img, pr_img).mean().item()
        score = compute_lpips(gt_img, pr_img).mean().item()
        ws_psnr = self.wspsnr_calculator.ws_psnr(
            pr_img, gt_img, max_val=1.0)  # input: B, H, W, C
        return {'wspsnr':ws_psnr.item(), 'psnr':float(psnr), 'ssim':float(ssim), 'lpips':score}
    

class Pinhole_Evaluator:
    def __init__(self, lpips_model):
        # self.loss_fn_lpips = lpips_model.eval()
        pass

    def eval_metrics_img(self,gt_img, pr_img):
        psnr = compute_psnr(gt_img, pr_img).mean().item()
        ssim = compute_ssim(gt_img, pr_img).mean().item()
        score = compute_lpips(gt_img, pr_img).mean().item()
        return {'psnr':float(psnr), 'ssim':float(ssim), 'lpips':score}


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    # import pdb;pdb.set_trace()
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    mse[mse == 0.0] = 1e-10 # avoid Inf (PSNR=100)
    return -10 * mse.log10()

@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]

class WSPSNR:
    """Weighted to spherical PSNR"""

    def __init__(self):
        self.weight_cache = {}

    def get_weights(self, height=1080, width=1920):
        """Gets cached weights.
        Args:
            height: Height.
            width: Width.
        Returns:
          Weights as H, W tensor.
        """
        key = f"{height};{width}"
        if key not in self.weight_cache:
            v = (np.arange(0, height) + 0.5) * (np.pi / height)
            v = np.sin(v).reshape(height, 1)
            v = np.broadcast_to(v, (height, width))
            self.weight_cache[key] = v.copy()
        return self.weight_cache[key]

    def calculate_wsmse(self, reconstructed, reference):
        """
        Calculates weighted mse for a single channel.
        Args:
            reconstructed: Image as B, C, H, W tensor.
            reference: Image as B, C, H, W tensor.
        Returns:
            wsmse
        """
        batch_size, channels, height, width = reconstructed.shape
        weights = torch.tensor(
            self.get_weights(height, width),
            device=reconstructed.device,
            dtype=reconstructed.dtype
        )
        weights = weights.view(1, 1, height, width).expand(batch_size, channels, -1, -1)
        
        squared_error = torch.pow((reconstructed - reference), 2.0)
        wmse = torch.sum(weights * squared_error, dim=(2, 3)) / torch.sum(weights, dim=(2, 3))
        wmse = wmse.mean(dim=1)  # Mean over channels
        return wmse

    def ws_psnr(self, y_pred, y_true, max_val=1.0):
        """
        Weighted to spherical PSNR.
        Args:
          y_pred: First image as B, C, H, W tensor.
          y_true: Second image as B, C, H, W tensor.
          max_val: Maximum possible value for the pixel intensity.
        Returns:
          Tensor of weighted spherical PSNR values for each image in the batch.
        """
        wmse = self.calculate_wsmse(y_pred, y_true)
        ws_psnr = 10 * torch.log10(max_val * max_val / wmse)
        return ws_psnr



def compute_depth_metrics_batched(gt_bN, pred_bN, valid_masks_bN, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths, 
    batched. Abuses nan behavior in torch.
    """

    gt_bN = gt_bN.clone()
    pred_bN = pred_bN.clone()

    gt_bN[~valid_masks_bN] = torch.nan
    pred_bN[~valid_masks_bN] = torch.nan

    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a_dict = {}
    
    a_val = (thresh_bN < (1.0+0.05)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a5"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a10"] = torch.nanmean(a_val, dim=1) 

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a25"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a0"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a1"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 2).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a2"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 3).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a3"] = torch.nanmean(a_val, dim=1)

    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse_bN = (gt_bN - pred_bN) ** 2
    rmse_b = torch.sqrt(torch.nanmean(rmse_bN, dim=1))

    rmse_log_bN = (torch.log(gt_bN) - torch.log(pred_bN)) ** 2
    rmse_log_b = torch.sqrt(torch.nanmean(rmse_log_bN, dim=1))

    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1)

    sq_rel_b = torch.nanmean((gt_bN - pred_bN) ** 2 / gt_bN, dim=1)

    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1)

    metrics_dict = {
                    "abs_diff": abs_diff_b,
                    "abs_rel": abs_rel_b,
                    "sq_rel": sq_rel_b,
                    "rmse": rmse_b,
                    "rmse_log": rmse_log_b,
                }
    metrics_dict.update(a_dict)

    return metrics_dict