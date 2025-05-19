import omegaconf
import sys
sys.path.append('/home/renjiahui/PanoSplatt3r')
from model.encoder.gs_encoder import EncoderNoPoSplat
from dataset.dataset_hm3d import DatasetHM3D
from torch.utils.data import DataLoader
import argparse
from model.decoder import get_decoder
from dataset.database import get_data_shim
import torch
from tqdm import tqdm
from einops import rearrange
from utils.cubemap_process import Cube2Equirec
from collections import OrderedDict
import cv2
import composer.functional as cf
from utils.metric import Evaluator
import lpips as lpips_lib
from utils.metric import Pinhole_Evaluator, compute_depth_metrics_batched
from dataset.view_sampler import get_view_sampler
from visualization.color_map import apply_color_map_to_image
from utils.pose_scale_align import get_scale
import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from misc.image_io import save_image

def depth_map(result):
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        result = result.log()
        result = 1 - (result - near) / (far - near)
    except Exception:
        near = result.min()
        far = result.max()
        result = 1 - (result - near) / (far - near)
    
    return apply_color_map_to_image(result, "turbo")

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cuda(item))
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            results[k]=to_cuda(v)
        return results
    elif type(data).__name__ == "Tensor":
        return data.cuda()
    else:
        return data

def main(cfg):
    torch.use_deterministic_algorithms(True)
    input_h = 512
    input_w = 1024
    encoder = EncoderNoPoSplat(cfg.encoder).to('cuda')
    ckpt = torch.load("outputs/random_with_pretrain_no_pose_2dgs_full_reso_xformers/step_100000_psnr_28.180999046148255.pth", map_location='cpu')['model']
    encoder.load_state_dict(ckpt,strict=True)
    decoder = get_decoder(cfg.decoder).to('cuda')
    data_shim = get_data_shim(encoder)
    c2e = Cube2Equirec(256, 512, 1024).cuda()

    to_save_image = False
    if to_save_image:
        save_path = os.path.join('result', cfg.test_dataset.name)
        os.makedirs(save_path, exist_ok=True)

    stage = "test"
    step_tracker = None
    view_sampler = get_view_sampler(
            cfg.test_dataset.view_sampler,
            stage,
            cfg.test_dataset.overfit_to_scene is not None,
            cfg.test_dataset.cameras_are_circular,
            step_tracker,
        )
    test_dataset = DatasetHM3D(cfg.test_dataset, stage, view_sampler, resized_shape=(input_h, input_w))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data_loader.test.batch_size,
        num_workers=cfg.data_loader.test.num_workers,
        persistent_workers=cfg.data_loader.test.persistent_workers)
    
    lpips_model = lpips_lib.LPIPS(net='vgg').to('cuda').eval()

    evaluator = Pinhole_Evaluator(lpips_model=lpips_model)
    
    encoder.eval()
    local_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'abs_diff': [], 'abs_rel': [], 'rmse':[], 'a25': []}

    for batch in tqdm(test_dataloader, desc='eval'):
        batch = data_shim(batch)
        batch = to_cuda(batch)

        with torch.no_grad():
            gaussians = encoder(batch["context"], mode='test')
            point_num = gaussians.means.shape[1] // 2

            scale_pred = get_scale(gaussians.means[0, point_num:].cpu().numpy(), input_h, input_w, batch["context"]["pano_extrinsics"][0,1].cpu().detach().numpy())
            gaussians.means = gaussians.means * scale_pred
            gaussians.scales = gaussians.scales * scale_pred

            output = decoder.forward(
                        gaussians,
                        rearrange(batch["target"]["extrinsics"], "b v c r1 r2 -> b (v c) r1 r2"),
                        rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)"),
                        rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)") ,
                        (256, 256),
                        render_mode='pinhole'
                    )
            gt_image = rearrange(batch["target"]["image"], 'b n v c h w -> (b n v) c h w')
            gt_depth = rearrange(batch["target"]["depth"][:, :, 1:], 'b n v h w c -> (b n v) c h w') # drop top view due to empty depth (cause NaNs)

            valid_mask_b = (gt_depth > 0.1)
            valid_mask_batch = torch.any(rearrange(gt_depth >= 0.1, 'n c h w -> n (c h w)'), dim=-1)
            assert (valid_mask_b).any(), 'no valid depth in batch'

            pred_image = rearrange(output.color, 'b n c h w -> (b n) c h w')
            pred_depth = rearrange(rearrange(output.depth, 'b (n v) c h w -> b n v c h w', v=6)[:, :, 1:], 'b n v c h w -> (b n v) c h w')

            if to_save_image:
                saved_gt_depth = rearrange(batch["target"]["depth"], 'b n v h w c -> (b n v) c h w')
                saved_pred_depth = rearrange(rearrange(output.depth, 'b (n v) c h w -> b n v c h w', v=6), 'b n v c h w -> (b n v) c h w')
                for cube_index in range(len(gt_image)):
                    save_image(pred_image[cube_index], os.path.join(save_path,  batch["scene"][0], f"color/{cube_index:0>3}.png"))
                    save_image(gt_image[cube_index], os.path.join(save_path,  batch["scene"][0], f"color/{cube_index:0>3}_gt.png"))
                    save_image(depth_map(saved_pred_depth[cube_index]), os.path.join(save_path , batch["scene"][0], f"depth/{cube_index:0>3}_depth.png"))
                    save_image(depth_map(saved_gt_depth[cube_index]), os.path.join(save_path , batch["scene"][0], f"depth/{cube_index:0>3}_depth_gt.png"))

                pred_panorama = c2e(rearrange(pred_image, '(b v) c h w -> b c v h w', v=6))
                gt_panorama = batch["target"]["pano_image"].squeeze()
                for i in range(len(gt_panorama)):
                    save_image(pred_panorama[i], os.path.join(save_path,  batch["scene"][0], f"color/{i:0>6}_erp.png"))
                    # save_image(gt_panorama[i], os.path.join(save_path,  batch["scene"][0], f"color/{i:0>6}_erp_gt.png"))

            rgb_metric = evaluator.eval_metrics_img(gt_image, pred_image)
            depth_metric = compute_depth_metrics_batched(gt_depth.flatten(start_dim=1).float(),
                                                            pred_depth.flatten(start_dim=1).float(), 
                                                            valid_mask_b.flatten(start_dim=1), 
                                                            mult_a=True)

            for key in depth_metric.keys():
                if key in ["abs_diff", "abs_rel", "rmse", "a25"]: # only evaluate with four metrics
                    depth_metric[key][~valid_mask_batch] = 0
                    # non zero
                    batch_valid_cnt = valid_mask_batch.count_nonzero()
                    assert batch_valid_cnt > 0
                    local_metrics[key].append((depth_metric[key].sum() / batch_valid_cnt).item())

            for k, v in rgb_metric.items():
                local_metrics[k].append(rgb_metric[k])
    total_metrics = {}
    for k in local_metrics.keys():
        total_metrics[k] = torch.tensor(local_metrics[k]).mean().item()

    metric_line = f'psnr {total_metrics["psnr"]}, ssim {total_metrics["ssim"]}, lpips {total_metrics["lpips"]}, abs_diff {total_metrics["abs_diff"]}, abs_rel {total_metrics["abs_rel"]}, rmse {total_metrics["rmse"]}, a25 {total_metrics["a25"]} \n'
    print(metric_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='replica', choices=['replica', 'hm3d'],
                        help='Which dataset config to load')
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load("config/defalts.yaml")
    if args.dataset == 'replica':
        cfg.test_dataset = omegaconf.OmegaConf.load("config/dataset/replica.yaml")
    elif args.dataset == 'hm3d':
        cfg.test_dataset = omegaconf.OmegaConf.load("config/dataset/hm3d_eval.yaml")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    cfg.encoder = omegaconf.OmegaConf.load("config/model/encoder/noposplat.yaml")
    cfg.encoder.croco = omegaconf.OmegaConf.load("config/model/encoder/backbone/croco.yaml")
    cfg.decoder = omegaconf.OmegaConf.load("config/model/decoder/splatting_cuda.yaml")
    main(cfg)