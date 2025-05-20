import omegaconf
import sys
sys.path.append('/home/renjiahui/PanoSplatt3R')
from model.encoder.gs_encoder import EncoderNoPoSplat
from dataset.dataset_hm3d import DatasetHM3D
from torch.utils.data import DataLoader
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
from estimate_pose.pose_eval import rotation_angle, translation_angle
from estimate_pose.lib_est_rel_pos import estimate_relative_pose_from_matches, estimate_relative_pose
from estimate_pose.example_estimate_pose import get_pose_pnp, xy_grid, find_reciprocal_matches
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
    Ours_PnP = {'rra':[], 'rta':[]}
    Ours_8PA = {'rra':[], 'rta':[]}
    SIFT_8PA = {'rra':[], 'rta':[]}

    for batch in tqdm(test_dataloader, desc='eval'):
        batch = data_shim(batch)
        batch = to_cuda(batch)

        with torch.no_grad():
            gaussians = encoder(batch["context"], mode='test')

            pts = rearrange(gaussians.means.squeeze(0), '(b h w) c -> b h w c', b=2, h=512, w=1024)
            pcd0 = rearrange(pts[0], 'h w c -> (h w) c').cpu().detach().numpy()
            pcd1 = rearrange(pts[1], 'h w c -> (h w) c').cpu().detach().numpy()
            gt_extrinsic = (np.linalg.inv(batch['context']["pano_extrinsics"][0,0].cpu().detach().numpy()) @ batch['context']["pano_extrinsics"][0,1].cpu().detach().numpy()).astype(np.float32)

        est_pose = get_pose_pnp(pcd1, 512, 1024)
        Ours_PnP['rra'].append(rotation_angle(est_pose[None, :3, :3], gt_extrinsic[None, :3, :3], batch_size=1).numpy())
        Ours_PnP['rta'].append(translation_angle(est_pose[None, :3, 3], gt_extrinsic[None, :3, 3], batch_size=1).numpy())

        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(pcd0, pcd1)

        pts2d0 = xy_grid(1024, 512).reshape(-1, 2)
        pts2d1 = xy_grid(1024, 512).reshape(-1, 2)

        matches_1 = pts2d1[reciprocal_in_P2][::100]
        matches_0 = pts2d0[nn2_in_P1][reciprocal_in_P2][::100]

        est_pose = estimate_relative_pose_from_matches(matches_0, matches_1, (1024, 512))
        Ours_8PA['rra'].append(rotation_angle(est_pose[None, :3, :3], gt_extrinsic[None, :3, :3], batch_size=1).numpy())
        Ours_8PA['rta'].append(translation_angle(est_pose[None, :3, 3], gt_extrinsic[None, :3, 3], batch_size=1).numpy())

        img1 = (rearrange(batch['context']['pano_image'][0,0], 'c h w -> h w c').cpu().numpy() * 255).astype(np.uint8)
        img2 = (rearrange(batch['context']['pano_image'][0,1], 'c h w -> h w c').cpu().numpy() * 255).astype(np.uint8)

        est_pose = estimate_relative_pose(img1, img2)
        SIFT_8PA['rra'].append(rotation_angle(est_pose[None, :3, :3], gt_extrinsic[None, :3, :3], batch_size=1).numpy())
        SIFT_8PA['rta'].append(translation_angle(est_pose[None, :3, 3], gt_extrinsic[None, :3, 3], batch_size=1).numpy())

    print("Ours-PnP")
    print("rra:", np.mean(Ours_PnP['rra']))
    print("rta:", np.mean(Ours_PnP['rta']))
    print("Ours-8PA")
    print("rra:", np.mean(Ours_8PA['rra']))
    print("rta:", np.mean(Ours_8PA['rta']))
    print("SIFT-8PA")
    print("rra:", np.mean(SIFT_8PA['rra']))
    print("rta:", np.mean(SIFT_8PA['rta']))



if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load("config/defalts.yaml")
    cfg.test_dataset = omegaconf.OmegaConf.load("config/dataset/replica.yaml")
    cfg.encoder = omegaconf.OmegaConf.load("config/model/encoder/noposplat.yaml")
    cfg.encoder.croco = omegaconf.OmegaConf.load("config/model/encoder/backbone/croco.yaml")
    cfg.decoder = omegaconf.OmegaConf.load("config/model/decoder/splatting_cuda.yaml")
    main(cfg)
