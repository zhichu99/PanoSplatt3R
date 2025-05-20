import omegaconf
import sys
sys.path.append('/home/renjiahui/PanoSplatt3R')
import numpy as np
import cv2
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from model.encoder.gs_encoder import EncoderNoPoSplat
from torch.utils.data import DataLoader
from model.decoder import get_decoder
from dataset.database import get_data_shim
from utils.cubemap_process import Cube2Equirec
import torch
from tqdm import tqdm
from misc.image_io import save_image
from dataset.dataset_mp3d import DatasetMP3D
from dataset.view_sampler import get_view_sampler
from einops import rearrange
from utils.metric import Evaluator
from utils.pose_scale_align import get_scale
import lpips as lpips_lib

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
    to_save_image = False
    if to_save_image:
        save_path = os.path.join('result', cfg.dataset.test_datasets[0].name)
        os.makedirs(save_path, exist_ok=True)
    encoder = EncoderNoPoSplat(cfg.encoder).to('cuda')
    ckpt = torch.load('outputs/random_with_pretrain_no_pose_2dgs_full_reso_xformers/step_100000_psnr_28.180999046148255.pth')['model']
    encoder.load_state_dict(ckpt,strict=True)
    decoder = get_decoder(cfg.decoder).to('cuda')
    data_shim = get_data_shim(encoder)
    c2e = Cube2Equirec(256, 512, 1024).to('cuda')
    total_dataloader = []
    stage = "test"
    for data_info in cfg.dataset.test_datasets:
        cfg.dataset.test_datasets = [data_info]
        test_dataset = DatasetMP3D(cfg.dataset, stage)
        total_dataloader.append([data_info, DataLoader(
            test_dataset,
            batch_size=cfg.data_loader.test.batch_size,
            num_workers=cfg.data_loader.test.num_workers,
            persistent_workers=cfg.data_loader.test.persistent_workers)])
    
    evaluator = Evaluator()
    
    encoder.eval()
    final_results = []
    for data_info, test_dataloader in total_dataloader:
        local_metrics = {'wspsnr': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}
        for batch in tqdm(test_dataloader, desc='eval'):
            batch = data_shim(batch)
            batch = to_cuda(batch)

            with torch.no_grad():
                gaussians = encoder(batch["context"], mode='test')
                point_num = gaussians.means.shape[1] // 2
                scale_pred = get_scale(gaussians.means[0, point_num:].cpu().numpy(), 512, 1024, batch["context"]["pano_extrinsics"][0,1].cpu().detach().numpy())
                gaussians.means = gaussians.means * scale_pred
                gaussians.scales = gaussians.scales * scale_pred
                output = decoder.forward(
                        gaussians,
                        batch["target"]["extrinsics"][:,0],
                        batch["target"]["near"][0],
                        batch["target"]["far"][0],
                        (256, 256),
                        render_mode='pinhole'
                    )
                gt_image = rearrange(batch["target"]["pano_image"], 'b v c h w -> (b v) c h w')
                pred_cube = output.color
                pred_cube[:, [0,5]] = torch.rot90(pred_cube[:, [0,5]], 2, [3,4])
                
                pano_image = c2e(rearrange(pred_cube, 'b v c h w -> b c v h w'))

                if to_save_image:
                    pred_cube = rearrange(pred_cube, 'b n c h w -> (b n) c h w')
                    gt_cube = rearrange(batch["target"]["image"], 'b n v c h w -> (b n v) c h w')
                    for cube_index in range(len(gt_cube)):
                        save_image(pred_cube[cube_index], os.path.join(save_path,  batch["scene"][0], f"color/{cube_index:0>3}.png"))
                        save_image(gt_cube[cube_index], os.path.join(save_path,  batch["scene"][0], f"color/{cube_index:0>3}_gt.png"))
                    save_image(pano_image[0], os.path.join(save_path,  batch["scene"][0], f"pano_color.png"))
                    save_image(gt_image[0], os.path.join(save_path,  batch["scene"][0], f"pano_color_gt.png"))
                single_metric = evaluator.eval_metrics_img(gt_image, pano_image)

            for k, v in single_metric.items():
                local_metrics[k] += v / len(test_dataloader)

        metric_line = f' wspsnr {local_metrics["wspsnr"]}, psnr {local_metrics["psnr"]}, ssim {local_metrics["ssim"]}, lpips {local_metrics["lpips"]}'
        final_results.append([data_info, metric_line])

    print("====== Final Results ===")
    for data_info, metric_line in final_results:
        print(data_info['name'], data_info['dis'])
        print(metric_line)
        print("========================")

if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load("config/defalts.yaml")
    cfg.dataset = omegaconf.OmegaConf.load("config/dataset/mp3d.yaml")
    cfg.encoder = omegaconf.OmegaConf.load("config/model/encoder/noposplat.yaml")
    cfg.encoder.croco = omegaconf.OmegaConf.load("config/model/encoder/backbone/croco.yaml")
    cfg.decoder = omegaconf.OmegaConf.load("config/model/decoder/splatting_cuda.yaml")
    main(cfg)
