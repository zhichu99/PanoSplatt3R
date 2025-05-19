import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from einops import rearrange, repeat
import omegaconf
from tqdm import tqdm
import math
import random
import numpy as np
import torch
from torch import Generator, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import composer.functional as cf

from dataset.dataset_hm3d import DatasetHM3D
from dataset.view_sampler import get_view_sampler
from model.encoder.common.gaussians import build_covariance
from model.decoder import get_decoder
from model.dust3r.model import AsymmetricCroCo3DStereo
from model.types import Gaussians
from dataset.database import get_data_shim
from misc.step_tracker import StepTracker
from utils.loss import l2_loss, l1_loss
from utils.log import make_figure
from utils.metric import compute_depth_metrics_batched

class Trainer:
    def __init__(self, cfg):
        ### cfg
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.cfg = cfg
        self.cfg.gpu_id = self.gpu_id
        self.h = 512
        self.w = 1024

        ### data
        stage = "train"
        self.step_tracker = StepTracker()
        view_sampler = get_view_sampler(
                cfg.train_dataset.view_sampler,
                stage,
                cfg.train_dataset.overfit_to_scene is not None,
                cfg.train_dataset.cameras_are_circular,
                self.step_tracker,
            )
        
        self.train_dataset = DatasetHM3D(self.cfg.train_dataset, stage, view_sampler, resized_shape=(self.h, self.w))
        train_generator = Generator()
        train_generator.manual_seed(self.cfg.data_loader.train.seed + self.gpu_id)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.data_loader.train.batch_size,
            pin_memory=True,
            num_workers=cfg.data_loader.train.num_workers,
            worker_init_fn=worker_init_fn,
            generator=train_generator,
            persistent_workers=cfg.data_loader.train.persistent_workers)
        
        stage = "test"
        view_sampler = get_view_sampler(
                cfg.test_dataset.view_sampler,
                stage,
                cfg.test_dataset.overfit_to_scene is not None,
                cfg.test_dataset.cameras_are_circular,
                self.step_tracker,
            )
        self.test_dataset = DatasetHM3D(self.cfg.test_dataset, stage, view_sampler, resized_shape=(self.h, self.w))
        test_generator = Generator()
        test_generator.manual_seed(self.cfg.data_loader.test.seed)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=cfg.data_loader.test.batch_size,
            num_workers=cfg.data_loader.test.num_workers,
            generator=test_generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=cfg.data_loader.test.persistent_workers,
            shuffle=False)
        
        y_coords = torch.linspace(-math.pi / 2, math.pi / 2, steps=self.h)
        weights = torch.cos(y_coords).abs()
        self.loss_weights = weights.unsqueeze(1).expand(-1, self.w).unsqueeze(0).repeat(cfg.data_loader.train.batch_size,1,1).unsqueeze(-1).to(self.gpu_id)
        
        print(' -> dataset initialized on rank', self.gpu_id)
        
        ### network
        self.encoder = AsymmetricCroCo3DStereo(cfg.croco)
        self.encoder = self.encoder.to(self.gpu_id)
        if not cfg.pretrain_path is None:
            ckpt = torch.load(cfg.pretrain_path, map_location='cpu')['model']
            self.encoder.load_state_dict(ckpt, strict=False)
            
        cf.apply_low_precision_layernorm(self.encoder, precision='amp')

        self.data_shim = get_data_shim(self.encoder)
        self.encoder = DDP(self.encoder,device_ids=[self.gpu_id], find_unused_parameters=True)

        self.decoder = get_decoder(cfg.decoder).to(self.gpu_id)

        print(' -> network initialized on rank', self.gpu_id)

        ### opt
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.cfg.optimizer.lr, weight_decay=0.05, betas=(0.9, 0.95))

        warm_up_steps = self.cfg.optimizer.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            self.optimizer ,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.total_epochs*self.cfg.checkpointing.every_n_train_steps, eta_min=self.cfg.optimizer.lr * 0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        print(' -> optimizer initialized on rank', self.gpu_id)
        self.scaler = GradScaler()
        
        ### log
        if self.gpu_id == 0:
            os.makedirs(os.path.join(cfg.log_path, cfg.exp_name), exist_ok=True)
            self.logger = SummaryWriter(log_dir=os.path.join(cfg.log_path, cfg.exp_name))
            print(' -> logger initialized on rank', self.gpu_id)
            os.system(f'cp -r config {cfg.log_path}')
        
        ### misc
        self.current_step = 0
        self.current_epoch = 0

        if self.cfg.resume:
            ckpt = torch.load(self.cfg.resume, map_location='cpu')
            self.encoder.module.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.current_step = ckpt['step']
            self.current_epoch = self.current_step // self.cfg.checkpointing.every_n_train_steps
            print(' -> resume from', self.cfg.resume)

    def train(self):
        for epoch in range(self.current_epoch, self.cfg.train.total_epochs):
            self.current_epoch = epoch
            print(f' -> start training epoch {epoch} on GPU_id={self.gpu_id}')
            self.train_one_epoch()
            dist.barrier()
            if self.gpu_id == 0:
                abs_rel = self.eval()
                self.save_ckpt(abs_rel)
            torch.cuda.empty_cache()

    def train_one_epoch(self):
        for batch in tqdm(self.train_dataloader, disable=(not self.gpu_id == 0), desc=f'epoch_{self.current_epoch}'):
            batch = self.data_shim(batch)

            batch = self.to_cuda(batch)
            with autocast('cuda', dtype=torch.float16):
                res1, res2, dec1, dec2, shape1, shape2 = self.encoder(batch["context"], mode='test')
            gt_pts3d_1 = self.depth_to_pts3d(batch["context"]["pano_depth"][:,0])
            gt_pts3d_1 = torch.einsum('bhwij,bhwj->bhwi', 
                                      batch["context"]['pano_extrinsics'][:,0].unsqueeze(1).unsqueeze(1).repeat(1, shape1[0,0], shape1[0,1], 1, 1),
                                      gt_pts3d_1)[..., :3]
            mask_1 = batch["context"]["pano_mask"][:,0].unsqueeze(-1)
            loss_1 = l1_loss(gt_pts3d_1, res1['pts3d'], weight=self.loss_weights, mask=mask_1) * 0.5 + l2_loss(gt_pts3d_1, res1['pts3d'], weight=self.loss_weights, mask=mask_1) * 0.5
            gt_pts3d_2 = self.depth_to_pts3d(batch["context"]["pano_depth"][:,1])
            gt_pts3d_2 = torch.einsum('bhwij,bhwj->bhwi',
                                        batch["context"]['pano_extrinsics'][:,1].unsqueeze(1).unsqueeze(1).repeat(1, shape2[0,0], shape2[0,1], 1, 1),
                                        gt_pts3d_2)[..., :3]
            mask_2 = batch["context"]["pano_mask"][:,1].unsqueeze(-1)
            loss_2 = l1_loss(gt_pts3d_2, res2['pts3d'], weight=self.loss_weights, mask=mask_2) * 0.5 + l2_loss(gt_pts3d_2, res2['pts3d'], weight=self.loss_weights, mask=mask_2) * 0.5

            loss_total = loss_1 + loss_2

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()
            dist.barrier()
            
            if self.current_step % self.cfg.train.print_log_every_n_steps == 0 and self.gpu_id == 0:
                mean3d = torch.cat([rearrange(res1['pts3d'], 'b h w c -> b (h w) c'), rearrange(res2['pts3d'], 'b h w c -> b (h w) c')], dim=1).cuda()[:1]

                color_1 = rearrange(batch["context"]["pano_image"][:,0], 'b c h w -> b (h w) c')[:1]
                color_2 = rearrange(batch["context"]["pano_image"][:,1], 'b c h w -> b (h w) c')[:1]
                sh = torch.cat([color_1, color_2], dim=1).cuda().unsqueeze(-1)[:1] * 0.5 + 0.5
                invalid_pixels = sh.squeeze().sum(-1) < 0.1

                eps = 1e-8
                rotations = torch.ones(1, mean3d.shape[1], 4).cuda()
                rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
                scales = torch.ones(1, mean3d.shape[1], 3).cuda() * 0.01
                scales = 0.001 * F.softplus(scales)
                scales = scales.clamp_max(0.3)

                opacities = torch.ones(1, mean3d.shape[1]).cuda().sigmoid()
                opacities[0, invalid_pixels] = 0.0
                covariances = build_covariance(scales, rotations)
                gaussians = Gaussians(means=mean3d, covariances=covariances, opacities=opacities, rotations=rotations, scales=scales, harmonics=sh)
                with torch.no_grad():
                    output = self.decoder.forward(
                            gaussians,
                            rearrange(batch["target"]["extrinsics"], "b v c r1 r2 -> b (v c) r1 r2")[:1],
                            rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)")[:1],
                            rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)")[:1],
                            (256, 256),
                            render_mode='pinhole'
                        )

                log_pack = {
                    'loss': loss_total,
                    'loss_details': {
                        'loss_1': loss_1.detach().data,
                        'loss_2': loss_2.detach().data,
                        },
                    }
                
                log_pack.update({
                        'out_depth_0': output.depth[0][0], 'gt_depth_0': batch['target']['depth'][0][0][0],
                        'out_depth_1': output.depth[0][1], 'gt_depth_1': batch['target']['depth'][0][0][1],
                        'out_depth_2': output.depth[0][2], 'gt_depth_2': batch['target']['depth'][0][0][2],
                        'out_depth_3': output.depth[0][3], 'gt_depth_3': batch['target']['depth'][0][0][3],
                        'out_depth_4': output.depth[0][4], 'gt_depth_4': batch['target']['depth'][0][0][4], 
                        'out_depth_5': output.depth[0][5], 'gt_depth_5': batch['target']['depth'][0][0][5],

                        'input_pano_0': batch['context']['pano_image'][0][0], 'input_pano_1': batch['context']['pano_image'][0][1],
                        'input_depth_0': batch['context']['pano_depth'][0][0], 'input_depth_1': batch['context']['pano_depth'][0][1],
                        'input_mask_0': batch['context']['pano_mask'][0][0], 'input_mask_1': batch['context']['pano_mask'][0][1],
                    })
                
                self.log(log_pack)

                torch.cuda.empty_cache()
            
            self.current_step += 1

            if self.step_tracker is not None:
                self.step_tracker.set_step(self.current_step)

            if self.current_step % self.cfg.checkpointing.every_n_train_steps == 0:
                break

    def depth_to_pts3d(self, depth):
        b,_,h, w = depth.shape
        depth = depth.squeeze(1)
        lat = torch.linspace(-torch.pi/2,torch.pi/2,h)
        lon = torch.linspace(0,2 * torch.pi,w)
        lat, lon = torch.meshgrid(lat, lon)
        lat = lat.unsqueeze(0).expand(b, -1, -1).to(depth.device)
        lon = lon.unsqueeze(0).expand(b, -1, -1).to(depth.device)
        x = depth * torch.cos(lat) * torch.sin(lon)
        y = depth * torch.sin(lat)
        z = depth * torch.cos(lat) * torch.cos(lon)
        return torch.stack([x,y,z,torch.ones_like(x)],dim=-1)
        

    def to_cuda(self,data):
        if type(data)==list:
            results = []
            for i, item in enumerate(data):
                results.append(self.to_cuda(item))
            return results
        elif type(data)==dict:
            results={}
            for k,v in data.items():
                results[k]=self.to_cuda(v)
            return results
        elif type(data).__name__ == "Tensor":
            return data.cuda()
        else:
            return data
    
    def log(self, log_pack):
        self.logger.add_figure('out_depth_0', make_figure(log_pack['out_depth_0'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_0' , make_figure(log_pack['gt_depth_0'].detach().cpu()) , self.current_step)

        self.logger.add_figure('out_depth_1', make_figure(log_pack['out_depth_1'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_1' , make_figure(log_pack['gt_depth_1'].detach().cpu()) , self.current_step)

        self.logger.add_figure('out_depth_2', make_figure(log_pack['out_depth_2'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_2' , make_figure(log_pack['gt_depth_2'].detach().cpu()) , self.current_step)

        self.logger.add_figure('out_depth_3', make_figure(log_pack['out_depth_3'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_3' , make_figure(log_pack['gt_depth_3'].detach().cpu()) , self.current_step)

        self.logger.add_figure('out_depth_4', make_figure(log_pack['out_depth_4'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_4' , make_figure(log_pack['gt_depth_4'].detach().cpu()) , self.current_step)

        self.logger.add_figure('out_depth_5', make_figure(log_pack['out_depth_5'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_5' , make_figure(log_pack['gt_depth_5'].detach().cpu()) , self.current_step)

        self.logger.add_image('input_pano_0', log_pack['input_pano_0'].detach().cpu()            , self.current_step)
        self.logger.add_image('input_pano_1', log_pack['input_pano_1'].detach().cpu()            , self.current_step)

        self.logger.add_figure('input_depth_0', make_figure(log_pack['input_depth_0'].detach().cpu()), self.current_step)
        self.logger.add_figure('input_depth_1', make_figure(log_pack['input_depth_1'].detach().cpu()), self.current_step)

        self.logger.add_figure('input_mask_0', make_figure(log_pack['input_mask_0'].detach().cpu()), self.current_step)
        self.logger.add_figure('input_mask_1', make_figure(log_pack['input_mask_1'].detach().cpu()), self.current_step)

        self.logger.add_scalar('loss_total', log_pack['loss'].detach().cpu().data, self.current_step)
        for k,v in log_pack['loss_details'].items():
            self.logger.add_scalar(k, v, self.current_step)

    def save_ckpt(self, abs_rel):
        ckpt = {
            'model'  : self.encoder.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'step'    : self.current_step}
        torch.save(ckpt, os.path.join(self.cfg.log_path, f'step_{self.current_step}_abs_rel_{abs_rel}.pth'))
        print(' -> saved checkpoint')

    def eval(self):
        self.encoder.eval()
        local_metrics = {'abs_diff': 0, 'abs_rel': 0, 'rmse':0, 'a25': 0}
        valid_data_count = 0
        for batch in tqdm(self.test_dataloader, desc='eval', disable=(not self.gpu_id == 0)):
            batch = self.to_cuda(batch)
            batch = self.data_shim(batch)

            with torch.no_grad():
                res1, res2, dec1, dec2, shape1, shape2 = self.encoder(batch["context"], mode='test')
                mean3d = torch.cat([rearrange(res1['pts3d'], 'b h w c -> b (h w) c'), rearrange(res2['pts3d'], 'b h w c -> b (h w) c')], dim=1).cuda()

                color_1 = rearrange(batch["context"]["pano_image"][:,0], 'b c h w -> b (h w) c')
                color_2 = rearrange(batch["context"]["pano_image"][:,1], 'b c h w -> b (h w) c')
                sh = torch.cat([color_1, color_2], dim=1).cuda().unsqueeze(-1) * 0.5 + 0.5
                invalid_pixels = sh.squeeze().sum(-1) < 0.1

                eps = 1e-8
                rotations = torch.ones(1, mean3d.shape[1], 4).cuda()
                rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
                scales = torch.ones(1, mean3d.shape[1], 3).cuda()
                scales = 0.005 * F.softplus(scales)
                scales = scales.clamp_max(0.3)

                opacities = torch.ones(1, mean3d.shape[1]).cuda().sigmoid()
                opacities[0, invalid_pixels] = 0.0
                covariances = build_covariance(scales, rotations)
                gaussians = Gaussians(means=mean3d, covariances=covariances, opacities=opacities, rotations=rotations, scales=scales, harmonics=sh)
                output = self.decoder.forward(
                        gaussians,
                        rearrange(batch["target"]["extrinsics"], "b v c r1 r2 -> b (v c) r1 r2"),
                        rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)"),
                        rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)") ,
                        (256, 256),
                        render_mode='pinhole'
                    )
                
                gt_depth = rearrange(batch["target"]["depth"][:, :, 1:], 'b n v h w c -> (b n v) c h w') # drop top view due to empty depth (cause NaNs)
                pred_depth = rearrange(rearrange(output.depth, 'b (n v) c h w -> b n v c h w', v=6)[:, :, 1:], 'b n v c h w -> (b n v) c h w')

                valid_mask_b = (gt_depth > 0.1) * (pred_depth > 0.1)
                valid_mask_batch = torch.any(rearrange(gt_depth >= 0.1, 'n c h w -> n (c h w)'), dim=-1) 
                assert (valid_mask_b).any(), 'no valid depth in batch'

                depth_metric = compute_depth_metrics_batched(gt_depth.flatten(start_dim=1).float(),
                                                             pred_depth.flatten(start_dim=1).float(), 
                                                             valid_mask_b.flatten(start_dim=1), 
                                                             mult_a=True)
                
                if any(torch.isnan(value).any() for value in depth_metric.values()):
                    continue

                for key in depth_metric.keys():
                    if key in ["abs_diff", "abs_rel", "rmse", "a25"]: # only evaluate with three metrics
                        depth_metric[key][~valid_mask_batch] = 0
                        # non zero
                        batch_valid_cnt = valid_mask_batch.count_nonzero()
                        assert batch_valid_cnt > 0
                        local_metrics[key] += (depth_metric[key].sum() / batch_valid_cnt).item()
                valid_data_count += 1

        for k in local_metrics.keys():
            local_metrics[k] /= valid_data_count

        if self.gpu_id == 0:
            metric_line = f'step:{self.current_step}, abs_diff {local_metrics["abs_diff"]}, abs_rel {local_metrics["abs_rel"]}, rmse {local_metrics["rmse"]}, a25 {local_metrics["a25"]} \n'
            for k, v in local_metrics.items():
                self.logger.add_scalar(k, v, self.current_step)
            with open(os.path.join(self.cfg.log_path, f'metricsm.txt'), 'a') as f:
                f.write(metric_line)
            print(metric_line)
        abs_rel = local_metrics["abs_rel"]

        self.encoder.train()

        return abs_rel

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
                
if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load("config/defalts_pretrain.yaml")
    cfg.train_dataset = omegaconf.OmegaConf.load("config/dataset/hm3d.yaml")
    cfg.test_dataset = omegaconf.OmegaConf.load("config/dataset/replica.yaml")
    cfg.croco = omegaconf.OmegaConf.load("config/model/encoder/backbone/croco.yaml")
    cfg.decoder = omegaconf.OmegaConf.load("config/model/decoder/splatting_cuda.yaml")
    ### DDP setup
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

    trainer = Trainer(cfg)
    trainer.train()
    destroy_process_group()