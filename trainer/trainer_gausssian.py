import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from einops import rearrange, repeat
import omegaconf
import lpips as lpips_lib
from tqdm import tqdm
import cv2
import random
import torch
from torch import Generator, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.autograd import profiler
from ema_pytorch import EMA
import composer.functional as cf
from collections import OrderedDict

from dataset.dataset_hm3d import DatasetHM3D
from dataset.view_sampler import get_view_sampler
from model.encoder.gs_encoder import EncoderNoPoSplat
from model.decoder import get_decoder
from misc.step_tracker import StepTracker
from dataset.database import get_data_shim
from utils.loss import l1_loss, l2_loss, lpips_loss, convert_to_buffer
from utils.log import unnorm_img, make_figure
from utils.metric import Pinhole_Evaluator, compute_depth_metrics_batched
from utils.cubemap_process import spherical_to_cubemap, bilinear_interpolate

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

class Trainer:
    def __init__(self, cfg):
        ### cfg
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.cfg = cfg
        self.cfg.gpu_id = self.gpu_id

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
        
        self.train_dataset = DatasetHM3D(self.cfg.train_dataset, stage, view_sampler, resized_shape=(512, 1024), data_augmentation=self.cfg.train.data_augmentation)
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
        self.test_dataset = DatasetHM3D(self.cfg.test_dataset, stage, view_sampler, resized_shape=(512, 1024))
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

        print(' -> dataset initialized on rank', self.gpu_id)
        
        ### network
        self.encoder = EncoderNoPoSplat(cfg.encoder).to(self.gpu_id)
        cf.apply_low_precision_layernorm(self.encoder.backbone, precision='amp')

        backbone_ckpt = torch.load(cfg.weights_path, map_location='cpu')['model']

        if cfg.encoder.circular_pad:
            backbone_ckpt = OrderedDict(
                    (k.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias'), v)
                    if ('downstream_head' in k ) and not ('.0.1' in k or '.1.1' in k) and not ('downstream_head2.dpt.mlp.' in k) else (k, v)
                    for k, v in backbone_ckpt.items()
                )

        self.encoder.backbone.load_state_dict(backbone_ckpt, strict=False) 
        print(f"Loaded weights from {cfg.weights_path}")

        self.data_shim = get_data_shim(self.encoder)

        if self.cfg.use_ema:
            self.ema_encoder = EMA(self.encoder, beta=self.cfg.beta, update_every=self.cfg.update_every, update_after_step=self.cfg.update_after_steps)
            self.ema_encoder.ema_model.to('cpu')
        
        self.encoder = DDP(self.encoder,device_ids=[self.gpu_id], find_unused_parameters=True)
        self.decoder = get_decoder(cfg.decoder).to(self.gpu_id)

        ### LPIPS loss
        self.lpips = lpips_lib.LPIPS(net='vgg').to(self.gpu_id).eval()
        convert_to_buffer(self.lpips, persistent=False)

        self.evaluator = Pinhole_Evaluator(lpips_model=self.lpips)

        print(' -> network initialized on rank', self.gpu_id)

        ### opt
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        self.head_optimizer = torch.optim.AdamW(new_params, lr=self.cfg.optimizer.lr, weight_decay=0.05, betas=(0.9, 0.95))
        self.backbone_optimizer = torch.optim.AdamW(pretrained_params, lr=self.cfg.optimizer.lr * self.cfg.optimizer.backbone_lr_multiplier, weight_decay=0.05, betas=(0.9, 0.95))

        warm_up_steps = self.cfg.optimizer.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            self.head_optimizer,
            1 /  warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        head_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.head_optimizer, T_max=self.cfg.train.total_epochs*self.cfg.checkpointing.every_n_train_steps, eta_min=self.cfg.optimizer.lr * self.cfg.optimizer.lr_decay_rate)
        self.head_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.head_optimizer, schedulers=[warm_up, head_lr_scheduler], milestones=[warm_up_steps])

        backbone_warm_up = torch.optim.lr_scheduler.LinearLR(
            self.backbone_optimizer ,
            1 /  warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        backbone_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.backbone_optimizer, T_max=self.cfg.train.total_epochs*self.cfg.checkpointing.every_n_train_steps, eta_min=self.cfg.optimizer.lr * self.cfg.optimizer.backbone_lr_multiplier * self.cfg.optimizer.lr_decay_rate)
        self.backbone_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.backbone_optimizer, schedulers=[backbone_warm_up, backbone_lr_scheduler], milestones=[warm_up_steps])

        self.scaler = GradScaler()

        print(' -> optimizer initialized on rank', self.gpu_id)
        
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
            self.head_optimizer.load_state_dict(ckpt['head_optimizer'])
            self.head_lr_scheduler.load_state_dict(ckpt['head_lr_scheduler'])
            self.backbone_optimizer.load_state_dict(ckpt['backbone_optimizer'])
            self.backbone_lr_scheduler.load_state_dict(ckpt['backbone_lr_scheduler'])
            self.current_step = ckpt['step']
            self.current_epoch = self.current_step // self.cfg.checkpointing.every_n_train_steps
            if self.cfg.use_ema:
                self.ema_encoder.ema_model.load_state_dict(ckpt['ema_model'])
                self.ema_encoder.step = torch.tensor([self.current_step])
            print(' -> resume from', self.cfg.resume)

    def train(self):
        for epoch in range(self.current_epoch, self.cfg.train.total_epochs):
            self.current_epoch = epoch
            print(f' -> start training epoch {epoch} on GPU_id={self.gpu_id}')
            self.train_one_epoch()

            torch.cuda.empty_cache()
            psnr = self.eval()
            if self.gpu_id == 0:
                self.save_ckpt(psnr)
            torch.cuda.empty_cache()
    
    def train_one_epoch(self):
        for batch in tqdm(self.train_dataloader, disable=(not self.gpu_id == 0), desc=f'epoch_{self.current_epoch}'):
            batch = self.data_shim(batch)
            batch = self.to_cuda(batch)

            with autocast('cuda', dtype=torch.float16):
                gaussians = self.encoder(batch["context"], self.current_step, mode='train')

            output = self.decoder.forward(
                        gaussians,
                        rearrange(batch["target"]["extrinsics"], "b v c r1 r2 -> b (v c) r1 r2"),
                        rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)"),
                        rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)") ,
                        (256, 256),
                        render_mode='pinhole'
                    )
            
            loss_lpips = lpips_loss(output.color, rearrange(batch["target"]["image"], 'b n v c h w -> b (n v) c h w'), self.lpips) * self.cfg.train.lambda_lpips 
            loss_rgb = l2_loss(output.color, rearrange(batch["target"]["image"], 'b n v c h w -> b (n v) c h w')) * self.cfg.train.lambda_rgb 
            loss_depth = l2_loss(output.depth, rearrange(batch["target"]["depth"], 'b n v h w c -> b (n v) c h w')) * self.cfg.train.lambda_depth

            loss_total = (loss_lpips + loss_rgb + loss_depth)

            self.scaler.scale(loss_total).backward()

            self.scaler.unscale_(self.head_optimizer)
            self.scaler.unscale_(self.backbone_optimizer)

            torch.nn.utils.clip_grad_norm_(self.encoder.module.parameters(), max_norm=10.0)

            self.scaler.step(self.head_optimizer)
            self.scaler.step(self.backbone_optimizer)
            self.head_lr_scheduler.step()
            self.backbone_lr_scheduler.step()
            self.scaler.update()

            self.head_optimizer.zero_grad()
            self.backbone_optimizer.zero_grad()
            
            if self.current_step % self.cfg.train.print_log_every_n_steps == 0:
                if self.gpu_id == 0:
                    log_pack = {}

                    log_pack.update({
                        'out_0': output.color[0][0], 'out_depth_0': output.depth[0][0],
                        'gt_0': batch['target']['image'][0][0][0], 'gt_depth_0': batch['target']['depth'][0][0][0],
                        'out_1': output.color[0][1], 'out_depth_1': output.depth[0][1],
                        'gt_1': batch['target']['image'][0][0][1], 'gt_depth_1': batch['target']['depth'][0][0][1],
                        'out_2': output.color[0][2], 'out_depth_2': output.depth[0][2],
                        'gt_2': batch['target']['image'][0][0][2], 'gt_depth_2': batch['target']['depth'][0][0][2],
                        'out_3': output.color[0][3], 'out_depth_3': output.depth[0][3],
                        'gt_3': batch['target']['image'][0][0][3], 'gt_depth_3': batch['target']['depth'][0][0][3],
                        'out_4': output.color[0][4], 'out_depth_4': output.depth[0][4],
                        'gt_4': batch['target']['image'][0][0][4], 'gt_depth_4': batch['target']['depth'][0][0][4], 
                        'out_5': output.color[0][5], 'out_depth_5': output.depth[0][5],
                        'gt_5': batch['target']['image'][0][0][5], 'gt_depth_5': batch['target']['depth'][0][0][5],

                        'input_pano_0': batch['context']['pano_image'][0][0], 'input_depth_0': batch['context']['pano_depth'][0][0],
                        'input_pano_1': batch['context']['pano_image'][0][1], 'input_depth_1': batch['context']['pano_depth'][0][1],

                        'loss': loss_total,
                        'loss_details': {
                            'lpips_loss': loss_lpips.detach().data,
                            'image_loss': loss_rgb.detach().data,
                            'depth_loss': loss_depth.detach().data,
                            }}
                        )
                    
                    self.log(log_pack)
            if self.current_step % self.cfg.train.empty_cache_every_n_steps == 0:
                torch.cuda.empty_cache()

            self.current_step += 1

            if self.cfg.use_ema:
                self.ema_encoder.update()

            if self.step_tracker is not None:
                self.step_tracker.set_step(self.current_step)
            
            if self.current_step % self.cfg.checkpointing.every_n_train_steps == 0:
                break
            
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

        self.logger.add_image('out_0'       , log_pack['out_0'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_0'        , log_pack['gt_0'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_0', make_figure(log_pack['out_depth_0'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_0' , make_figure(log_pack['gt_depth_0'].detach().cpu()) , self.current_step)

        self.logger.add_image('out_1'       , log_pack['out_1'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_1'        , log_pack['gt_1'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_1', make_figure(log_pack['out_depth_1'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_1' , make_figure(log_pack['gt_depth_1'].detach().cpu()) , self.current_step)

        self.logger.add_image('out_2'       , log_pack['out_2'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_2'        , log_pack['gt_2'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_2', make_figure(log_pack['out_depth_2'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_2' , make_figure(log_pack['gt_depth_2'].detach().cpu()) , self.current_step)

        self.logger.add_image('out_3'       , log_pack['out_3'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_3'        , log_pack['gt_3'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_3', make_figure(log_pack['out_depth_3'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_3' , make_figure(log_pack['gt_depth_3'].detach().cpu()) , self.current_step)

        self.logger.add_image('out_4'       , log_pack['out_4'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_4'        , log_pack['gt_4'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_4', make_figure(log_pack['out_depth_4'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_4' , make_figure(log_pack['gt_depth_4'].detach().cpu()) , self.current_step)

        self.logger.add_image('out_5'       , log_pack['out_5'].detach().cpu()                   , self.current_step)
        self.logger.add_image('gt_5'        , log_pack['gt_5'].detach().cpu()                    , self.current_step)
        self.logger.add_figure('out_depth_5', make_figure(log_pack['out_depth_5'].detach().cpu()), self.current_step)
        self.logger.add_figure('gt_depth_5' , make_figure(log_pack['gt_depth_5'].detach().cpu()) , self.current_step)

        self.logger.add_image('input_pano_0', log_pack['input_pano_0'].detach().cpu()            , self.current_step)
        self.logger.add_figure('input_depth_0', make_figure(log_pack['input_depth_0'].detach().cpu()), self.current_step)
        self.logger.add_image('input_pano_1', log_pack['input_pano_1'].detach().cpu()            , self.current_step)
        self.logger.add_figure('input_depth_1', make_figure(log_pack['input_depth_1'].detach().cpu()), self.current_step)

        self.logger.add_scalar('loss_total', log_pack['loss'].detach().cpu().data, self.current_step)
        for k,v in log_pack['loss_details'].items():
            self.logger.add_scalar(k, v, self.current_step)

    def save_ckpt(self, psnr):
        ckpt = {
            'model'  : self.encoder.module.state_dict(),
            'head_optimizer': self.head_optimizer.state_dict(),
            'backbone_optimizer': self.backbone_optimizer.state_dict(),
            'head_lr_scheduler': self.head_lr_scheduler.state_dict(),
            'backbone_lr_scheduler': self.backbone_lr_scheduler.state_dict(),
            'step'    : self.current_step}
        if self.cfg.use_ema:
            ckpt['ema_model'] = self.ema_encoder.ema_model.state_dict()
        torch.save(ckpt, os.path.join(self.cfg.log_path, f"step_{self.current_step}_psnr_{psnr}.pth"))
        print(' -> saved checkpoint')

    def eval(self):
        if self.cfg.use_ema:
            self.ema_encoder.ema_model.eval()
            self.ema_encoder.ema_model.to(self.gpu_id)
        else:
            self.encoder.eval()
        local_metrics = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'abs_diff': 0, 'abs_rel': 0, 'rmse':0, 'a25': 0}
        valid_data_count = 0
        for batch in tqdm(self.test_dataloader, desc='eval', disable=(not self.gpu_id == 0)):
            batch = self.data_shim(batch)
            batch = self.to_cuda(batch)

            with torch.no_grad():
                if self.cfg.use_ema:
                    gaussians = self.ema_encoder.ema_model(batch["context"], self.current_step)
                else:
                    gaussians = self.encoder(batch["context"], self.current_step, mode='test')
                    
                output = self.decoder.forward(
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

                rgb_metric = self.evaluator.eval_metrics_img(gt_image, pred_image)
                depth_metric = compute_depth_metrics_batched(gt_depth.flatten(start_dim=1).float(),
                                                             pred_depth.flatten(start_dim=1).float(), 
                                                             valid_mask_b.flatten(start_dim=1), 
                                                             mult_a=True)
                
                for key in depth_metric.keys():
                    if key in ["abs_diff", "abs_rel", "rmse", "a25"]: # only evaluate with three metrics
                        depth_metric[key][~valid_mask_batch] = 0
                        # non zero
                        batch_valid_cnt = valid_mask_batch.count_nonzero()
                        assert batch_valid_cnt > 0
                        local_metrics[key] += (depth_metric[key].sum() / batch_valid_cnt).item()

                for k, v in rgb_metric.items():
                    local_metrics[k] += rgb_metric[k]
                
                valid_data_count += 1

        for k in local_metrics.keys():
            local_metrics[k] /= valid_data_count

        if self.gpu_id == 0:
            metric_line = f'step:{self.current_step}, psnr {local_metrics["psnr"]}, ssim {local_metrics["ssim"]}, lpips {local_metrics["lpips"]}, abs_diff {local_metrics["abs_diff"]}, abs_rel {local_metrics["abs_rel"]}, rmse {local_metrics["rmse"]}, a25 {local_metrics["a25"]} \n'
            for k, v in local_metrics.items():
                self.logger.add_scalar(k, v, self.current_step)
            with open(os.path.join(self.cfg.log_path, f'metricsm.txt'), 'a') as f:
                f.write(metric_line)
            print(metric_line)
        psnr = local_metrics["psnr"]

        if self.cfg.use_ema:
            self.ema_encoder.ema_model.to('cpu')
        else:
            self.encoder.train()

        return psnr
                
if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load("config/defalts.yaml")
    cfg.train_dataset = omegaconf.OmegaConf.load("config/dataset/hm3d.yaml")
    cfg.test_dataset = omegaconf.OmegaConf.load("config/dataset/replica.yaml")
    cfg.encoder = omegaconf.OmegaConf.load("config/model/encoder/noposplat.yaml")
    cfg.encoder.croco = omegaconf.OmegaConf.load("config/model/encoder/backbone/croco.yaml")
    cfg.decoder = omegaconf.OmegaConf.load("config/model/decoder/splatting_cuda.yaml")
    ### DDP setup
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend='nccl')

    trainer = Trainer(cfg)
    trainer.train()
    destroy_process_group()
