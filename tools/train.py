import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import sys
import time
import random
import argparse
import warnings
import yaml
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info

sys.path.append('.')
from utils import Adder, Timer, DualOutput
import custom_utils
from custom_utils.dataset_utils import DataLoaderX
from custom_utils.data_loaders.dehaze_datasets import DeblurDataset
from custom_utils.warmup_scheduler.scheduler import GradualWarmupScheduler
from model.loss import FFTloss
from model.model_builder import DehazeNet

# ------------------------------
# 工具函数
# ------------------------------
def setup_seeds(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def is_main_process(rank):
    return rank == 0

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_sd[k[len('module.'):]] = v
        else:
            new_sd[k] = v
    return new_sd

def parse_args():
    parser = argparse.ArgumentParser(description='FFTRWKV — DDP Training')
    parser.add_argument('--mode', type=str, default='ITS') 
    parser.add_argument('--yml_path', type=str, default='./configs/ITS_hazy.yaml')
    parser.add_argument('--pretrain_weights', type=str, default='')
    parser.add_argument('--channel', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr_init', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fft_weight', type=float, default=0.15)
    parser.add_argument('--clip_grad', type=float, default=0.001)
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--init_port', type=str, default='23456', help='Master node port')
    parser.add_argument('--gpus', type=str, default='0,1')
    return parser.parse_args()


# ------------------------------
# 训练 worker
# ------------------------------
def train_worker(rank, world_size, args, gpu_list):
    local_rank = gpu_list[rank]
    torch.cuda.set_device(local_rank)
    setup_seeds(args.seed, rank)

    if world_size > 1:
        init_port = args.init_port
        dist.init_process_group(
            backend=args.backend,
            init_method=f'tcp://127.0.0.1:{init_port}',
            rank=rank,
            world_size=world_size
        )

    # ------------------------------
    # 配置与路径
    # ------------------------------
    with open(args.yml_path, 'r') as f:
        config = yaml.safe_load(f)
    Train = config['TRAINING']

    mode = args.mode
    save_root = os.path.join(Train['SAVE_DIR'], mode)
    model_dir = os.path.join(save_root, 'models')
    log_dir = os.path.join(save_root, 'log')
    if is_main_process(rank):
        custom_utils.mkdir(model_dir)
        custom_utils.mkdir(log_dir)
        log_file_path = os.path.join(model_dir, "training_log.txt")
        sys.stdout = DualOutput(log_file_path)
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')
    else:
        writer = None

    # ------------------------------
    # 模型初始化 & 移动到 GPU
    # ------------------------------
    model = DehazeNet(dim=args.channel).to(local_rank)

    # ------------------------------
    # 优化器 & Scheduler
    # ------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs - warmup_epochs, eta_min=float(args.lr_min)
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    scheduler.step()

    # ------------------------------
    # 加载预训练 & 恢复训练
    # ------------------------------
    start_epoch = 1
    # 恢复训练
    if Train.get('RESUME', False):
        chk = custom_utils.get_last_path(model_dir, '_latest.pth')
        if chk is not None:
            checkpoint = torch.load(chk, map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 将 optimizer state 张量搬到 GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(local_rank)
            start_epoch = checkpoint['epoch'] + 1
            for _ in range(1, start_epoch):
                scheduler.step()

    # 加载预训练权重（可选）
    if args.pretrain_weights:
        ckpt = torch.load(args.pretrain_weights, map_location=f'cuda:{local_rank}')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        if is_main_process(rank):
            print(f"Loaded pretrained weights from {args.pretrain_weights}")

    # ------------------------------
    # DDP 包装
    # ------------------------------
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ------------------------------
    # 数据集 & Dataloader
    # ------------------------------
    patch_size = Train['PATCH_SIZE']
    train_dataset = DeblurDataset(Train['TRAIN_DIR'], patch_size)
    val_dataset = DeblurDataset(Train['VAL_DIR'], is_valid=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoaderX(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=8, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    max_iter = len(train_loader)

    # ------------------------------
    # 打印训练信息 (rank 0)
    # ------------------------------
    if is_main_process(rank):
        # 计算参数量和FLOPs
        flops, params = get_model_complexity_info(
            model.module if isinstance(model, DDP) else model,
            (3, 256, 256),
            as_strings=False,
            print_per_layer_stat=False
        )
        params_m = params / 1e6
        flops_g = flops / 1e9
        iters_per_epoch = len(train_loader)

        print(f"==> Training details:")
        print(f"------------------------------------------------------------------")
        print(f"Restoration mode:   {args.mode}")
        print(f"Train patch size:   {Train['PATCH_SIZE']}x{Train['PATCH_SIZE']}")
        print(f"Model parameters:   {params_m:.2f} M")
        print(f"Model FLOPs:        {flops_g:.2f} G")
        print(f"Start/End epochs:   {start_epoch} ~ {args.epochs}")
        print(f"Batch size:         {args.batch_size * world_size}")
        print(f"Learning rate:      {args.lr_init}")
        print(f"FFT loss weight:    {args.fft_weight}")
        print(f"Grad clip value:    {args.clip_grad}")
        print(f"Iters per epoch:    {iters_per_epoch}")
        print(f"GPUs used:          {args.gpus}")
        print(f"------------------------------------------------------------------")
        if Train.get('RESUME', False):
            print(f"Resuming from epoch {start_epoch}, lr={scheduler.get_lr()[0]:.6f}")

    # ------------------------------
    # 损失函数
    # ------------------------------
    l1_loss = nn.L1Loss().to(local_rank)
    fft_loss = FFTloss().to(local_rank)

    best_psnr, best_ssim = 0.0, 0.0
    total_start_time = time.time()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    # ------------------------------
    # 训练循环
    # ------------------------------
    for epoch_idx in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)

        model.train()
        epoch_timer.tic()
        iter_timer.tic()

        for iter_idx, batch_data in enumerate(train_loader):
            inp, tar = batch_data
            inp = inp.to(local_rank, non_blocking=True)
            tar = tar.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(inp)
            l_cont = l1_loss(out, tar)
            l_fft = fft_loss(out, tar)
            loss = l_cont + args.fft_weight * l_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)  #1.0
            optimizer.step()

            # 多卡同步 loss
            if world_size > 1:
                metrics = torch.tensor([loss.item(), l_cont.item(), l_fft.item()], device=local_rank)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics /= world_size
                l_cont_avg, l_fft_avg = metrics[1].item(), metrics[2].item()
            else:
                l_cont_avg, l_fft_avg = l_cont.item(), l_fft.item()

            iter_pixel_adder(l_cont_avg)
            iter_fft_adder(l_fft_avg)
            epoch_pixel_adder(l_cont_avg)
            epoch_fft_adder(l_fft_avg)

            if is_main_process(rank) and ((iter_idx + 1) % args.print_freq == 0):
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Pixel Loss: %7.4f FFT Loss: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter,
                    optimizer.param_groups[0]['lr'], iter_pixel_adder.average(), iter_fft_adder.average()))
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_timer.tic()

        # epoch结束记录
        if is_main_process(rank):
            print(f"Epoch: {epoch_idx}\nElapsed Time: {epoch_timer.toc():.2f}s "
                    f"Epoch Pixel Loss: {epoch_pixel_adder.average():.4f} "
                    f"Epoch FFT Loss: {epoch_fft_adder.average():.4f}")
            writer.add_scalar('Epoch/Pixel_Loss', epoch_pixel_adder.average(), epoch_idx)
            writer.add_scalar('Epoch/FFT_Loss', epoch_fft_adder.average(), epoch_idx)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch_idx)
            epoch_pixel_adder.reset()
            epoch_fft_adder.reset()

        # 验证
        if is_main_process(rank) and (epoch_idx % Train['VAL_AFTER_EVERY'] == 0):
            model.eval()
            with torch.no_grad():
                psnr_adder = Adder()
                ssim_adder = Adder()
                for inp, tar in val_loader:
                    inp, tar = inp.to(local_rank), tar.to(local_rank)
                    h, w = inp.shape[2], inp.shape[3]
                    factor = 8
                    H = ((h + factor) // factor) * factor
                    W = ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    inp_pad = F.pad(inp, (0, padw, 0, padh), 'reflect')
                    out = model(inp_pad)
                    out = out[:, :, :tar.shape[2], :tar.shape[3]]
                    out = torch.clamp(out, 0, 1)
                    psnr_adder(custom_utils.torchPSNR(out, tar))
                    ssim_adder(custom_utils.torchSSIM(out, tar, H, W))
            psnr = psnr_adder.average()
            ssim = ssim_adder.average()
            writer.add_scalar('val/PSNR', psnr, epoch_idx)
            writer.add_scalar('val/SSIM', ssim, epoch_idx)

            # 保存最优模型
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                }, os.path.join(model_dir, f'{mode}_bestPSNR.pth'))

            if ssim > best_ssim:
                best_ssim = ssim
                torch.save({
                    'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                }, os.path.join(model_dir, f'{mode}_bestSSIM.pth'))

            print(f"PSNR: {psnr:.4f} (Best {best_psnr:.4f}) SSIM: {ssim:.4f} (Best {best_ssim:.4f})")

            # 保存周期模型
            if epoch_idx % Train['SAVE_AFTER_EVERY'] == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f'{mode}_epoch_{epoch_idx}.pth'))
                

        # 保存最新模型
        if is_main_process(rank):
            torch.save({
                'epoch': epoch_idx,
                'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, f'{mode}_latest.pth'))

        scheduler.step()

    if is_main_process(rank):
        total_time = time.time() - total_start_time
        print(f"Total training time: {total_time/3600:.2f} hours")
        writer.close()

    if world_size > 1:
        dist.destroy_process_group()


# ------------------------------
# Entry
# ------------------------------
if __name__ == '__main__':
    args = parse_args()
    gpu_list = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_list)
    mp.spawn(train_worker, nprocs=world_size, args=(world_size, args, gpu_list))
