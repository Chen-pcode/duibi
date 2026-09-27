"""EGE-UNet 训练脚本（Kaggle / 本地通用，设备自动检测）。
复用原仓库的模型/损失/数据增强，把 .cuda() 替换为自动设备(cuda/cpu)，
并可通过命令行控制 epoch 数量与数据路径。

用法:
    python train_cpu.py                       # 默认: 60 epoch, 数据 ../data/isic2017
    python train_cpu.py --epochs 300          # Kaggle GPU 上跑官方 300 epoch
    python train_cpu.py --data /kaggle/input/xxx/isic2017
    python train_cpu.py --outdir ./train_output

目录约定(与脚本位置相对):
    <script_dir>/EGE-UNet/            # EGE-UNet 源码(嵌套目录)
    <script_dir>/../data/isic2017     # 默认数据
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
EGE_ROOT = SCRIPT_DIR / "EGE-UNet"          # 嵌套源码目录
sys.path.insert(0, str(EGE_ROOT))

from datasets.dataset import NPY_datasets  # noqa: E402
from models.egeunet import EGEUNet  # noqa: E402
from utils import (get_optimizer, get_scheduler, GT_BceDiceLoss, myNormalize,  # noqa: E402
                   myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip,
                   myRandomRotation, myResize, set_seed)


class Cfg:
    datasets = 'isic17'
    data_path = None
    network = 'egeunet'
    model_config = dict(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True)
    criterion = None
    input_size_h = 256
    input_size_w = 256
    seed = 42
    batch_size = 8
    num_workers = 0
    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False
    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 0.00001
    last_epoch = -1
    epochs = 60
    threshold = 0.5
    outdir = None


def build_transforms(cfg, train):
    if train:
        return transforms.Compose([
            myNormalize('isic17', train=True),
            myToTensor(),
            myRandomHorizontalFlip(p=0.5),
            myRandomVerticalFlip(p=0.5),
            myRandomRotation(p=0.5, degree=[0, 360]),
            myResize(cfg.input_size_h, cfg.input_size_w),
        ])
    else:
        return transforms.Compose([
            myNormalize('isic17', train=False),
            myToTensor(),
            myResize(cfg.input_size_h, cfg.input_size_w),
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", default=None, help="数据根(含 train/val), 默认 <script>../data/isic2017")
    ap.add_argument("--outdir", default=None, help="输出目录, 默认 <script>/train_output")
    args = ap.parse_args()

    cfg = Cfg()
    cfg.criterion = GT_BceDiceLoss(wb=1, wd=1)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch
    cfg.data_path = Path(args.data) if args.data else SCRIPT_DIR.parent / "data" / "isic2017"
    cfg.outdir = Path(args.outdir) if args.outdir else SCRIPT_DIR / "train_output"

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(6)
    print(f"[device] {device}  (cuda_available={torch.cuda.is_available()})", flush=True)

    cfg.data_path = str(cfg.data_path)
    os.makedirs(cfg.outdir, exist_ok=True)

    cfg.train_transformer = build_transforms(cfg, True)
    cfg.test_transformer = build_transforms(cfg, False)

    print('#---- dataset ----#', flush=True)
    train_ds = NPY_datasets(cfg.data_path, cfg, train=True)
    val_ds = NPY_datasets(cfg.data_path, cfg, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    print('#---- model ----#', flush=True)
    mc = cfg.model_config
    model = EGEUNet(num_classes=mc['num_classes'], input_channels=mc['input_channels'],
                    c_list=mc['c_list'], bridge=mc['bridge'], gt_ds=mc['gt_ds']).to(device)
    criterion = cfg.criterion
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    min_loss = 999.0
    best_state = None
    step = 0
    print(f'#---- training {cfg.epochs} epochs ----#', flush=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for it, (img, msk) in enumerate(train_loader):
            img, msk = img.to(device).float(), msk.to(device).float()
            optimizer.zero_grad()
            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if it % 200 == 0:
                print(f'  epoch {epoch} iter {it} loss {np.mean(losses[-50:]):.4f} '
                      f'lr {optimizer.param_groups[0]["lr"]:.6f}', flush=True)
        scheduler.step()

        model.eval()
        vlosses = []
        with torch.no_grad():
            for img, msk in val_loader:
                img, msk = img.to(device).float(), msk.to(device).float()
                gt_pre, out = model(img)
                vlosses.append(criterion(gt_pre, out, msk).item())
        vloss = float(np.mean(vlosses))
        dt = time.time() - t0
        print(f'EPOCH {epoch}: train_loss={np.mean(losses):.4f} val_loss={vloss:.4f} time={dt:.0f}s', flush=True)
        if vloss < min_loss:
            min_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, str(cfg.outdir / 'best.pth'))
            print(f'  -> new best val_loss {vloss:.4f}', flush=True)

    if best_state is not None:
        torch.save(best_state, str(cfg.outdir / 'egeunet_best.pth'))
    torch.save(model.state_dict(), str(cfg.outdir / 'egeunet_last.pth'))
    print(f'#---- done. min_val_loss={min_loss:.4f}. saved to {cfg.outdir} ----#', flush=True)


if __name__ == '__main__':
    main()
