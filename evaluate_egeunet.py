"""EGE-UNet 评估运行器。

直接用 evaluation/ 文件夹里的指标实现（metrics.py: segmentation_metrics/aggregate_metrics；
profiling.py: profile_model）来评估一个已训练好的 EGE-UNet 模型，
在 isic2017 / isic2018 / ph2 三个测试集上输出 per-image 与汇总指标。

用法:
    python evaluate_egeunet.py --checkpoint <EGE-UNet权重.pth> \
        --evalData <数据根目录> --out <输出目录> [--profile] [--savePred] [--viz]

    # 例（Kaggle）:
    python evaluate_egeunet.py \
        --checkpoint /kaggle/working/EGE-UNet/train_output/egeunet_best.pth \
        --evalData /kaggle/input/datasets/nero20260505/data-20260925 \
        --out /kaggle/working/results \
        --profile --savePred --viz

    # --savePred: 保存预测二值掩膜 PNG 到 <out>/predictions/<domain>/<id>.png
    # --viz:      保存 原图/真值/预测 三连对比图到 <out>/visuals/<domain>/<id>.png

数据根目录(--evalData)的结构与附件 data 文件夹一致:
    <evalData>/isic2017/{train,val,test}/{images,masks}
    <evalData>/isic2018/{train,val,test}/{images,masks}
    <evalData>/ph2/test/{images,masks}

checkpoint 支持两种格式:
    1. 官方 train.py 保存的 best-epochX-lossY.pth / latest.pth（纯 state_dict，或含 model_state_dict 键的 dict）
    2. 本仓库 train_cpu.py 保存的 egeunet_best.pth / egeunet_last.pth

说明:
    - 预处理与官方 EGE-UNet 测试一致: 按 isic17 测试统计量(mean=148.429,std=25.748)做 min-max 到 0-255,
      resize 到 256x256。跨域(isic2018/ph2)复用同一归一化, 保持输入分布一致。
    - 模型 gt_ds=True, forward 返回 (gt_pre_tuple, out), 取 out 作为最终概率图, 阈值 0.5 二值化。
    - 指标均为 per-image 计算后 macro 平均 (与 evaluation/metrics.py 约定一致)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# 导入 evaluation 文件夹的指标实现
EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
sys.path.insert(0, str(EVAL_DIR))
from metrics import segmentation_metrics, aggregate_metrics  # noqa: E402
from profiling import profile_model  # noqa: E402

# EGE-UNet 模型源码
EGE_ROOT = Path(__file__).resolve().parent / "EGE-UNet" / "EGE-UNet"
sys.path.insert(0, str(EGE_ROOT))
from models.egeunet import EGEUNet  # noqa: E402

IMAGE_SIZE = 256
THRESHOLD = 0.5
# isic17 测试统计量（官方 test_transformer 使用 train=False 的统计量）
MEAN, STD = 148.429, 25.748

DOMAINS = ("isic2017", "isic2018", "ph2")


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """EGE-UNet myNormalize(train=False) 对 isic17 的处理: 减均值除标准差 -> min-max 到 0-255。"""
    x = (img - MEAN) / STD
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return ((x - lo) / (hi - lo)) * 255.0


def load_sample(img_path: Path, msk_path: Path, device):
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    msk = np.array(Image.open(msk_path).convert("L")).astype(np.float32) / 255.0

    img = normalize_minmax(img)
    img_t = torch.from_numpy(img).permute(2, 0, 1)          # [3,H,W]
    msk_t = torch.from_numpy(msk).unsqueeze(0)              # [1,H,W]

    img_t = TF.resize(img_t, [IMAGE_SIZE, IMAGE_SIZE], TF.InterpolationMode.BILINEAR)
    msk_t = TF.resize(msk_t, [IMAGE_SIZE, IMAGE_SIZE], TF.InterpolationMode.NEAREST)
    return img_t.to(device), msk_t


def find_mask(mask_dir: Path, img_name: str):
    """按多种命名规则匹配图像对应的掩膜文件。

    规则:
      1) 同名（ph2: 图 IMD002.png 与掩膜 IMD002.png 同名）
      2) 同主名+.png
      3) ISIC 惯例: <图主名>_segmentation.png（isic2017/2018: 图 ISIC_xxxx.jpg -> 掩膜 ISIC_xxxx_segmentation.png）
    """
    img_stem = Path(img_name).stem
    for candidate in (mask_dir / img_name,
                      mask_dir / f"{img_stem}.png",
                      mask_dir / f"{img_stem}_segmentation.png"):
        if candidate.exists():
            return candidate
    for mp in mask_dir.glob("*"):
        if mp.name.lower().startswith(img_stem.lower()):
            return mp
    return None


def save_pred_mask(pred: np.ndarray, out_dir: Path, domain: str, name: str):
    """保存预测二值掩膜 PNG（0/255）。name 可为带扩展名的文件名，自动去扩展名。"""
    d = out_dir / "predictions" / domain
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    Image.fromarray(pred.astype(np.uint8) * 255).save(d / f"{stem}.png")


def save_visual(img_path: Path, msk_path: Path, pred: np.ndarray,
                out_dir: Path, domain: str, name: str):
    """保存 原图 / 真值 / 预测 三连对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    gt = Image.open(msk_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    d = out_dir / "visuals" / domain
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(orig);           axes[0].set_title("Image")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("GT")
    axes[2].imshow(pred * 255, cmap="gray"); axes[2].set_title("Pred")
    fig.tight_layout()
    fig.savefig(d / f"{stem}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_weights(model: torch.nn.Module, ckpt: Path):
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def evaluate_checkpoint(ckpt: Path, data_root: Path, out_dir: Path, run_profile: bool, device,
                        save_pred: bool = False, viz: bool = False):
    model = EGEUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True)
    model = load_weights(model, ckpt)
    model.to(device)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    complexity = {}
    if run_profile:
        try:
            complexity = profile_model(model, input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                                       device=str(device), precision="fp32", warmup=5, iterations=20)
            (out_dir / "profile.json").write_text(json.dumps(complexity, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] profiling failed: {e}")

    summary_rows = []
    all_rows = []
    with torch.inference_mode():
        for domain in DOMAINS:
            img_dir = data_root / domain / "test" / "images"
            msk_dir = data_root / domain / "test" / "masks"
            if not img_dir.exists() or not msk_dir.exists():
                print(f"[skip] {domain}: no test dir")
                continue
            ids = sorted(p.name for p in img_dir.glob("*"))
            rows = []
            for name in ids:
                msk_path = find_mask(msk_dir, name)
                if msk_path is None:
                    continue
                img_t, msk_t = load_sample(img_dir / name, msk_path, device)
                out = model(img_t.unsqueeze(0))            # (gt_pre, out)
                out_t = out[1] if isinstance(out, tuple) else out
                prob = out_t[0, 0].cpu().numpy()           # [H,W] in [0,1] (已 sigmoid)
                pred = (prob >= THRESHOLD)
                truth = (msk_t[0] >= 0.5).numpy()           # msk_t: [1,H,W] -> [H,W]
                if save_pred:
                    save_pred_mask(pred, out_dir, domain, name)
                if viz:
                    save_visual(img_dir / name, msk_path, pred, out_dir, domain, name)
                rows.append({"id": name, **segmentation_metrics(pred, truth)})
            if not rows:
                print(f"[skip] {domain}: no valid image/mask pairs (n=0)")
                continue
            scores = aggregate_metrics(rows)
            all_rows.extend(rows)
            summary_rows.append({"domain": domain, "n": len(rows), **scores})
            with open(out_dir / f"{domain}_per_image.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"{domain}: n={len(rows)}, dice={scores['dice']:.4f}, iou={scores['iou']:.4f}, "
                  f"hd95={scores['hd95']:.2f} (finite {scores['hd95_finite_count']}, fail {scores['hd95_failed_count']})")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    report = {"checkpoint": str(ckpt.resolve()), "image_size": IMAGE_SIZE, "threshold": THRESHOLD,
              "normalization": {"mean": MEAN, "std": STD}, "profile": complexity, "results": summary_rows}
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="EGE-UNet 训练好的权重 .pth 路径")
    ap.add_argument("--evalData", default=None,
                    help="评估数据根目录(含 isic2017/isic2018/ph2, 结构与附件 data 文件夹一致)。默认: <脚本>/data")
    ap.add_argument("--out", default=None,
                    help="输出目录(存 summary.csv / *_per_image.csv / results.json)。默认: checkpoint 同目录/evaluation_egeunet")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--profile", action="store_true", help="是否运行 profiling (延迟/FLOPs 估算)")
    ap.add_argument("--savePred", action="store_true",
                    help="保存每张测试图的预测二值掩膜 PNG 到 <out>/predictions/<domain>/")
    ap.add_argument("--viz", action="store_true",
                    help="保存 原图/真值/预测 三连对比图到 <out>/visuals/<domain>/")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        sys.exit(f"checkpoint 不存在: {ckpt}")
    data_root = Path(args.evalData) if args.evalData else Path(__file__).resolve().parent / "data"
    out_dir = Path(args.out) if args.out else ckpt.parent / "evaluation_egeunet"

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[device] {device}  (cuda_available={torch.cuda.is_available()})", flush=True)

    rows = evaluate_checkpoint(ckpt, data_root, out_dir, args.profile, device,
                               save_pred=args.savePred, viz=args.viz)
    print("\n=== 汇总 ===")
    for r in rows:
        print(f"{r['domain']:>9}: dice={r['dice']:.4f} iou={r['iou']:.4f} acc={r['accuracy']:.4f} "
              f"sen={r['sensitivity']:.4f} spe={r['specificity']:.4f} hd95={r['hd95']:.2f}")
    print(f"\n结果已写入: {out_dir}")


if __name__ == "__main__":
    main()
