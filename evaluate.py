#!/usr/bin/env python3
"""
Evaluate the released PFM4WSSS segmentation models on the test set of
LUAD-HistoSeg or BCSS-WSSS, and optionally save the predicted masks.

Only the trained segmentation network is needed at inference time.

Evaluation protocol:
  - per-class IoU accumulated over the whole test set, mIoU over the four
    tissue classes;
  - pixels with label 4 (white background) are ignored;
  - test-time augmentation: mean of the softmax over the original patch and
    its horizontal and vertical flips (disable with --no-tta).

Examples
--------
    python evaluate.py --dataset luad --data-root data/LUAD-HistoSeg
    python evaluate.py --dataset bcss --data-root data/BCSS-WSSS --seeds 0
    python evaluate.py --dataset luad --data-root data/LUAD-HistoSeg \
        --seeds 0 --save-pred predictions/luad_seed0
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import segmentation_models_pytorch as smp  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_SIZE = 224
NUM_CLASSES = 4
IGNORE_IDX = 4

# Official class order of each benchmark (they differ: do not swap).
CLASS_NAMES = {
    "luad": ["TE", "NEC", "LYM", "TAS"],
    "bcss": ["TUM", "STR", "LYM", "NEC"],
}
CKPT_NAME = "unet_mitb2_{dataset}_seed{seed}.pth"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def load_mask(path):
    """Test masks are palette (P-mode) images: the palette index is the class."""
    mask = Image.open(path)
    if mask.size != (IMG_SIZE, IMG_SIZE):
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


def to_tensor(img_rgb):
    t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ((t - IMAGENET_MEAN) / IMAGENET_STD).to(DEVICE)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.tp = np.zeros(NUM_CLASSES)
        self.fp = np.zeros(NUM_CLASSES)
        self.fn = np.zeros(NUM_CLASSES)
        self.gt = np.zeros(NUM_CLASSES)

    def update(self, pred, gt):
        valid = gt != IGNORE_IDX
        for c in range(NUM_CLASSES):
            p = (pred == c) & valid
            g = (gt == c) & valid
            self.tp[c] += (p & g).sum()
            self.fp[c] += (p & ~g).sum()
            self.fn[c] += (~p & g).sum()
            self.gt[c] += g.sum()

    def results(self, class_names):
        eps = 1e-8
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        present = self.gt > 0
        freq = np.where(present, self.gt / (self.gt.sum() + eps), 0.0)
        out = {
            "mIoU": float(np.nanmean(np.where(present, iou, np.nan))) * 100,
            "fwIoU": float((freq * iou).sum()) * 100,
        }
        for c, name in enumerate(class_names):
            out[f"IoU_{name}"] = float(iou[c]) * 100
        return out


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def load_model(ckpt_path):
    model = smp.Unet(encoder_name="mit_b2", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES, activation=None)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, img_rgb, use_tta=True):
    t = to_tensor(img_rgb)
    prob = torch.softmax(model(t), dim=1)
    if use_tta:
        p_h = torch.flip(torch.softmax(model(torch.flip(t, dims=[3])), dim=1), dims=[3])
        p_v = torch.flip(torch.softmax(model(torch.flip(t, dims=[2])), dim=1), dims=[2])
        prob = (prob + p_h + p_v) / 3.0
    return prob.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)


def evaluate(ckpt_path, img_dir, mask_dir, files, class_names, use_tta,
             save_dir=None):
    model = load_model(ckpt_path)
    metrics = Metrics()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for fname in tqdm(files, desc=os.path.basename(ckpt_path), leave=False):
        img = load_image(os.path.join(img_dir, fname))
        if img is None:
            continue
        pred = predict(model, img, use_tta)
        metrics.update(pred, load_mask(os.path.join(mask_dir, fname)))
        if save_dir:
            Image.fromarray(pred).save(
                os.path.join(save_dir, os.path.splitext(fname)[0] + ".png"))
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics.results(class_names)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["luad", "bcss"])
    ap.add_argument("--data-root", required=True,
                    help="dataset folder that contains test/img and test/mask")
    ap.add_argument("--checkpoints", default="checkpoints",
                    help="folder with the downloaded .pth files")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-tta", action="store_true",
                    help="disable test-time augmentation")
    ap.add_argument("--save-pred", default=None,
                    help="optional folder to save the predicted label maps "
                         "(one sub-folder per seed)")
    ap.add_argument("--out", default="outputs",
                    help="folder for the result CSV files")
    args = ap.parse_args()

    class_names = CLASS_NAMES[args.dataset]
    img_dir = os.path.join(args.data_root, "test", "img")
    mask_dir = os.path.join(args.data_root, "test", "mask")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        sys.exit(f"[X] expected {img_dir} and {mask_dir}")

    # Only images with a ground-truth mask are evaluated (this also skips
    # hidden macOS files such as "._name.png").
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".png", ".jpg"))
                   and not f.startswith("._")
                   and os.path.exists(os.path.join(mask_dir, f)))
    use_tta = not args.no_tta
    print(f"[INFO] {args.dataset.upper()} | {len(files)} test images | "
          f"device {DEVICE} | TTA {'on' if use_tta else 'off'}")

    rows = []
    for seed in args.seeds:
        ckpt = os.path.join(args.checkpoints,
                            CKPT_NAME.format(dataset=args.dataset, seed=seed))
        if not os.path.exists(ckpt):
            print(f"[WARN] seed {seed}: {ckpt} not found, skipped")
            continue
        save_dir = (os.path.join(args.save_pred, f"seed{seed}")
                    if args.save_pred else None)
        res = evaluate(ckpt, img_dir, mask_dir, files, class_names, use_tta,
                       save_dir)
        res = {"seed": seed, **res}
        rows.append(res)
        print(f"  seed {seed}: mIoU {res['mIoU']:.2f}  "
              + "  ".join(f"{n} {res['IoU_' + n]:.2f}" for n in class_names))

    if not rows:
        sys.exit("[X] no checkpoint evaluated; see checkpoints/README.md")

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.dataset}_{'tta' if use_tta else 'notta'}"
    csv_path = os.path.join(args.out, f"{tag}_per_seed.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print("-" * 60)
    for col in [c for c in df.columns if c != "seed"]:
        std = df[col].std(ddof=1) if len(df) > 1 else 0.0
        print(f"  {col:<10} {df[col].mean():6.2f} ± {std:.2f}")
    print("-" * 60)
    print(f"[OK] {csv_path}")


if __name__ == "__main__":
    main()
