<div align="center">

# PFM4WSSS

### Weakly Supervised Semantic Segmentation of Histopathology Images

Ahmed Benariba · Nassima Dif · Sidi Mohammed Benslimane

LabRI-SBA Lab., École Supérieure en Informatique 8 Mai 1945, Sidi Bel Abbès, Algeria

![Status](https://img.shields.io/badge/paper-under%20review-orange)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c)
![Release](https://img.shields.io/badge/release-v0.1-informational)

</div>

---

This is the official repository of **PFM4WSSS**, a framework for tissue
segmentation in H&E histopathology images trained with image-level labels
only. The paper is currently under review.

> **About this release.** To support the review process, this first release
> provides the trained segmentation models and the evaluation code needed to
> reproduce the test results reported in the paper on two public benchmarks,
> **LUAD-HistoSeg** and **BCSS-WSSS**. The description of the method and the
> complete source code will be published here upon acceptance.

## What this release provides

- **10 trained segmentation models**: five independent runs (seeds 0–4) on
  each benchmark, so that both the mean and the run-to-run variability of
  the reported results can be checked.
- **A single evaluation script** that runs inference on the official test
  sets, computes the metrics used in the paper, and can save the predicted
  masks.
- **SHA-256 checksums** for every model file.
- **Pinned package versions** for an exact reproduction.

## Release status

| Version | Content | Availability |
|---|---|---|
| **v0.1** | Trained models, evaluation code | Now |
| **v1.0** | Method description, full training and analysis code, citation | Upon publication |

## Installation

Tested with Python 3.10 on Windows and an NVIDIA RTX 4070 (12 GB). A GPU is
recommended but not required.

```bash
git clone https://github.com/ahmedESISBA/PFM4WSSS.git
cd PFM4WSSS

conda create -n pfm4wsss python=3.10 -y
conda activate pfm4wsss

# PyTorch 2.4.1 with CUDA 11.8 (see https://pytorch.org for other platforms)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Keep `segmentation-models-pytorch==0.3.4`: other versions may name the model
weights differently and fail to load the checkpoints.

## Data

### Source

Both benchmarks were released by Han et al. [1] and are available from their
official Google Drive folders:

| Dataset | Download | Test patches |
|---|---|:---:|
| LUAD-HistoSeg | [Google Drive](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr) | 307 |
| BCSS-WSSS | [Google Drive](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77) | 4,986 |

BCSS-WSSS is derived from the Breast Cancer Semantic Segmentation (BCSS)
dataset of Amgad et al. [2], available at
<https://github.com/PathologyDataScience/BCSS>. Please cite these works if
you use the data.

Only the **test split** is needed: from each Drive folder, download the
`test` folder (it contains `img/` and `mask/`).

### Dataset summary

| | LUAD-HistoSeg | BCSS-WSSS |
|---|---|---|
| Organ | Lung (adenocarcinoma) | Breast (carcinoma) |
| Resolution | 10× | 0.25 µm/pixel |
| Patch size | 224 × 224 | 224 × 224 |
| Training patches (image-level labels) | 16,678 | 23,422 |
| Validation patches (pixel-level masks) | 300 | 3,418 |
| Test patches (pixel-level masks) | 307 | 4,986 |

### Folder layout

```
data/
├── LUAD-HistoSeg/
│   └── test/
│       ├── img/     # 307 H&E patches (.png)
│       └── mask/    # 307 masks, same file names
└── BCSS-WSSS/
    └── test/
        ├── img/     # 4,986 H&E patches (.png)
        └── mask/    # 4,986 masks, same file names
```

### Mask encoding

Masks are palette (P-mode) PNG images in which the pixel value is the class
index. **The class order differs between the two datasets.**

| Index | LUAD-HistoSeg | BCSS-WSSS |
|:---:|---|---|
| 0 | Tumor epithelium (TE) | Tumor (TUM) |
| 1 | Necrosis (NEC) | Stroma (STR) |
| 2 | Lymphocyte (LYM) | Lymphocytic infiltrate (LYM) |
| 3 | Tumor-associated stroma (TAS) | Necrosis (NEC) |
| 4 | White background, ignored | White background, ignored |

### References

[1] C. Han et al., "Multi-layer pseudo-supervision for histopathology tissue
semantic segmentation using patch-level classification labels," *Medical
Image Analysis*, vol. 80, 102487, 2022.

[2] M. Amgad et al., "Structured crowdsourcing enables convolutional
segmentation of histology images," *Bioinformatics*, vol. 35, no. 18,
pp. 3461–3467, 2019.

## Trained models

The ten checkpoints (about 105 MB each) are attached to the
[v0.1 release](https://github.com/ahmedESISBA/PFM4WSSS/releases/tag/v0.1).

| Seed | LUAD-HistoSeg | BCSS-WSSS |
|:---:|---|---|
| 0 | [unet_mitb2_luad_seed0.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_luad_seed0.pth) | [unet_mitb2_bcss_seed0.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_bcss_seed0.pth) |
| 1 | [unet_mitb2_luad_seed1.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_luad_seed1.pth) | [unet_mitb2_bcss_seed1.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_bcss_seed1.pth) |
| 2 | [unet_mitb2_luad_seed2.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_luad_seed2.pth) | [unet_mitb2_bcss_seed2.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_bcss_seed2.pth) |
| 3 | [unet_mitb2_luad_seed3.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_luad_seed3.pth) | [unet_mitb2_bcss_seed3.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_bcss_seed3.pth) |
| 4 | [unet_mitb2_luad_seed4.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_luad_seed4.pth) | [unet_mitb2_bcss_seed4.pth](https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_bcss_seed4.pth) |

Download all of them into `checkpoints/`:

```bash
# Linux / macOS
for d in luad bcss; do for s in 0 1 2 3 4; do
  wget -P checkpoints https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/unet_mitb2_${d}_seed${s}.pth
done; done
```

```powershell
# Windows PowerShell
foreach ($d in "luad","bcss") { foreach ($s in 0..4) {
  $f = "unet_mitb2_${d}_seed$s.pth"
  Invoke-WebRequest "https://github.com/ahmedESISBA/PFM4WSSS/releases/download/v0.1/$f" -OutFile "checkpoints\$f"
}}
```

Verify each download against the SHA-256 checksums listed in
[`checkpoints/README.md`](checkpoints/README.md).

## Evaluation

Evaluate the five models of each benchmark, as in the paper:

```bash
python evaluate.py --dataset luad --data-root data/LUAD-HistoSeg
python evaluate.py --dataset bcss --data-root data/BCSS-WSSS
```

The script prints the per-class IoU, mIoU and fwIoU of each model, then the
mean and standard deviation over the five seeds, and writes them to a CSV
file in `outputs/`. These values match those reported in the paper.

Other options:

```bash
# a single seed
python evaluate.py --dataset bcss --data-root data/BCSS-WSSS --seeds 0

# without test-time augmentation
python evaluate.py --dataset luad --data-root data/LUAD-HistoSeg --no-tta

# save the predicted label maps (one PNG per patch, pixel value = class index)
python evaluate.py --dataset luad --data-root data/LUAD-HistoSeg --seeds 0 --save-pred predictions/luad
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `luad` or `bcss` |
| `--data-root` | required | dataset folder that contains `test/img` and `test/mask` |
| `--checkpoints` | `checkpoints` | folder with the downloaded `.pth` files |
| `--seeds` | `0 1 2 3 4` | seeds to evaluate |
| `--no-tta` | off | disable test-time augmentation |
| `--save-pred` | none | folder where the predicted masks are saved |
| `--out` | `outputs` | folder for the result CSV file |

On an RTX 4070, evaluation takes about 10 s per model on LUAD-HistoSeg and
about 2.5 min per model on BCSS-WSSS.

### Evaluation protocol

- IoU is accumulated over **all pixels of the test set** for each class, then
  averaged over the four tissue classes (mIoU). The frequency-weighted IoU
  (fwIoU) weights each class by its share of annotated pixels.
- Pixels labeled as white background (index 4) are **ignored**, following the
  released protocol of both benchmarks.
- Test-time augmentation averages the softmax outputs over the original patch
  and its horizontal and vertical flips.
- Inference is deterministic (`cudnn.deterministic = True`). With the package
  versions of `requirements.txt`, the scores match the reported values to
  within 0.01 points.

## Citation

The paper is under review. The BibTeX entry will be added upon publication.

## Acknowledgements

We thank the authors of LUAD-HistoSeg and BCSS-WSSS for releasing the
benchmarks, and the maintainers of
[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).

## License

The license of the code and models will be specified with the full release
(v1.0). The datasets remain under the terms of their original authors.

## Contact

Ahmed Benariba — a.benariba@esi-sba.dz
