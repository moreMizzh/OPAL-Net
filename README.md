# OPAL-Net

This repository contains the code used for the main OPAL-Net experimental pipeline, organized into two stages:
- `cascaded_reg`: atlas-guided cascaded registration
- `seg`: atlas-aware segmentation trained with registration-generated synthetic pairs

The code is kept as a research pipeline entry release rather than a packaged Python library. The main purpose of this README is to help readers reproduce training and inference with the provided entry scripts.

## Environment

Recommended local environment:
- Python `3.10`
- CUDA-enabled PyTorch environment for both training and inference

Install the pinned core dependencies with:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` is pinned from the core packages used in the local `prosnet` environment and is intended for stronger reproducibility than an unconstrained dependency list.

## Repository Layout

```text
OPAL-Net/
├── dataset.py
├── losses.py
├── utils.py
├── model_reg.py
├── model_seg.py
├── train_reg.py
├── inference_reg.py
├── train_seg.py
├── inference_seg.py
├── requirements.txt
└── README.md
```

Main entry scripts:
- `train_reg.py`: registration training
- `inference_reg.py`: registration inference, export, and metric summary
- `train_seg.py`: segmentation training
- `inference_seg.py`: segmentation inference, label export, Dice summary, and HD95 summary

## Data Preparation

The scripts follow a relative-path workflow by default. In practice, it is recommended to place datasets under a project-level `data/` directory and override paths through environment variables when needed.

Example neutral layout:

```text
../data/
├── IXI_Dataset/
├── IXI_Dataset_NCC/
├── ADNI_Dataset/
├── BCV/
├── OASIS/
└── spine/
```

Notes:
- Registration scripts read the original atlas-target dataset directly.
- Segmentation scripts require both real data and registration-generated synthetic outputs.
- The default presets inside the scripts may still contain older internal path conventions, but all important paths can be overridden through environment variables.

## Registration

### Train Registration

Minimal example:

```bash
cd OPAL-Net
OPAL_REG_DATASET_KEY=BCV python train_reg.py
```

Common environment variables:
- `OPAL_REG_DATASET_KEY`: dataset preset key such as `BCV`, `OASIS`, `IXI`, or `ADNI`
- `OPAL_REG_SOURCE_ROOT_DIR`: registration dataset root
- `OPAL_REG_SAVE_ROOT_DIR`: output root for registration experiments
- `OPAL_REG_EXP_NAME`: experiment name used in output directory naming
- `OPAL_REG_BCV_ATLAS_ID`: atlas id for BCV experiments
- `OPAL_REG_EPOCHS`: training epochs
- `OPAL_REG_CONTINUE_TRAIN`: resume from `resume_latest.pth` when set

Training flow:
1. Build configuration from dataset preset and environment overrides.
2. Construct the cascaded registration model.
3. Train with stage-A coarse registration and optional stage-B refinement.
4. Save `config.json`, checkpoints, and training curves.
5. Optionally launch automatic registration inference after training.

### Run Registration Inference

Minimal example:

```bash
cd OPAL-Net
OPAL_REG_INF_MODEL_DIR=./output/Reg/BCV_cascaded_reg python inference_reg.py
```

Common environment variables:
- `OPAL_REG_INF_MODEL_DIR`: training output directory containing `config.json` and checkpoints
- `OPAL_REG_INF_CHECKPOINT`: checkpoint selector such as `best_dice`, `best_loss`, or a specific `.pth` file
- `OPAL_REG_INF_EXPORT_TRAIN`: whether to export the train split
- `OPAL_REG_INF_EXPORT_VAL`: whether to export the val split
- `OPAL_REG_INF_EXPORT_TEST`: whether to export the test split
- `OPAL_REG_INF_COMPUTE_METRICS`: whether to compute Dice / HD95 / folding rate
- `OPAL_REG_INF_COMPUTE_HD95`: whether to compute HD95
- `OPAL_REG_INF_COMPUTE_FOLDING_RATE`: whether to compute folding rate

`inference_reg.py` performs two roles:
- export warped images, labels, and flows
- summarize split-level metrics into `reg_metrics.json`

## Segmentation

### Train Segmentation

Minimal example:

```bash
cd OPAL-Net
OPAL_SEG_DATASET_KEY=BCV python train_seg.py
```

Segmentation depends on two data roots:
- `real_root_dir`: original labeled dataset used for validation and testing
- `syn_root_dir`: synthetic data exported by the registration stage

Common environment variables:
- `OPAL_SEG_DATASET_KEY`: dataset preset key
- `OPAL_SEG_REAL_ROOT_DIR`: root of the real labeled dataset
- `OPAL_SEG_SYN_ROOT_DIR`: root of the registration-generated synthetic outputs
- `OPAL_SEG_SAVE_ROOT_DIR`: output root for segmentation experiments
- `OPAL_SEG_EXP_NAME`: experiment name used in output directory naming
- `OPAL_SEG_BCV_ATLAS_ID`: atlas id for BCV experiments
- `OPAL_SEG_ENABLE_ATLAS_TOKEN_PATH`: enable atlas token path
- `OPAL_SEG_SHARE_ATLAS_ENCODER_WITH_SEG`: share atlas encoder with segmentation encoder
- `OPAL_SEG_EPOCHS`: training epochs

Training flow:
1. Build segmentation configuration from preset and environment overrides.
2. Load real data and synthetic registration outputs.
3. Train segmentation on synthetic supervision.
4. Optionally regularize with MMD, entropy minimization, and class-ratio prior terms.
5. Save checkpoints, plots, and summary metrics.
6. Optionally launch automatic segmentation inference after training.

### Run Segmentation Inference

Minimal example:

```bash
cd OPAL-Net
OPAL_SEG_INF_MODEL_DIR=./output/Seg/BCV_seg_cascaded_reg_hd python inference_seg.py
```

Common environment variables:
- `OPAL_SEG_INF_MODEL_DIR`: segmentation training output directory
- `OPAL_SEG_INF_CHECKPOINT`: checkpoint selector such as `best_dice`, `best_loss`, or a specific `.pth` file
- `OPAL_SEG_INF_EXPORT_TRAIN`: whether to evaluate the train split
- `OPAL_SEG_INF_EXPORT_VAL`: whether to evaluate the val split
- `OPAL_SEG_INF_EXPORT_TEST`: whether to evaluate the test split
- `OPAL_SEG_INF_COMPUTE_HD95`: whether to compute HD95
- `OPAL_SEG_INF_HD95_MODE`: HD95 summary mode
- `OPAL_SEG_INF_SAVE_PRED_LABEL`: whether to save predicted labels
- `OPAL_SEG_INF_DEVICE`: target device

`inference_seg.py` performs the following tasks:
- run segmentation inference split by split
- optionally save predicted label NIfTI files
- compute per-class Dice summaries
- compute HD95 summaries
- manage atlas token cache reuse for repeated inference

## Outputs

### Registration Outputs

Typical registration experiment directory contains:
- `config.json`
- `resume_latest.pth`
- `best_loss_*.pth`
- `best_dice_*.pth`
- `img/reg_metrics_latest.png`
- `reg_metrics.json`
- split export directories such as `train_output/` or `test_output/`

### Segmentation Outputs

Typical segmentation experiment directory contains:
- `config.json`
- `resume_latest.pth`
- `best_loss_*.pth`
- `best_real_dice_*.pth`
- `img/seg_latest.png`
- `metrics_summary.json`
- `inference_seg_<split>_dice_metrics.json`
- `inference_seg_<split>_hd95_metrics.json`
- `inference_seg_<split>_metrics.json`
- optional predicted-label directories such as `inference_seg_test_label/`

## Reproducibility Notes

- Both training entry scripts save the resolved runtime configuration to `config.json`.
- Most important path and behavior changes can be controlled through environment variables without editing source files.
- Random seeds are set inside the training entry scripts for more stable reruns.
- Registration and segmentation training both require CUDA in the current release.
- Segmentation inference may create and reuse an atlas token cache file inside the model directory.
- If the data layout does not match the built-in presets, it is safer to override paths explicitly rather than editing the preset table during reproduction.

## Practical Reproduction Order

For a full pipeline reproduction, the expected order is:
1. Prepare the original dataset.
2. Train registration with `train_reg.py`.
3. Export or compute registration results with `inference_reg.py` if needed.
4. Use the registration output directory as `OPAL_SEG_SYN_ROOT_DIR`.
5. Train segmentation with `train_seg.py`.
6. Evaluate segmentation with `inference_seg.py`.

## Scope

This repository is intended to expose the exact executable research pipeline used for the main OPAL-Net experiments. It is not yet organized as a general-purpose package, and readers should expect script-driven training and inference rather than a high-level public API.
