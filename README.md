# OPAL-Net

Official repository for **OPAL-Net**, an atlas-guided medical image segmentation framework with cascaded registration, atlas prior tokens, bidirectional atlas-image cross-attention, and synthetic-real feature alignment.

> This is a lightweight review-stage repository. The complete training and inference implementation is being organized for release.

## Overview

OPAL-Net follows a two-stage pipeline:

1. **Atlas-guided registration** generates synthetic image-label pairs from an atlas.
2. **Atlas-aware segmentation** uses atlas prior tokens and feature alignment to improve target-domain segmentation.

The current public repository provides the project description, dependency list, and release structure. Full executable scripts, model definitions, and reproduction details are temporarily withheld during peer review to protect unpublished implementation details.

## Code Preparation

We are organizing the following materials for release:

- registration training and inference scripts;
- segmentation training and inference scripts;
- model definitions for registration and OPAL-Net segmentation;
- dataset loaders and preprocessing utilities;
- evaluation code for Dice and HD95;
- experiment configuration examples;
- instructions for reproducing the reported results.

## Training Workflow

The full release will follow the reproduction order below. The key dependency is that segmentation training reads the synthetic image-label pairs exported by the registration stage.

1. Train the registration model. This creates a registration experiment directory under `output/Reg/`.

```bash
python train_reg.py
```

2. Run registration inference/export with the trained registration directory. This writes warped synthetic pairs such as `train_output/` under the same registration experiment directory.

```bash
OPAL_REG_INF_MODEL_DIR=output/Reg/<reg_experiment> \
OPAL_REG_INF_EXPORT_TRAIN=1 \
OPAL_REG_INF_EXPORT_TEST=1 \
python inference_reg.py
```

3. Train the segmentation model using the real dataset root and the exported registration directory. `OPAL_SEG_SYN_ROOT_DIR` should point to the registration experiment directory from step 1/2, not directly to an individual file.

```bash
OPAL_SEG_REAL_ROOT_DIR=<real_dataset_root> \
OPAL_SEG_SYN_ROOT_DIR=output/Reg/<reg_experiment> \
python train_seg.py
```

4. Run segmentation inference from the segmentation experiment directory under `output/Seg/`.

```bash
OPAL_SEG_INF_MODEL_DIR=output/Seg/<seg_experiment> \
OPAL_SEG_INF_EXPORT_TEST=1 \
OPAL_SEG_INF_COMPUTE_HD95=1 \
python inference_seg.py
```

Dataset-specific presets are selected by `dataset_key` in the training configs. The review-stage repository keeps the workflow description lightweight while the complete configuration files and executable reproduction scripts are being organized.

## Environment

The development environment uses:

- Python 3.10;
- PyTorch with CUDA support;
- MONAI, SimpleITK, NumPy, SciPy, and related medical-imaging utilities.

A preliminary dependency list is provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data

Datasets are not distributed with this repository. Users should prepare the corresponding public or institutionally approved datasets according to their own access permissions. Path conventions and configuration examples for supported datasets are being organized.

## Code Availability

The complete source code and reproduction materials are being organized. At this stage, the repository provides project information, dependency tracking, and the high-level reproduction workflow.

## Citation

Citation information will be added after publication.

## Contact

For questions about OPAL-Net, please contact the authors through the paper correspondence channel.
