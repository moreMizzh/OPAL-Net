# OPAL-Net

OPAL-Net is an atlas-guided medical image segmentation framework with cascaded registration, atlas prior tokens, bidirectional atlas-image cross-attention, and synthetic-real feature alignment.

## Overview

OPAL-Net uses a two-stage pipeline:

1. Atlas-guided registration generates synthetic image-label pairs from an atlas.
2. Atlas-aware segmentation uses real images together with the exported synthetic pairs and atlas prior information.

The repository contains the training, inference, model, dataset, loss, and utility implementations for both stages.

## Requirements

- Python 3.10
- CUDA-enabled PyTorch
- MONAI
- SimpleITK
- NumPy
- Matplotlib
- tqdm

Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

Training and the default evaluation workflow require a CUDA-capable PyTorch installation.

## Data preparation

Datasets are not distributed with this repository. Prepare each dataset according to its access terms and pass the dataset root explicitly through environment variables.

The supported layouts are:

### BCV

BCV uses a flat directory containing image-label pairs:

```text
<bcv_root>/
|-- img0001_bcv_CT.nii.gz
|-- seg0001_bcv_CT.nii.gz
|-- img0002_bcv_CT.nii.gz
|-- seg0002_bcv_CT.nii.gz
`-- ...
```

The default atlas is case 27. Use `OPAL_REG_BCV_ATLAS_ID` and `OPAL_SEG_BCV_ATLAS_ID` to select another atlas explicitly.

### ADNI

ADNI expects separate atlas, training, and test directories:

```text
<adni_root>/
|-- atlas/
|-- Train/
|   |-- img/
|   `-- label/
`-- Test/
    |-- img/
    `-- label/
```

### OASIS

OASIS expects one directory per subject. Each subject directory must contain the normalized image and segmentation label:

```text
<oasis_root>/
|-- <subject_001>/
|   |-- aligned_norm.nii.gz
|   `-- aligned_seg35.nii.gz
`-- ...
```

## Reproduction workflow

Run the commands below from the repository root. The example uses BCV; replace `DATA_ROOT` and the dataset-specific settings for ADNI or OASIS.

```bash
export DATA_ROOT=/path/to/data
```

### 1. Train the registration model

```bash
OPAL_REG_DATASET_KEY=BCV \
OPAL_REG_SOURCE_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_REG_BCV_ATLAS_ID=27 \
python train_reg.py
```

The experiment is written under `output/Reg/`. The training script can automatically run the registration inference stage after training.

### 2. Export synthetic registration pairs

If a separate export run is needed, point the inference script to the registration experiment directory:

```bash
OPAL_REG_INF_MODEL_DIR=output/Reg/<reg_experiment> \
OPAL_REG_INF_DATA_ROOT="$DATA_ROOT/BCV" \
OPAL_REG_INF_BCV_ATLAS_ID=27 \
OPAL_REG_INF_EXPORT_TRAIN=1 \
OPAL_REG_INF_EXPORT_TEST=1 \
python inference_reg.py
```

The exported synthetic image-label pairs are stored in the registration experiment directory, typically under `train_output/` and `test_output/`.

### 3. Train the segmentation model

Set the real dataset root and the registration experiment directory. The atlas-token path is enabled explicitly for the main OPAL-Net configuration.

```bash
OPAL_SEG_REAL_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_SEG_SYN_ROOT_DIR=output/Reg/<reg_experiment> \
OPAL_SEG_BCV_ATLAS_ID=27 \
OPAL_SEG_ENABLE_ATLAS_TOKEN_PATH=1 \
python train_seg.py
```

The segmentation experiment is written under `output/Seg/`.

### 4. Run segmentation inference

```bash
OPAL_SEG_INF_MODEL_DIR=output/Seg/<seg_experiment> \
OPAL_SEG_INF_REAL_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_SEG_INF_BCV_ATLAS_ID=27 \
OPAL_SEG_INF_ENABLE_ATLAS_TOKEN_PATH=1 \
OPAL_SEG_INF_EXPORT_TEST=1 \
OPAL_SEG_INF_COMPUTE_HD95=1 \
python inference_seg.py
```

The inference script writes prediction labels and metric summaries under the segmentation experiment directory.

## Outputs and reproducibility

Generated checkpoints, predictions, NIfTI files, logs, and experiment directories are intentionally excluded from version control. Reproduction requires the approved dataset, the pinned software environment, and the commands above.

The training and inference scripts expose additional configuration through `OPAL_REG_*`, `OPAL_REG_INF_*`, `OPAL_SEG_*`, and `OPAL_SEG_INF_*` environment variables.

## Citation

Citation information will be added after publication.
