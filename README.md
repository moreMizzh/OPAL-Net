# OPAL-Net

OPAL-Net is a cleaned public release snapshot extracted from the current research workspace.

This folder contains the code path we are currently using for two main components:
- `cascaded_reg`: atlas-guided cascaded registration
- `seg`: atlas-aware segmentation trained on registration-generated synthetic pairs

## What Is Included

Core source files:
- `dataset.py`
- `losses.py`
- `utils.py`
- `model_reg.py`
- `model_seg.py`

Registration entry points:
- `train_reg.py`
- `inference_reg.py`

Segmentation entry points:
- `train_seg.py`
- `inference_seg.py`

Basic regression tests:
- `test_bcv_atlas_id.py`
- `test_release_naming.py`
- `test_public_code_naming.py`

## What Is Not Included

This release folder intentionally excludes:
- experiment outputs and checkpoints
- plotting scripts and paper figures
- notebooks and temporary analysis files
- historical backups and abandoned variants
- dataset files

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

## Dependencies

Typical runtime dependencies:
- Python 3.10+
- PyTorch
- NumPy
- SimpleITK
- MONAI
- matplotlib
- tqdm

Install the lightweight dependency list with:

```bash
pip install -r requirements.txt
```

## Expected Data Layout

The current scripts still follow the original relative-path convention from the workspace. By default, datasets are expected under paths similar to:
- `../data/IXI_Dataset`
- `../data/IXI_Dataset_NCC`
- `../data/ADNI_Dataset`
- `../data/BCV_by_ZF/bcv_small`
- `../data/OASIS_by_ZF/OASIS`
- `../data/spine/spine_small`

If you move `OPAL-Net` into a standalone repository, you will usually want to:
- update the dataset presets in the training scripts
- or override paths through environment variables

## Quick Start

Most configuration is controlled through environment variables inside each training or inference script.

### 1. Train Cascaded Registration

Example for BCV:

```bash
cd OPAL-Net
OPAL_REG_DATASET_KEY=BCV python train_reg.py
```

Useful variables for registration training include:
- `OPAL_REG_DATASET_KEY`
- `OPAL_REG_SOURCE_ROOT_DIR`
- `OPAL_REG_SAVE_ROOT_DIR`
- `OPAL_REG_EXP_NAME`
- `OPAL_REG_BCV_ATLAS_ID`

### 2. Run Registration Inference

```bash
cd OPAL-Net
OPAL_REG_INF_MODEL_DIR=./output/Reg/BCV_cascaded_reg python inference_reg.py
```

Useful variables for registration inference include:
- `OPAL_REG_INF_MODEL_DIR`
- `OPAL_REG_INF_CHECKPOINT`
- `OPAL_REG_INF_EXPORT_TRAIN`
- `OPAL_REG_INF_EXPORT_TEST`
- `OPAL_REG_INF_COMPUTE_METRICS`

### 3. Train Segmentation

Example for BCV using registration-generated synthetic data:

```bash
cd OPAL-Net
OPAL_SEG_DATASET_KEY=BCV python train_seg.py
```

Useful variables for segmentation training include:
- `OPAL_SEG_REAL_ROOT_DIR`
- `OPAL_SEG_SYN_ROOT_DIR`
- `OPAL_SEG_SAVE_ROOT_DIR`
- `OPAL_SEG_EXP_NAME`
- `OPAL_SEG_BCV_ATLAS_ID`

### 4. Run Segmentation Inference

```bash
cd OPAL-Net
OPAL_SEG_INF_MODEL_DIR=./output/Seg/BCV_seg_cascaded_reg_hd python inference_seg.py
```

Useful variables for segmentation inference include:
- `OPAL_SEG_INF_MODEL_DIR`
- `OPAL_SEG_INF_CHECKPOINT`
- `OPAL_SEG_INF_EXPORT_TEST`
- `OPAL_SEG_INF_COMPUTE_HD95`
- `OPAL_SEG_INF_DEVICE`

## Notes On Naming

The public file names and main code paths have been cleaned to use `cascaded_reg` and `seg`.

Environment variable names are now unified to the `OPAL_REG_*`, `OPAL_REG_INF_*`, `OPAL_SEG_*`, and `OPAL_SEG_INF_*` prefixes for a cleaner public release.

## Notes

- This is still research code rather than a fully packaged Python library.
- The current release reflects the exact implementation branch we are actively using, instead of a redesigned public API.
- If you plan to release this as a polished open-source project, the next good step would be adding a `LICENSE`, a dataset preparation guide, and a minimal reproducibility checklist.
