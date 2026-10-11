# Single-case segmentation demo

This demo runs the final segmentation model on one preprocessed 3D NIfTI image. It does not need a target label or registration output. The matching atlas image and atlas label are required to build the model's prior tokens. Run commands from the repository root after installing `requirements.txt`. The private review repository contains one runnable BCV example; OASIS and ADNI require separately obtained data and checkpoints.

## Models

The three checkpoint specifications were selected by the best validation Dice among the reviewed `baseline_v14` / `v1305` experiments with the atlas-token path enabled. The recorded test Dice is shown for context; it was not used to choose the checkpoints. Only the BCV checkpoint is included in this private repository.

| Dataset | Checkpoint | Included here | Best validation Dice | Recorded test Dice |
| --- | --- | --- | ---: | ---: |
| BCV | `checkpoints/bcv.pth` | Yes, private review only | 0.913512 | 0.870447 |
| OASIS | `checkpoints/oasis.pth` | No | 0.862896 | 0.866259 |
| ADNI | `checkpoints/adni.pth` | No | 0.862027 | 0.855023 |

`models.json` records each checkpoint's SHA-256 and exact architecture. The demo checks the hash before loading. The OASIS and ADNI checkpoints remain local and are not available from this repository.

## Run

Each input must use the same preprocessing and label convention as its dataset's training data. The script preserves the input image's NIfTI geometry in the prediction. `--device` defaults to CUDA when available, otherwise CPU.

The bundled BCV example contains preprocessed test image 61, atlas image 27, and the atlas's label 27 under `demo/data/bcv/`. These are 160 x 128 x 128 volumes from the local `bcv_small` layout; no test ground-truth label is included. Run from the repository root with no arguments:

```bash
python demo/run_bcv.py
```

The script selects the bundled BCV files and checkpoint, uses CUDA when available (otherwise CPU), and prints the paths of `prediction.nii.gz` and `overlay.png`. Each run creates a separate directory under `demo/results/`.

For a different BCV case, supply its preprocessed image and the matching atlas pair explicitly:

```bash
python demo/run.py \
  --dataset bcv \
  --image /path/to/bcv/img0061_bcv_CT.nii.gz \
  --atlas-image /path/to/bcv/img0027_bcv_CT.nii.gz \
  --atlas-label /path/to/bcv/seg0027_bcv_CT.nii.gz \
  --output demo/results/bcv_case
```

### OASIS-1 (not bundled)

The reviewed example uses [Neurite-processed OASIS-1](https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar) data (about 7 GB), derived from [OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/). The raw OASIS-1 release is not a drop-in replacement for these aligned files. Neither the scans nor the selected checkpoint are in this repository, and permission to rehost this processed subset has not been confirmed. Use full-size `aligned_norm.nii.gz`, not `aligned_norm_small.nii.gz`, with its corresponding `aligned_seg35.nii.gz` atlas label. The reviewed atlas is `OASIS_OAS1_0012_MR1` and the example target is `OASIS_OAS1_0417_MR1`.

```bash
OASIS_ROOT=/path/to/OASIS
python demo/run.py \
  --dataset oasis \
  --image "$OASIS_ROOT/OASIS_OAS1_0417_MR1/aligned_norm.nii.gz" \
  --atlas-image "$OASIS_ROOT/OASIS_OAS1_0012_MR1/aligned_norm.nii.gz" \
  --atlas-label "$OASIS_ROOT/OASIS_OAS1_0012_MR1/aligned_seg35.nii.gz" \
  --checkpoint /path/to/oasis.pth \
  --output demo/results/oasis_case
```

### ADNI (not bundled)

[ADNI data access](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData) requires an application. Its [Data Use Agreement](https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf) prohibits redistribution of individual participant-level data, so no ADNI image or atlas is committed here, even though the repository is private. The selected checkpoint also remains local pending a separate sharing review. Authorized users can supply a preprocessed image from `Test/img/` and the training atlas image/label pair specified in `dataset.py`:

```bash
ADNI_ROOT=/path/to/ADNI_Dataset
python demo/run.py \
  --dataset adni \
  --image "$ADNI_ROOT/Test/img/<preprocessed-subject>.nii.gz" \
  --atlas-image "$ADNI_ROOT/atlas/<training-atlas-image>.nii.gz" \
  --atlas-label "$ADNI_ROOT/atlas/<matching-atlas-label>.nii.gz" \
  --checkpoint /path/to/adni.pth \
  --output demo/results/adni_case
```

The output directory contains `prediction.nii.gz` and `overlay.png`. The preview shows the input and one axial slice with predicted labels overlaid. It is an inspection image, not a quantitative evaluation. Existing output files are not overwritten.

For a checkpoint outside `demo/checkpoints/`, provide `--checkpoint /path/to/model.pth`. It must match the selected dataset's SHA-256 in `models.json`.
