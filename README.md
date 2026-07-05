# OPAL-Net

Official repository for **OPAL-Net**, an atlas-guided medical image segmentation framework with cascaded registration, atlas prior tokens, bidirectional atlas-image cross-attention, and synthetic-real feature alignment.

> This is a lightweight review-stage repository. The complete training and inference implementation will be released after paper acceptance.

## Overview

OPAL-Net follows a two-stage pipeline:

1. **Atlas-guided registration** generates synthetic image-label pairs from an atlas.
2. **Atlas-aware segmentation** uses atlas prior tokens and feature alignment to improve target-domain segmentation.

The current public repository provides the project description, dependency list, and release structure. Full executable scripts, model definitions, and reproduction details are temporarily withheld during peer review to protect unpublished implementation details.

## Planned Full Release

After acceptance, this repository will include:

- registration training and inference scripts;
- segmentation training and inference scripts;
- model definitions for registration and OPAL-Net segmentation;
- dataset loaders and preprocessing utilities;
- evaluation code for Dice and HD95;
- experiment configuration examples;
- instructions for reproducing the reported results.

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

Datasets are not distributed with this repository. Users should prepare the corresponding public or institutionally approved datasets according to their own access permissions. The full release will provide path conventions and configuration examples for supported datasets.

## Code Availability

The complete source code will be made available in this repository after acceptance. Until then, this repository serves as a placeholder for project information and dependency tracking.

## Citation

Citation information will be added after publication.

## Contact

For questions about OPAL-Net, please contact the authors through the paper correspondence channel.
