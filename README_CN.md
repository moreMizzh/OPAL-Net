# OPAL-Net

[English](README.md) | 中文

OPAL-Net 是一个基于图谱（atlas）的医学图像分割框架，包含级联配准、图谱先验 token、双向图谱-图像交叉注意力，以及合成与真实图像的特征对齐。

## 项目概览

OPAL-Net 采用两阶段流程：

1. **图谱引导配准**：将图谱图像与标签配准到目标图像空间，生成合成图像-标签对。
2. **图谱感知分割**：结合真实图像、配准阶段导出的合成数据和图谱先验进行分割。

仓库包含这两个阶段的训练、推理、模型、数据集、损失函数和工具代码。完整流程依次使用 `train_reg.py`、`inference_reg.py`、`train_seg.py` 和 `inference_seg.py`；下面的单例 demo 只运行最终分割模型，不需要先训练或运行配准。

## 环境要求

- Python 3.10
- 支持 CUDA 的 PyTorch
- MONAI
- SimpleITK
- NumPy
- Matplotlib
- tqdm

在仓库根目录安装固定版本依赖：

```bash
pip install -r requirements.txt
```

完整训练与默认评估流程需要可用的 CUDA 环境；下面的 BCV demo 在没有 CUDA 时也能使用 CPU。

## 快速运行 BCV demo

在仓库根目录直接运行，无需设置参数：

```bash
python demo/run_bcv.py
```

脚本自动读取仓库内的 BCV 测试图像 61、图谱图像与标签 27，以及 `demo/checkpoints/bcv.pth`；有 CUDA 时使用 GPU，否则使用 CPU。每次运行都会在 `demo/results/bcv-.../` 新建结果目录，包含三维预测标签 `prediction.nii.gz` 和叠加预览图 `overlay.png`，不会覆盖已有结果。

这例测试图像**没有随附真实分割标签**，因此 demo 仅供定性查看，不能据此计算 Dice。仓库未提供 OASIS、ADNI 的图像或权重；它们的输入要求、所选模型记录及数据获取说明见 [demo/README.md](demo/README.md)。

## 数据准备

完整训练数据不随仓库提供。当前私有审阅仓库仅包含上述 BCV 单例的预处理测试图像、图谱图像与标签，以及 BCV 分割权重。上述文件的再分发权限尚待确认，确认之前不要将仓库转为公开；具体访问条件见 [demo/README.md](demo/README.md)。若要运行下面的完整训练与评估流程，需要按照各数据集的访问条件自行准备数据，并通过环境变量指定根目录。

### BCV

BCV 使用图像和标签成对存放的平铺目录：

```text
<bcv_root>/
|-- img0001_bcv_CT.nii.gz
|-- seg0001_bcv_CT.nii.gz
|-- img0002_bcv_CT.nii.gz
|-- seg0002_bcv_CT.nii.gz
`-- ...
```

默认图谱为病例 27；可用 `OPAL_REG_BCV_ATLAS_ID` 和 `OPAL_SEG_BCV_ATLAS_ID` 显式选择其他图谱。

### ADNI

ADNI 按图谱、训练集和测试集分别存放：

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

ADNI 需要申请访问；仓库不包含其受限图像或图谱。

### OASIS

OASIS 每个受试者一个目录，包含归一化图像和分割标签：

```text
<oasis_root>/
|-- <subject_001>/
|   |-- aligned_norm.nii.gz
|   `-- aligned_seg35.nii.gz
`-- ...
```

demo 文档中的 OASIS 示例使用完整分辨率的 `aligned_norm.nii.gz`，不是下采样的 `aligned_norm_small.nii.gz`；原始 OASIS-1 文件也不能直接替代该预处理版本。

## 完整复现流程

以下以 BCV 为例，所有命令都从仓库根目录运行。请先准备**完整 BCV 数据集**，将示例路径替换为实际路径；仓库内的单例 demo 文件不足以完成训练和定量评估。

```bash
export DATA_ROOT=/path/to/data
```

### 1. 训练配准模型

```bash
OPAL_REG_DATASET_KEY=BCV \
OPAL_REG_SOURCE_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_REG_BCV_ATLAS_ID=27 \
python train_reg.py
```

实验目录位于 `output/Reg/`。训练脚本也可以在训练结束后自动执行配准推理。

### 2. 导出配准生成的合成数据

将 `<reg_experiment>` 替换为实际配准实验目录名。`OPAL_REG_INF_MODEL_DIR` 指向整个实验目录；合成图像和标签通常导出到其中的 `train_output/` 与 `test_output/`。

```bash
OPAL_REG_INF_MODEL_DIR="output/Reg/<reg_experiment>" \
OPAL_REG_INF_DATA_ROOT="$DATA_ROOT/BCV" \
OPAL_REG_INF_BCV_ATLAS_ID=27 \
OPAL_REG_INF_EXPORT_TRAIN=1 \
OPAL_REG_INF_EXPORT_TEST=1 \
python inference_reg.py
```

### 3. 训练分割模型

`OPAL_SEG_SYN_ROOT_DIR` 同样指向**整个配准实验目录**，不是单独的 `train_output/` 或某个 NIfTI 文件。图谱先验 token 路径在此显式启用。

```bash
OPAL_SEG_REAL_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_SEG_SYN_ROOT_DIR="output/Reg/<reg_experiment>" \
OPAL_SEG_BCV_ATLAS_ID=27 \
OPAL_SEG_ENABLE_ATLAS_TOKEN_PATH=1 \
python train_seg.py
```

分割实验目录位于 `output/Seg/`。

### 4. 分割推理与评价

将 `<seg_experiment>` 替换为实际分割实验目录名：

```bash
OPAL_SEG_INF_MODEL_DIR="output/Seg/<seg_experiment>" \
OPAL_SEG_INF_REAL_ROOT_DIR="$DATA_ROOT/BCV" \
OPAL_SEG_INF_BCV_ATLAS_ID=27 \
OPAL_SEG_INF_ENABLE_ATLAS_TOKEN_PATH=1 \
OPAL_SEG_INF_EXPORT_TEST=1 \
OPAL_SEG_INF_COMPUTE_HD95=1 \
python inference_seg.py
```

预测标签和评价汇总写入分割实验目录。其他可配置项由 `OPAL_REG_*`、`OPAL_REG_INF_*`、`OPAL_SEG_*` 和 `OPAL_SEG_INF_*` 环境变量控制。

## 引用

论文发表后补充正式引用信息。
