"""OPAL-Net 数据集定义。

阅读提示：
- atlas-to-subject / atlas-to-case 类数据集主要服务于 registration；
- `*_Syn_Dataset` 主要服务于 segmentation，负责对齐 real 样本和 reg 导出的 syn 样本；
- 推理模式下会额外返回路径/文件名，方便把输出结果写回磁盘。
"""

from torch.utils.data import Dataset
import os
from pathlib import Path
import re
import SimpleITK as sitk
import torch
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 避免部分环境下 OpenMP 重复初始化报错
from monai.transforms import apply_transform


IXI_V11_ATLAS_DIR = "/home/administrator/MyItem/PROS-Net_02/data/IXI_Dataset/atlas_ood"
IXI_NCC_ATLAS_IMG_PATH = "/home/administrator/MyItem/PROS-Net_02/data/IXI_Dataset_NCC/subject_215.nii.gz"
IXI_NCC_ATLAS_LABEL_PATH = "/home/administrator/MyItem/PROS-Net_02/data/IXI_Dataset_NCC/subject_215_label.nii.gz"


def _resolve_ixi_atlas_paths(
    atlas_dir: str | None = None,
    atlas_img_path: str | None = None,
    atlas_label_path: str | None = None,
) -> tuple[str, str]:
    if atlas_img_path or atlas_label_path:
        if not atlas_img_path or not atlas_label_path:
            raise ValueError("atlas_img_path and atlas_label_path must be provided together")
        if not os.path.exists(atlas_img_path) or not os.path.exists(atlas_label_path):
            raise FileNotFoundError(
                f"IXI atlas files not found: {atlas_img_path} / {atlas_label_path}"
            )
        return atlas_img_path, atlas_label_path

    atlas_dir = atlas_dir or IXI_V11_ATLAS_DIR
    if not os.path.isdir(atlas_dir):
        raise FileNotFoundError(f"IXI atlas dir not found: {atlas_dir}")
    atlas_img_path = os.path.join(atlas_dir, "atlas.nii.gz")
    atlas_label_path = os.path.join(atlas_dir, "atlas_label.nii.gz")
    if not os.path.exists(atlas_img_path) or not os.path.exists(atlas_label_path):
        raise FileNotFoundError(
            f"IXI atlas files not found: {atlas_img_path} / {atlas_label_path}"
        )
    return atlas_img_path, atlas_label_path


def _stem_from_syn_path(file_path: str, suffix_key: str) -> str:
    return os.path.basename(file_path).replace(f"_{suffix_key}.nii.gz", "")


def _resolve_existing_dir(root_dir: str, preferred_children: tuple[str, ...]) -> str:
    for child in preferred_children:
        candidate = os.path.join(root_dir, child)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(root_dir):
        return root_dir
    raise FileNotFoundError(f"Directory not found: {root_dir}")


def _replace_syn_suffix(file_path: str, old_suffix: str, new_suffix: str) -> str:
    name = os.path.basename(file_path)
    expected_tail = f"_{old_suffix}.nii.gz"
    if not name.endswith(expected_tail):
        raise ValueError(f"Unexpected synthetic file name: {file_path}")
    stem = name[: -len(expected_tail)]
    return os.path.join(os.path.dirname(file_path), f"{stem}_{new_suffix}.nii.gz")


def _align_label_sitk_to_image(label_sitk: sitk.Image, image_sitk: sitk.Image) -> sitk.Image:
    """用最近邻把标签重采样到图像网格，主要用于 BCH 这类 image/label size 不一致的数据。"""
    if label_sitk.GetSize() == image_sitk.GetSize():
        return label_sitk
    return sitk.Resample(
        label_sitk,
        image_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        label_sitk.GetPixelID(),
    )


def _normalize_bch_image_np(image_np: np.ndarray) -> np.ndarray:
    """BCH CT 图像归一化到 [0, 1]，保留 0 背景。"""
    image_np = image_np.astype(np.float32, copy=False)
    max_val = float(image_np.max())
    if max_val <= 0:
        return image_np
    return image_np / max_val


def _normalize_bch2_image_np(image_np: np.ndarray) -> np.ndarray:
    """BCH2 使用原始 CT，按固定窗口裁剪后缩放到 [0, 1]。"""
    image_np = image_np.astype(np.float32, copy=False)
    image_np = np.clip(image_np, -1000.0, 2000.0)
    return (image_np + 1000.0) / 3000.0


def _adni_train_val_split(img_dir: str, val_ratio: float = 0.2, split_seed: int = 0):
    """固定随机种子，把 ADNI Train 再切成 train / val。"""
    all_img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".nii.gz"))
    n = len(all_img_files)
    if n == 0:
        raise RuntimeError(f"No ADNI images found in: {img_dir}")

    val_count = int(n * float(val_ratio))
    if n > 1 and val_count < 1:
        val_count = 1
    if val_count >= n:
        val_count = n - 1

    rng = np.random.RandomState(int(split_seed))
    perm = rng.permutation(n)
    val_idx = set(perm[:val_count].tolist())
    train_files = [all_img_files[i] for i in range(n) if i not in val_idx]
    val_files = [all_img_files[i] for i in range(n) if i in val_idx]
    return train_files, val_files


class ADNI_Dataset(Dataset):
    """ADNI atlas-to-subject 配准数据集。"""

    def __init__(self, source_root_dir, template_dir=None, transform2img=None, transform2both=None, is_training=True, is_val=False, inference=False):
        self.source_root_dir = source_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        self.atlas_path = os.path.join(source_root_dir, 'atlas', 'ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392-n3-resample-reorient-stripped-n3.nii.gz')
        self.atlas_label_path = os.path.join(source_root_dir, 'atlas', 'label_ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392-n3-resample-reorient-stripped-n3.nii.gz')

        if is_training:
            self.img_dir = os.path.join(source_root_dir, 'Train', 'img')
            self.label_dir = os.path.join(source_root_dir, 'Train', 'label')
        elif is_val:
            raise ValueError(
                "ADNI_Dataset has no standalone val split. "
                "Use ADNI_Split_Dataset for train/val/test separation."
            )
        else:
            self.img_dir = os.path.join(source_root_dir, 'Test', 'img')
            self.label_dir = os.path.join(source_root_dir, 'Test', 'label')
        
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.img_files)
    
    def getinfo(self):
        first_img_path = os.path.join(self.img_dir, self.img_files[0])
        img_size = sitk.GetArrayFromImage(sitk.ReadImage(first_img_path)).shape
        num_classes = int(sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path)).max().item()) + 1
        return img_size, num_classes

    def __getitem__(self, idx):
        # Target (Subject)
        target_img_name = self.img_files[idx]
        target_path = os.path.join(self.img_dir, target_img_name)
        label_name = 'label_' + target_img_name
        target_label_path = os.path.join(self.label_dir, label_name)

        # Source (Atlas)
        source_path = self.atlas_path
        source_label_path = self.atlas_label_path

        # Check if files exist
        if not os.path.exists(target_path):
             raise FileNotFoundError(f"Target path not found: {target_path}")
        
        source_image_sitk = sitk.ReadImage(source_path)
        target_image_sitk = sitk.ReadImage(target_path)
        source_label_sitk = sitk.ReadImage(source_label_path)
        target_label_sitk = sitk.ReadImage(target_label_path)
        
        source_image_np = sitk.GetArrayFromImage(source_image_sitk)
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk)

        source_label_np = source_label_np.astype(int)
        target_label_np = target_label_np.astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)
        
        source_data_dict = {'image': source_image, "label": source_label}
        target_data_dict = {'image': target_image, "label": target_label}
        # Transforms
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
             source_data = apply_transform(self.transform2both, source_data_dict)
             target_data = apply_transform(self.transform2both, target_data_dict)
             
             source_image = source_data['image']
             source_label = source_data['label']
             target_image = target_data['image']
             target_label = target_data['label']

        if self.inference:
            return source_image, target_image, source_label, target_label, target_path, target_img_name
        return source_image, target_image, source_label, target_label


class ADNI_Split_Dataset(Dataset):
    """ADNI 特化配准数据集：原 Train 固定切 train/val，原 Test 作为 test。"""

    def __init__(
        self,
        source_root_dir,
        template_dir=None,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=False,
        inference=False,
        val_ratio=0.2,
        split_seed=0,
    ):
        self.source_root_dir = source_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        self.atlas_path = os.path.join(source_root_dir, 'atlas', 'ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392-n3-resample-reorient-stripped-n3.nii.gz')
        self.atlas_label_path = os.path.join(source_root_dir, 'atlas', 'label_ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392-n3-resample-reorient-stripped-n3.nii.gz')

        train_img_dir = os.path.join(source_root_dir, 'Train', 'img')
        train_label_dir = os.path.join(source_root_dir, 'Train', 'label')
        test_img_dir = os.path.join(source_root_dir, 'Test', 'img')
        test_label_dir = os.path.join(source_root_dir, 'Test', 'label')

        if is_training or is_val:
            train_files, val_files = _adni_train_val_split(train_img_dir, val_ratio=val_ratio, split_seed=split_seed)
            self.img_dir = train_img_dir
            self.label_dir = train_label_dir
            self.img_files = train_files if is_training else val_files
        else:
            self.img_dir = test_img_dir
            self.label_dir = test_label_dir
            self.img_files = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.nii.gz'))

    def __len__(self):
        return len(self.img_files)

    def getinfo(self):
        first_img_path = os.path.join(self.img_dir, self.img_files[0])
        img_size = sitk.GetArrayFromImage(sitk.ReadImage(first_img_path)).shape
        num_classes = int(sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path)).max().item()) + 1
        return img_size, num_classes

    def __getitem__(self, idx):
        target_img_name = self.img_files[idx]
        target_path = os.path.join(self.img_dir, target_img_name)
        target_label_path = os.path.join(self.label_dir, 'label_' + target_img_name)

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target path not found: {target_path}")

        source_image_np = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))
        target_image_np = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        source_label_np = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path)).astype(int)
        target_label_np = sitk.GetArrayFromImage(sitk.ReadImage(target_label_path)).astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {"image": source_image, "label": source_label}
        target_data_dict = {"image": target_image, "label": target_label}
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data["image"]
            source_label = source_data["label"]
            target_image = target_data["image"]
            target_label = target_data["label"]

        if self.inference:
            return source_image, target_image, source_label, target_label, target_path, target_img_name
        return source_image, target_image, source_label, target_label

class ADNI_Split_Syn_Dataset(Dataset):
    """ADNI 特化 syn 数据集：原 Train 固定切 train/val，原 Test 作为 test。"""

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key="final",
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        val_ratio=0.2,
        split_seed=0,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        _suffixes = {"final": ("final_image", "final_label"), "coarse": ("coarse_image", "coarse_label")}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        train_img_dir = os.path.join(real_root_dir, "Train", "img")
        train_label_dir = os.path.join(real_root_dir, "Train", "label")
        test_img_dir = os.path.join(real_root_dir, "Test", "img")
        test_label_dir = os.path.join(real_root_dir, "Test", "label")

        train_files, val_files = _adni_train_val_split(train_img_dir, val_ratio=val_ratio, split_seed=split_seed)
        if is_training:
            self.image_dir = train_img_dir
            self.label_dir = train_label_dir
            selected_img_files = train_files
            split_dir_name = "train_output"
        elif is_val:
            self.image_dir = train_img_dir
            self.label_dir = train_label_dir
            selected_img_files = val_files
            split_dir_name = "val_output"
        else:
            self.image_dir = test_img_dir
            self.label_dir = test_label_dir
            selected_img_files = sorted(f for f in os.listdir(self.image_dir) if f.endswith('.nii.gz'))
            split_dir_name = "test_output"

        self.syn_data_dir = os.path.join(syn_root_dir, split_dir_name)
        if not os.path.isdir(self.syn_data_dir):
            raise FileNotFoundError(f"ADNI syn split dir not found: {self.syn_data_dir}")

        real_image_paths = [os.path.join(self.image_dir, f) for f in selected_img_files]
        real_label_paths = [os.path.join(self.label_dir, 'label_' + f) for f in selected_img_files]
        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_image_suffix}.nii.gz")
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_label_suffix}.nii.gz") or f.startswith('pred_label_')
        )

        real_image_map = {self._stem_from_real_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_label_path(p): p for p in real_label_paths if os.path.exists(p)}
        syn_image_map = {_stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {_stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}

        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
        )
        if len(common_keys) == 0:
            raise RuntimeError(
                f"No matched ADNI split syn samples found between {self.image_dir} and {self.syn_data_dir}"
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _stem_from_real_path(self, file_path: str) -> str:
        name = os.path.basename(file_path).replace('.nii.gz', '')
        return name

    def _stem_from_label_path(self, file_path: str) -> str:
        name = os.path.basename(file_path).replace('.nii.gz', '')
        if name.startswith('label_'):
            name = name[len('label_'):]
        return name

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_np = sitk.GetArrayFromImage(sitk.ReadImage(real_image_path))
        real_label_np = sitk.GetArrayFromImage(sitk.ReadImage(real_label_path)).astype(int)
        syn_image_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_image_path))
        syn_label_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_label_path)).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {"image": real_image, "label": real_label}
        syn_data_dict = {"image": syn_image, "label": syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data["image"]
            real_label = real_data["label"]
            syn_image = syn_data["image"]
            syn_label = syn_data["label"]

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


class IXI_Dataset(Dataset):
    """IXI atlas-to-subject 配准数据集，支持 atlas 目录或直接 atlas 文件路径。"""

    def __init__(
        self,
        root_dir,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        atlas_dir: str | None = None,
        atlas_img_path: str | None = None,
        atlas_label_path: str | None = None,
    ):
        self.root_dir = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        atlas_img_path, atlas_label_path = _resolve_ixi_atlas_paths(
            atlas_dir=atlas_dir,
            atlas_img_path=atlas_img_path,
            atlas_label_path=atlas_label_path,
        )

        if is_training:
            self.target_img_dir = os.path.join(root_dir, 'Train', 'img')
            self.target_label_dir = os.path.join(root_dir, 'Train', 'label')
        elif is_val:
            self.target_img_dir = os.path.join(root_dir, 'Val', 'img')
            self.target_label_dir = os.path.join(root_dir, 'Val', 'label')
        else:
            self.target_img_dir = os.path.join(root_dir, 'Test', 'img')
            self.target_label_dir = os.path.join(root_dir, 'Test', 'label')
        
        self.target_img_names = [f for f in os.listdir(self.target_img_dir) if f.endswith('.nii.gz')]
        # self.target_label_names = [f for f in os.listdir(self.target_label_dir) if f.endswith('.nii.gz')]
        self.target_img_names.sort()
        # self.target_label_names.sort()

        self.atlas_img_path = atlas_img_path
        self.atlas_label_path = atlas_label_path

        print(f"atlas_img_path: {self.atlas_img_path}")
        print(f"atlas_label_path: {self.atlas_label_path}")
        print(f"dataset len:{len(self.target_img_names)}")


    def __len__(self):
        return len(self.target_img_names)

    def __getitem__(self, idx):
        target_img_name = self.target_img_names[idx]
        target_img_path = os.path.join(self.target_img_dir, target_img_name)
        target_label_stem = target_img_name[:-7]
        target_label_name = target_label_stem + '_label.nii.gz'
        target_label_path = os.path.join(self.target_label_dir, target_label_name)

        if not os.path.exists(target_img_path):
            raise FileNotFoundError(f"Target path not found: {target_img_path}")
        if not os.path.exists(target_label_path):
            raise FileNotFoundError(f"Target label path not found: {target_label_path}")

        source_image_sitk = sitk.ReadImage(self.atlas_img_path)
        source_label_sitk = sitk.ReadImage(self.atlas_label_path)
        target_image_sitk = sitk.ReadImage(target_img_path)
        target_label_sitk = sitk.ReadImage(target_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk)
        source_label_np = sitk.GetArrayFromImage(source_label_sitk).astype(int)
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk).astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {'image': source_image, 'label': source_label}
        target_data_dict = {'image': target_image, 'label': target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data['image']
            source_label = source_data['label']
            target_image = target_data['image']
            target_label = target_data['label']

        if self.inference:
            return source_image, target_image, source_label, target_label, target_img_path, target_img_name
        return source_image, target_image, source_label, target_label


class IXI_Syn_Dataset(Dataset):
    """IXI seg 训练集：real(train) + reg 导出的 syn(final/coarse)。"""

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key="final",
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        _suffixes = {"final": ("final_image", "final_label"), "coarse": ("coarse_image", "coarse_label")}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        self.image_dir = os.path.join(real_root_dir, "Train", "img")
        self.label_dir = os.path.join(real_root_dir, "Train", "label")
        self.syn_data_dir = os.path.join(syn_root_dir, "train_output")
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir

        real_image_paths = sorted(
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")
        )
        real_label_paths = sorted(
            os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir) if f.endswith(".nii.gz")
        )
        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_image_suffix}.nii.gz")
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_label_suffix}.nii.gz")
        )

        real_image_map = {self._stem_from_real_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_label_path(p): p for p in real_label_paths}
        syn_image_map = {_stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {_stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}

        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
        )
        if len(common_keys) == 0:
            raise RuntimeError(
                f"No matched IXI seg samples found between {real_root_dir} and {self.syn_data_dir}"
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _stem_from_real_path(self, file_path: str) -> str:
        return os.path.basename(file_path).replace(".nii.gz", "")

    def _stem_from_label_path(self, file_path: str) -> str:
        name = os.path.basename(file_path).replace(".nii.gz", "")
        if name.endswith("_label"):
            name = name[: -len("_label")]
        return name

    def __len__(self) -> int:
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_sitk = sitk.ReadImage(real_image_path)
        real_label_sitk = sitk.ReadImage(real_label_path)
        syn_image_sitk = sitk.ReadImage(syn_image_path)
        syn_label_sitk = sitk.ReadImage(syn_label_path)

        real_image_np = sitk.GetArrayFromImage(real_image_sitk)
        real_label_np = sitk.GetArrayFromImage(real_label_sitk).astype(int)
        syn_image_np = sitk.GetArrayFromImage(syn_image_sitk)
        syn_label_np = sitk.GetArrayFromImage(syn_label_sitk).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {"image": real_image, "label": real_label}
        syn_data_dict = {"image": syn_image, "label": syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data["image"]
            real_label = real_data["label"]
            syn_image = syn_data["image"]
            syn_label = syn_data["label"]

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


IXI_NCC_Dataset = IXI_Dataset
IXI_NCC_Syn_Dataset = IXI_Syn_Dataset

class OASIS_Dataset(Dataset):
    """OASIS atlas-to-subject 配准数据集。"""

    def __init__(
        self, 
        root_dir,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False
    ):
        super().__init__()
        self.data_root = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        self.data_list = sorted(
            [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
        )

        self.atlas_data = self.data_list[10]
        if is_training:
            self.data_list = self.data_list[:335]
        elif is_val:
            self.data_list = self.data_list[335:375]
        else:
            self.data_list = self.data_list[375:]

        # 删掉 atlas，避免 target 与 atlas 重合
        if self.atlas_data in self.data_list:
            self.data_list.remove(self.atlas_data)

        # OASIS aligned_seg35 默认已是连续标签，这里保留映射接口但默认不做重映射。
        self.label_mapping = {}
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dir = self.data_list[idx]
        target_name = os.path.basename(data_dir)
        target_image_name = f"{target_name}.nii.gz"

        source_image_path = os.path.join(self.data_root, self.atlas_data, "aligned_norm.nii.gz")
        source_label_path = os.path.join(self.data_root, self.atlas_data, "aligned_seg35.nii.gz")
        target_image_path = os.path.join(self.data_root, data_dir, "aligned_norm.nii.gz")
        target_label_path = os.path.join(self.data_root, data_dir, "aligned_seg35.nii.gz")

        for p in [source_image_path, source_label_path, target_image_path, target_label_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"OASIS file not found: {p}")

        
        source_image_sitk = sitk.ReadImage(source_image_path)
        source_label_sitk = sitk.ReadImage(source_label_path)
        target_image_sitk = sitk.ReadImage(target_image_path)
        target_label_sitk = sitk.ReadImage(target_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk)
        source_label_np = sitk.GetArrayFromImage(source_label_sitk).astype(int)
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk).astype(int)

        source_image_np = np.maximum(source_image_np, 0)
        target_image_np = np.maximum(target_image_np, 0)
        
        for old_label, new_label in self.label_mapping.items():
            source_label_np[source_label_np == old_label] = new_label
            target_label_np[target_label_np == old_label] = new_label

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {"image": source_image, "label": source_label}
        target_data_dict = {"image": target_image, "label": target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data["image"]
            source_label = source_data["label"]
            target_image = target_data["image"]
            target_label = target_data["label"]
        if self.inference:
            return source_image, target_image, source_label, target_label, target_image_path, target_image_name
        return source_image, target_image, source_label, target_label

class OASIS_Syn_Dataset(Dataset):
    """OASIS seg 训练集：real(target) + reg 导出的 syn(final/coarse)。"""

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key="final",
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        atlas_subject=None,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        _suffixes = {"final": ("final_image", "final_label"), "coarse": ("coarse_image", "coarse_label")}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        all_subjects = sorted(
            [d for d in os.listdir(real_root_dir) if os.path.isdir(os.path.join(real_root_dir, d))]
        )
        if len(all_subjects) <= 10:
            raise RuntimeError(f"OASIS subject count is too small in {real_root_dir}")
        atlas_subject = all_subjects[10]
        train_subjects = all_subjects[:335]
        if atlas_subject in train_subjects:
            train_subjects.remove(atlas_subject)

        self.syn_data_dir = os.path.join(syn_root_dir, 'train_output')
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir
        if not os.path.isdir(self.syn_data_dir):
            raise FileNotFoundError(f'BCH syn dir not found: {self.syn_data_dir}')

        real_image_paths = [
            os.path.join(real_root_dir, sub, 'aligned_norm.nii.gz')
            for sub in train_subjects
            if os.path.exists(os.path.join(real_root_dir, sub, 'aligned_norm.nii.gz'))
        ]
        real_label_paths = [
            os.path.join(real_root_dir, sub, 'aligned_seg35.nii.gz')
            for sub in train_subjects
            if os.path.exists(os.path.join(real_root_dir, sub, 'aligned_seg35.nii.gz'))
        ]
        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_image_suffix}.nii.gz")
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_label_suffix}.nii.gz")
        )

        real_image_map = {self._stem_from_real_image_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_real_label_path(p): p for p in real_label_paths}
        syn_image_map = {_stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {_stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}
        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
        )
        if len(common_keys) == 0:
            raise RuntimeError(
                f"No matched OASIS syn samples found between {real_root_dir} and {self.syn_data_dir}"
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _stem_from_real_image_path(self, file_path: str) -> str:
        return os.path.basename(os.path.dirname(file_path))

    def _stem_from_real_label_path(self, file_path: str) -> str:
        return os.path.basename(os.path.dirname(file_path))

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_np = sitk.GetArrayFromImage(sitk.ReadImage(real_image_path))
        real_label_np = sitk.GetArrayFromImage(sitk.ReadImage(real_label_path)).astype(int)
        syn_image_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_image_path))
        syn_label_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_label_path)).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {"image": real_image, "label": real_label}
        syn_data_dict = {"image": syn_image, "label": syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data["image"]
            real_label = real_data["label"]
            syn_image = syn_data["image"]
            syn_label = syn_data["label"]

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


def _bcv_train_pool_ids(atlas_id: int = 27) -> list[int]:
    return [i for i in range(1, 36) if i != atlas_id and not (11 <= i <= 20)]

def _bcv_val_pool_ids(atlas_id: int = 27) -> list[int]:
    return [i for i in range(36, 41) if i != atlas_id and not (i == 36)]


def _bcv_test_ids() -> list[int]:
    return list(range(61, 81))



def _bcv_kfold_train_val_ids(
    atlas_id: int = 27,
    fold_idx: int = 0,
    num_folds: int = 5,
    split_seed: int = 0,
) -> tuple[list[int], list[int]]:
    if num_folds < 2:
        raise ValueError(f"BCV k-fold expects num_folds >= 2, got {num_folds}")
    if not (0 <= fold_idx < num_folds):
        raise ValueError(f"BCV k-fold expects 0 <= fold_idx < {num_folds}, got {fold_idx}")

    ids = list(_bcv_train_pool_ids(atlas_id))
    rng = np.random.RandomState(split_seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    fold_sizes = [len(shuffled) // num_folds] * num_folds
    for i in range(len(shuffled) % num_folds):
        fold_sizes[i] += 1

    folds: list[list[int]] = []
    cursor = 0
    for fold_size in fold_sizes:
        folds.append(sorted(shuffled[cursor:cursor + fold_size]))
        cursor += fold_size

    val_ids = folds[fold_idx]
    train_ids = sorted(case_id for i, fold in enumerate(folds) if i != fold_idx for case_id in fold)
    return train_ids, val_ids


class BCV_Dataset(Dataset):
    """BCV 配准数据集（flat 目录版本，atlas 默认固定为 27，可按需覆写）。

    当前口径：
    - ``train_pool``: 返回完整 24 个训练池 case，给上游 reg 一次性训练/导出。
    - ``kfold``: 在这 24 个 case 上做可复现五折；test 仍固定 61..80。

    这里不再保留旧的固定 val=36..40 兼容逻辑。
    """

    default_atlas_id: int = 27

    def __init__(
        self,
        root_dir,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        split_mode: str = "train_pool",
        fold_idx: int = 0,
        num_folds: int = 5,
        split_seed: int = 0,
        atlas_id: int | None = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.split_mode = str(split_mode).strip().lower()
        self.fold_idx = int(fold_idx)
        self.num_folds = int(num_folds)
        self.split_seed = int(split_seed)
        self.atlas_id = self.default_atlas_id if atlas_id is None else int(atlas_id)

        self.atlas_img_path = os.path.join(root_dir, f"img{self.atlas_id:04d}_bcv_CT.nii.gz")
        self.atlas_label_path = os.path.join(root_dir, f"seg{self.atlas_id:04d}_bcv_CT.nii.gz")
        if not os.path.exists(self.atlas_img_path) or not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(
                f"BCV atlas not found: {self.atlas_img_path} / {self.atlas_label_path}"
            )

        target_ids = self._resolve_target_ids(is_training=is_training, is_val=is_val)

        self.target_img_paths: list[str] = []
        self.target_label_paths: list[str] = []
        self.target_img_names: list[str] = []
        for case_id in target_ids:
            img_name = f"img{case_id:04d}_bcv_CT.nii.gz"
            label_name = f"seg{case_id:04d}_bcv_CT.nii.gz"
            img_path = os.path.join(root_dir, img_name)
            label_path = os.path.join(root_dir, label_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_img_names.append(img_name)

        if len(self.target_img_paths) == 0:
            raise RuntimeError(f"No BCV samples found in {root_dir} for split_mode={self.split_mode}")

    def _resolve_target_ids(self, is_training: bool, is_val: bool) -> list[int]:
        if self.split_mode == "kfold":
            train_ids, val_ids = _bcv_kfold_train_val_ids(
                atlas_id=self.atlas_id,
                fold_idx=self.fold_idx,
                num_folds=self.num_folds,
                split_seed=self.split_seed,
            )
            if is_training:
                return train_ids
            if is_val:
                return val_ids
            return _bcv_test_ids()

        if self.split_mode == "train_pool":
            if is_training:
                return _bcv_train_pool_ids(self.atlas_id)
            elif is_val:
                return _bcv_val_pool_ids(self.atlas_id)
            return _bcv_test_ids()

        raise ValueError(
            f"Unsupported BCV split_mode: {self.split_mode}. Expected 'kfold' or 'train_pool'"
        )

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        target_img_path = self.target_img_paths[idx]
        target_label_path = self.target_label_paths[idx]
        target_img_name = self.target_img_names[idx]

        source_image_sitk = sitk.ReadImage(self.atlas_img_path)
        source_label_sitk = sitk.ReadImage(self.atlas_label_path)
        target_image_sitk = sitk.ReadImage(target_img_path)
        target_label_sitk = sitk.ReadImage(target_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk)
        source_label_np = sitk.GetArrayFromImage(source_label_sitk).astype(int)
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk).astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {"image": source_image, "label": source_label}
        target_data_dict = {"image": target_image, "label": target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            trans_source_data = apply_transform(self.transform2both, source_data_dict)
            trans_target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = trans_source_data["image"]
            source_label = trans_source_data["label"]
            target_image = trans_target_data["image"]
            target_label = trans_target_data["label"]

        if self.inference:
            return source_image, target_image, source_label, target_label, target_img_path, target_img_name
        return source_image, target_image, source_label, target_label


class BCV_Syn_Dataset(Dataset):
    """BCV seg 训练集：real(target) + reg 导出的 syn(final_* / coarse_*).

    split_mode:
    - ``train_pool``: 返回完整 24 个 train pool，适合先固定上游 reg 结果。
    - ``kfold``: 在 24 个 train pool case 上做可复现五折，切分 train/val。

    不再保留旧的 fixed-val 兼容模式。
    """

    default_atlas_id: int = 27

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key="final",
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        atlas_subject=None,
        split_mode: str = "train_pool",
        fold_idx: int = 0,
        num_folds: int = 5,
        split_seed: int = 0,
        atlas_id: int | None = None,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        self.split_mode = str(split_mode).strip().lower()
        self.fold_idx = int(fold_idx)
        self.num_folds = int(num_folds)
        self.split_seed = int(split_seed)
        self.atlas_id = self.default_atlas_id if atlas_id is None else int(atlas_id)
        _suffixes = {"final": ("final_image", "final_label"), "coarse": ("coarse_image", "coarse_label")}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference

        self.syn_data_dir = os.path.join(syn_root_dir, "train_output")
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir

        if self.split_mode == "train_pool":
            selected_ids = _bcv_train_pool_ids(self.atlas_id)
        elif self.split_mode == "kfold":
            train_ids, val_ids = _bcv_kfold_train_val_ids(
                atlas_id=self.atlas_id,
                fold_idx=self.fold_idx,
                num_folds=self.num_folds,
                split_seed=self.split_seed,
            )
            selected_ids = train_ids if is_training else val_ids if is_val else []
        else:
            raise ValueError(
                f"Unsupported BCV split_mode: {self.split_mode}. Expected 'kfold' or 'train_pool'"
            )

        selected_keys = {f"img{case_id:04d}_bcv_CT" for case_id in selected_ids}

        real_image_paths = []
        real_label_paths = []
        for case_id in _bcv_train_pool_ids(self.atlas_id):
            img_path = os.path.join(real_root_dir, f"img{case_id:04d}_bcv_CT.nii.gz")
            label_path = os.path.join(real_root_dir, f"seg{case_id:04d}_bcv_CT.nii.gz")
            if os.path.exists(img_path) and os.path.exists(label_path):
                real_image_paths.append(img_path)
                real_label_paths.append(label_path)

        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_image_suffix}.nii.gz")
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f"{self.syn_label_suffix}.nii.gz")
        )
        real_image_map = {self._stem_from_real_image_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_real_label_path(p): p for p in real_label_paths}
        syn_image_map = {self._stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {self._stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}

        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
            & selected_keys
        )
        if len(common_keys) == 0:
            raise RuntimeError(
                f"No matched BCV syn samples found between {real_root_dir} and {self.syn_data_dir} "
                f"for split_mode={self.split_mode}, fold_idx={self.fold_idx}"
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _normalize_subject_key(self, raw_name: str) -> str:
        name = raw_name
        if name.startswith("pred_image_"):
            name = name.replace("pred_image_", "", 1)
        if name.startswith("pred_label_"):
            name = name.replace("pred_label_", "", 1)
        if name.startswith("seg") and name.endswith("_bcv_CT"):
            name = "img" + name[3:]
        match = re.fullmatch(r"(?:img)?(\d{4})(?:_bcv_CT)?", name)
        if match:
            return f"img{int(match.group(1)):04d}_bcv_CT"
        return name

    def _stem_from_real_image_path(self, file_path: str) -> str:
        name = os.path.basename(file_path).replace(".nii.gz", "")
        return self._normalize_subject_key(name)

    def _stem_from_real_label_path(self, file_path: str) -> str:
        name = os.path.basename(file_path).replace(".nii.gz", "")
        return self._normalize_subject_key(name)

    def _stem_from_syn_path(self, file_path: str, suffix_key: str) -> str:
        name = os.path.basename(file_path).replace(f"_{suffix_key}.nii.gz", "")
        return self._normalize_subject_key(name)

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_np = sitk.GetArrayFromImage(sitk.ReadImage(real_image_path))
        real_label_np = sitk.GetArrayFromImage(sitk.ReadImage(real_label_path)).astype(int)
        syn_image_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_image_path))
        syn_label_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_label_path)).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {"image": real_image, "label": real_label}
        syn_data_dict = {"image": syn_image, "label": syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data["image"]
            real_label = real_data["label"]
            syn_image = syn_data["image"]
            syn_label = syn_data["label"]

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


class BCH_Dataset(Dataset):
    """BCH 配准数据集。

    主口径优先采用《数据集整理(1).pdf》里的 BCH-CT 规则：
    - 数据目录：``.../BCH_by_ZF/CT/BCH_CT001``
    - atlas：``BCH_CT009``
    - 划分：``1-70`` train，``71-88`` val，``89-118`` test

    同时保留旧版 fallback：
    1. ``train/val/test`` 目录结构；
    2. 参考 ``data/zf/dataset.py`` 的平铺 ``BCH_00001`` 结构。
    """

    image_name: str = "brain_w_bone_small.nii.gz"
    image_name_norm: str = "brain_w_bone_small_norm.nii.gz"
    train_label_name: str = "Cranial_Head_bone_premask_250702.nii.gz"
    val_label_name: str = "Cranial_Head_bone_gt.nii.gz"
    zf_ct_image_name: str = "brain_small.nii.gz"
    zf_ct_label_name: str = "label_small.nii.gz"
    zf_ct_atlas_subject: str = "BCH_CT009"
    legacy_image_name: str = "brain_small_norm.nii.gz"
    legacy_target_label_name: str = "label_small_rigid.nii.gz"
    legacy_atlas_label_name: str = "label_small.nii.gz"
    legacy_label_mapping: dict[float, int] = {
        0.0: 0,
        1.0: 1,
        2.0: 2,
        3.0: 3,
        4.0: 4,
        5.0: 5,
        6.0: 6,
        7.0: 7,
        8.0: 8,
        9.0: 9,
        10.0: 10,
        11.0: 11,
        12.0: 12,
        13.0: 13,
        14.0: 14,
        15.0: 15,
        16.0: 16,
        17.0: 17,
        18.0: 18,
        19.0: 19,
        20.0: 20,
        21.0: 21,
        24.0: 22,
        25.0: 23,
        28.0: 24,
        29.0: 25,
        30.0: 26,
    }

    def __init__(
        self,
        root_dir,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        atlas_subject=None,
        atlas_data_root=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.atlas_data_root = atlas_data_root
        self.label_mapping: dict[float, int] = {}
        self.target_img_paths: list[str] = []
        self.target_label_paths: list[str] = []
        self.target_names: list[str] = []

        zf_ct_root = self._resolve_zf_ct_root(root_dir)
        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val")
        if zf_ct_root is not None:
            self.layout_mode = "zf_ct"
            self._init_zf_ct_layout(zf_ct_root, is_training, is_val, atlas_subject)
        elif os.path.isdir(train_dir) and os.path.isdir(val_dir):
            self.layout_mode = "split"
            self._init_split_layout(train_dir, val_dir, root_dir, is_training, is_val, atlas_subject)
        else:
            self.layout_mode = "legacy_flat"
            self._init_legacy_flat_layout(root_dir, is_training, is_val, atlas_data_root)

    def _resolve_zf_ct_root(self, root_dir: str) -> str | None:
        candidates = [
            root_dir,
            os.path.join(root_dir, 'CT'),
        ]
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            subject_dir = os.path.join(candidate, 'BCH_CT001')
            if os.path.isdir(subject_dir):
                return candidate
        return None

    def _resolve_existing_file(self, candidate_paths: list[str]) -> str:
        for file_path in candidate_paths:
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"No file found among candidates: {candidate_paths}")

    def _list_subjects(self, split_dir: str) -> list[str]:
        return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

    def _init_zf_ct_layout(self, ct_root, is_training, is_val, atlas_subject):
        self.label_mapping = dict(self.legacy_label_mapping)
        if atlas_subject is None:
            atlas_subject = self.zf_ct_atlas_subject
        self.atlas_subject = atlas_subject
        atlas_dir = os.path.join(ct_root, atlas_subject)
        self.atlas_img_path = os.path.join(atlas_dir, self.zf_ct_image_name)
        self.atlas_label_path = os.path.join(atlas_dir, self.zf_ct_label_name)
        if not os.path.exists(self.atlas_img_path) or not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(
                f"BCH zf_ct atlas file not found: {self.atlas_img_path} / {self.atlas_label_path}"
            )

        if is_training:
            id_iter = range(1, 71)
        elif is_val:
            id_iter = range(71, 89)
        else:
            id_iter = range(89, 119)

        for case_id in id_iter:
            subject = f"BCH_CT{case_id:03d}"
            if subject == self.atlas_subject:
                continue
            case_dir = os.path.join(ct_root, subject)
            img_path = os.path.join(case_dir, self.zf_ct_image_name)
            label_path = os.path.join(case_dir, self.zf_ct_label_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_names.append(subject)

        if len(self.target_img_paths) == 0:
            raise RuntimeError(f"No BCH zf_ct samples found in {ct_root} for the requested split")

    def _init_split_layout(self, train_dir, val_dir, root_dir, is_training, is_val, atlas_subject):
        self.train_dir = train_dir
        self.val_dir = val_dir

        val_subjects = self._list_subjects(self.val_dir)
        if len(val_subjects) == 0:
            raise RuntimeError(f"No val subjects found in {self.val_dir}")
        if atlas_subject is None:
            atlas_subject = "BCH_00009" if "BCH_00009" in val_subjects else val_subjects[0]
        if atlas_subject not in val_subjects:
            raise ValueError(f"atlas_subject {atlas_subject} not found in {self.val_dir}")
        self.atlas_subject = atlas_subject
        eval_subjects = [sub for sub in val_subjects if sub != self.atlas_subject]

        self.atlas_img_path = self._resolve_existing_file([
            os.path.join(self.val_dir, atlas_subject, self.image_name),
            os.path.join(self.val_dir, atlas_subject, self.image_name_norm),
        ])
        self.atlas_label_path = os.path.join(self.val_dir, atlas_subject, self.val_label_name)
        if not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(
                f"BCH atlas file not found: {self.atlas_img_path} / {self.atlas_label_path}"
            )

        if is_training:
            subjects = self._list_subjects(self.train_dir)
            label_name = self.train_label_name
            base_dir = self.train_dir
        elif is_val:
            subjects = eval_subjects
            label_name = self.val_label_name
            base_dir = self.val_dir
        else:
            test_dir = os.path.join(root_dir, "test")
            if os.path.isdir(test_dir):
                subjects = self._list_subjects(test_dir)
                label_name = self.val_label_name
                base_dir = test_dir
            else:
                subjects = eval_subjects
                label_name = self.val_label_name
                base_dir = self.val_dir

        for sub in subjects:
            img_path = self._resolve_existing_file([
                os.path.join(base_dir, sub, self.image_name),
                os.path.join(base_dir, sub, self.image_name_norm),
            ])
            label_path = os.path.join(base_dir, sub, label_name)
            if not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_names.append(sub)

        if len(self.target_img_paths) == 0:
            raise RuntimeError(f"No BCH samples found in split under {base_dir}")

    def _init_legacy_flat_layout(self, root_dir, is_training, is_val, atlas_data_root):
        if atlas_data_root is None:
            atlas_data_root = os.path.join(root_dir, 'atlas')
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"BCH root dir not found: {root_dir}")
        if not os.path.isdir(atlas_data_root):
            raise FileNotFoundError(f"BCH atlas_data_root not found: {atlas_data_root}")

        self.atlas_subject = os.path.basename(atlas_data_root.rstrip('/'))
        self.atlas_img_path = os.path.join(atlas_data_root, self.legacy_image_name)
        self.atlas_label_path = os.path.join(atlas_data_root, self.legacy_atlas_label_name)
        if not os.path.exists(self.atlas_img_path) or not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(
                f"BCH legacy atlas file not found: {self.atlas_img_path} / {self.atlas_label_path}"
            )

        if is_training:
            id_iter = range(1, 801)
        elif is_val:
            id_iter = range(801, 1001)
        else:
            id_iter = range(801, 1001)

        self.label_mapping = dict(self.legacy_label_mapping)
        for case_id in id_iter:
            subject = f"BCH_{case_id:05d}"
            img_path = os.path.join(root_dir, subject, self.legacy_image_name)
            label_path = os.path.join(root_dir, subject, self.legacy_target_label_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_names.append(subject)

        if len(self.target_img_paths) == 0:
            split_name = 'train' if is_training else 'eval'
            raise RuntimeError(f"No BCH samples found in legacy flat split under {root_dir} for {split_name}")

    def __len__(self):
        return len(self.target_img_paths)

    def _remap_label_array(self, label_np: np.ndarray) -> np.ndarray:
        if not self.label_mapping:
            return label_np.astype(int)
        out = label_np.copy()
        for old_label, new_label in self.label_mapping.items():
            out[label_np == old_label] = new_label
        return out.astype(int)

    def __getitem__(self, idx):
        target_img_path = self.target_img_paths[idx]
        target_label_path = self.target_label_paths[idx]
        target_name = self.target_names[idx]

        source_image_sitk = sitk.ReadImage(self.atlas_img_path)
        source_label_sitk = _align_label_sitk_to_image(sitk.ReadImage(self.atlas_label_path), source_image_sitk)
        target_image_sitk = sitk.ReadImage(target_img_path)
        target_label_sitk = _align_label_sitk_to_image(sitk.ReadImage(target_label_path), target_image_sitk)

        source_image_np = _normalize_bch_image_np(sitk.GetArrayFromImage(source_image_sitk))
        source_label_np = self._remap_label_array(sitk.GetArrayFromImage(source_label_sitk))
        target_image_np = _normalize_bch_image_np(sitk.GetArrayFromImage(target_image_sitk))
        target_label_np = self._remap_label_array(sitk.GetArrayFromImage(target_label_sitk))

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {"image": source_image, "label": source_label}
        target_data_dict = {"image": target_image, "label": target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data["image"]
            source_label = source_data["label"]
            target_image = target_data["image"]
            target_label = target_data["label"]

        if self.inference:
            return source_image, target_image, source_label, target_label, target_img_path, target_name
        return source_image, target_image, source_label, target_label


class BCH_Syn_Dataset(Dataset):
    """BCH seg 训练集：real(train) + reg 导出的 syn(final_* / coarse_*)."""

    image_name: str = "brain_w_bone_small.nii.gz"
    train_label_name: str = "Cranial_Head_bone_premask_250702.nii.gz"
    zf_ct_image_name: str = "brain_small.nii.gz"
    zf_ct_label_name: str = "label_small.nii.gz"
    zf_ct_atlas_subject: str = "BCH_CT009"
    label_mapping: dict[float, int] = BCH_Dataset.legacy_label_mapping

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key="final",
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        atlas_subject=None,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        _suffixes = {"final": ("final_image", "final_label"), "coarse": ("coarse_image", "coarse_label")}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self._label_mapping = {}

        self.syn_data_dir = os.path.join(syn_root_dir, 'train_output')
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir

        zf_ct_root = self._resolve_zf_ct_root(real_root_dir)
        if zf_ct_root is not None:
            self.layout_mode = 'zf_ct'
            self._init_zf_ct_layout(zf_ct_root, atlas_subject)
        else:
            self.layout_mode = 'split'
            self._init_split_layout(real_root_dir, atlas_subject)

    def _resolve_zf_ct_root(self, root_dir: str) -> str | None:
        candidates = [root_dir, os.path.join(root_dir, 'CT')]
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            if os.path.isdir(os.path.join(candidate, 'BCH_CT001')):
                return candidate
        return None

    def _init_zf_ct_layout(self, ct_root: str, atlas_subject: str | None):
        if atlas_subject is None:
            atlas_subject = self.zf_ct_atlas_subject
        self.atlas_subject = atlas_subject
        self._label_mapping = dict(self.label_mapping)

        real_image_paths = []
        real_label_paths = []
        for case_id in range(1, 71):
            subject = f'BCH_CT{case_id:03d}'
            if subject == self.atlas_subject:
                continue
            case_dir = os.path.join(ct_root, subject)
            img_path = os.path.join(case_dir, self.zf_ct_image_name)
            label_path = os.path.join(case_dir, self.zf_ct_label_name)
            if os.path.exists(img_path) and os.path.exists(label_path):
                real_image_paths.append(img_path)
                real_label_paths.append(label_path)

        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_image_suffix}.nii.gz')
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_label_suffix}.nii.gz')
        )

        real_image_map = {self._stem_from_real_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_real_path(p): p for p in real_label_paths}
        syn_image_map = {self._stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {self._stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}
        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
        )
        common_keys = [k for k in common_keys if k != self.atlas_subject]
        if len(common_keys) == 0:
            raise RuntimeError(
                f'No matched BCH zf_ct syn samples found between {ct_root} and {self.syn_data_dir}'
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _init_split_layout(self, real_root_dir: str, atlas_subject: str | None):
        self.real_train_dir = os.path.join(real_root_dir, 'train')
        self.real_val_dir = os.path.join(real_root_dir, 'val')
        if not os.path.isdir(self.real_train_dir):
            raise FileNotFoundError(f'BCH train dir not found: {self.real_train_dir}')
        if not os.path.isdir(self.real_val_dir):
            raise FileNotFoundError(f'BCH val dir not found: {self.real_val_dir}')

        val_subjects = sorted(
            d for d in os.listdir(self.real_val_dir) if os.path.isdir(os.path.join(self.real_val_dir, d))
        )
        if len(val_subjects) == 0:
            raise RuntimeError(f'No BCH val subjects found in {self.real_val_dir}')
        if atlas_subject is None:
            atlas_subject = 'BCH_00009' if 'BCH_00009' in val_subjects else val_subjects[0]
        if atlas_subject not in val_subjects:
            raise ValueError(f'atlas_subject {atlas_subject} not found in {self.real_val_dir}')
        self.atlas_subject = atlas_subject

        train_subjects = sorted(
            d for d in os.listdir(self.real_train_dir) if os.path.isdir(os.path.join(self.real_train_dir, d))
        )
        real_image_paths = []
        real_label_paths = []
        for sub in train_subjects:
            img_path = os.path.join(self.real_train_dir, sub, self.image_name)
            label_path = os.path.join(self.real_train_dir, sub, self.train_label_name)
            if os.path.exists(img_path) and os.path.exists(label_path):
                real_image_paths.append(img_path)
                real_label_paths.append(label_path)

        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_image_suffix}.nii.gz')
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_label_suffix}.nii.gz')
        )

        real_image_map = {self._stem_from_real_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_real_path(p): p for p in real_label_paths}
        syn_image_map = {self._stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {self._stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}

        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
        )
        common_keys = [k for k in common_keys if k != self.atlas_subject]
        if len(common_keys) == 0:
            raise RuntimeError(
                f'No matched BCH syn samples found between {real_root_dir} and {self.syn_data_dir}'
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _normalize_subject_key(self, raw_name: str) -> str:
        name = raw_name
        if name.startswith('pred_image_'):
            name = name.replace('pred_image_', '', 1)
        if name.startswith('pred_label_'):
            name = name.replace('pred_label_', '', 1)
        match = re.search(r'(BCH_CT\d{3}|BCH_\d{5})', name)
        if match:
            return match.group(1)
        return name

    def _stem_from_real_path(self, file_path: str) -> str:
        return self._normalize_subject_key(os.path.basename(os.path.dirname(file_path)))

    def _stem_from_syn_path(self, file_path: str, suffix_key: str) -> str:
        name = os.path.basename(file_path).replace(f'_{suffix_key}.nii.gz', '')
        return self._normalize_subject_key(name)

    def _remap_label_array(self, label_np: np.ndarray) -> np.ndarray:
        if not self._label_mapping:
            return label_np.astype(int)
        out = label_np.copy()
        for old_label, new_label in self._label_mapping.items():
            out[label_np == old_label] = new_label
        return out.astype(int)

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_sitk = sitk.ReadImage(real_image_path)
        real_label_sitk = _align_label_sitk_to_image(sitk.ReadImage(real_label_path), real_image_sitk)
        syn_image_sitk = sitk.ReadImage(syn_image_path)
        syn_label_sitk = _align_label_sitk_to_image(sitk.ReadImage(syn_label_path), syn_image_sitk)

        real_image_np = _normalize_bch_image_np(sitk.GetArrayFromImage(real_image_sitk))
        real_label_np = self._remap_label_array(sitk.GetArrayFromImage(real_label_sitk))
        syn_image_np = _normalize_bch_image_np(sitk.GetArrayFromImage(syn_image_sitk))
        syn_label_np = self._remap_label_array(sitk.GetArrayFromImage(syn_label_sitk))

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {'image': real_image, 'label': real_label}
        syn_data_dict = {'image': syn_image, 'label': syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data['image']
            real_label = real_data['label']
            syn_image = syn_data['image']
            syn_label = syn_data['label']

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


class BCH2_Dataset(Dataset):
    """BCH2 配准数据集：使用 data/BCH，atlas 固定 BCH_00009。"""

    image_name: str = "brain_w_bone_small.nii.gz"
    train_label_name: str = "Cranial_Head_bone_premask_250702.nii.gz"
    eval_label_name: str = "Cranial_Head_bone_gt.nii.gz"
    atlas_subject: str = "BCH_00009"

    def __init__(self, root_dir, transform2img=None, transform2both=None, is_training=True, is_val=True, inference=False, atlas_subject=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.train_dir = os.path.join(root_dir, 'train')
        self.val_dir = os.path.join(root_dir, 'val')
        self.test_dir = os.path.join(root_dir, 'test')
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f'BCH2 split dir not found: {d}')

        self.atlas_subject = atlas_subject or self.atlas_subject
        self.atlas_img_path = os.path.join(self.val_dir, self.atlas_subject, self.image_name)
        self.atlas_label_path = os.path.join(self.val_dir, self.atlas_subject, self.eval_label_name)
        if not os.path.exists(self.atlas_img_path) or not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(f'BCH2 atlas not found: {self.atlas_img_path} / {self.atlas_label_path}')

        if is_training:
            base_dir = self.train_dir
            label_name = self.train_label_name
        elif is_val:
            base_dir = self.val_dir
            label_name = self.eval_label_name
        else:
            base_dir = self.test_dir
            label_name = self.eval_label_name

        self.target_img_paths = []
        self.target_label_paths = []
        self.target_names = []
        for sub in sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))):
            if base_dir == self.val_dir and sub == self.atlas_subject:
                continue
            img_path = os.path.join(base_dir, sub, self.image_name)
            label_path = os.path.join(base_dir, sub, label_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_names.append(sub)

        if len(self.target_img_paths) == 0:
            raise RuntimeError(f'No BCH2 samples found in split under {base_dir}')

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        target_img_path = self.target_img_paths[idx]
        target_label_path = self.target_label_paths[idx]
        target_name = self.target_names[idx]

        source_image_np = _normalize_bch2_image_np(sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_img_path)))
        source_label_np = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path)).astype(int)
        target_image_np = _normalize_bch2_image_np(sitk.GetArrayFromImage(sitk.ReadImage(target_img_path)))
        target_label_np = sitk.GetArrayFromImage(sitk.ReadImage(target_label_path)).astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {'image': source_image, 'label': source_label}
        target_data_dict = {'image': target_image, 'label': target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data['image']
            source_label = source_data['label']
            target_image = target_data['image']
            target_label = target_data['label']

        if self.inference:
            return source_image, target_image, source_label, target_label, target_img_path, target_name
        return source_image, target_image, source_label, target_label


class BCH2_Syn_Dataset(Dataset):
    """BCH2 seg 训练集：real(train) + reg 导出的 syn(final/coarse)。"""

    image_name: str = "brain_w_bone_small.nii.gz"
    train_label_name: str = "Cranial_Head_bone_premask_250702.nii.gz"
    atlas_subject: str = "BCH_00009"

    def __init__(self, real_root_dir, syn_root_dir, syn_key='final', transform2img=None, transform2both=None, is_training=True, is_val=True, inference=False, atlas_subject=None):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.atlas_subject = atlas_subject or self.atlas_subject
        suffixes = {'final': ('final_image', 'final_label'), 'coarse': ('coarse_image', 'coarse_label')}
        try:
            self.syn_image_suffix, self.syn_label_suffix = suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None

        self.real_train_dir = os.path.join(real_root_dir, 'train')
        if not os.path.isdir(self.real_train_dir):
            raise FileNotFoundError(f'BCH2 train dir not found: {self.real_train_dir}')

        self.syn_data_dir = os.path.join(syn_root_dir, 'train_output')
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir
        if not os.path.isdir(self.syn_data_dir):
            raise FileNotFoundError(f'BCH2 syn dir not found: {self.syn_data_dir}')

        real_image_paths = []
        real_label_paths = []
        for sub in sorted(d for d in os.listdir(self.real_train_dir) if os.path.isdir(os.path.join(self.real_train_dir, d))):
            img_path = os.path.join(self.real_train_dir, sub, self.image_name)
            label_path = os.path.join(self.real_train_dir, sub, self.train_label_name)
            if os.path.exists(img_path) and os.path.exists(label_path):
                real_image_paths.append(img_path)
                real_label_paths.append(label_path)

        syn_image_paths = sorted(os.path.join(self.syn_data_dir, f) for f in os.listdir(self.syn_data_dir) if f.endswith(f'{self.syn_image_suffix}.nii.gz'))
        syn_label_paths = sorted(os.path.join(self.syn_data_dir, f) for f in os.listdir(self.syn_data_dir) if f.endswith(f'{self.syn_label_suffix}.nii.gz'))

        real_image_map = {os.path.basename(os.path.dirname(p)): p for p in real_image_paths}
        real_label_map = {os.path.basename(os.path.dirname(p)): p for p in real_label_paths}
        syn_image_map = {os.path.basename(p).replace(f'_{self.syn_image_suffix}.nii.gz', '').replace('pred_image_', ''): p for p in syn_image_paths}
        syn_label_map = {os.path.basename(p).replace(f'_{self.syn_label_suffix}.nii.gz', '').replace('pred_label_', ''): p for p in syn_label_paths}

        common_keys = sorted(set(real_image_map) & set(real_label_map) & set(syn_image_map) & set(syn_label_map))
        if len(common_keys) == 0:
            raise RuntimeError(f'No matched BCH2 syn samples found between {real_root_dir} and {self.syn_data_dir}')

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_np = _normalize_bch2_image_np(sitk.GetArrayFromImage(sitk.ReadImage(real_image_path)))
        real_label_np = sitk.GetArrayFromImage(sitk.ReadImage(real_label_path)).astype(int)
        syn_image_np = _normalize_bch2_image_np(sitk.GetArrayFromImage(sitk.ReadImage(syn_image_path)))
        syn_label_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_label_path)).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {'image': real_image, 'label': real_label}
        syn_data_dict = {'image': syn_image, 'label': syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data['image']
            real_label = real_data['label']
            syn_image = syn_data['image']
            syn_label = syn_data['label']

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


def _spine_train_pool_ids(atlas_id: int = 41) -> list[int]:
    return [i for i in range(16, 619) if i != atlas_id]



def _spine_test_ids() -> list[int]:
    return list(range(640, 824))



def _spine_train_val_split(
    atlas_id: int = 41,
    val_ratio: float = 0.1,
    split_seed: int = 0,
) -> tuple[list[int], list[int]]:
    ids = list(_spine_train_pool_ids(atlas_id))
    n = len(ids)
    if n == 0:
        raise RuntimeError('Spine train pool is empty')

    val_count = int(n * float(val_ratio))
    if n > 1 and val_count < 1:
        val_count = 1
    if val_count >= n:
        val_count = n - 1

    rng = np.random.RandomState(int(split_seed))
    perm = rng.permutation(n)
    val_idx = set(perm[:val_count].tolist())
    train_ids = sorted(ids[i] for i in range(n) if i not in val_idx)
    val_ids = sorted(ids[i] for i in range(n) if i in val_idx)
    return train_ids, val_ids


class Spine_Dataset(Dataset):
    """Spine 配准数据集。

    规则：
    - atlas 固定为 41；
    - train/val 从 16..618 去掉 41 的训练池里按固定随机种子切分；
    - test 固定独立为 640..823。
    """

    atlas_id: int = 41

    def __init__(
        self,
        root_dir,
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        val_ratio: float = 0.1,
        split_seed: int = 0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'masks')
        self.atlas_img_path = os.path.join(self.image_dir, f'spine_image{self.atlas_id:03}.nii.gz')
        self.atlas_label_path = os.path.join(self.label_dir, f'spine_label{self.atlas_id:03}.nii.gz')
        if not os.path.exists(self.atlas_img_path) or not os.path.exists(self.atlas_label_path):
            raise FileNotFoundError(
                f'Spine atlas not found: {self.atlas_img_path} / {self.atlas_label_path}'
            )

        if is_training or is_val:
            train_ids, val_ids = _spine_train_val_split(
                atlas_id=self.atlas_id,
                val_ratio=self.val_ratio,
                split_seed=self.split_seed,
            )
            target_ids = train_ids if is_training else val_ids
        else:
            target_ids = _spine_test_ids()

        self.target_img_paths: list[str] = []
        self.target_label_paths: list[str] = []
        self.target_img_names: list[str] = []
        for case_id in target_ids:
            img_name = f'spine_image{case_id:03}.nii.gz'
            label_name = f'spine_label{case_id:03}.nii.gz'
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, label_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            self.target_img_paths.append(img_path)
            self.target_label_paths.append(label_path)
            self.target_img_names.append(img_name)

        if len(self.target_img_paths) == 0:
            raise RuntimeError(f'No Spine samples found in {root_dir}')

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        target_img_path = self.target_img_paths[idx]
        target_label_path = self.target_label_paths[idx]
        target_img_name = self.target_img_names[idx]

        source_image_np = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_img_path))
        source_label_np = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path)).astype(int)
        target_image_np = sitk.GetArrayFromImage(sitk.ReadImage(target_img_path))
        target_label_np = sitk.GetArrayFromImage(sitk.ReadImage(target_label_path)).astype(int)

        source_image = torch.Tensor(source_image_np).unsqueeze(dim=0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim=0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim=0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim=0)

        source_data_dict = {'image': source_image, 'label': source_label}
        target_data_dict = {'image': target_image, 'label': target_label}

        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            source_data = apply_transform(self.transform2both, source_data_dict)
            target_data = apply_transform(self.transform2both, target_data_dict)
            source_image = source_data['image']
            source_label = source_data['label']
            target_image = target_data['image']
            target_label = target_data['label']

        if self.inference:
            return source_image, target_image, source_label, target_label, target_img_path, target_img_name
        return source_image, target_image, source_label, target_label


class Spine_Syn_Dataset(Dataset):
    """Spine seg 训练集：real + reg 导出的 syn(final/coarse)。

    与 Spine_Dataset 保持一致：
    - train/val 在训练池内按固定随机种子切分；
    - test 独立固定，但这里主要用于 train/val 配对 syn。
    """

    atlas_id: int = 41

    def __init__(
        self,
        real_root_dir,
        syn_root_dir,
        syn_key='final',
        transform2img=None,
        transform2both=None,
        is_training=True,
        is_val=True,
        inference=False,
        val_ratio: float = 0.1,
        split_seed: int = 0,
    ):
        super().__init__()
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.syn_key = syn_key
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.inference = inference
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        _suffixes = {'final': ('final_image', 'final_label'), 'coarse': ('coarse_image', 'coarse_label')}
        try:
            self.syn_image_suffix, self.syn_label_suffix = _suffixes[str(syn_key).strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported syn_key: {syn_key}. Expected 'final' or 'coarse'") from None

        if is_training or is_val:
            train_ids, val_ids = _spine_train_val_split(
                atlas_id=self.atlas_id,
                val_ratio=self.val_ratio,
                split_seed=self.split_seed,
            )
            selected_ids = train_ids if is_training else val_ids
        else:
            selected_ids = _spine_test_ids()
        selected_keys = {f'spine_image{case_id:03}' for case_id in selected_ids}

        self.image_dir = os.path.join(real_root_dir, 'images')
        self.label_dir = os.path.join(real_root_dir, 'masks')
        self.syn_data_dir = os.path.join(syn_root_dir, 'train_output')
        if not os.path.isdir(self.syn_data_dir):
            self.syn_data_dir = syn_root_dir

        real_image_paths = sorted(
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.nii.gz')
        )
        real_label_paths = sorted(
            os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir) if f.endswith('.nii.gz')
        )
        syn_image_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_image_suffix}.nii.gz')
        )
        syn_label_paths = sorted(
            os.path.join(self.syn_data_dir, f)
            for f in os.listdir(self.syn_data_dir)
            if f.endswith(f'{self.syn_label_suffix}.nii.gz') or f.startswith('pred_label_')
        )

        real_image_map = {self._stem_from_real_image_path(p): p for p in real_image_paths}
        real_label_map = {self._stem_from_real_label_path(p): p for p in real_label_paths}
        syn_image_map = {self._stem_from_syn_path(p, self.syn_image_suffix): p for p in syn_image_paths}
        syn_label_map = {self._stem_from_syn_path(p, self.syn_label_suffix): p for p in syn_label_paths}
        common_keys = sorted(
            set(real_image_map.keys())
            & set(real_label_map.keys())
            & set(syn_image_map.keys())
            & set(syn_label_map.keys())
            & selected_keys
        )
        if len(common_keys) == 0:
            raise RuntimeError(
                f'No matched Spine syn samples found between {real_root_dir} and {self.syn_data_dir}'
            )

        self.real_image_paths = [real_image_map[k] for k in common_keys]
        self.real_label_paths = [real_label_map[k] for k in common_keys]
        self.syn_image_paths = [syn_image_map[k] for k in common_keys]
        self.syn_label_paths = [syn_label_map[k] for k in common_keys]

    def _normalize_subject_key(self, raw_name: str) -> str:
        name = raw_name
        if name.startswith('pred_image_'):
            name = name.replace('pred_image_', '', 1)
        if name.startswith('pred_label_'):
            name = name.replace('pred_label_', '', 1)
        if name.endswith('_affined'):
            name = name[:-len('_affined')]
        if name.endswith('_rigid'):
            name = name[:-len('_rigid')]
        if name.startswith('spine_label'):
            name = 'spine_image' + name[len('spine_label'):]
        m = re.fullmatch(r'(?:spine_image)?(\d{3})', name)
        if m:
            return f'spine_image{int(m.group(1)):03}'
        return name

    def _stem_from_real_image_path(self, file_path: str) -> str:
        return self._normalize_subject_key(Path(file_path).stem.replace('.nii', ''))

    def _stem_from_real_label_path(self, file_path: str) -> str:
        return self._normalize_subject_key(Path(file_path).stem.replace('.nii', ''))

    def _stem_from_syn_path(self, file_path: str, suffix_key: str) -> str:
        name = os.path.basename(file_path).replace(f'_{suffix_key}.nii.gz', '')
        return self._normalize_subject_key(name)

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image_path = self.real_image_paths[idx]
        real_label_path = self.real_label_paths[idx]
        syn_image_path = self.syn_image_paths[idx]
        syn_label_path = self.syn_label_paths[idx]

        real_image_np = sitk.GetArrayFromImage(sitk.ReadImage(real_image_path))
        real_label_np = sitk.GetArrayFromImage(sitk.ReadImage(real_label_path)).astype(int)
        syn_image_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_image_path))
        syn_label_np = sitk.GetArrayFromImage(sitk.ReadImage(syn_label_path)).astype(int)

        real_image = torch.Tensor(real_image_np).unsqueeze(dim=0)
        real_label = torch.Tensor(real_label_np).unsqueeze(dim=0)
        syn_image = torch.Tensor(syn_image_np).unsqueeze(dim=0)
        syn_label = torch.Tensor(syn_label_np).unsqueeze(dim=0)

        real_data_dict = {'image': real_image, 'label': real_label}
        syn_data_dict = {'image': syn_image, 'label': syn_label}

        if self.transform2img:
            real_image = apply_transform(self.transform2img, real_image)
            syn_image = apply_transform(self.transform2img, syn_image)

        if self.transform2both:
            real_data = apply_transform(self.transform2both, real_data_dict)
            syn_data = apply_transform(self.transform2both, syn_data_dict)
            real_image = real_data['image']
            real_label = real_data['label']
            syn_image = syn_data['image']
            syn_label = syn_data['label']

        return real_image, real_label, syn_image, syn_label, real_image_path, os.path.basename(real_image_path)


def resolve_atlas_paths_from_dataset(dataset) -> tuple[str, str]:
    if hasattr(dataset, "atlas_img_path") and hasattr(dataset, "atlas_label_path"):
        return str(dataset.atlas_img_path), str(dataset.atlas_label_path)

    if hasattr(dataset, "atlas_path") and hasattr(dataset, "atlas_label_path"):
        return str(dataset.atlas_path), str(dataset.atlas_label_path)

    if hasattr(dataset, "atlas_data") and hasattr(dataset, "data_root"):
        atlas_root = os.path.join(dataset.data_root, dataset.atlas_data)
        atlas_img_path = os.path.join(atlas_root, "aligned_norm.nii.gz")
        atlas_label_path = os.path.join(atlas_root, "aligned_seg35.nii.gz")
        return atlas_img_path, atlas_label_path

    raise AttributeError(
        f"Dataset {dataset.__class__.__name__} does not expose resolvable atlas paths"
    )
