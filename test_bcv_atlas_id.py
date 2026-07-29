import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import BCV_Dataset, BCV_Syn_Dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class BCVAtlasIdTests(unittest.TestCase):
    def test_bcv_dataset_none_atlas_id_falls_back_to_27(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id in (27, 28, 1, 2, 3):
                _touch(root / f"img{case_id:04d}_bcv_CT.nii.gz")
                _touch(root / f"seg{case_id:04d}_bcv_CT.nii.gz")

            ds = BCV_Dataset(str(root), is_training=True, atlas_id=None)

            self.assertEqual(ds.atlas_id, 27)
            self.assertTrue(ds.atlas_img_path.endswith("img0027_bcv_CT.nii.gz"))
            self.assertTrue(ds.atlas_label_path.endswith("seg0027_bcv_CT.nii.gz"))

    def test_bcv_dataset_explicit_atlas_id_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id in (27, 28, 1, 2, 3):
                _touch(root / f"img{case_id:04d}_bcv_CT.nii.gz")
                _touch(root / f"seg{case_id:04d}_bcv_CT.nii.gz")

            ds = BCV_Dataset(str(root), is_training=True, atlas_id=28)

            self.assertEqual(ds.atlas_id, 28)
            self.assertTrue(ds.atlas_img_path.endswith("img0028_bcv_CT.nii.gz"))
            self.assertTrue(ds.atlas_label_path.endswith("seg0028_bcv_CT.nii.gz"))

    def test_bcv_syn_dataset_none_atlas_id_falls_back_to_27(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            syn_root = root / "syn"
            syn_root.mkdir(parents=True, exist_ok=True)

            for case_id in (27, 28, 1, 2, 3):
                _touch(root / f"img{case_id:04d}_bcv_CT.nii.gz")
                _touch(root / f"seg{case_id:04d}_bcv_CT.nii.gz")

            for case_id in (1, 2, 3):
                stem = f"img{case_id:04d}_bcv_CT"
                _touch(syn_root / f"{stem}_final_image.nii.gz")
                _touch(syn_root / f"{stem}_final_label.nii.gz")

            ds = BCV_Syn_Dataset(str(root), str(syn_root), atlas_id=None)

            self.assertEqual(ds.atlas_id, 27)

    def test_bcv_syn_dataset_explicit_atlas_id_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            syn_root = root / "syn"
            syn_root.mkdir(parents=True, exist_ok=True)

            for case_id in (27, 28, 1, 2, 3):
                _touch(root / f"img{case_id:04d}_bcv_CT.nii.gz")
                _touch(root / f"seg{case_id:04d}_bcv_CT.nii.gz")

            for case_id in (1, 2, 3):
                stem = f"img{case_id:04d}_bcv_CT"
                _touch(syn_root / f"{stem}_final_image.nii.gz")
                _touch(syn_root / f"{stem}_final_label.nii.gz")

            ds = BCV_Syn_Dataset(str(root), str(syn_root), atlas_id=28)

            self.assertEqual(ds.atlas_id, 28)


if __name__ == "__main__":
    unittest.main()
