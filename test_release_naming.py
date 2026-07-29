from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


class TestReleaseNaming(unittest.TestCase):
    def test_public_filenames_use_reg_and_seg(self) -> None:
        expected = {
            "train_reg.py",
            "inference_reg.py",
            "train_seg.py",
            "inference_seg.py",
            "model_reg.py",
            "model_seg.py",
        }
        missing = [name for name in expected if not (ROOT / name).exists()]
        self.assertEqual(missing, [])

    def test_longer_public_reg_filenames_are_removed(self) -> None:
        legacy = {
            "train_cascaded_reg.py",
            "inference_cascaded_reg.py",
            "train_reg_v1100_demo.py",
            "inference_reg_v1100_demo.py",
            "train_seg_v1305_demo.py",
            "inference_seg_v1305_demo.py",
            "model_seg_v1305.py",
        }
        still_present = [name for name in legacy if (ROOT / name).exists()]
        self.assertEqual(still_present, [])


if __name__ == "__main__":
    unittest.main()
