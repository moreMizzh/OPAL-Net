from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [
    ROOT / "train_reg.py",
    ROOT / "inference_reg.py",
    ROOT / "train_seg.py",
    ROOT / "inference_seg.py",
    ROOT / "model_reg.py",
    ROOT / "model_seg.py",
    ROOT / "dataset.py",
    ROOT / "utils.py",
    ROOT / "README.md",
]
FORBIDDEN_TOKENS = {
    "v13_dataset",
    "v13_reg_model",
    "v13_utils",
    "v13_losses",
    "build_v11_0_0_roib_downsampled_model_bundle",
    "build_v9_3_0_roib_downsampled_model_bundle",
    "_v1100_",
    "v1305",
    "V1305",
    "V13_",
    "baseline_v13",
    "baseline_v14",
    "`v14`",
}


class TestPublicCodeNaming(unittest.TestCase):
    def test_public_source_files_do_not_expose_legacy_internal_tokens(self) -> None:
        found: dict[str, list[str]] = {}
        for path in SOURCE_FILES:
            text = path.read_text(encoding="utf-8")
            hits = [token for token in FORBIDDEN_TOKENS if token in text]
            if hits:
                found[path.name] = hits
        self.assertEqual(found, {})


if __name__ == "__main__":
    unittest.main()
