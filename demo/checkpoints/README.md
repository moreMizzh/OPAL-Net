# Checkpoint files

This private review repository includes `bcv.pth` for the bundled BCV example. The selected OASIS and ADNI checkpoints are not included. If you have authorized access to them, place them here as `oasis.pth` and `adni.pth`, respectively.

Their SHA-256 hashes and model settings are recorded in `../models.json`; `run.py` checks each hash before inference. A different checkpoint cannot be substituted without updating its recorded model specification and hash. Review data and model sharing rights before making this repository public.
