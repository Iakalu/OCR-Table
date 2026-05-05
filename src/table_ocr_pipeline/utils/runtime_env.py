from __future__ import annotations

import os


def configure_runtime_environment() -> None:
    """Set runtime flags before importing heavy OCR/DL libraries."""
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_onednn", "0")
    os.environ.setdefault("FLAGS_pir_apply_inplace_pass", "0")
    os.environ.setdefault("FLAGS_pir_apply_shape_optimization_pass", "0")