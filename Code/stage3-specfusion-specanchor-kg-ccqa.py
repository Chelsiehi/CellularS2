from __future__ import annotations

import importlib.util
from pathlib import Path


CORE_PATH = Path(__file__).with_name("stage3-specfusion-specanchor-kg-tfqa.py")
CORE_SPEC = importlib.util.spec_from_file_location("stage3_specanchor_kg_core", CORE_PATH)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load SpecAnchor KG implementation from {CORE_PATH}")
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_RRC_stage3-CCQA.json",
]
OUTPUT_FILE = Path("stage3-specfusion-specanchor-kg-ccqa.json")

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
JUDGE_API_KEY = ""
JUDGE_BASE_URL = ""
JUDGE_MODEL = ""


if __name__ == "__main__":
    core.API_KEY = API_KEY
    core.BASE_URL = BASE_URL
    core.MODEL_NAME = MODEL_NAME
    core.JUDGE_API_KEY = JUDGE_API_KEY
    core.JUDGE_BASE_URL = JUDGE_BASE_URL
    core.JUDGE_MODEL = JUDGE_MODEL
    core.run(
        "CCQA",
        DATASETS,
        OUTPUT_FILE,
        "Answer the cross-clause question by combining all relevant globally ranked specification and KG evidence.",
    )
