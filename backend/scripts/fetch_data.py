"""Download the public BRFSS2015 5050-split CSV used as the fixed evaluation/
holdout set for model promotion and drift-baseline computation.

Source is a no-auth HuggingFace mirror of the original Kaggle dataset
(alexteboul/diabetes-health-indicators-dataset), verified to have the exact
same schema and column order as scaler.feature_names_in_. Not committed to
the repo per its size/license; fetched at build or first boot into data/.

Tolerant of network failure: exits 0 with a warning. The /models/*/evaluate
endpoint reports 503 "evaluation dataset unavailable" if this file is missing.
"""
import sys
from pathlib import Path

import requests

URL = (
    "https://huggingface.co/datasets/jason1966/"
    "alexteboul_diabetes-health-indicators-dataset/resolve/main/"
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEST = DATA_DIR / "brfss2015_5050split.csv"

EXPECTED_HEADER = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"[fetch_data] {DEST.name} already present, skipping download")
        return 0

    print(f"[fetch_data] downloading evaluation dataset from {URL} ...")
    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = DEST.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            tmp.replace(DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch_data] WARNING: failed to download evaluation dataset: {exc}", file=sys.stderr)
        print("[fetch_data] service will start but /models/*/evaluate will report 503", file=sys.stderr)
        return 0

    # Sanity check the header matches what the ML layer expects.
    with open(DEST, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != EXPECTED_HEADER:
        print(f"[fetch_data] WARNING: unexpected CSV header, got: {header}", file=sys.stderr)

    print(f"[fetch_data] saved to {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
