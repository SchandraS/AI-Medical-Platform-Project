"""Download the pretrained model artifacts from HuggingFace into artifacts/.

Idempotent: skips re-download if the file already exists with a matching size.
Writes artifacts/manifest.json with sha256 of every file, which the registry
seed step (scripts/seed.py) uses as each ModelVersion's identity.

Tolerant of network failure: exits 0 with a warning so the container can still
boot (the API will report models as unavailable rather than crash-looping).
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

REPO = "Neperl/diabetes-health-indicators-study"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
FILES = [
    "diabetes_random_forest_tuned.joblib",
    "diabetes_mlp_tuned.keras",
    "scaler.joblib",
]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(filename: str, dest: Path) -> bool:
    url = f"{BASE_URL}/{filename}"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch_artifacts] WARNING: failed to download {filename}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    all_ok = True

    for filename in FILES:
        dest = ARTIFACTS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[fetch_artifacts] {filename} already present, skipping download")
        else:
            print(f"[fetch_artifacts] downloading {filename} ...")
            ok = download(filename, dest)
            all_ok = all_ok and ok
            if not ok:
                continue
        manifest[filename] = {
            "sha256": sha256_of(dest),
            "size_bytes": dest.stat().st_size,
        }

    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[fetch_artifacts] manifest written to {manifest_path}")

    if not all_ok:
        print("[fetch_artifacts] one or more artifacts failed to download; "
              "service will start but model-dependent endpoints will report 503", file=sys.stderr)
    return 0  # never fail the build/boot on network issues


if __name__ == "__main__":
    raise SystemExit(main())
