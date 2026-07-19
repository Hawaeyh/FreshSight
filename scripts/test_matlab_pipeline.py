"""Run the real MATLAB API on one real dataset image and save returned artifacts."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATLAB_ROOT = PROJECT_ROOT / "matlab"
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "matlab_test"
CLASSES = ("Fresh", "Unripe", "Rotten")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def choose_image(explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Requested test image does not exist: {candidate}")
        return candidate
    for class_name in CLASSES:
        folder = DATASET_ROOT / class_name
        candidates = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ) if folder.is_dir() else []
        if candidates:
            return candidates[0].resolve()
    raise FileNotFoundError("No supported image was found in dataset/Fresh, Unripe, or Rotten.")


def save_outputs(result: dict) -> list[Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    saved = []
    images = result.get("images", {})
    for name, encoded in images.items():
        destination = OUTPUT_ROOT / f"{name}.png"
        destination.write_bytes(base64.b64decode(encoded, validate=True))
        saved.append(destination)
    report = {key: value for key, value in result.items() if key != "images"}
    report["saved_images"] = [str(path) for path in saved]
    report_path = OUTPUT_ROOT / "matlab_test_result.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    saved.append(report_path)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional real source image; dataset is never modified.")
    args = parser.parse_args()
    try:
        import matlab.engine
        image_path = choose_image(args.image)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    engine = None
    try:
        print("=== FreshSight Real MATLAB Pipeline Test ===")
        print(f"Source image: {image_path}")
        print("Source handling: read-only; no dataset files will be modified.")
        engine = matlab.engine.start_matlab()
        engine.addpath(engine.genpath(str(MATLAB_ROOT)), nargout=0)
        resolved = str(engine.which("run_freshsight_api", nargout=1)).strip()
        if not resolved:
            raise RuntimeError("run_freshsight_api is missing from the MATLAB path.")
        print(f"MATLAB API: {resolved}")
        raw = engine.run_freshsight_api(str(image_path), nargout=1)
        result = json.loads(str(raw))
        if result.get("status") != "success" or result.get("error"):
            raise RuntimeError(result.get("error") or f"API status was {result.get('status')}")

        print("\n=== MATLAB Result Fields ===")
        for key, value in result.items():
            if key == "images":
                print(f"images: {', '.join(sorted(value))}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for nested_key, nested_value in value.items():
                    print(f"  {nested_key}: {nested_value}")
            else:
                print(f"{key}: {value}")
        saved = save_outputs(result)
        print("\n=== Saved Outputs ===")
        for path in saved:
            print(path)
        print("\nResult: PASSED")
        return 0
    except Exception as exc:
        print(f"ERROR: MATLAB pipeline test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            try:
                engine.quit()
                print("MATLAB Engine closed cleanly.")
            except Exception as exc:
                print(f"WARNING: MATLAB Engine close failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
