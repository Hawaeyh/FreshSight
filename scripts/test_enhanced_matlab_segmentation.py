"""Run enhanced MATLAB segmentation on declared train/validation development images."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Running ``python scripts/<name>.py`` otherwise exposes only scripts/ on
# sys.path. Add the verified project root before importing FreshSight modules.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATLAB_ROOT = PROJECT_ROOT / "matlab"
MANIFEST = PROJECT_ROOT / "evaluation" / "outputs" / "dataset_manifest.csv"
DEFAULT_SAMPLES = PROJECT_ROOT / "config" / "development_segmentation_samples.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "matlab_segmentation_test"
IMAGE_FILE_NAMES = {
    "original": "original.png", "background_removed": "background_removed.png",
    "papaya_mask_raw": "raw_mask.png", "papaya_mask_clean": "clean_mask.png",
    "papaya_mask_inner": "inner_mask.png", "colour_segmentation_overlay": "colour_overlay.png",
    "damage_mask": "damage_mask.png", "damage_highlight": "damage_highlight.png",
    "hue_channel": "hue.png", "saturation_channel": "saturation.png", "value_channel": "value.png",
    "l_channel": "l_channel.png", "a_channel": "a_channel.png", "b_channel": "b_channel.png",
    "texture_map": "texture_map.png", "edge_map": "edge_map.png", "rough_area_mask": "rough_area_mask.png",
}


def _manifest_lookup() -> dict[str, dict]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(Path(row["source_path"]).resolve()).lower(): row for row in csv.DictReader(handle)}


def load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = _manifest_lookup()
    samples = []
    for entry in data.get("samples", []):
        source = (PROJECT_ROOT / entry["path"]).resolve()
        row = lookup.get(str(source).lower())
        if row is None:
            raise ValueError(f"Development sample is absent from current manifest: {source}")
        if row["split"] not in {"train", "validation"}:
            raise ValueError(f"Held-out test image is forbidden for segmentation development: {source}")
        if row["class_name"] != entry["expected_class"]:
            raise ValueError(f"Expected class does not match manifest for {source}")
        samples.append({**entry, "source": source, "split": row["split"], "sha256": row["sha256"]})
    if len(samples) < 10:
        raise ValueError("At least ten declared development samples are required.")
    return samples


def save_result(result: dict, destination: Path, sample: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for key, filename in IMAGE_FILE_NAMES.items():
        encoded = (result.get("images") or {}).get(key)
        if encoded:
            (destination / filename).write_bytes(base64.b64decode(encoded, validate=True))
    report = {key: value for key, value in result.items() if key != "images"}
    report["development_sample"] = {key: str(value) if isinstance(value, Path) else value for key, value in sample.items()}
    (destination / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate imports and development samples without starting MATLAB or inference.",
    )
    args = parser.parse_args()
    try:
        import matlab.engine
        samples = load_samples(args.samples.resolve())
        from web.services.prediction_service import PredictionService
        model_config = json.loads((PROJECT_ROOT / "config" / "model_config.json").read_text(encoding="utf-8"))
        registry = json.loads((PROJECT_ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
        web_config = json.loads((PROJECT_ROOT / "config" / "web_config.json").read_text(encoding="utf-8"))
        prediction_service = PredictionService(model_config, registry, web_config)
        if args.preflight_only:
            print("=== FreshSight Enhanced MATLAB Segmentation Preflight ===")
            print(f"Project root: {PROJECT_ROOT}")
            print("FreshSight web import: PASS")
            print("MATLAB Engine import: PASS")
            print(f"Development samples: {len(samples)}")
            print("Manifest split validation: PASS (train/validation only)")
            print("No MATLAB session or AI inference was started.")
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    engine = None
    try:
        engine = matlab.engine.start_matlab()
        engine.addpath(engine.genpath(str(MATLAB_ROOT)), nargout=0)
        print("=== FreshSight Enhanced MATLAB Segmentation Test ===")
        print("Development splits only: train/validation")
        print("Threshold changes: disabled")
        for index, sample in enumerate(samples, 1):
            raw = engine.run_freshsight_api(str(sample["source"]), nargout=1)
            result = json.loads(str(raw))
            ai = prediction_service.predict(str(sample["source"]))
            matlab_reliable = (result.get("measurement_reliability") or {}).get("matlab_class_reliable") is True
            agreement = (
                ai.get("predicted_class") == result.get("rule_class")
                if ai.get("available") and matlab_reliable else None
            )
            result["development_ai_comparison"] = {
                "predicted_class": ai.get("predicted_class"), "confidence": ai.get("confidence"),
                "device": ai.get("device"), "agreement": agreement,
            }
            slug = f"{index:02d}_{sample['expected_class']}_{sample['source'].stem}"
            save_result(result, OUTPUT_ROOT / slug, sample)
            quality = result.get("segmentation_quality") or {}
            reliability = result.get("measurement_reliability") or {}
            print(f"\n[{index}/{len(samples)}] {sample['source'].name}")
            print(f"Expected development label: {sample['expected_class']} ({sample['split']})")
            print(f"Scenario: {sample['scenario']}")
            print(f"Segmentation: {quality.get('status', 'Unavailable')} / {quality.get('score', 'Unavailable')}")
            print(f"Papaya area: {quality.get('papaya_area_percentage', 'Unavailable')}")
            print(f"Border contact: {quality.get('border_touching_percentage', 'Unavailable')}")
            print(f"Damage: {result.get('damage_percentage', 'Unavailable')}")
            print(f"Healthy: {result.get('healthy_percentage', 'Unavailable')}")
            print(f"Reliability: {json.dumps(reliability, sort_keys=True)}")
            print(f"MATLAB class: {result.get('rule_class', 'Unavailable')}")
            print(f"MobileNetV2 class: {ai.get('predicted_class', 'Unavailable')}")
            print(f"AI confidence: {ai.get('confidence', 'Unavailable')}")
            print(f"Agreement: {agreement if agreement is not None else 'Not available'}")
            warnings = quality.get("warnings") or []
            review_reasons = list(warnings)
            if agreement is False: review_reasons.append("AI and reliable MATLAB classifications disagree.")
            if not matlab_reliable: review_reasons.append("MATLAB class is not reliable.")
            print(f"Review reasons: {review_reasons}")
        print(f"\nOutputs: {OUTPUT_ROOT}")
        print("Result: COMPLETED — visually review masks before claiming improvement.")
        return 0
    except Exception as exc:
        print(f"ERROR: enhanced MATLAB segmentation test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            try: engine.quit()
            except Exception as exc: print(f"WARNING: MATLAB Engine close failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
