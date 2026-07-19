import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.test_enhanced_matlab_segmentation import DEFAULT_SAMPLES, MANIFEST, load_samples

ROOT = Path(__file__).resolve().parent.parent


def test_enhanced_matlab_functions_exist_and_are_connected():
    required = [
        "matlab/segmentation/remove_papaya_background.m",
        "matlab/segmentation/segment_papaya.m",
        "matlab/features/detect_papaya_colours.m",
        "matlab/features/create_hsv_visualizations.m",
        "matlab/features/create_lab_visualizations.m",
        "matlab/features/analyze_papaya_texture.m",
    ]
    assert all((ROOT / path).is_file() for path in required)
    flow = (ROOT / "matlab/core/analyze_papaya.m").read_text(encoding="utf-8")
    for function in ("segment_papaya", "create_hsv_visualizations", "create_lab_visualizations",
                     "detect_papaya_colours", "analyze_papaya_texture", "extract_papaya_features"):
        assert function in flow


def test_inner_mask_and_exclusive_priority_are_explicit():
    segmentation = (ROOT / "matlab/segmentation/segment_papaya.m").read_text(encoding="utf-8")
    colours = (ROOT / "matlab/features/detect_papaya_colours.m").read_text(encoding="utf-8")
    features = (ROOT / "matlab/features/extract_papaya_features.m").read_text(encoding="utf-8")
    assert "papayaMaskInner" in segmentation
    assert "available(mask) = false" in colours
    assert "{'whiteMold', 'dark', 'redLesion', 'brown', 'orange', 'yellow', 'green'}" in colours
    assert "filteredDamage = filteredDamage & inner" in features
    assert "features.healthyPercentage = 100 - features.damagePercentage" in features


def test_development_samples_exclude_held_out_test_split():
    samples = load_samples(DEFAULT_SAMPLES)
    assert len(samples) >= 10
    assert all(sample["split"] in {"train", "validation"} for sample in samples)


def test_api_contains_quality_reliability_and_compatibility_fields():
    api = (ROOT / "matlab/api/run_freshsight_api.m").read_text(encoding="utf-8")
    for token in ("segmentation_quality", "measurement_reliability", "papaya_mask_raw",
                  "papaya_mask_clean", "papaya_mask_inner", "damage_mask", "damage_highlight"):
        assert token in api


def test_web_handles_optional_images_and_unavailable_values():
    template = (ROOT / "web/templates/index.html").read_text(encoding="utf-8")
    assert "Not available due to unreliable segmentation" in template
    assert "papaya_mask_inner" in template
    assert "colour_segmentation_overlay" in template


def test_runner_imports_project_modules_when_launched_by_file_path():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_enhanced_matlab_segmentation.py"), "--preflight-only"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "FreshSight web import: PASS" in completed.stdout
    assert "train/validation only" in completed.stdout
