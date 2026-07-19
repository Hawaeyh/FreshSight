import hashlib
import subprocess
import sys
from pathlib import Path

from scripts.test_enhanced_matlab_damage import COMPARISON_COLUMNS, IMAGE_MAP, load_samples

ROOT = Path(__file__).resolve().parent.parent


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper()


def test_damage_samples_are_balanced_and_never_held_out():
    samples = load_samples()
    assert len(samples) == 24
    assert {name: sum(s["expected_class"] == name for s in samples) for name in ("Fresh", "Unripe", "Rotten")} == {
        "Fresh": 8, "Unripe": 8, "Rotten": 8,
    }
    assert all(sample["manifest_split"] in {"train", "validation"} for sample in samples)


def test_segmentation_baseline_and_weights_are_immutable():
    assert digest("matlab/segmentation/segment_papaya.m") == "37576D7DD00A4F44708E59AF161DE1FA160667A08BBBC247D7B9CE25B784E838"
    assert digest("matlab/config/segmentation_config.m") == "82E76BA6350E0C443E96A0210BD8841D215D6B567FABF911E8781AF6613CFF91"


def test_damage_code_excludes_stem_shadow_reflection_and_boundary():
    source = (ROOT / "matlab/features/extract_papaya_features.m").read_text(encoding="utf-8")
    assert "excluded = stem.mask | shadow.mask | reflection.mask | ~inner" in source
    assert "combinedEvidence = combinedEvidence & inner & ~stem.mask & ~shadow.mask & ~reflection.mask" in source
    assert "filteredDamage = filteredDamage & inner" in source
    assert "features.healthyPercentage = 100 - features.damagePercentage" in source


def test_dark_decay_requires_support_and_smooth_shadow_is_separate():
    feature_source = (ROOT / "matlab/features/extract_papaya_features.m").read_text(encoding="utf-8")
    shadow_source = (ROOT / "matlab/features/detect_shadow_regions.m").read_text(encoding="utf-8")
    assert "colour.masks.dark & ~excluded" in feature_source
    assert "textured | contrasted | lesionSupport | brownSupport" in feature_source
    assert "localTexture < cfg.shadowMaximumTexture" in shadow_source
    assert "localContrast < cfg.shadowMaximumLocalContrast" in shadow_source


def test_specular_is_excluded_but_textured_white_cluster_is_mold():
    reflection = (ROOT / "matlab/features/detect_specular_highlights.m").read_text(encoding="utf-8")
    features = (ROOT / "matlab/features/extract_papaya_features.m").read_text(encoding="utf-8")
    assert "S < cfg.reflectionMaximumSaturation" in reflection
    assert "localTexture < cfg.reflectionMaximumTexture" in reflection
    assert "colour.masks.whiteMold & ~stem.mask & ~reflection.mask" in features
    assert "localTexture >= damageCfg.minimumDecayTexture" in features


def test_runner_outputs_and_comparison_schema_are_exact():
    assert set(IMAGE_MAP.values()) == {
        "original.png", "clean_mask.png", "inner_mask.png", "stem_mask.png", "shadow_mask.png",
        "reflection_mask.png", "brown_decay_mask.png", "dark_decay_mask.png", "white_mold_mask.png",
        "lesion_mask.png", "abnormal_texture_mask.png", "filtered_damage_mask.png", "damage_highlight.png",
    }
    assert COMPARISON_COLUMNS == [
        "sample", "expected_class", "ai_class", "matlab_class_before", "matlab_class_after",
        "damage_before", "damage_after", "healthy_before", "healthy_after", "stem_excluded",
        "shadow_excluded", "reflection_excluded", "agreement_before", "agreement_after",
        "review_required_before", "review_required_after",
    ]


def test_damage_runner_preflight_by_file_path():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_enhanced_matlab_damage.py"), "--preflight-only"],
        cwd=ROOT.parent, capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "24 (8 Fresh, 8 Unripe, 8 Rotten)" in completed.stdout
    assert "train/validation only" in completed.stdout
