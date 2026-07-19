from pathlib import Path

import pandas as pd

from ai.dataset import (
    CLASS_NAMES,
    MANIFEST_COLUMNS,
    SPLIT_NAMES,
    build_group_aware_manifest,
    calculate_class_weights,
    load_manifest,
)


def _inspection_frame(include_unsupported=False):
    rows = []
    sequence = 1
    for class_index, class_name in enumerate(CLASS_NAMES):
        for group_index in range(10):
            digest = f"{sequence:064x}"
            copies = 3 if group_index == 0 else 1
            for copy_index in range(copies):
                rows.append({
                    "image_path": str(Path("synthetic") / class_name / f"g{group_index}_c{copy_index}.jpg"),
                    "class_name": class_name,
                    "class_index": class_index,
                    "sha256": digest,
                    "corrupted": False,
                })
            sequence += 1
    if include_unsupported:
        for unsupported in ("semi_fresh", "Semi-Fresh", "unripen", "UnRippen"):
            rows.append({
                "image_path": str(Path("synthetic") / unsupported / "excluded.jpg"),
                "class_name": unsupported,
                "class_index": 99,
                "sha256": f"{sequence:064x}",
                "corrupted": False,
            })
            sequence += 1
    return pd.DataFrame(rows)


def test_duplicate_groups_stay_in_one_split():
    manifest = build_group_aware_manifest(_inspection_frame(), seed=42)
    assert manifest.groupby("duplicate_group_id")["split"].nunique().max() == 1


def test_no_sha256_hash_occurs_across_splits():
    manifest = build_group_aware_manifest(_inspection_frame(), seed=42)
    assert manifest.groupby("sha256")["split"].nunique().max() == 1


def test_unsupported_classes_are_excluded():
    manifest = build_group_aware_manifest(_inspection_frame(include_unsupported=True), seed=42)
    assert set(manifest["class_name"]) == set(CLASS_NAMES)
    assert not {"semi_fresh", "Semi-Fresh", "unripen", "UnRippen"} & set(manifest["class_name"])


def test_group_aware_splitting_is_deterministic():
    inspection = _inspection_frame()
    first = build_group_aware_manifest(inspection, seed=123)
    second = build_group_aware_manifest(inspection, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_all_classes_are_represented_in_every_split():
    manifest = build_group_aware_manifest(_inspection_frame(), seed=42)
    for class_name in CLASS_NAMES:
        for split_name in SPLIT_NAMES:
            assert not manifest[
                (manifest["class_name"] == class_name) & (manifest["split"] == split_name)
            ].empty


def test_manifest_has_exact_columns_and_can_be_loaded(tmp_path):
    manifest = build_group_aware_manifest(_inspection_frame(), seed=42)
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded_train = load_manifest(manifest_path, split="train")
    assert list(loaded_train.columns) == MANIFEST_COLUMNS
    assert set(loaded_train["split"]) == {"train"}

    weights = calculate_class_weights(manifest, split="train")
    assert set(weights) == set(CLASS_NAMES)
    assert all(weight > 0 for weight in weights.values())
