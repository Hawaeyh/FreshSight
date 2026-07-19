import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ai.dataset import CLASS_NAMES
from evaluation.duplicate_cleanup_common import (
    CleanupSafetyError,
    apply_cleanup,
    canonical_sort_key,
    prepare_cleanup_plan,
    rollback_cleanup,
    sha256_file,
)


def _create_image(path, color, size=(24, 18)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")


def _build_fixture(tmp_path):
    dataset_root = tmp_path / "dataset"
    backup_root = tmp_path / "dataset_duplicates_backup"
    report_path = tmp_path / "dataset_inspection.json"
    plan_path = tmp_path / "duplicate_cleanup_plan.csv"
    summary_path = tmp_path / "duplicate_cleanup_summary.json"
    audit_path = tmp_path / "duplicate_cleanup_audit.csv"
    images = []
    duplicate_groups = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        unique_path = dataset_root / class_name / "unique.png"
        _create_image(unique_path, color=(20 + class_index * 50, 80, 120))
        images.append({
            "image_path": str(unique_path.resolve()),
            "relative_path": unique_path.relative_to(dataset_root).as_posix(),
            "class_name": class_name,
            "class_index": class_index,
            "sha256": sha256_file(unique_path),
            "corrupted": False,
        })

    for class_name, color in (("Fresh", (40, 200, 80)), ("Rotten", (120, 40, 20))):
        canonical = dataset_root / class_name / "a_canonical.png"
        duplicate_b = dataset_root / class_name / "b_duplicate.png"
        duplicate_c = dataset_root / class_name / "c_duplicate.png"
        _create_image(canonical, color=color, size=(32, 24))
        shutil.copy2(canonical, duplicate_b)
        shutil.copy2(canonical, duplicate_c)
        group = []
        for path in (canonical, duplicate_b, duplicate_c):
            relative = path.relative_to(dataset_root).as_posix()
            group.append(relative)
            images.append({
                "image_path": str(path.resolve()),
                "relative_path": relative,
                "class_name": class_name,
                "class_index": CLASS_NAMES.index(class_name),
                "sha256": sha256_file(path),
                "corrupted": False,
            })
        duplicate_groups.append(group)

    counts = {
        class_name: sum(row["class_name"] == class_name for row in images)
        for class_name in CLASS_NAMES
    }
    report = {
        "dataset_path": str(dataset_root.resolve()),
        "classes": CLASS_NAMES,
        "class_counts": counts,
        "total_images": len(images),
        "corrupted_count": 0,
        "unsupported_folders": [],
        "unsupported_files": [],
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "images": images,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "dataset_root": dataset_root,
        "backup_root": backup_root,
        "report_path": report_path,
        "plan_path": plan_path,
        "summary_path": summary_path,
        "audit_path": audit_path,
    }


def _prepare(paths):
    return prepare_cleanup_plan(
        report_path=paths["report_path"],
        plan_path=paths["plan_path"],
        summary_path=paths["summary_path"],
        backup_root=paths["backup_root"],
    )


def _inventory(root):
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_canonical_selection_is_deterministic_and_uses_priority_order(tmp_path):
    fake = [
        {"readable": False, "width": 100, "height": 100, "file_size_bytes": 500, "path": Path("a")},
        {"readable": True, "width": 20, "height": 20, "file_size_bytes": 100, "path": Path("d")},
        {"readable": True, "width": 30, "height": 20, "file_size_bytes": 90, "path": Path("c")},
        {"readable": True, "width": 30, "height": 20, "file_size_bytes": 110, "path": Path("b")},
    ]
    assert min(fake, key=canonical_sort_key)["path"] == Path("b")

    paths = _build_fixture(tmp_path)
    first, _ = _prepare(paths)
    first_csv = paths["plan_path"].read_text(encoding="utf-8")
    second, _ = _prepare(paths)
    second_csv = paths["plan_path"].read_text(encoding="utf-8")
    assert first_csv == second_csv
    assert first.equals(second)
    for _, group in first.groupby("duplicate_group_id"):
        keep = group[group["action"] == "KEEP"].iloc[0]
        assert Path(keep["source_path"]).name == "a_canonical.png"


def test_only_exact_duplicates_are_quarantined_and_classes_never_cross(tmp_path):
    paths = _build_fixture(tmp_path)
    plan, _ = _prepare(paths)
    assert set(plan["action"]) == {"KEEP", "QUARANTINE"}
    assert not plan["source_path"].str.contains("unique.png", regex=False).any()
    for _, row in plan[plan["action"] == "QUARANTINE"].iterrows():
        source_relative = Path(row["source_path"]).relative_to(paths["dataset_root"].resolve())
        backup_relative = Path(row["destination_path"]).relative_to(paths["backup_root"].resolve())
        assert source_relative.parts[0] == row["class_name"]
        assert backup_relative.parts[0] == row["class_name"]


def test_dry_run_does_not_change_dataset_or_create_backup(tmp_path):
    paths = _build_fixture(tmp_path)
    before = _inventory(paths["dataset_root"])
    _prepare(paths)
    after = _inventory(paths["dataset_root"])
    assert before == after
    assert not paths["backup_root"].exists()
    assert not paths["audit_path"].exists()


def test_apply_refuses_to_overwrite_backup_file(tmp_path):
    paths = _build_fixture(tmp_path)
    plan, _ = _prepare(paths)
    quarantine_row = plan[plan["action"] == "QUARANTINE"].iloc[0]
    destination = Path(quarantine_row["destination_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"existing backup content")
    before = _inventory(paths["dataset_root"])

    with pytest.raises(CleanupSafetyError, match="overwrite refused"):
        apply_cleanup(paths["plan_path"], paths["summary_path"], paths["audit_path"])
    assert _inventory(paths["dataset_root"]) == before


def test_apply_detects_stale_plan_before_moving_any_file(tmp_path):
    paths = _build_fixture(tmp_path)
    plan, _ = _prepare(paths)
    stale_source = Path(plan[plan["action"] == "QUARANTINE"].iloc[0]["source_path"])
    _create_image(stale_source, color=(1, 2, 3), size=(10, 10))
    before = _inventory(paths["dataset_root"])

    with pytest.raises(CleanupSafetyError, match="Stale cleanup plan"):
        apply_cleanup(paths["plan_path"], paths["summary_path"], paths["audit_path"])
    assert _inventory(paths["dataset_root"]) == before
    assert not paths["audit_path"].exists()


def test_rollback_restores_every_quarantined_file(tmp_path):
    paths = _build_fixture(tmp_path)
    before = _inventory(paths["dataset_root"])
    plan, _ = _prepare(paths)
    expected_moves = int((plan["action"] == "QUARANTINE").sum())

    moved = apply_cleanup(paths["plan_path"], paths["summary_path"], paths["audit_path"])
    assert moved == expected_moves
    assert all((paths["backup_root"] / class_name).is_dir() for class_name in CLASS_NAMES)
    assert len(_inventory(paths["dataset_root"])) == len(before) - expected_moves
    restored = rollback_cleanup(paths["audit_path"])
    assert restored == expected_moves
    assert _inventory(paths["dataset_root"]) == before
