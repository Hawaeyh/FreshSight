import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image

from ai.dataset import CLASS_NAMES
from config.paths import BASE_DIR, EVALUATION_OUTPUTS_DIR


DEFAULT_INSPECTION_REPORT = EVALUATION_OUTPUTS_DIR / "dataset_inspection.json"
DEFAULT_PLAN_PATH = EVALUATION_OUTPUTS_DIR / "duplicate_cleanup_plan.csv"
DEFAULT_SUMMARY_PATH = EVALUATION_OUTPUTS_DIR / "duplicate_cleanup_summary.json"
DEFAULT_AUDIT_PATH = EVALUATION_OUTPUTS_DIR / "duplicate_cleanup_audit.csv"
DEFAULT_BACKUP_ROOT = BASE_DIR / "dataset_duplicates_backup"

PLAN_COLUMNS = [
    "duplicate_group_id",
    "class_name",
    "sha256",
    "source_path",
    "action",
    "canonical_path",
    "width",
    "height",
    "file_size_bytes",
    "destination_path",
]
ALLOWED_ACTIONS = {"KEEP", "QUARANTINE"}
AUDIT_COLUMNS = [
    "timestamp_utc",
    "operation",
    "duplicate_group_id",
    "class_name",
    "sha256",
    "source_path",
    "destination_path",
    "status",
]


class CleanupSafetyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(path: Path) -> Dict:
    if not path.is_file():
        raise CleanupSafetyError(f"Source image is missing: {path}")
    digest = sha256_file(path)
    file_size = path.stat().st_size
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        readable = True
    except Exception:
        width, height = 0, 0
        readable = False
    return {
        "path": path.resolve(),
        "sha256": digest,
        "width": int(width),
        "height": int(height),
        "file_size_bytes": int(file_size),
        "readable": readable,
    }


def canonical_sort_key(metadata: Dict):
    return (
        0 if metadata["readable"] else 1,
        -(metadata["width"] * metadata["height"]),
        -metadata["file_size_bytes"],
        str(metadata["path"]).casefold(),
    )


def _safe_resolve_under(root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise CleanupSafetyError(f"Unsafe relative path in inspection report: {relative_path}")
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CleanupSafetyError(f"Path escapes the dataset root: {relative_path}") from exc
    return candidate


def _write_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: Iterable[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_cleanup_plan(
    report_path: Union[str, Path] = DEFAULT_INSPECTION_REPORT,
    plan_path: Union[str, Path] = DEFAULT_PLAN_PATH,
    summary_path: Union[str, Path] = DEFAULT_SUMMARY_PATH,
    backup_root: Union[str, Path] = DEFAULT_BACKUP_ROOT,
) -> Tuple[pd.DataFrame, Dict]:
    report_path = Path(report_path).expanduser().resolve()
    plan_path = Path(plan_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    backup_root = Path(backup_root).expanduser().resolve()
    if not report_path.is_file():
        raise CleanupSafetyError(f"Dataset inspection report is missing: {report_path}")

    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    dataset_root = Path(report.get("dataset_path", "")).expanduser().resolve()
    if not dataset_root.is_dir():
        raise CleanupSafetyError(f"Inspection dataset path is unavailable: {dataset_root}")
    if report.get("corrupted_count"):
        raise CleanupSafetyError("The inspection report contains corrupted images. Cleanup planning stopped.")
    if report.get("unsupported_folders") or report.get("unsupported_files"):
        raise CleanupSafetyError("Unsupported dataset content is present. Rerun inspection after resolving it.")

    images_by_relative = {row["relative_path"]: row for row in report.get("images", [])}
    duplicate_groups = report.get("duplicate_groups", [])
    if len(duplicate_groups) != int(report.get("duplicate_group_count", -1)):
        raise CleanupSafetyError("Inspection report duplicate-group count is inconsistent.")

    prepared_groups = []
    seen_sources = set()
    for reported_group in duplicate_groups:
        metadata_rows = []
        for relative_path in reported_group:
            if relative_path in seen_sources:
                raise CleanupSafetyError(f"Inspection report repeats a duplicate source: {relative_path}")
            seen_sources.add(relative_path)
            report_row = images_by_relative.get(relative_path)
            if report_row is None:
                raise CleanupSafetyError(f"Duplicate source is missing from report images: {relative_path}")
            class_name = report_row.get("class_name")
            if class_name not in CLASS_NAMES:
                raise CleanupSafetyError(f"Unsupported class in duplicate report: {class_name}")
            source_path = _safe_resolve_under(dataset_root, relative_path)
            metadata = inspect_file(source_path)
            reported_hash = str(report_row.get("sha256", ""))
            if metadata["sha256"] != reported_hash:
                raise CleanupSafetyError(
                    f"Stale inspection report: SHA-256 changed for {source_path}. Rerun dataset inspection."
                )
            if not metadata["readable"]:
                raise CleanupSafetyError(
                    f"Stale inspection report: image is no longer readable: {source_path}. Rerun inspection."
                )
            metadata["class_name"] = class_name
            metadata["relative_path"] = relative_path
            metadata_rows.append(metadata)

        hashes = {row["sha256"] for row in metadata_rows}
        classes = {row["class_name"] for row in metadata_rows}
        if len(hashes) != 1:
            raise CleanupSafetyError("A reported duplicate group no longer has identical SHA-256 content.")
        if len(classes) != 1:
            raise CleanupSafetyError("Cross-class duplicate group detected; automatic cleanup is not allowed.")
        prepared_groups.append((next(iter(hashes)), next(iter(classes)), metadata_rows))

    rows = []
    group_summaries = []
    quarantined_by_class = {name: 0 for name in CLASS_NAMES}
    for group_number, (digest, class_name, metadata_rows) in enumerate(
        sorted(prepared_groups, key=lambda item: (CLASS_NAMES.index(item[1]), item[0])), start=1
    ):
        group_id = f"duplicate-{group_number:04d}"
        canonical = min(metadata_rows, key=canonical_sort_key)
        canonical_path = str(canonical["path"])
        redundant_paths = []
        for metadata in sorted(metadata_rows, key=lambda item: str(item["path"]).casefold()):
            keep = metadata["path"] == canonical["path"]
            if keep:
                action = "KEEP"
                destination_path = ""
            else:
                action = "QUARANTINE"
                relative_parts = PurePosixPath(metadata["relative_path"]).parts
                if len(relative_parts) < 2:
                    raise CleanupSafetyError(f"Image path has no class-relative filename: {metadata['relative_path']}")
                destination = backup_root / class_name / Path(*relative_parts[1:])
                destination_path = str(destination.resolve())
                redundant_paths.append(str(metadata["path"]))
                quarantined_by_class[class_name] += 1
            rows.append({
                "duplicate_group_id": group_id,
                "class_name": class_name,
                "sha256": digest,
                "source_path": str(metadata["path"]),
                "action": action,
                "canonical_path": canonical_path,
                "width": metadata["width"],
                "height": metadata["height"],
                "file_size_bytes": metadata["file_size_bytes"],
                "destination_path": destination_path,
            })
        group_summaries.append({
            "duplicate_group_id": group_id,
            "class_name": class_name,
            "sha256": digest,
            "canonical_path": canonical_path,
            "redundant_paths": redundant_paths,
            "files_kept": 1,
            "files_to_quarantine": len(redundant_paths),
        })

    counts_before = {name: int(report["class_counts"][name]) for name in CLASS_NAMES}
    counts_after = {name: counts_before[name] - quarantined_by_class[name] for name in CLASS_NAMES}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "inspection_report_path": str(report_path),
        "dataset_root": str(dataset_root),
        "backup_root": str(backup_root),
        "duplicate_group_count": len(group_summaries),
        "files_kept": len(group_summaries),
        "files_to_quarantine": sum(quarantined_by_class.values()),
        "counts_before_cleanup": counts_before,
        "expected_counts_after_cleanup": counts_after,
        "quarantine_counts_by_class": quarantined_by_class,
        "groups": group_summaries,
        "source_files_modified": False,
        "backup_directories_created": False,
        "obsolete_outputs": [
            str(EVALUATION_OUTPUTS_DIR / "dataset_manifest.csv"),
            str(EVALUATION_OUTPUTS_DIR / "dataset_split_summary.json"),
        ],
    }
    _write_csv_atomic(plan_path, rows, PLAN_COLUMNS)
    _write_json_atomic(summary_path, summary)
    return pd.DataFrame(rows, columns=PLAN_COLUMNS), summary


def load_plan_and_summary(
    plan_path: Union[str, Path] = DEFAULT_PLAN_PATH,
    summary_path: Union[str, Path] = DEFAULT_SUMMARY_PATH,
) -> Tuple[pd.DataFrame, Dict]:
    plan_path = Path(plan_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    if not plan_path.is_file():
        raise CleanupSafetyError(f"Cleanup plan is missing: {plan_path}")
    if not summary_path.is_file():
        raise CleanupSafetyError(f"Cleanup summary is missing: {summary_path}")
    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False)
    if list(plan.columns) != PLAN_COLUMNS:
        raise CleanupSafetyError(f"Cleanup plan columns must be exactly: {', '.join(PLAN_COLUMNS)}")
    invalid_actions = sorted(set(plan["action"]) - ALLOWED_ACTIONS)
    if invalid_actions:
        raise CleanupSafetyError(f"Cleanup plan has invalid actions: {', '.join(invalid_actions)}")
    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return plan, summary


def _relative_to_or_error(path: Path, root: Path, message: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CleanupSafetyError(f"{message}: {path}") from exc


def preflight_cleanup(plan: pd.DataFrame, summary: Dict) -> List[Dict]:
    dataset_root = Path(summary["dataset_root"]).resolve()
    backup_root = Path(summary["backup_root"]).resolve()
    if plan["source_path"].duplicated().any():
        raise CleanupSafetyError("Cleanup plan repeats source paths.")
    operations = []
    for group_id, group in plan.groupby("duplicate_group_id", sort=True):
        if group["class_name"].nunique() != 1 or group["sha256"].nunique() != 1:
            raise CleanupSafetyError(f"Cleanup group {group_id} mixes classes or hashes.")
        keep_rows = group[group["action"] == "KEEP"]
        if len(keep_rows) != 1:
            raise CleanupSafetyError(f"Cleanup group {group_id} must contain exactly one KEEP row.")
        canonical_path = Path(keep_rows.iloc[0]["source_path"]).resolve()
        if set(group["canonical_path"]) != {str(canonical_path)}:
            raise CleanupSafetyError(f"Cleanup group {group_id} has inconsistent canonical paths.")

        for _, row in group.iterrows():
            class_name = row["class_name"]
            if class_name not in CLASS_NAMES:
                raise CleanupSafetyError(f"Unsupported class in cleanup plan: {class_name}")
            source_path = Path(row["source_path"]).resolve()
            dataset_relative = _relative_to_or_error(source_path, dataset_root, "Source escapes dataset root")
            if not dataset_relative.parts or dataset_relative.parts[0].casefold() != class_name.casefold():
                raise CleanupSafetyError(f"Source class folder does not match plan class: {source_path}")
            metadata = inspect_file(source_path)
            if metadata["sha256"] != row["sha256"]:
                raise CleanupSafetyError(
                    f"Stale cleanup plan: SHA-256 changed for {source_path}. No files were moved."
                )
            if (
                metadata["width"] != int(row["width"])
                or metadata["height"] != int(row["height"])
                or metadata["file_size_bytes"] != int(row["file_size_bytes"])
            ):
                raise CleanupSafetyError(f"Stale cleanup plan: file metadata changed for {source_path}.")
            if row["action"] == "QUARANTINE":
                destination = Path(row["destination_path"]).resolve()
                backup_relative = _relative_to_or_error(
                    destination, backup_root, "Destination escapes quarantine root"
                )
                if not backup_relative.parts or backup_relative.parts[0].casefold() != class_name.casefold():
                    raise CleanupSafetyError(f"Destination class folder does not match plan class: {destination}")
                if destination.exists():
                    raise CleanupSafetyError(f"Quarantine destination already exists; overwrite refused: {destination}")
                operations.append({
                    "duplicate_group_id": group_id,
                    "class_name": class_name,
                    "sha256": row["sha256"],
                    "source_path": source_path,
                    "destination_path": destination,
                })
    destinations = [str(operation["destination_path"]).casefold() for operation in operations]
    if len(destinations) != len(set(destinations)):
        raise CleanupSafetyError("Cleanup plan repeats quarantine destinations.")
    return operations


def _append_audit(audit_path: Path, row: Dict) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not audit_path.exists() or audit_path.stat().st_size == 0
    with audit_path.open("a", encoding="utf-8", newline="") as audit_file:
        writer = csv.DictWriter(audit_file, fieldnames=AUDIT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        audit_file.flush()
        os.fsync(audit_file.fileno())


def apply_cleanup(
    plan_path: Union[str, Path] = DEFAULT_PLAN_PATH,
    summary_path: Union[str, Path] = DEFAULT_SUMMARY_PATH,
    audit_path: Union[str, Path] = DEFAULT_AUDIT_PATH,
) -> int:
    plan, summary = load_plan_and_summary(plan_path, summary_path)
    audit_path = Path(audit_path).expanduser().resolve()
    if audit_path.exists() and audit_path.stat().st_size > 0:
        raise CleanupSafetyError(
            f"Audit log already exists. Roll back or archive the previous cleanup before applying again: {audit_path}"
        )
    operations = preflight_cleanup(plan, summary)
    backup_root = Path(summary["backup_root"]).resolve()
    for class_name in CLASS_NAMES:
        (backup_root / class_name).mkdir(parents=True, exist_ok=True)
    for operation in operations:
        destination = operation["destination_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise CleanupSafetyError(f"Quarantine destination appeared during apply; overwrite refused: {destination}")
        shutil.move(str(operation["source_path"]), str(destination))
        _append_audit(audit_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "MOVE_TO_QUARANTINE",
            "duplicate_group_id": operation["duplicate_group_id"],
            "class_name": operation["class_name"],
            "sha256": operation["sha256"],
            "source_path": str(operation["source_path"]),
            "destination_path": str(destination),
            "status": "COMPLETED",
        })
    return len(operations)


def rollback_cleanup(audit_path: Union[str, Path] = DEFAULT_AUDIT_PATH) -> int:
    audit_path = Path(audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise CleanupSafetyError(f"Cleanup audit log is missing: {audit_path}")
    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    if list(audit.columns) != AUDIT_COLUMNS:
        raise CleanupSafetyError("Cleanup audit log has unexpected columns.")
    moves = audit[
        (audit["operation"] == "MOVE_TO_QUARANTINE") & (audit["status"] == "COMPLETED")
    ]
    rollbacks = set(
        audit.loc[
            (audit["operation"] == "ROLLBACK") & (audit["status"] == "COMPLETED"), "source_path"
        ]
    )
    pending = [row for _, row in moves.iterrows() if row["source_path"] not in rollbacks]
    if not pending:
        raise CleanupSafetyError("The audit log has no pending moves to roll back.")

    for row in pending:
        original = Path(row["source_path"]).resolve()
        quarantined = Path(row["destination_path"]).resolve()
        if original.exists():
            raise CleanupSafetyError(f"Rollback would overwrite an existing dataset file: {original}")
        if not quarantined.is_file():
            raise CleanupSafetyError(f"Quarantined file is missing: {quarantined}")
        if sha256_file(quarantined) != row["sha256"]:
            raise CleanupSafetyError(f"Quarantined file changed; rollback stopped: {quarantined}")

    restored = 0
    for row in reversed(pending):
        original = Path(row["source_path"]).resolve()
        quarantined = Path(row["destination_path"]).resolve()
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quarantined), str(original))
        _append_audit(audit_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "ROLLBACK",
            "duplicate_group_id": row["duplicate_group_id"],
            "class_name": row["class_name"],
            "sha256": row["sha256"],
            "source_path": str(original),
            "destination_path": str(quarantined),
            "status": "COMPLETED",
        })
        restored += 1
    return restored
