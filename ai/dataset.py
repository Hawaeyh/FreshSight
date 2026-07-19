import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config.paths import DATASET_DIR, EVALUATION_OUTPUTS_DIR


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
CLASS_NAMES = ["Fresh", "Unripe", "Rotten"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ["train", "validation", "test"]
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMBALANCE_RATIO_THRESHOLD = 1.5
MANIFEST_COLUMNS = [
    "source_path",
    "class_name",
    "class_index",
    "split",
    "sha256",
    "duplicate_group_id",
]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS


def _resolve_class_folders(dataset_dir: Path) -> Dict[str, Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root folder does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    folders: Dict[str, Path] = {}
    directory_entries = [entry for entry in dataset_dir.iterdir() if entry.is_dir()]
    for class_name in CLASS_NAMES:
        matches = [entry for entry in directory_entries if entry.name.casefold() == class_name.casefold()]
        if not matches:
            raise FileNotFoundError(f"Required dataset class folder missing: {class_name}")
        if len(matches) > 1:
            names = ", ".join(entry.name for entry in matches)
            raise ValueError(f"Multiple folders match required class {class_name}: {names}")
        folders[class_name] = matches[0]
    return folders


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    return_summary: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """Inspect the dataset without modifying any source file or directory."""
    dataset_dir = Path(dataset_path).expanduser().resolve() if dataset_path else DATASET_DIR.resolve()
    class_folders = _resolve_class_folders(dataset_dir)
    supported_names = {name.casefold() for name in CLASS_NAMES}
    unsupported_folders = sorted(
        entry.name
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and entry.name.casefold() not in supported_names
    )

    records: List[Dict] = []
    unsupported_files: List[str] = []
    hashes: Dict[str, List[str]] = {}
    counts = {class_name: 0 for class_name in CLASS_NAMES}

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = class_folders[class_name]
        for file_path in sorted(path for path in class_dir.rglob("*") if path.is_file()):
            relative_path = file_path.relative_to(dataset_dir).as_posix()
            if not _is_image_file(file_path):
                unsupported_files.append(relative_path)
                continue

            counts[class_name] += 1
            corrupted = False
            error = ""
            file_hash = ""
            try:
                file_hash = _sha256(file_path)
            except OSError as exc:
                corrupted = True
                error = f"Unable to read file: {exc}"

            if not corrupted:
                try:
                    with Image.open(file_path) as image:
                        image.verify()
                except Exception as exc:
                    corrupted = True
                    error = f"Invalid or corrupted image: {exc}"

            records.append({
                "image_path": str(file_path.resolve()),
                "relative_path": relative_path,
                "class_name": class_name,
                "class_index": class_index,
                "extension": file_path.suffix.lower(),
                "sha256": file_hash,
                "corrupted": corrupted,
                "error": error,
                "duplicate": False,
                "duplicate_of": "",
            })
            if file_hash:
                hashes.setdefault(file_hash, []).append(relative_path)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    first_path_by_hash = {digest: paths[0] for digest, paths in hashes.items() if len(paths) > 1}
    for record in records:
        digest = record["sha256"]
        if digest in first_path_by_hash:
            record["duplicate"] = True
            if record["relative_path"] != first_path_by_hash[digest]:
                record["duplicate_of"] = first_path_by_hash[digest]

    empty_classes = [class_name for class_name, count in counts.items() if count == 0]
    minimum = min(counts.values()) if counts else 0
    maximum = max(counts.values()) if counts else 0
    imbalance_ratio = None if minimum == 0 else round(maximum / minimum, 4)
    class_imbalance = minimum == 0 or (imbalance_ratio is not None and imbalance_ratio > IMBALANCE_RATIO_THRESHOLD)

    columns = [
        "image_path", "relative_path", "class_name", "class_index", "extension",
        "sha256", "corrupted", "error", "duplicate", "duplicate_of",
    ]
    dataframe = pd.DataFrame(records, columns=columns)
    summary = {
        "dataset_path": str(dataset_dir),
        "classes": CLASS_NAMES,
        "class_folders": {name: str(path.resolve()) for name, path in class_folders.items()},
        "class_counts": counts,
        "total_images": len(records),
        "corrupted_count": sum(bool(record["corrupted"]) for record in records),
        "unsupported_folders": unsupported_folders,
        "unsupported_files": unsupported_files,
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "empty_classes": empty_classes,
        "class_imbalance": class_imbalance,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_ratio_threshold": IMBALANCE_RATIO_THRESHOLD,
    }
    dataframe.attrs["inspection_summary"] = summary
    if return_summary:
        return dataframe, summary
    return dataframe


def _validate_ratios(ratios: Dict[str, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must define exactly: {', '.join(SPLIT_NAMES)}")
    if any(ratios[name] <= 0 for name in SPLIT_NAMES):
        raise ValueError("All split ratios must be greater than zero.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")


def _assignment_error(counts: Dict[str, int], targets: Dict[str, float]) -> float:
    return sum((counts[name] - targets[name]) ** 2 for name in SPLIT_NAMES)


def _assign_class_groups(class_frame: pd.DataFrame, ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    group_sizes = class_frame.groupby("sha256", sort=True).size().to_dict()
    if len(group_sizes) < len(SPLIT_NAMES):
        class_name = str(class_frame["class_name"].iloc[0])
        raise ValueError(
            f"Class {class_name} has only {len(group_sizes)} unique SHA-256 groups; "
            f"at least {len(SPLIT_NAMES)} are required to represent it in every split."
        )

    class_name = str(class_frame["class_name"].iloc[0])
    rng = random.Random(f"{seed}:{class_name}")
    groups = list(group_sizes.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)
    tie_order = SPLIT_NAMES.copy()
    rng.shuffle(tie_order)
    tie_rank = {name: index for index, name in enumerate(tie_order)}

    total = len(class_frame)
    targets = {name: ratios[name] * total for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    assignments: Dict[str, str] = {}

    for index, (digest, size) in enumerate(groups):
        remaining_after = len(groups) - index - 1
        empty_splits = [name for name in SPLIT_NAMES if group_counts[name] == 0]
        candidates = empty_splits if len(empty_splits) > remaining_after else SPLIT_NAMES

        def candidate_key(split_name: str):
            proposed = counts.copy()
            proposed[split_name] += size
            return _assignment_error(proposed, targets), tie_rank[split_name]

        selected = min(candidates, key=candidate_key)
        assignments[digest] = selected
        counts[selected] += size
        group_counts[selected] += 1

    improved = True
    while improved:
        improved = False
        current_error = _assignment_error(counts, targets)
        best_move = None
        for digest, source in sorted(assignments.items()):
            if group_counts[source] <= 1:
                continue
            size = group_sizes[digest]
            for destination in SPLIT_NAMES:
                if destination == source:
                    continue
                proposed = counts.copy()
                proposed[source] -= size
                proposed[destination] += size
                error = _assignment_error(proposed, targets)
                move = (error, digest, source, destination, proposed)
                if error + 1e-12 < current_error and (best_move is None or move[:4] < best_move[:4]):
                    best_move = move
        if best_move is not None:
            _, digest, source, destination, proposed = best_move
            assignments[digest] = destination
            counts = proposed
            group_counts[source] -= 1
            group_counts[destination] += 1
            improved = True

    return assignments


def build_group_aware_manifest(
    inspection_frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a deterministic manifest while keeping each SHA-256 group indivisible."""
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    _validate_ratios(ratios)
    required_columns = {"image_path", "class_name", "sha256"}
    missing_columns = required_columns - set(inspection_frame.columns)
    if missing_columns:
        raise ValueError(f"Inspection data is missing columns: {', '.join(sorted(missing_columns))}")

    supported = inspection_frame[inspection_frame["class_name"].isin(CLASS_NAMES)].copy()
    if supported.empty:
        raise ValueError("Inspection data contains no supported FreshSight classes.")
    if "corrupted" in supported.columns and supported["corrupted"].fillna(False).astype(bool).any():
        raise ValueError("Dataset contains corrupted images. Split generation was stopped.")
    if supported["sha256"].isna().any() or (supported["sha256"].astype(str).str.len() != 64).any():
        raise ValueError("Every supported image must have a valid SHA-256 hash.")
    if supported["image_path"].duplicated().any():
        duplicates = supported.loc[supported["image_path"].duplicated(), "image_path"].tolist()
        raise ValueError(f"Inspection data repeats source paths: {duplicates[:3]}")

    missing_classes = [name for name in CLASS_NAMES if name not in set(supported["class_name"])]
    if missing_classes:
        raise ValueError(f"Required classes missing from inspection data: {', '.join(missing_classes)}")
    cross_class_hashes = supported.groupby("sha256")["class_name"].nunique()
    conflicts = cross_class_hashes[cross_class_hashes > 1]
    if not conflicts.empty:
        raise ValueError(f"Cross-class SHA-256 conflicts detected: {', '.join(conflicts.index.tolist())}")

    split_by_hash: Dict[str, str] = {}
    for class_name in CLASS_NAMES:
        class_frame = supported[supported["class_name"] == class_name]
        split_by_hash.update(_assign_class_groups(class_frame, ratios, seed))

    manifest = pd.DataFrame({
        "source_path": supported["image_path"].map(lambda value: str(Path(value).resolve())),
        "class_name": supported["class_name"],
        "class_index": supported["class_name"].map(CLASS_TO_INDEX),
        "split": supported["sha256"].map(split_by_hash),
        "sha256": supported["sha256"],
        "duplicate_group_id": supported["sha256"].map(lambda digest: f"sha256:{digest}"),
    })
    manifest = manifest[MANIFEST_COLUMNS].sort_values(
        ["split", "class_index", "duplicate_group_id", "source_path"], kind="stable"
    ).reset_index(drop=True)
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: pd.DataFrame) -> Dict[str, bool]:
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"Manifest columns must be exactly: {', '.join(MANIFEST_COLUMNS)}")
    normalized_paths = manifest["source_path"].map(
        lambda value: str(Path(value).expanduser().resolve()).casefold()
    )
    if normalized_paths.duplicated().any():
        raise ValueError("Manifest contains repeated source paths.")
    unsupported = sorted(set(manifest["class_name"]) - set(CLASS_NAMES))
    if unsupported:
        raise ValueError(f"Manifest contains unsupported classes: {', '.join(unsupported)}")
    if not manifest["class_name"].map(CLASS_TO_INDEX).equals(manifest["class_index"]):
        raise ValueError("Manifest class indexes do not match normalized class names.")
    hash_split_counts = manifest.groupby("sha256")["split"].nunique()
    if (hash_split_counts > 1).any():
        raise ValueError("At least one SHA-256 hash occurs in more than one split.")
    group_split_counts = manifest.groupby("duplicate_group_id")["split"].nunique()
    if (group_split_counts > 1).any():
        raise ValueError("At least one duplicate group occurs in more than one split.")
    invalid_splits = sorted(set(manifest["split"]) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError(f"Manifest contains invalid splits: {', '.join(invalid_splits)}")
    missing_pairs = [
        f"{class_name}/{split_name}"
        for class_name in CLASS_NAMES
        for split_name in SPLIT_NAMES
        if manifest[(manifest["class_name"] == class_name) & (manifest["split"] == split_name)].empty
    ]
    if missing_pairs:
        raise ValueError(f"Required class/split combinations are empty: {', '.join(missing_pairs)}")
    return {
        "no_cross_split_hashes": True,
        "duplicate_groups_indivisible": True,
        "all_classes_in_every_split": True,
        "supported_classes_only": True,
        "unique_source_paths": True,
    }


def _build_split_summary(
    manifest: pd.DataFrame,
    ratios: Dict[str, float],
    seed: int,
    inspection_summary: Dict,
) -> Dict:
    total = len(manifest)
    split_counts = {name: int((manifest["split"] == name).sum()) for name in SPLIT_NAMES}
    achieved = {name: split_counts[name] / total for name in SPLIT_NAMES}
    exact_ratios_achieved = all(abs(achieved[name] - ratios[name]) < 1e-12 for name in SPLIT_NAMES)
    counts_by_class_and_split = {
        class_name: {
            split_name: int(((manifest["class_name"] == class_name) & (manifest["split"] == split_name)).sum())
            for split_name in SPLIT_NAMES
        }
        for class_name in CLASS_NAMES
    }
    class_counts = {name: int((manifest["class_name"] == name).sum()) for name in CLASS_NAMES}
    max_count = max(class_counts.values())
    future_class_weights = {name: round(max_count / count, 6) for name, count in class_counts.items()}
    warnings = []
    if not exact_ratios_achieved:
        warnings.append(
            "Exact requested ratios were not achievable with integer image counts and indivisible SHA-256 groups."
        )
    if inspection_summary.get("class_imbalance"):
        warnings.append(
            "Moderate class imbalance detected; class weights are available for future training, but no oversampling was applied."
        )
    return {
        "dataset_path": inspection_summary.get("dataset_path"),
        "source_class_counts": inspection_summary.get("class_counts"),
        "source_duplicate_group_count": inspection_summary.get("duplicate_group_count"),
        "seed": seed,
        "total_images": total,
        "requested_ratios": ratios,
        "achieved_ratios": achieved,
        "split_counts": split_counts,
        "counts_by_class_and_split": counts_by_class_and_split,
        "duplicate_group_count": int(manifest["duplicate_group_id"].nunique()),
        "exact_duplicate_group_count": int((manifest.groupby("duplicate_group_id").size() > 1).sum()),
        "validation": validate_split_manifest(manifest),
        "class_imbalance": {
            "detected": bool(inspection_summary.get("class_imbalance", False)),
            "ratio": inspection_summary.get("imbalance_ratio"),
            "future_class_weights_all_images": future_class_weights,
            "oversampling_applied": False,
        },
        "warnings": warnings,
    }


def split_dataset(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    dataset_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict, Path, Path]:
    """Reinspect safely, create a manifest, and never copy or modify source images."""
    inspection_frame, inspection_summary = inspect_dataset(dataset_path, return_summary=True)
    manifest = build_group_aware_manifest(
        inspection_frame,
        train_ratio=train_ratio,
        validation_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    ratios = {"train": train_ratio, "validation": val_ratio, "test": test_ratio}
    summary = _build_split_summary(manifest, ratios, seed, inspection_summary)
    destination = Path(output_dir).expanduser().resolve() if output_dir else EVALUATION_OUTPUTS_DIR.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "dataset_manifest.csv"
    summary_path = destination / "dataset_split_summary.json"
    temporary_manifest = destination / "dataset_manifest.csv.tmp"
    temporary_summary = destination / "dataset_split_summary.json.tmp"
    manifest.to_csv(temporary_manifest, index=False)
    with temporary_summary.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    temporary_manifest.replace(manifest_path)
    temporary_summary.replace(summary_path)
    return manifest, summary, manifest_path, summary_path


def load_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    validate: bool = True,
) -> pd.DataFrame:
    manifest = pd.read_csv(Path(manifest_path).expanduser().resolve())
    if validate:
        validate_split_manifest(manifest)
    if split is not None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(SPLIT_NAMES)}")
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    return manifest


def calculate_class_weights(manifest: pd.DataFrame, split: str = "train") -> Dict[str, float]:
    """Return balanced inverse-frequency weights; training may opt into them later."""
    subset = manifest[manifest["split"] == split]
    counts = subset["class_name"].value_counts()
    missing = [name for name in CLASS_NAMES if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(f"Cannot calculate class weights; missing classes in {split}: {', '.join(missing)}")
    total = len(subset)
    return {name: total / (len(CLASS_NAMES) * int(counts[name])) for name in CLASS_NAMES}


class ManifestImageDataset(Dataset):
    """PyTorch dataset that loads source images directly from a FreshSight manifest."""

    def __init__(self, manifest_path: Union[str, Path], split: str, transform=None):
        self.records = load_manifest(manifest_path, split=split, validate=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        source_path = Path(row["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Manifest source image is missing: {source_path}")
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["class_index"])
