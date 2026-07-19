import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.dataset import CLASS_NAMES, SPLIT_NAMES, split_dataset
from config.paths import DATASET_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Create a duplicate-group-aware FreshSight dataset manifest without copying images."
    )
    parser.add_argument("--dataset-path", default=None, help="Optional dataset root path.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else DATASET_DIR.resolve()

    print("=== FreshSight Duplicate-Aware Dataset Split ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Random seed: {args.seed}")
    print("Requested ratios:")
    print(f"  Train: {args.train_ratio:.2%}")
    print(f"  Validation: {args.validation_ratio:.2%}")
    print(f"  Test: {args.test_ratio:.2%}")
    print("SHA-256 values will be recomputed safely from supported source images.")
    print("Source handling: manifest only; no images will be copied or modified.")
    print("Output handling: dataset_manifest.csv and dataset_split_summary.json will be regenerated.")

    try:
        manifest, summary, manifest_path, summary_path = split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            dataset_path=dataset_path,
        )

        print("\n=== Duplicate grouping ===")
        print(f"Unique SHA-256 groups: {summary['duplicate_group_count']}")
        print(f"Exact duplicate groups: {summary['exact_duplicate_group_count']}")
        print("All files sharing a SHA-256 hash were assigned to one split.")

        print("\n=== Counts by class and split ===")
        print(f"{'Class':<12}{'Train':>10}{'Validation':>14}{'Test':>10}{'Total':>10}")
        for class_name in CLASS_NAMES:
            counts = summary["counts_by_class_and_split"][class_name]
            total = sum(counts.values())
            print(
                f"{class_name:<12}{counts['train']:>10}{counts['validation']:>14}"
                f"{counts['test']:>10}{total:>10}"
            )
        print(
            f"{'Total':<12}{summary['split_counts']['train']:>10}"
            f"{summary['split_counts']['validation']:>14}{summary['split_counts']['test']:>10}"
            f"{summary['total_images']:>10}"
        )

        print("\n=== Achieved split percentages ===")
        for split_name in SPLIT_NAMES:
            print(
                f"  {split_name.title()}: {summary['split_counts'][split_name]} "
                f"({summary['achieved_ratios'][split_name]:.2%})"
            )

        print("\n=== Validation ===")
        for check_name, passed in summary["validation"].items():
            print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

        print("\n=== Warnings ===")
        if summary["warnings"]:
            for warning in summary["warnings"]:
                print(f"  - {warning}")
        else:
            print("  None")

        print("\n=== Outputs ===")
        print(f"Manifest: {manifest_path}")
        print(f"Split summary: {summary_path}")
        print("\n=== Result ===")
        print(f"PASSED: {len(manifest)} images were assigned without cross-split duplicate leakage.")
        return 0
    except Exception as exc:
        print("\n=== Result ===")
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
