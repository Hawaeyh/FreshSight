import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.dataset import inspect_dataset
from config.paths import BASE_DIR, DATASET_DIR


REPORT_DIR = BASE_DIR / "evaluation" / "outputs"
JSON_REPORT = REPORT_DIR / "dataset_inspection.json"
CSV_REPORT = REPORT_DIR / "dataset_inspection.csv"


def _print_items(items):
    if not items:
        print("  None")
        return
    for item in items:
        print(f"  - {item}")


def _write_reports(dataframe, summary):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(CSV_REPORT, index=False)
    payload = dict(summary)
    payload["images"] = json.loads(dataframe.to_json(orient="records"))
    with JSON_REPORT.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Inspect the FreshSight dataset structure and image integrity.")
    parser.add_argument("--dataset-path", help="Optional dataset root path.", default=None)
    args = parser.parse_args()
    requested_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else DATASET_DIR.resolve()

    print("=== FreshSight Dataset Inspection ===")
    print(f"Dataset path: {requested_path}")

    try:
        dataframe, summary = inspect_dataset(requested_path, return_summary=True)
        _write_reports(dataframe, summary)

        print("\n=== Unsupported folders (excluded) ===")
        _print_items(summary["unsupported_folders"])

        print("\n=== Class counts ===")
        for class_name, count in summary["class_counts"].items():
            print(f"  {class_name}: {count}")
        print(f"  Total: {summary['total_images']}")

        print("\n=== Integrity checks ===")
        corrupted = dataframe.loc[dataframe["corrupted"], "relative_path"].tolist()
        print(f"Corrupted or unreadable images: {len(corrupted)}")
        _print_items(corrupted)
        print(f"Unsupported file extensions: {len(summary['unsupported_files'])}")
        _print_items(summary["unsupported_files"])
        print(f"Exact duplicate groups (SHA-256): {summary['duplicate_group_count']}")
        for index, group in enumerate(summary["duplicate_groups"], start=1):
            print(f"  Group {index}:")
            _print_items(group)
        print(f"Empty class folders: {len(summary['empty_classes'])}")
        _print_items(summary["empty_classes"])
        ratio = summary["imbalance_ratio"] if summary["imbalance_ratio"] is not None else "undefined (empty class)"
        print(f"Class imbalance detected: {'Yes' if summary['class_imbalance'] else 'No'}")
        print(f"  Max/min ratio: {ratio}")
        print(f"  Warning threshold: {summary['imbalance_ratio_threshold']}")

        print("\n=== Reports ===")
        print(f"JSON report: {JSON_REPORT.resolve()}")
        print(f"CSV report: {CSV_REPORT.resolve()}")

        fatal_issues = bool(corrupted or summary["empty_classes"])
        print("\n=== Result ===")
        if fatal_issues:
            print("FAILED: Fix corrupted images or empty required class folders, then inspect again.")
            return 1
        if summary["unsupported_folders"] or summary["unsupported_files"] or summary["duplicate_groups"] or summary["class_imbalance"]:
            print("COMPLETED WITH WARNINGS: Supported class images were inspected; excluded content was not included.")
        else:
            print("PASSED: All required class folders and images passed inspection.")
        return 0
    except Exception as exc:
        print("\n=== Result ===")
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
