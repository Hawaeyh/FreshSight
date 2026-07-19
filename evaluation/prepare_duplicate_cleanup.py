import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.duplicate_cleanup_common import (
    DEFAULT_PLAN_PATH,
    DEFAULT_SUMMARY_PATH,
    prepare_cleanup_plan,
)


def main():
    parser = argparse.ArgumentParser(description="Preview exact-duplicate cleanup without moving source images.")
    parser.add_argument("--inspection-report", default=None)
    args = parser.parse_args()
    print("=== FreshSight Duplicate Cleanup Preview ===")
    print("Mode: DRY RUN — no dataset image or quarantine directory will be changed.")
    try:
        kwargs = {"report_path": args.inspection_report} if args.inspection_report else {}
        plan, summary = prepare_cleanup_plan(**kwargs)
        for group in summary["groups"]:
            print(f"\n[{group['duplicate_group_id']}] {group['class_name']}")
            print(f"SHA-256: {group['sha256']}")
            print(f"Canonical KEEP: {group['canonical_path']}")
            print("Redundant QUARANTINE:")
            for path in group["redundant_paths"]:
                print(f"  - {path}")
            print(f"Files kept: {group['files_kept']}")
            print(f"Files to move: {group['files_to_quarantine']}")

        print("\n=== Counts before cleanup ===")
        for class_name, count in summary["counts_before_cleanup"].items():
            print(f"  {class_name}: {count}")
        print(f"  Total: {sum(summary['counts_before_cleanup'].values())}")
        print("\n=== Expected counts after cleanup ===")
        for class_name, count in summary["expected_counts_after_cleanup"].items():
            print(f"  {class_name}: {count}")
        print(f"  Total: {sum(summary['expected_counts_after_cleanup'].values())}")
        print("\n=== Dry-run totals ===")
        print(f"Duplicate groups: {summary['duplicate_group_count']}")
        print(f"Files kept: {summary['files_kept']}")
        print(f"Files to quarantine: {summary['files_to_quarantine']}")
        print("\n=== Outputs ===")
        print(f"Cleanup plan: {DEFAULT_PLAN_PATH.resolve()}")
        print(f"Cleanup summary: {DEFAULT_SUMMARY_PATH.resolve()}")
        print("WARNING: Existing dataset_manifest.csv and dataset_split_summary.json are obsolete and must not be reused.")
        print("\n=== Result ===")
        print(f"PREVIEW COMPLETE: {len(plan)} duplicate-file rows planned; no source files were changed.")
        return 0
    except Exception as exc:
        print("\n=== Result ===")
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
