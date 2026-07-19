import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.duplicate_cleanup_common import DEFAULT_AUDIT_PATH, rollback_cleanup


def main():
    print("=== FreshSight Rollback Duplicate Cleanup ===")
    try:
        restored = rollback_cleanup()
        print(f"Rollback completed successfully. Files restored: {restored}")
        print(f"Audit log updated: {DEFAULT_AUDIT_PATH.resolve()}")
        print("Rerun dataset inspection now:")
        print("powershell -ExecutionPolicy Bypass -File inspect_dataset.ps1")
        return 0
    except Exception as exc:
        print(f"Rollback was not completed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
