import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.duplicate_cleanup_common import DEFAULT_AUDIT_PATH, apply_cleanup


def main():
    print("=== FreshSight Apply Duplicate Cleanup ===")
    try:
        moved = apply_cleanup()
        print(f"Cleanup applied successfully. Files moved to quarantine: {moved}")
        print(f"Audit log: {DEFAULT_AUDIT_PATH.resolve()}")
        print("Rerun dataset inspection now:")
        print("powershell -ExecutionPolicy Bypass -File inspect_dataset.ps1")
        print("Do not reuse the previous dataset manifest or split summary.")
        return 0
    except Exception as exc:
        print(f"Cleanup was not completed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
