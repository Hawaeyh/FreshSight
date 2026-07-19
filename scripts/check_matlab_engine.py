"""Real MATLAB Engine smoke check. This is intentionally not a mocked test."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATLAB_ROOT = PROJECT_ROOT / "matlab"
API_FUNCTION = "run_freshsight_api"


def main() -> int:
    try:
        import matlab.engine
    except Exception as exc:
        print(f"ERROR: MATLAB Engine import failed: {exc}", file=sys.stderr)
        return 1

    engine = None
    try:
        print("=== FreshSight MATLAB Engine Check ===")
        print("MATLAB Engine import: PASS")
        engine = matlab.engine.start_matlab()
        print("MATLAB Engine start: PASS")
        print(f"MATLAB version: {engine.version(nargout=1)}")
        generated_path = engine.genpath(str(MATLAB_ROOT))
        engine.addpath(generated_path, nargout=0)
        print(f"MATLAB path root: {MATLAB_ROOT}")
        resolved = str(engine.which(API_FUNCTION, nargout=1)).strip()
        print(f"which {API_FUNCTION}: {resolved or 'NOT FOUND'}")
        if not resolved:
            print(f"ERROR: {API_FUNCTION} is missing from the MATLAB path.", file=sys.stderr)
            return 1
        print("Result: PASSED")
        return 0
    except Exception as exc:
        print(f"ERROR: MATLAB Engine check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            try:
                engine.quit()
                print("MATLAB Engine closed cleanly.")
            except Exception as exc:
                print(f"WARNING: MATLAB Engine close failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
