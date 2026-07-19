"""Lazy, reusable integration with the active FreshSight MATLAB subsystem."""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from config.paths import BASE_DIR

LOGGER = logging.getLogger(__name__)
MATLAB_ROOT = BASE_DIR / "matlab"
MATLAB_API_FUNCTION = "run_freshsight_api"

try:
    import matlab.engine as _matlab_engine
except (ImportError, OSError):
    _matlab_engine = None


def _unavailable(error: str) -> dict:
    return {"available": False, "status": "unavailable", "error": error}


def _apply_reliability_gate(result: dict) -> dict:
    """Defence in depth: unreliable MATLAB values must not reach the UI as evidence."""
    reliability = result.get("measurement_reliability") or {}
    if reliability.get("segmentation_reliable") is False:
        measurements = result.get("measurements") or {}
        for key in (
            "green_percentage", "yellow_percentage", "orange_percentage", "brown_percentage",
            "dark_percentage", "white_mold_percentage", "lesion_percentage", "rough_percentage",
            "damage_percentage", "healthy_percentage", "largest_damage_percentage",
            "largest_lesion_percentage",
        ):
            measurements[key] = None
        result["measurements"] = measurements
        result["damage_percentage"] = None
        result["healthy_percentage"] = None
        result["freshness_score"] = None
        if not reliability.get("matlab_class_reliable", False):
            result["rule_class"] = "Unavailable"
            result["grade"] = "Unavailable"
    return result


class MatlabService:
    """Start MATLAB on first use and reuse the same session for later requests."""

    def __init__(self, engine_module: Any = None, timeout_seconds: float = 120.0):
        self._engine_module = _matlab_engine if engine_module is None else engine_module
        self._timeout_seconds = timeout_seconds
        self._engine: Optional[Any] = None
        self._start_lock = threading.Lock()
        self._call_lock = threading.Lock()

    @property
    def engine_started(self) -> bool:
        return self._engine is not None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_module is None:
            raise RuntimeError(
                "MATLAB Engine for Python is unavailable in this environment. "
                "Install matlabengine==25.2.2 in .venv."
            )

        with self._start_lock:
            if self._engine is not None:
                return self._engine
            LOGGER.info("Starting one reusable MATLAB Engine session")
            engine = self._engine_module.start_matlab()
            generated_path = engine.genpath(str(MATLAB_ROOT))
            engine.addpath(generated_path, nargout=0)
            resolved_api = str(engine.which(MATLAB_API_FUNCTION, nargout=1)).strip()
            if not resolved_api:
                try:
                    engine.quit()
                finally:
                    raise RuntimeError(
                        f"MATLAB API '{MATLAB_API_FUNCTION}' was not found under {MATLAB_ROOT}."
                    )
            LOGGER.info("Resolved MATLAB API %s at %s", MATLAB_API_FUNCTION, resolved_api)
            self._engine = engine
            return engine

    def _call_api(self, engine: Any, image_path: str) -> Any:
        function = getattr(engine, MATLAB_API_FUNCTION)
        try:
            future = function(image_path, nargout=1, background=True)
        except TypeError:
            LOGGER.warning("MATLAB background futures unsupported; using synchronous API call")
            return function(image_path, nargout=1)
        if hasattr(future, "result"):
            return future.result(timeout=self._timeout_seconds)
        return future

    def analyze_image(self, image_path: str) -> dict:
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            return _unavailable(f"Image file does not exist: {source}")

        started_at = time.perf_counter()
        try:
            engine = self._ensure_engine()
        except Exception as exc:
            LOGGER.exception("MATLAB Engine initialization failed")
            return _unavailable(str(exc))

        LOGGER.info("Calling MATLAB function %s for %s", MATLAB_API_FUNCTION, source)
        try:
            with self._call_lock:
                matlab_result = self._call_api(engine, str(source))
            result = json.loads(matlab_result) if isinstance(matlab_result, str) else dict(matlab_result)
            result = _apply_reliability_gate(result)
            result["available"] = True
            result.setdefault("status", "success")
            result.setdefault("error", "")
            result["service_processing_time_seconds"] = round(
                time.perf_counter() - started_at, 4
            )
            return result
        except Exception as exc:
            LOGGER.exception("MATLAB function %s failed", MATLAB_API_FUNCTION)
            return {
                "available": True,
                "status": "error",
                "error": f"MATLAB processing failed: {exc}",
                "service_processing_time_seconds": round(
                    time.perf_counter() - started_at, 4
                ),
            }

    def shutdown(self) -> None:
        with self._start_lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
                LOGGER.info("MATLAB Engine session closed")
            except Exception:
                LOGGER.exception("MATLAB Engine did not close cleanly")


_SERVICE = MatlabService()
atexit.register(_SERVICE.shutdown)


def get_matlab_service() -> MatlabService:
    return _SERVICE
