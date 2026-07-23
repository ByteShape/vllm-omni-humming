"""Patch point 1 — real-``humming`` pre-import (beats vLLM-Omni's factory stub).

WHY IT EXISTS
    ``vllm_omni.quantization.factory`` runs ``_register_humming_stubs()`` at import
    time, which installs *stub* ``humming`` / ``humming.*`` modules into
    ``sys.modules`` UNLESS the real ``humming`` package is already imported. If the
    stub wins, vLLM's lazy ``from .humming import HummingConfig`` and humming's
    ``transform_humming_layer`` bind to the stub and break. We import the REAL
    ``humming`` first, and purge any stub that beat us to ``sys.modules``.

UPSTREAM TARGET
    vllm-omni ==0.24.0  ·  ``vllm_omni/quantization/factory.py::_register_humming_stubs``
    (lines ~28-69; guarded by ``if "humming" in sys.modules``). The real ``humming``
    top-level module is provided by the ``humming-kernels`` distribution (>=0.1.11).

    Stable across the 0.24.x line; re-verify the stub-install guard if it moves.
"""
import logging
import sys

logger = logging.getLogger("vllm_omni_humming")


def preimport_real_humming() -> bool:
    """Ensure the REAL ``humming`` package is in ``sys.modules`` before (or instead
    of) vLLM-Omni's factory stubs. Idempotent and exception-safe: returns True on
    success, False (never raises) if the ``humming`` wheel is absent.
    """
    try:
        hm = sys.modules.get("humming")
        if hm is not None and getattr(hm, "__file__", None) is None:
            # A stub (bare ModuleType, no __file__) is installed -> purge the whole
            # humming.* subtree so the real package imports cleanly.
            for name in [k for k in sys.modules
                         if k == "humming" or k.startswith("humming.")]:
                del sys.modules[name]
        import humming  # noqa: F401  (the real wheel, from humming-kernels)
        return True
    except Exception as e:  # pragma: no cover - keep host startup non-fatal
        logger.debug("vllm-omni-humming: real-humming pre-import skipped (%s)", e)
        return False
