"""vllm-omni-humming — companion plugin that makes a PRE-QUANTIZED humming
Qwen-Image DiT pipeline load & serve on STOCK vLLM-Omni (0.24.x), with no vendor
patches, no ``.pth`` hand-edits, and no ``VLLM_HUMMING_*`` env vars.

It applies, at runtime and additively, FIVE small patches — one per upstream patch
point, each isolated in its own module with a WHY + which-upstream-version comment so
they are easy to audit and to re-verify when vLLM-Omni revs:

  1. ``_preimport``  — import the REAL ``humming`` before/instead of the factory's
                       import stubs (purge a stub if it beat us).
  2. ``_factory``    — register a ``humming`` builder in the quant factory that wraps
                       the flat on-disk ``quantization_config`` into a full config AND
                       marks an offline config ``is_checkpoint_quantized = True``.
  3. ``_loader``     — (a) per-module CUDA-staged repack for the CPU-loaded offline
                       path (matched pair with the flag from #2), and
                       (b) allow-list ``.zero_point`` / ``.global_scale`` in the
                       strict diffusers loader.
  4. ``_modpin``     — route the HummingConfig to the ``img_mod.1`` / ``txt_mod.1``
                       modulation Linears (humming-only).

Entry point: registered under ``vllm_omni.general_plugins`` (the group STOCK
vLLM-Omni loads in every process) and ``vllm.general_plugins`` (an earlier
pre-import in vLLM core's main process). ``register()`` is idempotent, and every
patch is exception-safe — a missing dependency degrades gracefully, it never crashes
the host process.

Keep this module import-light (stdlib only at top level): the heavy vllm / vllm_omni
/ torch imports all live lazily inside the patch functions.
"""
import logging

from . import _factory, _loader, _modpin, _preimport
from ._preflight import UnsupportedVllmOmniError

__version__ = "0.2.1"
__all__ = ["register", "__version__"]

logger = logging.getLogger("vllm_omni_humming")

# Which patches are currently in place (populated by register()).
_STATE = {"preimport": False, "factory": False,
          "process_weights": False, "suffix": False, "modpin": False}


def register() -> None:
    """Plugin entry point. Idempotent; called once per process by vLLM-Omni's
    ``load_omni_general_plugins()`` (and, if wired, vLLM core's general plugins).

    Order matters only for the pre-import: it must run FIRST so everything below —
    and vLLM's own lazy ``import humming`` inside ``HummingConfig`` — binds to the
    real ``humming`` package, never the omni factory stub.
    """
    _STATE["preimport"] = _preimport.preimport_real_humming()

    steps = (
        ("factory", _factory.register_factory_override),
        ("process_weights", _loader.patch_process_weights_after_loading),
        ("suffix", _loader.patch_suffix_allowlist),
        ("modpin", _modpin.patch_modpin),
    )
    for key, step in steps:
        try:
            _STATE[key] = step()
        except UnsupportedVllmOmniError:
            # A private symbol we patch has moved or changed signature. Fail LOUD
            # rather than degrade silently: applying only some patches would serve
            # humming checkpoints incorrectly. Already logged at CRITICAL by _fail().
            _STATE[key] = False
            raise
        except Exception:  # pragma: no cover - never crash the host on unrelated bugs
            logger.exception("vllm-omni-humming: %s failed", step.__name__)
            _STATE[key] = False

    logger.info("vllm-omni-humming %s registered: %s", __version__, _STATE)
