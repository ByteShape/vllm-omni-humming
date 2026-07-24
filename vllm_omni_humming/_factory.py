"""Patch point 2 — register a ``humming`` builder in vLLM-Omni's quant factory.

WHY IT EXISTS
    vLLM-Omni's factory builds a quant config by calling ``builder(**flat_kwargs)``,
    where ``flat_kwargs`` is the on-disk ``quantization_config`` dict. But
    ``HummingConfig.__init__`` only accepts ``full_config=`` -> ``TypeError`` /
    ``Cannot instantiate HummingConfig with kwargs {...}`` on a flat offline config.
    We register a ``humming`` builder that wraps the flat dict into a full config.

    This builder ALSO carries the offline-load fix (see below): it marks a
    pre-quantized (offline) config ``is_checkpoint_quantized = True`` so the loader
    takes its CPU-load branch. That flag is one half of a matched pair — the other
    half is ``_loader.patch_process_weights_after_loading`` (per-module CUDA repack).

UPSTREAM TARGET
    vllm-omni ==0.24.0  ·  ``vllm_omni/quantization/factory.py``
      - ``_OVERRIDES`` (dict) — the ONE documented extension point (module docstring:
        "The only extension point is _OVERRIDES"). ``_build_single`` consults it first.
      - ``SUPPORTED_QUANTIZATION_METHODS`` (list), ``_CACHED_ALIAS_MAP`` (cache).

    IMPORTANT VERSION NOTE: the published 0.24.0 wheel does NOT expose a
    ``register_quantization_override()`` helper — only the raw ``_OVERRIDES`` dict.
    (A ``register_quantization_override`` convenience wrapper exists on the upstream
    post-0.24.0 dev branch / a future release.) We therefore write ``_OVERRIDES``
    directly, but PREFER the upstream helper if a newer omni provides it. To rev for
    a new omni: confirm ``_OVERRIDES`` / ``SUPPORTED_QUANTIZATION_METHODS`` still live
    in ``factory`` and adjust ``_install_override`` if the helper's contract changes.
"""
import logging

from ._preflight import _fail

logger = logging.getLogger("vllm_omni_humming")


def build_humming(**kw):
    """The ``humming`` factory builder.

    online (no kwargs)  -> empty ``HummingConfig()`` (the online-quant branch).
    offline (flat on-disk ``quantization_config`` kwargs) -> wrap into a full config
        and mark it ``is_checkpoint_quantized = True``.

    ``is_checkpoint_quantized = True`` makes vLLM-Omni's diffusers loader take its
    OFFLINE placement branch — load the packed weights directly on CPU — instead of
    staging the whole pipeline (bf16 text encoder ~16.6 GB + packed DiT) on CUDA at
    once, which OOMs a 32 GiB card once the packed transformer exceeds ~16 GB. The
    CUDA-only humming repack that this CPU-load path would otherwise break on is
    handled per-module by ``_loader.patch_process_weights_after_loading``.
    """
    from vllm.model_executor.layers.quantization.humming import HummingConfig
    if not kw:
        return HummingConfig()  # online branch: no checkpoint flag
    cfg = HummingConfig.from_config({"quant_method": "humming", **kw})
    cfg.is_checkpoint_quantized = True  # loader: load packed weights directly on CPU
    return cfg


def _install_override(factory, method: str, builder) -> None:
    """Install ``builder`` for ``method`` via vLLM-Omni's ``_OVERRIDES`` extension
    point. Prefer the upstream ``register_quantization_override()`` helper when a
    newer omni provides it; otherwise write the three fields that helper would (the
    dict entry + the supported-methods list + alias-cache invalidation) directly.
    Either way we only touch the documented ``_OVERRIDES`` surface.
    """
    reg = getattr(factory, "register_quantization_override", None)
    if callable(reg):
        reg(method, builder)  # upstream dev / >=0.24.1 path
    else:
        factory._OVERRIDES[method] = builder  # published 0.24.0 path
        methods = getattr(factory, "SUPPORTED_QUANTIZATION_METHODS", None)
        if methods is not None and method not in methods:
            methods.append(method)
    factory._CACHED_ALIAS_MAP = None  # invalidate the reverse-alias cache


def register_factory_override() -> bool:
    """Register the ``humming`` builder into vLLM-Omni's quant factory. Idempotent;
    returns True on success, False (never raises) if omni isn't importable yet."""
    try:
        from vllm_omni.quantization import factory
    except ImportError as e:
        logger.debug("vllm-omni-humming: quant factory not importable yet (%s)", e)
        return False
    # Contract: the documented `_OVERRIDES` dict, or the newer helper that writes it.
    # If neither is present, the factory has been restructured -> fail loud.
    has_helper = callable(getattr(factory, "register_quantization_override", None))
    has_overrides = isinstance(getattr(factory, "_OVERRIDES", None), dict)
    if not (has_helper or has_overrides):
        raise _fail(
            "vllm_omni.quantization.factory._OVERRIDES",
            "to be a dict (or register_quantization_override() to exist)",
        )
    _install_override(factory, "humming", build_humming)
    return True
