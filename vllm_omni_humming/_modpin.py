"""Patch point 4 — mod-unpin: route a HummingConfig to the modulation Linears.

WHY IT EXISTS
    A pre-quantized humming Qwen-Image DiT ships ``img_mod.1`` / ``txt_mod.1`` as
    packed int tensors, but stock vLLM-Omni pins those two modulation Linears to bf16
    (``img_mod.1`` hard ``quant_config=None``; ``txt_mod.1`` via ``safe_quant_config``,
    which returns None for non-INC methods). Under a HummingConfig that both (a)
    mismatches the packed on-disk shape/dtype at load and (b) forces ~a third of the
    DiT to stay full precision (a +9 GB footprint). We monkeypatch the block so those
    two Linears receive the HummingConfig — HUMMING ONLY; every other quant method
    keeps upstream behavior untouched.

UPSTREAM TARGET
    vllm-omni ==0.24.0  ·
    ``vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py``
      - ``QwenImageTransformerBlock.__init__`` builds ``img_mod`` / ``txt_mod`` from
        ``ReplicatedLinear`` with the modulation Linears pinned (~lines 738-777).
      - module-level ``ReplicatedLinear`` symbol we transiently shadow during the ctor.
    To rev: confirm the block still builds the two mods via the module-level
    ``ReplicatedLinear`` with prefixes ``.img_mod.1`` / ``.txt_mod.1``.
"""
import inspect
import logging

from ._preflight import require_attr, require_params

logger = logging.getLogger("vllm_omni_humming")


def patch_modpin() -> bool:
    """Route the HummingConfig to ``img_mod.1`` / ``txt_mod.1`` in every
    ``QwenImageTransformerBlock``. Idempotent; returns True if the patch is in place,
    False (never raises) if the qwen_image module isn't importable yet."""
    try:
        import vllm_omni.diffusion.models.qwen_image.qwen_image_transformer as M
    except ImportError as e:
        logger.debug("vllm-omni-humming: qwen_image module not importable yet (%s)", e)
        return False

    Block = require_attr(M, "QwenImageTransformerBlock", "qwen_image_transformer")
    if getattr(Block, "_bshumming_modpin", False):
        return True
    # Contract: the block is still built from a module-level `ReplicatedLinear` we can
    # shadow, and its ctor still takes a `quant_config` we route to the mod Linears.
    require_attr(M, "ReplicatedLinear", "qwen_image_transformer")
    require_params(Block.__init__, ["quant_config"],
                   "QwenImageTransformerBlock.__init__")

    from vllm.model_executor.layers.quantization.humming import HummingConfig

    orig_init = Block.__init__
    sig = inspect.signature(orig_init)

    def patched_init(self, *args, **kwargs):
        try:
            bound = sig.bind_partial(self, *args, **kwargs)
            qc = bound.arguments.get("quant_config", None)
        except Exception:
            qc = kwargs.get("quant_config", None)
        if not isinstance(qc, HummingConfig):
            return orig_init(self, *args, **kwargs)
        # Transiently shadow the module-level ReplicatedLinear so the two modulation
        # Linears (prefix .img_mod.1 / .txt_mod.1) built inside the block body receive
        # the humming quant_config. Restored in `finally` so nothing else is affected.
        OrigRL = M.ReplicatedLinear

        class _ModRoutedReplicatedLinear(OrigRL):
            def __init__(self, *a, **k):
                pfx = k.get("prefix", "")
                if pfx.endswith(".img_mod.1") or pfx.endswith(".txt_mod.1"):
                    k["quant_config"] = qc
                super().__init__(*a, **k)

        M.ReplicatedLinear = _ModRoutedReplicatedLinear
        try:
            return orig_init(self, *args, **kwargs)
        finally:
            M.ReplicatedLinear = OrigRL

    patched_init._bshumming_wrapped = True
    Block.__init__ = patched_init
    Block._bshumming_modpin = True
    return True
