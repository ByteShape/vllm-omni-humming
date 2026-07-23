"""Patch points 3 & 5 — both target vLLM-Omni's diffusers loader
(``vllm_omni/diffusion/model_loader/diffusers_loader.py::DiffusersPipelineLoader``).

They are grouped in one file because they patch the same upstream class; each is a
separate, self-contained function.

--------------------------------------------------------------------------------
Patch point 3 — CPU-staged offline load (per-module CUDA repack).

WHY IT EXISTS
    Companion to the ``is_checkpoint_quantized = True`` flag set in
    ``_factory.build_humming``. With that flag the loader loads packed weights on CPU
    (``load_device == "cpu"`` -> ``target_device == cpu``) instead of staging the
    whole pipeline on CUDA. But humming's ``process_weights_after_loading`` runs the
    CUDA-only ``humming::repack_weight`` -> ``NotImplementedError`` on the CPU backend.
    We re-point the processing device at CUDA *only for HummingConfig*: the loader's
    own per-module move logic then stages ONE fused Linear at a time through the GPU
    (move -> repack -> move back to CPU), so the load-time GPU peak is ~30 MB instead
    of TE+DiT co-residency. End state is identical to the CUDA-staged path (repacked
    weights resident on CPU, swapped in per request by MODEL_LEVEL cpu-offload).

UPSTREAM TARGET
    vllm-omni ==0.24.0
      - ``DiffusersPipelineLoader._process_weights_after_loading(self, model, target_device)``
        (~line 400), called from ``load_model`` (~line 392) with ``target_device``.
      - ``self.quant_config`` (set ~line 140) — read to detect a HummingConfig.
      - offline branch keyed on ``getattr(quant_cfg, "is_checkpoint_quantized", False)``
        (~line 358).
    To rev: re-check the ``_process_weights_after_loading`` signature and that the
    loader still holds ``self.quant_config``.

--------------------------------------------------------------------------------
Patch point 5 — quant-suffix allowlist (asymmetric / online recipe parity).

WHY IT EXISTS
    The strict loader raises on any checkpoint tensor whose name isn't a known
    quant artifact. Upstream 0.24.0 already allow-lists ``.weight_scale`` (and
    ``.g_idx``, ``.weight_scale_inv``, ``.input_scale``, ``.qweight_type``) but NOT
    ``.zero_point`` / ``.global_scale``. We ADD exactly those two — verified still
    absent from upstream 0.24.0's ``_QUANTIZED_WEIGHT_SUFFIXES``, so nothing is
    redundantly re-added. Not exercised by the symmetric g32 artifact (weight +
    weight_scale only); present for asymmetric recipes and the online path.

UPSTREAM TARGET
    vllm-omni ==0.24.0
      - ``DiffusersPipelineLoader._is_expected_quantized_weight(name)`` (staticmethod,
        ~line 454) and its ``_QUANTIZED_WEIGHT_SUFFIXES`` tuple (~line 463).
    To rev: if upstream adds ``.zero_point`` / ``.global_scale`` to that tuple, drop
    them here (our wrapper is already additive, so it stays correct regardless).
"""
import logging

logger = logging.getLogger("vllm_omni_humming")

# Suffixes upstream 0.24.0 does NOT yet allow-list (verified against the pristine
# wheel's _QUANTIZED_WEIGHT_SUFFIXES). Keep only names still missing upstream.
_EXTRA_QUANT_SUFFIXES = (".zero_point", ".global_scale")


def patch_process_weights_after_loading() -> bool:
    """Re-point the post-load weight processing device to CUDA for HummingConfig, so
    humming's GPU-only repack runs while the loader stages one module at a time.
    Idempotent; returns True on success, False (never raises) if unavailable."""
    try:
        import torch
        from vllm_omni.diffusion.model_loader import diffusers_loader as dl
    except Exception as e:
        logger.debug("vllm-omni-humming: diffusers_loader not importable yet (%s)", e)
        return False

    L = dl.DiffusersPipelineLoader
    orig = L._process_weights_after_loading
    if getattr(orig, "_bshumming_cuda_staged", False):
        return True

    def _process_weights_after_loading(self, model, target_device):
        qc = getattr(self, "quant_config", None)
        # Detect HummingConfig by class name to avoid importing humming/vllm here.
        if (target_device.type == "cpu" and qc is not None
                and type(qc).__name__ == "HummingConfig"
                and torch.cuda.is_available()):
            target_device = torch.device("cuda")
        return orig(self, model, target_device)

    _process_weights_after_loading._bshumming_cuda_staged = True
    L._process_weights_after_loading = _process_weights_after_loading
    return True


def patch_suffix_allowlist() -> bool:
    """Add ``.zero_point`` / ``.global_scale`` to the strict loader's expected
    quant-suffix set (additive; upstream matches are honored first). Idempotent;
    returns True on success, False (never raises) if unavailable."""
    try:
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader as L,
        )
    except Exception as e:
        logger.debug("vllm-omni-humming: diffusers_loader not importable yet (%s)", e)
        return False
    if getattr(L, "_bshumming_suffix", False):
        return True

    orig = L._is_expected_quantized_weight

    def patched(name):
        try:
            if orig(name):
                return True
        except Exception:
            pass
        return name.endswith(_EXTRA_QUANT_SUFFIXES)

    L._is_expected_quantized_weight = staticmethod(patched)
    L._bshumming_suffix = True
    return True
