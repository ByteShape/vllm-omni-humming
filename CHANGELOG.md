# Changelog

All notable changes to `vllm-omni-humming`. This project adheres to
[Semantic Versioning](https://semver.org/).

## 0.3.0

- Fail loud on an unsupported vLLM-Omni version: when a private vLLM-Omni symbol this
  plugin patches is missing or has changed signature, `register()` now raises
  `UnsupportedVllmOmniError` (logged at CRITICAL) instead of silently skipping the patch
  — a partially-patched server would load humming checkpoints incorrectly. The
  "not imported yet" case still soft-skips. Verified against vLLM-Omni 0.24.0.
- README: compatibility notes (Linux; NVIDIA 30-/40-/50-series + RTX Pro 6000) and a
  keyless serve example.

## 0.2.1

Documentation-only release: conda-first install instructions. No code changes.

## 0.2.0

First public release (GitHub + PyPI).

- Serve pre-quantized humming Qwen-Image checkpoints on stock vLLM-Omni `0.24.x` —
  auto-detected from the checkpoint's `quantization_config`, no flags or environment
  variables.
- CPU-staged loading with per-module GPU repack, so large checkpoints load on
  32 GiB cards.
- Quantized modulation-layer (`img_mod.1`/`txt_mod.1`) support.
- Registers automatically via the `vllm_omni.general_plugins` /
  `vllm.general_plugins` entry points; idempotent and exception-safe.
- Optional `[kernels]` extra pinning `humming-kernels>=0.1.11`.
