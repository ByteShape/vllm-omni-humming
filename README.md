# vllm-omni-humming

Plugin that lets stock **vLLM-Omni** serve **pre-quantized
[humming](https://github.com/inclusionAI/humming) Qwen-Image** checkpoints.

Install it into the vLLM-Omni environment and serve normally — the plugin registers
itself through vLLM-Omni's standard plugin entry point, and quantized checkpoints are
detected automatically from their `transformer/config.json`. No flags, no environment
variables, no configuration.

## Compatibility

| vllm-omni-humming | vLLM-Omni  | humming-kernels | vLLM       | Python           |
| ----------------- | ---------- | --------------- | ---------- | ---------------- |
| **0.2.1**   | `0.24.x` | `>=0.1.11`    | `0.24.0` | `>=3.10,<3.14` |

`humming-kernels >= 0.1.11` is required (older versions produce corrupted images on
some GPUs). Install it explicitly as shown below, or via this package's `[kernels]`
extra.

## What it patches

Five small, humming-only runtime patches against stock vLLM-Omni; other quantization
methods are untouched. Each is a separate source module documenting the exact upstream
symbol it targets:

- `_preimport.py` — import the real `humming` package before vLLM-Omni's stub.
- `_factory.py` — register `humming` as a quantization method for checkpoints with a
  `quantization_config`.
- `_loader.py` — load packed weights on CPU and repack per module on GPU (keeps the
  load peak low enough for 32 GiB cards), and accept humming's tensor-name suffixes.
- `_modpin.py` — route quantized modulation layers (`img_mod.1`/`txt_mod.1`), which
  stock vLLM-Omni keeps in bf16.

## Install

Shown for CUDA 12.9 wheels; adapt the wheel URLs for your platform.

```bash
# 1) conda env (skip the first two lines if you already have conda)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && source ~/miniforge3/bin/activate
conda create -y -n qwen-image-humming python=3.12
conda activate qwen-image-humming

# 2) torch trio (cu129)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129

# 3) vLLM 0.24.0 cu129 wheel
printf 'torch==2.11.0+cu129\ntorchvision==0.26.0+cu129\ntorchaudio==2.11.0+cu129\n' > constraints.txt
PIP_CONSTRAINT=constraints.txt pip install \
  https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl

# 4) vLLM-Omni 0.24.0
PIP_CONSTRAINT=constraints.txt pip install vllm-omni==0.24.0

# 5) this plugin
PIP_CONSTRAINT=constraints.txt pip install vllm-omni-humming==0.2.1

# 6) humming kernels (required)
pip install --no-deps "humming-kernels>=0.1.11"
```

Requirements: NVIDIA GPU (SM75+), driver ≥ 575, gcc. No CUDA toolkit needed — kernels
are JIT-compiled on first serve (a few minutes once, cached afterwards). Any Python
3.10–3.13 environment works in place of conda (a system venv additionally needs the
`python3-venv` and `python3-dev` packages).

## Serve

```bash
vllm-omni serve /path/to/checkpoint \
  --omni --served-model-name Qwen/Qwen-Image-2512 \
  --enable-cpu-offload \
  --port 8124 --api-key "$KEY"
```

Keep `--enable-cpu-offload` on GPUs with ≤ 32 GiB; drop it on larger cards for full
speed. Generate an image (OpenAI images API):

```bash
curl -s http://127.0.0.1:8124/v1/images/generations \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen-Image-2512","prompt":"A red panda on a mossy log at dawn","size":"1024x1024","num_inference_steps":20,"true_cfg_scale":2.5,"negative_prompt":"blurry, low quality, distorted, watermark","seed":1234}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
```

Always send a `negative_prompt` — vLLM-Omni disables classifier-free guidance without
one.

## Troubleshooting

- **`TypeError: HummingConfig.__init__() got an unexpected keyword argument ...`** —
  the plugin isn't installed in the serving venv (`pip show vllm-omni-humming`).
- **Shape/dtype mismatch on `img_mod.1` / `txt_mod.1` at load** — same cause.
- **Corrupted images** — humming-kernels too old; `pip install --no-deps "humming-kernels>=0.1.11"`.

## License

Apache-2.0. See [LICENSE](LICENSE).
