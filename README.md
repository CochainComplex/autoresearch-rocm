# autoresearch ROCm

This is a ROCm-oriented fork of `autoresearch`, tuned for native Linux on AMD Radeon-class GPUs instead of the original H100/CUDA setup.

The core idea is unchanged: an agent edits `train.py`, launches a fixed 5-minute run, checks `val_bpb`, and keeps or discards the change. The code stays intentionally small so research loops remain easy to inspect and automate.

## What changed in this fork

- The runtime now assumes native Linux plus ROCm PyTorch, not CUDA.
- Attention uses PyTorch `scaled_dot_product_attention`, so this fork does not depend on `kernels` or external FlashAttention builds.
- The default model and evaluation settings are reduced for broader 8-24 GB Radeon compatibility.
- Mixed precision is fixed to `fp16`; this fork does not use `bf16`.
- `WINDOW_PATTERN` is restricted to `"L"` in v1 because the original sliding-window path depended on a CUDA-specific kernel.
- `torch.compile` is intentionally disabled in this fork to avoid another ROCm-specific failure surface.

## Project structure

```text
backend.py      - ROCm runtime detection and execution helpers
prepare.py      - fixed data prep, dataloader, and evaluation utilities
train.py        - model, optimizer, and 5-minute training loop
program.md      - agent instructions for the experiment loop
pyproject.toml  - ROCm-oriented dependencies
```

## Quick start

Requirements:
- Native Linux with a working ROCm installation
- A supported AMD GPU visible to ROCm/PyTorch
- `x86_64`
- Python `3.10` on Ubuntu 22.04 or Python `3.12` on Ubuntu 24.04
- `uv`

```bash
# 1. Install the AMD-tested ROCm 7.2 / PyTorch 2.9.1 environment
uv sync

# 2. Download data and train the tokenizer
uv run prepare.py

# 3. Run one 5-minute experiment
uv run train.py
```

## Default ROCm fork settings

- `MAX_SEQ_LEN = 512`
- `EVAL_TOKENS = 4 * 524288`
- `DEPTH = 4`
- `WINDOW_PATTERN = "L"`
- `DEVICE_BATCH_SIZE = 8`
- `TOTAL_BATCH_SIZE = 2**14`

These defaults are intentionally conservative so the repo is more likely to come up on the lower-memory Radeon cards in AMD's supported matrix. If you have a 16-24 GB card and want more throughput, scale `MAX_SEQ_LEN`, `DEPTH`, `DEVICE_BATCH_SIZE`, and `TOTAL_BATCH_SIZE` back up.

## Notes

- PyTorch ROCm still uses the `torch.cuda` API surface. That is expected in this fork.
- `fp16` is fixed because AMD's current Radeon support matrix explicitly lists `FP16` and mixed `FP32/FP16` as the supported path.
- The training summary reports throughput and peak memory on all runs. `mfu_percent` is reported as `n/a` unless you add a device-specific FLOPs constant.
- The base install does not depend on Triton.
- This fork is meant to be a practical bring-up path first, not a benchmark-parity clone of the original CUDA implementation.

## License

MIT
