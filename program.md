# autoresearch ROCm

This fork runs autonomous research on native Linux with AMD ROCm.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `backend.py` — ROCm runtime detection and execution helpers. Do not modify.
   - `prepare.py` — fixed data prep, tokenizer, dataloader, and evaluation. Do not modify.
   - `train.py` — the file you modify.
4. **Verify data exists**: check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: create `results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off experimentation.

## Experimentation

Each experiment runs on a single ROCm GPU. The training script runs for a fixed 5-minute wall-clock budget, excluding startup and compilation. Launch it as:

```bash
uv run train.py
```

What you CAN do:
- Modify `train.py`. Everything there is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, and model size.

What you CANNOT do:
- Modify `backend.py`. It is the fixed ROCm fork infrastructure.
- Modify `prepare.py`. It is read-only and defines the benchmark setup.
- Install new packages or add dependencies.
- Modify the evaluation harness in `prepare.py`.

ROCm constraints:
- Keep `WINDOW_PATTERN = "L"` unless you are explicitly implementing a new ROCm-safe attention backend.
- Precision is fixed to `fp16`. Do not add a `bf16` path back into this fork.
- `torch.compile` is disabled in this fork.
- The runtime uses PyTorch ROCm through the `torch.cuda` API surface. That is expected.

The goal is the same as the original project: get the lowest `val_bpb` that still fits the time budget.

## Output format

At the end of a successful run the script prints a summary like:

```text
---
val_bpb:          1.234567
training_seconds: 300.0
total_seconds:    322.1
peak_vram_mb:     15234.5
mfu_percent:      n/a
total_tokens_M:   65.5
num_steps:        2048
num_params_M:     11.5
depth:            4
```

Extract the key metrics from the log with:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Logging results

Record every experiment in `results.tsv` as tab-separated values:

```text
commit	val_bpb	memory_gb	status	description
```

Use:
1. short git hash
2. `val_bpb`, or `0.000000` for crashes
3. peak memory in GB, rounded to one decimal place, or `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. a short experiment description

## The experiment loop

1. Inspect git state.
2. Change `train.py`.
3. Commit.
4. Run `uv run train.py > run.log 2>&1`.
5. Extract results from the log.
6. If it crashed, read the traceback with `tail -n 50 run.log` and decide whether to retry or discard.
7. Append a row to `results.tsv` without committing that file.
8. Keep commits that improve `val_bpb`.
9. Reset back from regressions.

Once the loop starts, keep going until the human interrupts you.
