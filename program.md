# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Detect hardware**: Run a quick GPU probe to determine GPU model, VRAM, and compute capability. Record the results — they set hard constraints on what experiments are feasible:
   ```python
   python -c "import torch; print(torch.cuda.get_device_name()); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print(f'Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')"
   ```
   Use VRAM to estimate the maximum model size that fits with `DEVICE_BATCH_SIZE=64` and `seq_len=2048`. A rough rule: peak activation memory ≈ `6 * VRAM_per_layer_GB * n_layers` plus fixed overhead for embeddings, optimizer state, and logits buffer (`vocab_size * batch * seq * 4 bytes`). Do NOT attempt model sizes that would exceed ~80% of total VRAM — it will just OOM.
5. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

### Experiment strategy

With a fixed 5-minute time budget, **the number of optimizer steps is the single most important factor**. An experiment that gets 1500 steps will almost always beat one that gets 800 steps, regardless of model quality per step. This has been validated across 100+ experiments: every major improvement came from increasing throughput (more steps), not from architectural cleverness.

**Priority order** (do these in sequence, each builds on the previous):

1. **Throughput optimization** (free wins, no quality tradeoff):
   - `torch.compile` mode: try `max-autotune` first, then `reduce-overhead`. These find better CUDA kernels and use CUDA graphs. On some hardware `max-autotune` gives 5-10% more MFU for free.
   - SDPA backend selection: try forcing CuDNN (`torch.backends.cuda.enable_cudnn_sdp(True)` + disable others) — it can be faster than the default on newer GPUs. Also try `torch.backends.cudnn.benchmark = True`.
   - These are zero-risk changes: same model, same math, just faster execution.

2. **Batch size / step count tuning**:
   - Reduce `TOTAL_BATCH_SIZE` to get more optimizer steps (fewer tokens per step but more steps total). The optimal point depends on model size — too small loses gradient quality, too large loses steps.
   - Keep `DEVICE_BATCH_SIZE` as large as VRAM allows (avoid gradient accumulation — it effectively halves your step count since each accumulation micro-step takes nearly as long as a full step).

3. **LR schedule tuning**: `WARMDOWN_RATIO` is the most impactful schedule parameter. Tune it in ~0.05 increments.

4. **Architecture / hyperparameter tuning**: Only after throughput is maximized. Most hyperparameters have narrow sweet spots — expect diminishing returns quickly.

**Anti-patterns** (things that consistently fail — avoid wasting experiments on these):

- **Gradient accumulation** to fit larger models: kills step count. 2x micro-steps ≈ 2x slower, not 1x. If a model doesn't fit in VRAM at `DEVICE_BATCH_SIZE=64`, it's too big for this GPU.
- **Gradient checkpointing**: breaks torch.compile CUDA graph optimization, reducing MFU by 30-50%. Only viable if the MFU loss is small (test first).
- **Weight tying** (shared embedding/unembedding): embedding and unembedding require very different learning rates (e.g. 0.6 vs 0.004). Tying forces a single LR, which destroys one or the other.
- **Label smoothing**: trains on soft targets but eval uses hard targets — the mismatch causes large quality drops.
- **Auxiliary losses** (multi-token prediction, z-loss): the extra compute per step reduces step count, and the added gradient signal doesn't compensate in short training runs.
- **Reduced precision flags** (e.g. `allow_bf16_reduced_precision_reduction`): loses quality without meaningful speed gains on modern GPUs.

**Diminishing returns**: If 10+ consecutive experiments are discarded and all within ±0.002 of the best, the hyperparameter space is likely exhausted at the current architecture. At that point, don't keep grid-searching — either try a fundamentally different approach (new architecture, new optimizer, new training technique) or accept the current result.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Meta-review (every 100 experiments)

Every 100 experiments (count rows in `results.tsv`), pause the experiment loop and perform a meta-review:

1. **Analyze results**: Read `results.tsv` and categorize all experiments. For each `keep`, identify the *mechanism* behind the improvement (more steps? better LR schedule? architectural change?). For clusters of `discard` experiments, identify what approach they share and why it failed.
2. **Update `program.md`**: Create a new git branch (e.g. `improve-program-md-N` where N is the review number) from the current experiment branch. Update the "Experiment strategy", "Anti-patterns", and "Priority order" sections of this file based on what you learned. Add new anti-patterns, refine priorities, remove advice that turned out to be wrong. Commit, push, and return to the experiment branch.
3. **Reset strategy**: After updating, return to the experiment branch and use the updated guidance to plan the next 100 experiments. If the meta-review reveals that the current approach has plateaued, pivot — e.g. switch from hyperparameter tuning to architectural exploration, or vice versa.

The goal is to make each batch of 100 experiments smarter than the last. The `program.md` is a living document — it should accumulate the wisdom of all previous experiments so that the agent never repeats known dead ends.