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
   - `torch.compile` mode: try `max-autotune` first, then `reduce-overhead`. `max-autotune` was validated as ~7% faster (1510 vs 1414 steps). Stick with it.
   - SDPA backend selection: let PyTorch auto-select all backends — forcing CuDNN was worse. Don't override SDPA backend choices.
   - These are zero-risk changes: same model, same math, just faster execution.

2. **Model depth/width scaling** (biggest single improvement category):
   - DEPTH=9 at 512 dim (ASPECT_RATIO=56) was the single biggest improvement (1.036→1.033). More layers beat wider layers at fixed VRAM.
   - On RTX 5090 (32 GB), DEPTH=9 at 512 dim with BS=64 uses ~24.7 GB. DEPTH=10 at 512 dim uses ~27 GB but loses too many steps (1158 vs 1350). 640 dim OOMs at BS=64.
   - The sweet spot is the deepest model that still gets ≥1300 steps at BS=64 without gradient accumulation.

3. **Schedule shape tuning** (second biggest improvement category):
   - **Weight decay schedule**: Quadratic decay `(1-p)^2` beat linear `(1-p)`. This was the second-best improvement (1.033068→1.032967). The key insight: quadratic WD decays faster, so WD/LR ratio stays low at end of training — model is free for fine-tuning in the warmdown phase.
   - Cubic `(1-p)^3` was too aggressive. Exponent 1.5 was insufficient. Cosine WD was worse. Quadratic (exponent 2.0) is optimal.
   - **LR warmdown**: Linear warmdown is optimal. Cosine warmdown was much worse (1.038 vs 1.033). Don't change the warmdown shape.
   - `WARMDOWN_RATIO` has a sweet spot at 0.65. Tested 0.6, 0.65, 0.7, 0.75 — 0.65 wins consistently across model sizes.

4. **Hyperparameter fine-tuning** (diminishing returns — most params are at tight optima):
   - LR values (MATRIX_LR=0.04, EMBEDDING_LR=0.5, SCALAR_LR=0.25, UNEMBEDDING_LR=0.004) are tightly optimized. ±0.005-0.01 changes all hurt.
   - WEIGHT_DECAY=0.15 is optimal. Tested 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3.
   - ADAM_BETAS=(0.8, 0.95) is optimal. Higher/lower beta1 and higher beta2 all worse.
   - Muon: ns_steps=5, momentum ramp 0.85→0.95 over 300 steps. All variations tested, all worse.
   - softcap=12 is optimal. Tested 10, 11, 12, 13, 14, and no cap. Removing cap is very bad (1.047).

5. **Unexplored territory for next 100 experiments**:
   - **Data-side optimizations**: sequence packing, curriculum learning, data mixing strategies (within prepare.py constraints).
   - **New attention patterns**: sliding window + global hybrid, local attention for some layers.
   - **Initialization schemes**: different init scales, layer-dependent scaling.
   - **New optimizer ideas**: LR warmup reintroduction with different schedule, cyclical LR, per-layer LR schedules beyond current groups.
   - **Architectural novelties**: mixture of depths (early exit), different normalization, attention modifications (e.g. differential attention, multi-head latent attention).

**Anti-patterns** (things that consistently fail — avoid wasting experiments on these):

- **Gradient accumulation** to fit larger models: kills step count. 2x micro-steps ≈ 2x slower, not 1x. If a model doesn't fit in VRAM at `DEVICE_BATCH_SIZE=64`, it's too big for this GPU.
- **Gradient checkpointing**: breaks torch.compile CUDA graph optimization, reducing MFU by 30-50%. Only viable if the MFU loss is small (test first).
- **Weight tying** (shared embedding/unembedding): embedding and unembedding require very different learning rates (e.g. 0.6 vs 0.004). Tying forces a single LR, which destroys one or the other.
- **Label smoothing**: trains on soft targets but eval uses hard targets — the mismatch causes large quality drops.
- **Auxiliary losses** (multi-token prediction, z-loss): the extra compute per step reduces step count, and the added gradient signal doesn't compensate in short training runs.
- **Reduced precision flags** (e.g. `allow_bf16_reduced_precision_reduction`): loses quality without meaningful speed gains on modern GPUs.
- **Parallel attention+MLP** (PaLM-style): tested, hurts quality despite gaining ~31 steps. The interaction between attn and MLP outputs matters.
- **GQA with halved KV heads** (n_kv_head=2 from 4): quality drop (1.044) far exceeds the memory savings at this model size. KV cache is not the bottleneck.
- **Shared value embeddings**: per-layer VE is important. Sharing a single VE across layers loses quality (1.038).
- **Removing QK norm**: catastrophic — causes 0.018 bpb loss. QK norm is critical for training stability.
- **Cosine LR warmdown**: much worse than linear (1.038 vs 1.033). Linear warmdown is the right shape.
- **Gradient clipping with Muon**: unnecessary, Muon's Newton step already controls update magnitudes.
- **Constant weight decay** (no decay to 0): WD decay schedule is important for end-of-training refinement.
- **Combining near-misses**: changes that individually score close to the best rarely combine for improvement. They are not orthogonal — they often compete for the same "slack" in the loss landscape.
- **SiLU activation**: much worse than ReLU² (1.048 vs 1.033) and uses more memory.
- **SwiGLU MLP**: worse quality (1.042) and more memory than ReLU² with standard MLP.
- **RoPE base frequency changes**: base=50000 was worse than default 10000.

**Diminishing returns**: If 10+ consecutive experiments are discarded and all within ±0.002 of the best, the hyperparameter space is likely exhausted at the current architecture. At that point, don't keep grid-searching — either try a fundamentally different approach (new architecture, new optimizer, new training technique) or accept the current result.

**Key learnings from first 100 experiments (mar22 run)**:
- 8 keeps out of 100 experiments (8% success rate). Most improvements came in the first 50 experiments.
- Total improvement: 1.036433 → 1.032967 (Δ=-0.003466, ~0.33% relative improvement).
- Biggest wins by category: depth scaling (Δ=-0.003), schedule shape (Δ=-0.001), individual hyperparams (Δ=-0.001 cumulative from 5 small wins).
- After ~experiment 88 (quadratic WD), 12 consecutive experiments were discarded — the current config is at a local optimum for conventional hyperparameter tuning.
- The next 100 experiments should focus on architectural novelties and training technique changes, not hyperparameter grid search.

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