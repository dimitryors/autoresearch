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

With a fixed 5-minute time budget, **the number of optimizer steps is the single most important factor**. An experiment that gets 1500 steps will almost always beat one that gets 800 steps, regardless of model quality per step. This has been validated across 200+ experiments: every major improvement came from increasing throughput (more steps), not from architectural cleverness.

**Priority order** (do these in sequence, each builds on the previous):

1. **Throughput optimization** (free wins, no quality tradeoff):
   - `torch.compile` mode: `max-autotune` is ~7% faster than `reduce-overhead` (confirmed). Stick with it.
   - SDPA backend selection: let PyTorch auto-select all backends. Don't override.
   - These are zero-risk changes: same model, same math, just faster execution.

2. **Model depth/width scaling** (biggest single improvement category):
   - DEPTH=9 at 512 dim (ASPECT_RATIO=56) was the single biggest improvement (1.036→1.033). More layers beat wider layers at fixed VRAM.
   - On RTX 5090 (32 GB), DEPTH=9 at 512 dim with BS=64 uses ~24.7 GB. DEPTH=10 at 512 dim uses ~27 GB but loses too many steps. DEPTH=12 dim=384 loses quality despite 1607 steps — narrower is worse.
   - The sweet spot is the deepest model that still gets ≥1300 steps at BS=64 without gradient accumulation.
   - HEAD_DIM=128 (4 heads) beats HEAD_DIM=64 (8 heads) at the same dim=512.

3. **Schedule shape tuning** (second biggest improvement category):
   - **Weight decay schedule**: `(1-p)^1.5` is optimal at current config. `(1-p)^2` was originally best, but after adding x0 connections and re-tuning, exponent 1.5 wins.
   - **LR warmdown**: Linear warmdown is optimal. Cosine warmdown was much worse (1.038 vs 1.033). Don't change the warmdown shape.
   - `WARMDOWN_RATIO=0.65` is optimal. Tested 0.6, 0.62, 0.65, 0.67, 0.7, 0.75 — 0.65 wins.
   - `FINAL_LR_FRAC=0.02` is optimal. Zero end-LR (0.0) and larger (0.04, 0.005) are both worse.

4. **Confirmed optimal hyperparameters** (exhaustively tested — do not re-test):
   - `MATRIX_LR=0.055` — tested 0.04–0.06 in fine steps, all worse.
   - `EMBEDDING_LR=0.5` — tested 0.4, 0.6, all worse.
   - `SCALAR_LR=0.25` — tested 0.30, much worse.
   - `UNEMBEDDING_LR=0.004` — tested 0.003, 0.005, both worse.
   - `WEIGHT_DECAY=0.18` — tested 0.14–0.20, all worse.
   - `WD exponent=1.5` — tested 1.25, 1.4, 1.75, 2.0, all worse.
   - `ADAM_BETAS=(0.8, 0.95)` — tested (0.75, 0.78, 0.82, 0.9) × 0.95, all worse.
   - `Muon ns_steps=5, momentum ramp 0.85→0.95 over 300 steps` — tested 4, 6 ns_steps; 250, 400 step ramps; start=0.88; all worse.
   - `NorMuon beta2=0.95` — tested 0.90, 0.99, both worse.
   - `softcap=13.25` — tested 12.0, 13.0, 13.5, 13.75, no cap. 13.25 is best.
   - `lm_head init std=0.001` — tested 0.01, much worse.
   - `VE gate channels=32 (first 32)` — tested 16, 64, last-32, all worse.
   - `x0_lambdas=0.1` — tested 0.0 early layers, 0.05, 0.12, 0.15, all worse.
   - `VE in alternating layers (0,2,4,6,8)` — tested all-9, all-5-early, all-5-late, all worse.
   - `MLP expansion 4x` — tested 3x (fewer steps help but quality hurt), 5x (fewer steps, quality hurt).
   - `DEVICE_BATCH_SIZE=64` — tested 32 (2682 steps but noisier gradient signal), worse.
   - `RoPE base=10000` — tested 4096, 50000, multi-scale, all worse.
   - `ReLU²` activation — tested plain ReLU (catastrophic 1.044), ReLU^1.5, ReLU^3, SiLU, SwiGLU, ReGLU², all worse.
   - `RMSNorm` — LayerNorm worse and more VRAM.
   - Muon (not AdamW) for VE gate matrix.
   - QK norm before RoPE (not after).
   - ATTN before MLP order (MLP-first was tested, worse).
   - Pre-block x0 injection (not post-block, not mid-block after attn).
   - No LR warmup — any warmup (even 3%) wastes steps and hurts.
   - No lm_head bias — tested, worse.

5. **Unexplored territory for next 100 experiments** (architectural innovation only — hyperparameters are exhausted):
   - **Sparse / local attention**: sliding window attention (e.g., window=256) with global tokens, alternating local+global layers. Could reduce per-step compute and allow slightly larger model.
   - **ALiBi position encoding**: replace RoPE with ALiBi learned biases — different generalization properties, might interact better with the short training.
   - **Attention sinks**: one or two "sink" tokens that receive excess attention mass (as in StreamingLLM), preventing attention entropy collapse.
   - **Mixture of Experts (MoE)**: replace some MLP layers with 2-expert MoE — same compute, more parameter capacity. Top-1 routing to avoid throughput hit.
   - **Stochastic depth (LayerDrop)**: randomly skip layers during training with survival prob ~0.9. Acts as implicit ensembling, may improve generalization.
   - **Grouped Query Attention (GQA) with n_kv_head=2 revisited at DEPTH=9**: previously tried at different baseline — worth one more try with current best config.
   - **New optimizer class**: SOAP (Shampoo with Adam Preconditioner), Adan — might converge faster than Muon+Adam combo.
   - **Pre-norm vs sandwich norm**: current pre-norm is standard; sandwich norm (norm before and after each sublayer) could stabilize training further.
   - **Attention temperature learned per layer** (not per head): single scalar per layer rather than per head; lower complexity than the per-head version that failed.
   - **Sequence length curriculum**: train on shorter sequences first, then longer — could pack more steps in fixed budget.

**Anti-patterns** (things that consistently fail — avoid wasting experiments on these):

- **Gradient accumulation** to fit larger models: kills step count. 2x micro-steps ≈ 2x slower. If model doesn't fit at `DEVICE_BATCH_SIZE=64`, it's too big.
- **Gradient checkpointing**: breaks torch.compile CUDA graph optimization, reducing MFU by 30-50%.
- **Weight tying** (shared embedding/unembedding): embedding and unembedding need very different LRs (0.6 vs 0.004). Tying destroys one of them.
- **Label smoothing**: eval uses hard targets — mismatch causes large quality drops.
- **Auxiliary losses** (multi-token prediction, z-loss): extra compute per step reduces step count, added gradient doesn't compensate.
- **Reduced precision flags** (`allow_bf16_reduced_precision_reduction`): loses quality without speed gains.
- **Parallel attention+MLP** (PaLM-style): hurts quality despite gaining ~31 steps. Attn/MLP interaction matters.
- **GQA with halved KV heads** (n_kv_head=2 from 4): quality drop (1.044) far exceeds memory savings. KV cache is not the bottleneck.
- **Shared value embeddings**: per-layer VE is important. Sharing a single VE loses quality (1.038).
- **Removing QK norm**: catastrophic — causes 0.018+ bpb loss. QK norm is critical for stability.
- **Cosine LR warmdown**: much worse than linear (1.038 vs 1.033).
- **Gradient clipping with Muon**: unnecessary, Muon's Newton step already controls update magnitudes.
- **Constant weight decay** (no decay to 0): WD decay schedule is important for end-of-training refinement.
- **Combining near-misses**: changes individually close to best rarely combine. They compete for the same slack in the loss landscape. Confirmed with 3+ experiments.
- **SiLU activation**: much worse than ReLU² (1.048) and uses more memory.
- **SwiGLU MLP**: worse quality (1.042) and more memory.
- **RoPE base frequency changes**: base=50000 worse. Base=4096 worse. Multi-scale (per head group) worse.
- **Differential attention**: two SDPA calls doubles compute → only 821 steps → catastrophic 1.097. Never viable in 5-min budget.
- **Learnable softcap**: gradient drives softcap toward clamp minimum (1.0) → catastrophic 2.588. Keep softcap as fixed constant.
- **V norm before or after VE mixing**: adds compute, hurts quality (1.039–1.040). VE values should mix with raw v.
- **Mid-layer skip connection (x_mid)**: extra parameters/compute hurt both quality and throughput (1.039+).
- **ReGLU² MLP**: worse quality (1.048) despite being similar in concept to ReLU². Gate mechanism adds cost.
- **Plain ReLU** (no squaring): catastrophic 1.044 — the squaring in ReLU² is essential.
- **ResidLambda depth-dependent init**: `1/sqrt(layer+1)` for resid_lambdas — catastrophic 1.044, disrupts skip balance.
- **EMA weights for eval with torch.compile**: parameter copy at eval time breaks CUDA graph → catastrophic 1.081. Avoid EMA entirely.
- **MLP order change**: MLP-before-ATTN is worse. ATTN→MLP is optimal.
- **LR warmup**: any warmup (even 3%) wastes steps and hurts vs. no warmup. The model trains well cold.
- **Half batch size** (BS=32): gets 2x steps but gradient noise overwhelms the signal — worse quality.
- **LayerNorm instead of RMSNorm**: worse quality and significantly more VRAM.
- **MLP expansion 5x**: more VRAM, fewer steps, worse quality. 4x is optimal.
- **DEPTH=12 with narrower dim=384**: gets more steps (1607) but narrower model underperforms. Depth without width doesn't help.
- **VE gate with full n_embd input**: 512 channels is worse than 32. The gate doesn't need all features.
- **2-layer VE gate** (MLP with ReLU): extra capacity doesn't help gating — simple linear gate is sufficient.
- **AdamW for VE gate matrix**: Muon (orthogonal updates) is better for the gate matrix than AdamW.
- **Cyclical LR in warmdown or steady state**: any LR cycling was worse — monotonic warmdown is correct.
- **Learnable attn temperature per head**: worse quality and adds params. Per-layer might work but per-head doesn't.
- **Multi-scale RoPE** (different bases per head group): worse quality.
- **Learnable final RMSNorm scale**: stateless norm is sufficient.
- **AdamW eps=1e-8** (from 1e-10): worse — lower epsilon is better for numerical stability.
- **WD=0 during warmdown**: WD during warmdown is helpful, zero hurts.
- **WD constant in steady-state then decay in warmdown**: worse than decaying from start.
- **Partial RoPE** (50% of head dims): worse — all dims benefit from position encoding.
- **Softcap in bf16** (removing `.float()` before tanh): cross_entropy upcasts anyway but the computation itself loses precision, slightly worse.
- **x0 injections with AdamW betas=(0.9, 0.95)** instead of default: barely worse, confirms default is optimal.

**Diminishing returns**: After 200 experiments total, **0 improvements in the last 100**. The current config (val_bpb=1.028292) is at an extremely deep local optimum. Do not attempt any more hyperparameter grid-searching — the entire conventional search space is exhausted. Focus exclusively on structural architectural changes or new training techniques.

**Key learnings from experiments 1–100 (mar22 run)**:
- 16 keeps out of 100 experiments (16% success rate, but most in first 50).
- Total improvement: 1.036433 → 1.028292 (Δ=-0.008141, ~0.79% relative improvement).
- Biggest wins: depth scaling (Δ=-0.003), schedule shape (Δ=-0.001), QK norm placement (Δ=-0.0009), softcap tuning (Δ=-0.0005), individual hyperparams (Δ=-0.001 cumulative).

**Key learnings from experiments 101–200 (mar24 run)**:
- 0 keeps out of 100 experiments. The model is at a true local optimum for conventional approaches.
- Every hyperparameter tested: all LRs, WD, betas, schedule ratios, architecture variants — none improved.
- The 5 near-misses (closest to baseline) were: WD=0.16 (1.028335, Δ=+0.000043), WD=0.17 (1.028689), WARMDOWN=0.67 (1.028567), ns_steps=6 (1.028540), ADAM_BETAS=(0.75,0.95) x0 betas (1.028913). None combined for improvement.
- The next 100 experiments MUST try fundamentally different architectural approaches. Do not touch hyperparameters.

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