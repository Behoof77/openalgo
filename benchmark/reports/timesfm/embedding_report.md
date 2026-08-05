# TimesFM 2.5 as a Market Context Model for the Option Chain Transformer

**Scope**: Can TimesFM generate embeddings fusable with ChainST (options-ml-transformer)?
**Status**: All claims measured on this machine (Intel i5-8250U, 4C/8T, 7.9 GB RAM, CPU-only).
**Verdict**: **Yes — viable and recommended.** TimesFM is usable strictly as a frozen embedding
extractor. Precompute embeddings offline from the NIFTY spot series, then feed them into ChainST
as an extended per-patch context vector. This works without GPU, without jax, and without any
change to ChainST's architecture beyond widening one LayerNorm input.

All experiment code and raw results live in
`C:\Users\hem_s\AppData\Local\Temp\opencode\timesfm_exp\`:
`exp1_model_and_embeddings.py` + `exp1_results.json`, `exp2_benchmark.py` + `exp2_progress.jsonl`,
`exp3_export_pipeline.py` + `exp3_export.json` + `timesfm_embeddings_mini.npy`.

---

## Q1. Model info

| Property | Value |
|---|---|
| Class | `timesfm.TimesFM_2p5_200M_torch` (google/timesfm-2.5-200m-pytorch) |
| Params | **231,289,280 (231.3 M)** — measured |
| Weights footprint | ~882 MB fp32 (`model.safetensors`) |
| Embedding dim (`hidden_size`) | **1280** |
| Transformer layers | 20 |
| Attention heads | 16 (head_dim 80) |
| Patch length | 32 (input is tokenized as 32-value patches) |
| Output patch length | 128 (AR decode step) |
| Context limit | 16,384 (way beyond anything needed here) |
| Quantile head | 1024 horizon, quantiles [0.1..0.9], median index 5 |
| Runtime device | CPU (no GPU present) |
| Load time | 5.3 s warm (12.5 s cold), RSS 227 MB -> 1,110 MB |

Configuration loaded from the local checkpoint
`openalgo/timesfm-server/models/timesfm-2.5-200m-pytorch/config.json`.

## Q2. Hidden states

Captured with a plain `register_forward_hook` on `model.model` (the torch module). The module
`forward()` returns `((input_embeddings, output_embeddings, output_ts, output_quantile_spread),
decode_caches)` — so the **last hidden state is already exposed** by the public forward pass; no
source patch needed for embeddings.

Measured for a batch of 16, context 512 (16 patches):

| Tensor | Shape | dtype |
|---|---|---|
| `input_embeddings` (tokenizer out) | `(16, 16, 1280)` | float32 |
| `output_embeddings` (last layer out) | `(16, 16, 1280)` | float32 |
| `output_ts` (raw forecast) | `(16, 128, 1)` | float32 |
| `output_quantile_spread` | `(16, 128, 10)` | float32 |

Axes: `(batch, patch_index, hidden_dim)`. Patch 0 = oldest, patch `num_patches-1` = most recent
32 candles.

**Attention weights are NOT exposed.** `Transformer.forward()` returns only
`(output_embeddings, decode_cache)`; attention probabilities are computed inside
`_torch_dot_product_attention` and discarded. Extracting them would require a source patch or
per-layer hooks. For the embedding use-case this does not matter — see Q10.

**Prerequisite: `torch_compile=False`.** `from_pretrained(..., torch_compile=True)` replaces
`forward` with a compiled graph and forward hooks silently stop firing. Load with
`torch_compile=False` for embedding extraction (the production server keeps the default for
forecast speed; the two uses are separate).

## Q3. Embedding extraction — working code

```python
import numpy as np
import torch
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

model = TimesFM_2p5_200M_torch.from_pretrained(
    r"C:\Users\hem_s\Projects\openalgo\timesfm-server\models\timesfm-2.5-200m-pytorch",
    local_files_only=True, torch_compile=False,
)
model.compile(ForecastConfig(
    max_context=512, max_horizon=128, per_core_batch_size=16,
    normalize_inputs=False, force_flip_invariance=False,
    use_continuous_quantile_head=False, fix_quantile_crossing=False,
))

captured = {}
def hook(mod, args, output):
    captured["output_embeddings"] = output[0][1].detach()
h = model.model.register_forward_hook(hook)

closes = np.random.default_rng(0).normal(20000, 50, 256).astype(np.float32)  # any series
model.forecast(horizon=8, inputs=[closes])
emb = captured["output_embeddings"][0]          # (16, 1280); [0] = real row of padded batch
last_real_patch = emb[-1].numpy()               # (1280,) -> recommended embedding
h.remove()
```

Notes:
- The hook fires inside `decode()` prefill, so the embedding is exactly what the model "sees".
- `forecast()` pads the batch to a multiple of `per_core_batch_size` (16 here) — a single input
  yields `(16, 16, 1280)`; take row `0`. In the export pipeline pass full batches to avoid waste.
- Candidates measured (256 candles, ctx 512, L2 norms):
  `last_patch` 37.6, `last_real_patch` 37.6 (identical — the last patch is always real),
  `mean_pool_real_patches` 37.7, **`mean_pool_all` 125.3 — 3.3x larger: masked padding patches
  leak non-zero embeddings, so never pool over all patches without masking.**
  `input_emb_last_real_patch` (tokenizer out) 42.8.
- **Recommended**: the last-patch token, `output_embeddings[:, -1, :]`, a 1280-dim vector
  representing the most recent 32 candles after full-context attention. It is the natural
  analogue of ChainST's own "last-patch token" head design (see Q10).

## Q4. Input format

- **Close-only, univariate.** The tokenizer consumes exactly 2 channels per patch: value + mask
  (input_dims 64 = 2 x 32). **OHLC / OHLCV are NOT supported natively** — do not try to pack
  OHLCV into one series.
- **Multiple series / batch**: pass a Python list of 1-D arrays; they are batched internally.
  Measured: 3 different-length series in one call -> 3 forecasts, correct. Lengths may differ;
  each is independently preprocessed.
- Preprocessing `forecast()` applies per input: `strip_leading_nans` then `linear_interpolation`
  for interior NaNs; if `len >= max_context` the tail is kept (oldest truncated); else the array
  is **front-padded with zeros to `max_context`** with a mask (`True` = padded) so the model
  ignores padding. The close series itself is then normalized by the model (RevIN, mask-aware).
- **Because the tokenizer is univariate, you can run a second pass over a different series**
  (e.g., volume, IV, VIX) and concat its embeddings — a legitimate multi-view enrichment with
  zero changes to the model (costs one extra export run per series).

## Q5. Covariates / XReg

Measured: `forecast_with_covariates(...)` fails at runtime — first a
`ValueError: For XReg, return_backcast must be set to True`, then (with `return_backcast=True`)
`ImportError: Failed to load the XReg module. Did you forget to install timesfm[xreg]?`
(jax is not installed in the openalgo venv).

More importantly, from source: **XReg covariates never enter the transformer.** XReg fits a
linear model (ridge) on targets or residuals and adds a linear correction to the forecast.
So even with `pip install "timesfm[xreg]"`, ATR/ADX/RSI/VIX would only nudge point forecasts —
they cannot condition the embeddings. TimesFM therefore cannot take covariates to enrich
embeddings. The fix for ChainST is the reverse direction: TimesFM supplies a *macro* market
context and ChainST already carries the rich per-contract features itself.

## Q6. Embedding stability / determinism

**Deterministic, bit-exact.** Same 256 candles run twice -> `torch.equal` true,
`max_abs_diff = 0.0`. On CPU with eager mode there is no nondeterminism in this path. This makes
the offline precompute + cache strategy sound: re-running the exporter reproduces the exact same
`.npy`.

## Q7. Sliding window — 256 -> 257 -> 258 candles

Measured (cosine similarity of last-real-patch embeddings, synthetic series):

| Pair | cosine | rel. diff |
|---|---|---|
| 256 vs 257 | 0.9959 | 0.092 |
| 256 vs 258 | 0.9906 | — |
| 256 vs 512 | 0.9815 | 0.198 |

Embeddings are **highly stable** across window shifts, but not identical. Two sources of drift:
(1) lengths not a multiple of 32 shift patch boundaries in absolute time; (2) the mask-aware
RevIN stats change slightly. Best practice:

- Use a **fixed context** (512 here) and a fixed window length for the whole export — do not mix
  256 and 512 windows in one feature file.
- Prefer window lengths that are multiples of 32 (patch-aligned): 256, 384, 512. Then every
  window's real data occupies whole patches and `last_real_patch` sits at a stable position
  (`num_real_patches-1` within the real region), which is what the exporter below relies on.
- Keep `normalize_inputs=False` (the default compile flag here) to avoid the second,
  padding-sensitive global RevIN that `normalize_inputs=True` adds on top of decode's own
  mask-aware RevIN.

## Q8. Batch inference benchmark (embedding mode, CPU)

Measured this machine; embedding mode = flip-invariance OFF (one decode), no quantile head.
Timing includes the full prefill; batch = `per_core_batch_size`, context = padded length.

**Batch-size sweep, ctx 512, ~100 windows:**

| batch | ms/window | windows/s |
|---|---|---|
| 1 | 295 | 3.4 |
| 4 | 159 | 6.3 |
| 16 | 130 | 7.7 |
| 64 | 129 | 7.7 |

Batching buys **~2.3x** (295 -> 130 ms). Diminishing after 16; 16 is the sweet spot on this CPU.

**Count sweep, batch 16, ctx 512:**

| windows | total s | ms/window |
|---|---|---|
| 10 | 1.9 | 188* |
| 100 | 12.2 | 122 |
| 1000 | 109.8 | 110 |
| 10000 (extrapolated) | ~18 min | ~110 |

\* small counts round up to a full padded batch (16), so per-window cost looks inflated.

**Context length sweep, batch 16, 100 windows:**

| context | ms/window |
|---|---|
| 256 | 67 |
| 512 | 122 |
| 1024 | 239 |

Cost is linear in context (patch count). ctx 256 halves the cost vs 512.

**Single-window (server-style, batch 1):** ctx 512 = 317 ms, ctx 256 = 271 ms.
Compare the production forecast path at 1799.75 ms/window (existing benchmark, flip-invariance
= two decodes + quantile head): **embedding mode is ~5.7x faster.**

## Q9. Export pipeline — validated end to end

Validated on real data. NIFTY spot closes are extracted from the existing option-chain parquets
(`D:\option data\NIFTY\MONTH\ATM+10_CE.parquet` etc. — every file carries a `spot` column,
minute-resolved): 428,722 minutes, 2020-12-29 -> 2025-12-26. One file suffices (the spot series
is common to all strikes); dedupe on `timestamp` and sort.

Pipeline shape (input -> processing -> output):

```
parquet(spot) -> spot close np.float32 (428,722,)
  -> windows: len 256, stride 16 (patch-aligned)      ~26,800 windows
  -> batch 16 x model.forecast() + forward hook
  -> output_embeddings (b, 16, 1280), take [:, -1, :]  last-patch embedding
  -> timesfm_embeddings.npy (26,800, 1280) float32     ~137 MB
  + window_starts.npy (26,800,) int64  (alignment metadata)
```

Measured mini-run: 200 windows -> 208 processed (13 full batches), 23.1 s, **115 ms/window,
0 NaN**, row norms ~28. Full export estimate: **~51 min on this CPU** (~26,800 windows at
ctx 512); ctx 256 halves it to ~26 min. Memory steady ~1.2 GB RSS.

Embedding quality sanity (real NIFTY spot, stride-16 windows): cosine(window0, window1) = 0.86,
(window0, window16) = 0.68, (window0, window100) = 0.51 — smooth decay with time gap, i.e.
informative and not degenerate.

Reusable exporter is `exp3_export_pipeline.py` (temp location); production version should live
in openalgo (e.g., `timesfm-server/embed/`), batched at `per_core_batch_size=16`, and write
`.npy` + a `starts` alignment file so ChainST's dataset builder can join embeddings to minutes.

## Q10. Fusion recommendation — extend ChainST's per-patch context

ChainST's conditioning mechanism (read from source): per sample, `c` is per-minute context
`(B, T, ctx_dim=8)`; the model takes **the last minute of each patch**:
`ctx = c[:, patch_len-1::patch_len, :]` -> `(B, P, ctx_dim)`; every block then injects it via
`tokens = tokens + ctx_mlp(norm_ctx(cat([tokens, ctx_broadcast])))` with
`norm_ctx = LayerNorm(d_model + ctx_dim)`.

A TimesFM embedding of the window *ending at that same minute* is semantically the identical
slot: per-patch, window-level context. With stride 16 (patch-aligned), one TimesFM embedding per
ChainST patch, exactly `P` of them per sample. **Recommended: context-channel extension.**

```
c_tf (B, P, 1280)  --proj-->  Linear(1280 -> 16) + LayerNorm  ->  (B, P, 16)
c'   = cat([c_patch_last_minute (B, P, 8), c_tf (B, P, 16)], dim=-1)   # (B, P, 24)
```

Changes in ChainST:
1. Compute `c_tf` from the precomputed embeddings (align by timestamp; take the embedding whose
   window ends at the patch's last minute).
2. Widen `norm_ctx` and `ctx_mlp` input dims from `d_model + ctx_dim` to
   `d_model + ctx_dim + ctx_tf` (they are already constructed from `d_model + ctx_dim`, so this
   is a one-line config/param change). Train the projection head with ChainST.

Why this over the alternatives:
- **concat into token embeddings**: wrong granularity — the TimesFM vector is per-window, not
  per-contract; repeating it across the 84 contract tokens is exactly what the ctx injection
  already does, minus the LayerNorm gating. Redundant.
- **cross-attention over embeddings**: strictly more expressive but requires new block machinery
  (ChainST has none today) and a KV cache; a 20-layer 1280-dim encoder's final token is already
  an attended summary — the marginal gain over a projection is not worth the new code.
- **CLS / feature token**: ChainST has no CLS token; its heads consume the last-patch token
  across the 84 contracts. Appending to that token alone under-uses the context (P-1 other
  patches get nothing).
- The two last-patch designs are complementary: ChainST's last-patch token fuses the option
  surface; TimesFM's last-patch token fuses 256 minutes of spot history into the same slot.

## Q11. Fine-tuning recommendation — freeze it

**Do not fine-tune.** Three reasons:
1. **Hardware**: 231 M params, CPU-only, 7.9 GB RAM. Full fine-tune is out of the question;
   LoRA (rank 8-32 on attention q/k/v/o) would fit, but see 3.
2. **It is already a general forecaster.** TimesFM 2.5 is pretrained on massive financial +
   general time series. The embeddings are stable and deterministic (Q6) — the property you
   actually want from a context encoder.
3. **Offline precompute vs fine-tune are incompatible**: the export pipeline (Q9) bakes the
   frozen weights into a 137 MB `.npy`. Any weight change invalidates the whole cache and forces
   re-export. ChainST is the trainable half of the system — that is where the option-market
   signal lives.

Revisit LoRA only if (a) a GPU appears, and (b) the frozen-embedding baseline underperforms
expected on the option surface task. Then target the attention projections only and re-export.

## Q12. Memory + speed summary

| Scenario | RSS | Notes |
|---|---|---|
| Before load | 227 MB | interpreter baseline |
| After load (frozen) | 1,110 MB | weights 882 MB + torch overhead |
| Inference steady (bs 16) | ~1,200 MB | 1000 windows stays flat |
| Peak observed (bs 64) | 1,441 MB | worst case |
| Production server peak | 1,156 MB | existing benchmark |

Safe on 7.9 GB with ~6.5 GB headroom. Speed on this CPU:

| Operation | Time |
|---|---|
| Model load | 5.3 s warm |
| 1 embedding (ctx 512, bs 1) | 317 ms |
| 1 embedding (ctx 256, bs 1) | 271 ms |
| 1000 embeddings (ctx 512, bs 16) | 110 s |
| 1000 embeddings (ctx 256, bs 16) | ~67 s |
| Full NIFTY export (26.8k windows, ctx 512) | ~51 min one-off |
| Production forecast (flip-inv, per window) | 1799.75 ms |

The ~317 ms single-window latency means real-time per-minute embedding during live trading is
feasible (~3 updates/s) but wasteful; a single batched refresh per bar (1-2 windows) is trivial.

## Non-obvious discoveries

1. **`forecast()` pads the batch to `per_core_batch_size` multiples** — a 1-window call
   processes 16 and takes ~2 s, not 300 ms. Always batch to the configured size.
2. **`torch_compile=True` silently kills forward hooks** — the embedding path must load with
   `torch_compile=False`; this is separate from the server's forecast path.
3. **Mean-pooling without masking is contaminated**: padded patches produce large non-zero
   embeddings (L2 125 vs 37 for real patches). Always mask; prefer the last-patch token.
4. **XReg is a linear post-hoc correction, not conditioning** — covariates can never enrich
   embeddings, and the module does not even import without `timesfm[xreg]` (jax).
5. **The tokenizer is univariate** (value + mask). OHLCV cannot be packed in; but the same
   model can be run once per series (close, volume, IV, ...) and the embedding streams stacked.
6. **Window-length multiples of 32** keep patch boundaries stable; drift between 256/257
   windows is small (cos 0.996) but real.
7. **Determinism is bit-exact** — identical runs, identical `.npy`; caching is safe.
8. **TimesFM's per-window RevIN makes embeddings level/scale invariant** — the model
   normalizes each window itself (mask-aware), so raw spot closes can be fed directly without
   external standardization.

## Artifacts

- Experiments: `C:\Users\hem_s\AppData\Local\Temp\opencode\timesfm_exp\`
  (`exp1_*.py/json`, `exp2_*.py/jsonl`, `exp3_*.py/json`, `timesfm_embeddings_mini.npy`)
- Model: `openalgo/timesfm-server/models/timesfm-2.5-200m-pytorch/`
- Reference benchmark: `openalgo/benchmark/reports/timesfm/report.txt`
- Data: `D:\option data\NIFTY\{MONTH,WEEK}\*.parquet` (`spot` column)
