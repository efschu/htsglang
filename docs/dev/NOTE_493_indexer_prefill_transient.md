# Task #493 — the corridor breach is a transient, and the cap for it was inert

Desk task. No GPU window taken (window 4 was running); everything below is code
read at the tree, arithmetic derived from the allocation sites, and tests run
under `CUDA_VISIBLE_DEVICES=99`. The one measurement that is left is packaged as
`scripts/dev/493_indexer_transient/`.

## 1 — What window 3 actually showed

Evidence: `/spinning/gpu-battery-results/2026-08-03_w3_t1_474_wreemit/`
(`corridor.csv`, 2737 samples at 1 Hz; `boot474.sh`; `RESULTS.md`).

| card | steady free (mode) | min free | samples < 400 MiB |
|---|---|---|---|
| gpu0 (3080) | 873 MiB | **271 MiB** | 28 |
| gpu1 (5090) | 1680 MiB | 1118 MiB | 0 |
| gpu2 (3080) | 873 MiB | **271 MiB** | 186 |

Three readings of that table are load-bearing and none of them were in the
window's own report:

* **The steady state is not the problem.** The mode is 873 MiB on both 3080s
  through the entire deep prefill; the floor is an excursion of 602 MiB off that
  mode. `873 - 602 = 271`, exactly the floor.
* **Both 3080s were repaired, and it did not help.** The two boot logs settle
  it, not the scripts:

  ```
  2026-08-02_394_linkshards/boot394_equal.log:  reserve per GPU: {0: 2200, 1: 1400, 2: 1400}
  2026-08-03_w3_t1_474_wreemit/boot474.log:     reserve per GPU: {0: 2700, 1: 1900, 2: 1900}
  ```

  +500 MiB on *all three* ranks. RESULTS.md records gpu2 as "never repaired,
  its reserve stayed at 1900" — 1900 *is* the repaired value; the baseline was
  1400. The correct conclusion is stronger than the reported one: the repair
  reached every violating card and the floor still did not move. (The rule in
  runbook §4.5.4 item 4 is worth keeping either way — under the reported
  reading it is the direct lesson, and under the true reading it is what makes
  the negative result trustworthy.)
* **The 1 Hz trace undersamples.** The excursion recurs once per prefill chunk
  (~12 s apart at 128 chunks over 1507 s) and lasts well under a second, so most
  samples land on the plateau and the few that catch it land at random points on
  the rise and fall. 602 MiB is therefore a **lower bound** on the peak, and the
  apparent growth of the dip over t=1300..1900 s is extreme-value statistics of
  undersampling, not a ramp — the mode never moves.

## 2 — The transient, sized from its allocation sites

`fp8_paged_mqa_logits_torch_sm120`
(`python/sglang/srt/layers/attention/dsv4/indexer.py:351`) is the production
paged-MQA-logits path on every card of this rig since #417 Cut 3. Its loop body
holds, per query row and per KV position of the chunk
(`_indexer_logits_step_bytes`, `indexer.py:277`):

```
per_position = (head_dim + 4)      gathered page block (fp8 + fp32 scale)
             + head_dim * 4        its fp32 dequantization
             + 4                   the reshaped fp32 scales
             + num_heads * 4 * 2   the [rows, chunk, heads] bmm product
                                   plus the `score * weight_row` temporary
```

and on top of the loop, allocated before it and live across it, the returned
`[rows, max_seq_len]` fp32 logits (`indexer.py:432`, counted by
`_indexer_logits_output_bytes`, added in this change) — the one term **neither**
chunking axis bounds, because it is the return value.

At the window-3 geometry (DeepSeek-V4-Flash C4 indexer: `index_n_heads=64`,
`index_head_dim=128`, heads replicated so this is rank-invariant;
`--chunked-prefill-size 256`; `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK=2048`; C4
span 8196, the compress_ratio-4 image of the 32768-token prompt):

```
per_position = 132 + 512 + 4 + 512            = 1160 bytes
per row      = 2048 * 1160                    = 2.2656 MiB
loop         = 256 rows * 2.2656 MiB          = 580.0 MiB
logits       = 256 * 8196 * 4                 =   8.0 MiB
                                              -------------
modelled peak                                    588.0 MiB
```

against a measured excursion of **602 MiB** — and the measurement is a lower
bound, so the model sitting 2.4 % under it is the agreement. The residual is the
top-k stage's own `[rows, span]` buffers plus allocator block rounding.

## 3 — Why the reserve knob could never have fixed it

`--rank-auto-reserve-mib` is subtracted from the NVML total to form the rank
BUDGET; the KV pool then takes whatever the reserve leaves. So raising it buys
steady-state free memory **by giving up KV capacity** — which is exactly what the
window measured: `max_total_num_tokens` fell 90624 → 41984, a 54 % cut, and the
floor stayed at 271 MiB against the 273 MiB it was trying to fix.

A runtime allocation lands on top of both the budget and the pool. It is capped
where it is made, or not at all. `ServerArgs.pinned_reserve_shortfall_note`
already listed five terms no reserve charges (CUDA context, NCCL buffers, the
flashinfer workspace, graph capture, GDN prefill scratch); this is the sixth, and
on this recipe it is the largest.

## 4 — The root cause: the cap existed and did not bind

#449 built exactly the right mechanism — a per-rank MiB budget on the query axis,
under the #395 budget discipline (`_indexer_logits_chunk_rows`, `indexer.py:300`)
— and shipped it at `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = 2048`.
`NOTE_449_dsv4_indexer_query_chunk.md` section 5 names the value for what it was:
*"the default is a ceiling picked at desk, not a tuned value."*

It is above the peak it was meant to bound:

| `SEQ_CHUNK` | per row | 256 rows cost | 2048 MiB permits | binds? |
|---|---|---|---|---|
| 2048 (the recipe) | 2.27 MiB | 580 MiB | 903 rows | **no** |
| 8192 (the shipped default) | 9.06 MiB | 2320 MiB | 225 rows | trims 12 % |

Two further facts complete the picture:

* Window 3 ran `PYTHONPATH=/spinning/wt-441-night/python`, a tree that predates
  #449 entirely — it has no `_indexer_logits_chunk_rows` and no
  `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`. So the trace is the pre-#449 shape.
* Had it run the current integration branch, **nothing would have changed**: at
  2048 MiB the budget returns all 256 rows. That is the finding — a mechanism the
  catalog lists as present, whose reach at its shipped default is zero on the
  geometry this fork serves. (MECHANISM REACH, `CLAUDE.md`: the gate predicate is
  `rows = budget_bytes // step_bytes`, `indexer.py:347`, and it is what decides,
  not the catalog line.)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the first candidate in the
briefing — was **already set** by `boot474.sh:37` for the breaching run. It is
falsified by the evidence, not by argument, and was not pursued.

## 5 — What was changed

**The cap now binds** — `python/sglang/srt/environ.py`,
`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` 2048 → **256 MiB**: the largest
power-of-two budget that still binds on the reference geometry at both `SEQ_CHUNK`
settings. At the window-3 geometry the peak falls 588 → 261.8 MiB and the free
memory at peak rises 285 → 611 MiB, clearing the 400 MiB corridor floor. This
costs no capacity: the query-axis regrouping is exact (no reduction crosses a
chunk boundary, pinned at atol=0/rtol=0 by the #449 suite), it only turns one
loop step into three.

**The transient is nameable at boot** — `indexer_prefill_scratch_bytes`
(`indexer.py`, new) composes the two terms using the same functions the loop
uses, so the estimate cannot drift from the allocation; `ServerArgs.
dsv4_indexer_prefill_scratch_mib` (new, mirroring `gdn_prefill_scratch_mib`)
reads the checkpoint's `index_n_heads`/`index_head_dim` and returns `None`
off-family; `pinned_reserve_shortfall_note` itemizes it as its own post and its
docstring now states the reserve semantics with the window-3 numbers.

**Not done, deliberately:** no additional charge in
`derived_rank_auto_reserve_mib`. Charging the transient there would shrink the KV
pool by its size — the refuted move. Capping it costs nothing, so the arithmetic
runs the other way: the transient drops by 326 MiB and the pool keeps every
token.

## 6 — Tests

`test/registered/unit/layers/attention/test_dsv4_indexer_transient_493.py`,
**14 tests**, CPU-only, `CUDA_VISIBLE_DEVICES=99`, ~12 s. Reuses the #449
fixture and its two probes by path rather than restating them.

| class | pins |
|---|---|
| `TestTheShippedDefaultBinds` | the default binds at both `SEQ_CHUNK` settings; the pre-#493 default bound nothing; small/golden shapes stay single-pass |
| `TestTheModelMatchesTheMeasuredBreach` | the pre-#449 model reproduces the measured excursion within 5 %; the old default breaches the measured corridor allowance and the new one does not; the output term is counted and is not chunkable; the predicted A/B delta is stated as a number |
| `TestTheModelAndTheAllocationShareOneFormula` | EXECUTED — the real call's gathered block and bmm result fit inside the modelled step, with a can-fail arm proving the bound is the knob's doing |
| `TestTheReserveDiagnosticNamesIt` | the launcher's estimate equals the loop's bound; `None` off-family and without a context length; it follows the knob it names |

**Falsifier, executed:** with `environ.py` reverted to `EnvInt(2048)` and the test
file kept, **4 of 14 fail**, each on its own terms, no import or attribute errors:

```
self.assertLess(rows, REF_ROWS)                                    # 256 !< 256
self.assertLessEqual(fixed, CORRIDOR_ALLOWANCE_MIB)                # 588.0 !<= 473
self.assertAlmostEqual(delta, 326.0, delta=2.0)                    # delta was 0.0
self.assertAlmostEqual(got, expected, delta=1.0)                   # launcher vs loop
```

The third line is the whole task in one assertion: with the shipped default the
A/B delta is **zero**, because the cap did not bind.

**Suite state**, `test/registered/unit/layers/attention/`: **152 passed, 9
skipped, 480 subtests**, with the new default in place — the #449 exactness and
invariance suites are unaffected by the lower budget.
`test/registered/unit/server_args*` + `test/registered/unit/server_args/`: **526
passed, 1 failed**, the failure being `test_server_args_mutation_ratchet` with a
count of 10 against a baseline of 0 — **pre-existing**, reproduced identically on
the unmodified base tree at `/spinning/wt-merge-ops`.

**Lint:** `black` and `isort` (the repo's formatters, per `.pre-commit-config.yaml`)
and `ruff check --select=F401,F821,UP037` (the repo's ruff invocation) are clean
on all six touched files; `codespell` clean on those plus the three documents.
A note for the next agent: `ruff format` is **not** this repo's formatter and
reformats ~676 files under `python/sglang/srt/` if run — use `black`.

## 7 — The GPU arm, packaged not run

`scripts/dev/493_indexer_transient/` — `predict.py` (production formula, prints
both arms and the predicted delta), `sample_corridor.sh` (100 ms via
`nvidia-smi -lms`), `verdict.py` (reads `forward_peak` JSON + the corridor CSVs,
two-part gate, exit 1 on failure), `README.md` (the procedure).

Both python scripts were executed here — `predict.py` against the real formula,
`verdict.py` against synthetic arms including a can-fail arm that returns exit 1.

The falsifier for the whole task: **`peak_bytes_max` must fall by ~326 MiB per
rank between the budget-off and budget-on arms.** If it does not, the corridor
breach is not this transient and this note is wrong.
