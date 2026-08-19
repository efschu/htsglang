# #735 Step-1 in the FLIP WORLD — it boots, flips, and keeps the prefill gain

GPU window 2026-08-19 06:02-06:16Z, rig CT999, tip `4ce6ec2fc5`. Config: the
Step-1 argv (PP cut `--pp-stage-ratio 31,17,16`, `--pp-attn-stage-ratio 7,5,4`)
with the phase-flip controller and NEXTN spec it already carried, TP-phase
vector unchanged at the ship `32,16,16`, checkpoint swapped to
`Qwen3.8-27B-INT8-vocabint8-embed` (valid since #763, and it banks the 1211 MiB).

## Two carried values had to be dropped, and one of them invalidates a prior claim

**The token vector was carried.** `restore_r2_env.txt:81` sets
`SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8` — the vector solved for the INCUMBENT cut —
and the runner feeds that file to every boot. Per the hard law (uneven DCP is
token-sharded; the vector is co-solved with any layout change) that is the
silent-misfit class, so it was cleared for this boot.

It also means **the earlier +10.1% Step-1 measurement (`step1ctg`/`step1ctgB`)
ran the carried 14,10,8 against a 31,17,16 cut.** That number was taken on a
config that was not co-solved. The gain reproduces here anyway (below), but the
earlier figure should not be quoted as if its config was clean.

Clearing is the documented derivation path, not a hack:
`parse_flip_token_vector` (`phase_flip_boot.py:117-119`) reads
`if not raw: return flip_vec`, and its docstring states the token split must
follow "each rank's REMAINING memory once its weights are placed", NOT the
compute/layer cut — so the naive co-solve `31,17,16` would have been wrong too.
The server computes and logs a token-proportional vector after profiling; on
this boot it logged none, so the resolved value is the flip vector `32,16,16`
(the documented default), and that is what ran.

**The pool number was carried too.** `--max-total-tokens 436275` is the PP-arm
Step-1 figure and was removed so the planner could size the flip world itself.

## Boot A: the refusal, with numbers

Unpinned, the planner sized the pool to **880936 / 915692 / 1007656 tokens** —
far above the 550000 the flip world is built around — and the boot then died
with `torch.OutOfMemoryError: Tried to allocate 810.00 MiB` on ALL THREE ranks,
with 270 / 396 / 412 MiB free. The pool had absorbed the headroom the later
allocation needed.

Two structural numbers came out of that boot and stand on their own:

- **`ARMING FLOOR 1728 MiB` per rank** — corridor law plus this rank's measured
  one-leg seam draw plus load margin. Higher than the configured 1536.
- **Ranks 1 and 2 receive NOTHING at the flip.** `received-layer derivation --
  attention map [7, 5, 4] over 16 total, tp vector [32.0, 16.0, 16.0] -> this
  rank RECEIVES 0 layer(s) at the flip`, and both then went `COLD and the seam
  slope could not be derived (received=0)`, sizing with no per-token seam term.
  Only rank 0 receives a layer (1 layer, 2048 B/token, `R' = 13764 MiB`).
  The Step-1 FA map and the ship TP vector interact so that the whole seam
  lands on one rank. This did not stop the boot below from flipping, but it
  means two of three ranks are pricing the seam blind.

## Boot B: acceptance

Same config, pool bounded at 550000 (the flip world's own number, which the
sizing code names as the value that "arms and flips" on this rig).

| # | acceptance | result |
|---|---|---|
| a | boots, health 200 | **yes, 174 s**; guard markers 0 |
| b | >=3 PP->TP->PP cutovers commit cleanly | **7 cutovers**, strictly alternating, **0 aborts** |
| c | prefill gain holds in the flip world | **5699 tok/s** median vs incumbent 5113.5 = **+11.4%** |
| d | TP-decode unchanged | **76.0 tok/s** at bs=1, accept len 2.48-2.88 |
| e | coherence (content + reasoning, or /generate) | banana / 42 / The Pacific Ocean, all `stop`; `/generate` returns real prose |

Cutover epochs, one line per rank: `pp_to_tp`(1), `tp_to_pp`(2), `pp_to_tp`(3),
`tp_to_pp`(4), `pp_to_tp`(5), `tp_to_pp`(6), `pp_to_tp`(7) — three full round
trips, 2.43-2.75 s each, driven by a session-load probe (multi-turn sessions
alternating a prefill-heavy turn with a decode-heavy turn) rather than a manual
fill battery.

The int8 vocab loads correctly under PP: `weight_scale not found` appears twice,
on PP1 and PP2 only — the ranks that do not own the embedding — and NOT on PP0,
which owns it. That is the same benign pattern the healthy known-good shows.

Corridor after boot: 3059 / 1560 / 3055 MiB free. Rank 1 (the 5090) sits at
1560 MiB, above the ~1024 MiB target but BELOW the 1728 MiB arming floor; flips
armed and committed anyway, so the floor is charged against the pool rather
than against NVML free, but this is the number to watch if the cut is pushed.

## One watch item, not a failure

`PHASE-POLICY ARM-VERDICT-WRONG` fired once: it armed `pp_to_tp` on idle
(3.0 s >= 3 s, returning to the decode resting layout), the cutover COMMITTED,
and it still built no batch in 8 rounds, on inputs `running_bs=0 pending=0
nothing_can_run=False target_can_admit=False ready_carriers=0`. The log names
the risk itself: if that repeats in ALTERNATING directions it is the
2026-08-16 ping-pong. It fired once here, not alternating.

## Status

Evidence only. The standing ship config was NOT switched — that is a user
decision. What is shown: the Step-1 cut survives the flip world on the
requantized checkpoint, keeps its prefill advantage, and flips repeatedly and
cleanly; the open questions are the one-rank seam concentration and the 550000
pool being a carried constant rather than a solved one.

## #767 — the early release fired against an in-flight copy (fixed), and the poison outlived the boot

Root: `_mamba_early_release_admissible` accepted `mamba_backuped`, which reads
`mamba_host_value is not None`. The write-through path publishes that value in
the same block that HANDS the transfer to the cache controller and records the
node in `ongoing_write_through` / `_write_through_inflight`
(`hi_mamba_radix_cache.py:412-432`). Between queue and ack the anchor is an
intention, not a copy, so #755 released the pin against it and the node became
evictable before its bytes landed. The predicate now also requires the copy to
have LANDED (`_mamba_host_copy_complete`, overridden on the hierarchical pool).

Measured with salted greedy "capital of France" probes, so no probe could be
served from a cache key:

| arm | degenerate |
|---|---|
| full boot, reorder ON, pre-fix | **9 / 10** |
| bisect: `SGLANG_MAMBA_SLOT_REORDER=0`, pre-fix | 1 / 10 |
| full boot, reorder ON, post-fix | **0 / 10**, then 1 / 10 on a later run |

TWO SILENT SYMPTOMS CONFIRMED IT: zero #755 refusals were ever logged (every
node claimed to be backed) while there were zero completed-backup markers, and a
deliberate prefix repeat reported `cached=None`. A contract that never refuses
and never completes is not being met.

**THE POISON IS DURABLE, AND THE FIX ALONE DOES NOT CLEAR IT.** With the fix in
place the SALTED probes were clean while the CACHED prompt stayed degenerate on
all 4 attempts -- deterministically, which is what a stored bad entry looks
like. `hicache_storage_backend=file` persists to `/tmp/hicache`: 3.5 GB, 19642
files, 14187 of them written during the defective boots. Pointing the next boot
at a fresh store (`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR`) returned the cached
path to `' Paris.'` on 5 of 5. So any rig that ran the defective build must have
its disk tier purged; correctness does not come back with the patch alone.

Residual: about 1 in 10 salted probes still degenerates. Smaller than the
original defect and NOT closed -- next item, alongside #768.
