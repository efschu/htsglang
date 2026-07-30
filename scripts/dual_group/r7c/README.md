# #274 round 7c boot queue

Four boots, none of them run yet. Budget is 6, so there is room for one retry
plus one boot the results ask for.

Each script is self-contained: it resolves the card order at runtime (CUDA and
NVML order differ on this rig and the mapping can shift), refuses to start if
any card is above 500 MiB, claims and releases the arbitration `holder`, and
samples free VRAM per card throughout. Run one at a time.

```bash
cd /spinning/wt-lane-r7c/scripts/dual_group/r7c
./boot_a_fp8_reference.sh        # posten 0, the one-axis falsifier
./boot_b_dense_head.sh           # the head-precision axis, closed at BF16
./boot_c_dflash_solo_q8.sh       # posten 1, quantised drafter on a 3080
./boot_d_lane_reseed.sh          # posten 2, re-seed A/B
```

## Order, and why

**A first.** It is the only boot whose outcome changes what the others mean.
If the reference reproduces, the GGUF accept ceiling is a target-quantisation
property and every lane-spec number on that vehicle has to be read against a
ceiling rather than against the reference. If it does not reproduce, the
reference was a property of the old measuring path and B and C are measuring
against nothing.

**B second**, because it is the other half of the same question and shares A's
apparatus: A moves the target quantisation with the head following it, B holds
the target coarse and lifts only the head. Read together they separate "the
head's precision is the lever" from "the target's is". Run alone, neither does.

**C third.** Independent of A and B; it is placed third because it is the one
most likely to fail in a way that costs a retry, and a retry is cheaper once
the accept questions are settled.

**D last** and cheapest. It is the round-7b configuration exactly, so it is the
one boot known to come up.

## Summary table

| Boot | Vehicle | What moves | Window | Expected |
|---|---|---|---|---|
| A | Qwen3.6-27B-**FP8** (base) | target quant | ~35 min | accept 2.6-3.3, position 0 ~65% |
| B | **Huihui-AWQ-MTP** (INT4 body, BF16 head) | head precision only | ~40 min | decides head-vs-target |
| C | GGUF-Q3 + **DFLASH-Q8_0 solo** on a 3080 | drafter architecture + placement | ~45 min | drafter loads and generates |
| D | GGUF-Q3 + NEXTN + lane | re-seed on/off | ~30 min | little or nothing; the price is the point |

Total ~2h30 of card time, plus load contention if anything is retried.

## Reference column

Every accept number in these boots is reported against **2.688 (prose) /
3.279 (code)** — `docs/benchmarks/htsglang_tp3.json:87-90`, the same FP8
vehicle at pure NEXTN and K=3. Not against 2.75-2.82: that pair is two cells of
a five-axis cross-algo battery and is not a comparison for a K=3 measurement.
The GGUF-Q3 serving group's own band, 1.15-1.53 (round 7b), is the other end of
the scale.

All four boots run at **K = 3** with the same five contents (alphabet, squares,
repeat, code, prose) and report the **per-position curve**, not just the mean.

## What each boot answers, in one line

- **A** — does the accept reference exist on this rig, on one axis?
- **B** — is it the head's precision or the target's?
- **C** — can a quantised DFLASH drafter live on a second card, and is it worth it?
- **D** — what does aligning the lane's chain with the serving group's cost and buy?

## Queue item 4: free MiB per card

Every boot samples `nvidia-smi` every 5 s for its whole life and prints the
minimum free MiB per card at the end (`vram_summary.txt` in each output
directory). That is the input every multi-card placement decision has been
missing, and it costs nothing in a boot that is happening anyway.

Sampled throughout rather than at the end on purpose: the peak is what decides
whether a drafter fits, and the steady state after warmup hides the prefill
scratch that the 2700 MiB reserve exists for.

Boots A, B and D take it on the standard reserve, so they measure the headroom
as the rig normally runs. Boot C measures it with the reserve deliberately
raised on the hosting card, which is the configuration a drafter would actually
live in.

## Known risks, stated before the run

- **Boot B has never been booted in this shape.** AWQ x uneven TP x MTP is
  untested on this branch, and the round-7b GPTQ arm died on a unit-count
  mismatch. If it rejects the shape, that is one consumed boot and the answer
  is "not on this vehicle" — do not tune the ratio inside the window.
- **Boot C's reserve arithmetic is derived, not measured.** `RESERVE_HOST`
  defaults to 5000 MiB on the hosting card (1753 weights + headroom). If the
  load OOMs, raise it and shorten `CTX` rather than retrying unchanged.
- **Boot C cannot also carry the lane.** The lane's NEXTN head nests into the
  head the serving group runs; a DFLASH serving group builds none. That is why
  the re-seed A/B is boot D and not an arm of C.
- **The 5090 arm of boot C is one variable away**: `DRAFT_GPU=$CUDA_BIG`. Worth
  it only if the 3080 arm shows the drafter working — placement is the second
  question, not the first.

## Dry run, before the window opens

`R7C_DRY_RUN=1` walks any recipe from the top to the launch line without a GPU,
without touching the arbitration files and without starting a server. It prints
the fully assembled launch command instead, so everything the recipe derived at
runtime — boot C's drafter card and its per-rank reserve string above all — is
readable before a boot window is spent on it.

```
R7C_DRY_RUN=1 R7C_CARDS_FILE=/path/to/recorded-cards.txt \
  bash scripts/dual_group/r7c/boot_c_dflash_solo_q8.sh
```

`R7C_CARDS_FILE` feeds a recorded `resolve_cards` output in place of the live
query; without it the dry run still asks the real cards. Missing model files
are a warning in a dry run and an abort in a real one.

This exists because round 7c's boot C spent a window to fail at
`CUDA_SMALL: unbound variable` — the call site piped `load_card_order` into
`tee`, so the resolver ran in a pipeline subshell and its assignments never
reached the recipe, while the echoed lines still made the log look correct.
`test/registered/unit/test_r7c_recipe_dry_run.py` now runs all four recipes
through the dry run under `bash -u`, against two different card orders.
