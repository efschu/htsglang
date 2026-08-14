# HANDOFF MERGE-M584 — the gate can be fed now, and it says no

Shift `m584`. Worktree `/spinning/wt-m584`, branch `feat/584-cardrates`, based
on `origin/feat/route-a-631` at `4d5419609a` — the tip both lines carried, and a
clean **fast-forward**, so there is no merge commit and no conflict resolution
to audit. Frozen pre-merge baseline: `/spinning/wt-merge-r12`, detached at
`4d5419609a`, clean tree (`dirty=0`, recorded in the run's own SUMMARY).
Evidence `/spinning/evidence-631/m584/`.

Landed, four commits:

| SHA | |
|---|---|
| `8149d963ec` | `[#485]` port `certify_485.py` onto the line, and stop it certifying superseded criteria |
| `3816240f53` | `[#584]` the card-rate measurement pass, and the store's missing location |
| `b2b61e9ca1` | `[#485]` pin which census calibrates the gate (`RUNSHEET` §6b) |
| `c9ffa2a342` | `[#584]` window verdict |

ERRORS FIRST.

---

## 1. THE ARM LIST WAS SHORT AGAIN — ninth consecutive round

`test/registered/unit/tools/` is named by **no arm** in R12's canonical set and
by no line of `run_631_flip_family.sh`. This shift lands
`test_certify_485_port.py` there, and that file is the **only** thing in the
tree that pins the ported certification judge — the tool the merged runsheet's
§7a and §9 both invoke. Shipped as-is, it would have been a test that never ran.

This is R12 §2's shape one directory over, and R12's own §2 called it "the
eighth consecutive round with a short family list". Fixed **in this batch**, not
deferred: `tools` is added as an **eleventh arm on BOTH sides**, so the
transition is a measured delta rather than a number that appears.

**Attribution, measured not asserted.** `tools` base **11P**, tip **25P**: the
+14 is exactly `test_certify_485_port.py`'s own collected count, confirmed by
running the file alone.

## 2. THE INSTRUMENT LIED ABOUT ONE ARM, AND IT WOULD HAVE MIS-REPORTED IT

R12's runner records each arm's result with `tail -1` of its log. On
`model_executor` the last line is **not** the pytest summary — it is a stray
`forward-peak dump failed: [Errno 2] ... peak.tp0.json`, emitted after the
summary by a teardown. A shift reading `SUMMARY.txt` alone would have recorded
`model_executor` as having no counts at all.

Caught by re-extracting every arm with a regex anchored on pytest's own
`... in N.NNs` summary form rather than on line position
(`/tmp/m584_counts.py`, output preserved in the evidence directory). This is
R12 §3's ANSI trap in a third location: **the log's last line is not the
summary, and the summary's name grep is not the return code.** All three keys
are now known to disagree with each other on at least one arm each.

The family log's ANSI colouring (R12 §3) is handled the same way — the
extractor strips `\x1b[...m` before matching, because a raw grep finds nothing
in that file.

## 3. THE BLACK RATCHET DOES NOT GOVERN THIS BATCH, CHECKED RATHER THAN ASSUMED

R12 §1 is the round where that ratchet first ever fired, so this shift checked
its own files against it instead of assuming. `black --check` (26.1.0, verified
to be the pinned version in the run interpreter) reports four of this batch's
files as unformatted.

**None of them is in the ratchet's scope**, which is the phase-flip surface
plus `test/registered/**/*_631.py` and `*_656.py`. This batch touches
`planner/`, `environ.py`, `scripts/cert_485/` and `*_584.py` / `*_port.py`
tests — no glob matches.

Nor would formatting them match local convention: of seven neighbouring files
sampled, **five** are dirty under the same pinned black, including
`planner/card_library.py`, `rigmon/card_probe.py`, `planner/transient_census.py`
and `scripts/cert_485/excursion_485.py` — the very files this batch extends.
The ratchet's own docstring puts the tree-wide figure at 762 of 6211.

So the files are left as they are, and the decision is **recorded rather than
silent**, which is what the ratchet asks for. A shift that wants them formatted
should widen the scope glob and take the neighbourhood with it, not format four
files into inconsistency with the six around them.

## 4. SUITE — eleven arms, both sides, failure sets diffed by NAME

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner `/spinning/evidence-631/m584/arms.sh`,
which differs from R12's in exactly one documented way: `tools` is an arm, on
both sides.

| suite | BASE `4d5419609a` | TIP `c9ffa2a342` | delta |
|---|---|---|---|
| #631 flip family | **1345P 7S** | **1345P 7S** | — |
| `unit/managers` | 9F 1467P 18S | 9F 1467P 18S | — |
| `unit/mem_cache` | 940F 748P 707S | 940F 748P 707S | — |
| `unit/mem_ledger` | 444P | 444P | — |
| `unit/model_executor` | 15F 594P | 15F 594P | — |
| `unit/planner` | 2F **2389P** 123S | 2F **2405P** 123S | **+16** |
| `unit/server_args` | 627P | 627P | — |
| `unit/tools` (**new arm**) | **11P** | **25P** | **+14** |
| `unit/turnkey` | 116P | 116P | — |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | — |
| `unit/docker` | 4P | 4P | — |

**+30 tests, 0 new failures, 0 reds found.**

**Continuity.** The BASE column reproduces R12's post-merge column **exactly** on
every arm R12 recorded: family 1345P 7S, `managers` 9F 1467P 18S, `mem_cache`
940F 748P 707S, `mem_ledger` 444P, `model_executor` 15F 594P, `planner` 2F
2389P 123S, `server_args` 627P, `turnkey` 116P, `utils` 46F 348P 4S, `docker`
4P. The chain back through R12, R11, R10, R9, R8, R7 and R6's frozen bases is
unbroken. The eleventh arm's base, `tools` 11P, is established here for the
first time.

**Failure SETS diffed by name:** **all eleven identical, zero new failures.**

| arm | names at base | at tip | diff |
|---|---|---|---|
| `mem_cache` | 940 | 940 | **identical** |
| `utils` | 46 | 46 | **identical** |
| `model_executor` | 15 | 15 | **identical** |
| `managers` | 10 | 10 | **identical** |
| `planner` | 2 | 2 | **identical** |
| `mem_ledger` / `server_args` / `tools` / `turnkey` / `docker` | 0 | 0 | identical |

**Predicted vs delivered.** The two new test files were predicted to land in
`unit/planner` and `unit/tools`, and the deltas are **exactly** their own
collected counts — 16 and 14, each confirmed by running the file alone. Nothing
else moved by a single test, which is what a bridge-plus-location change should
look like: `server_args.py` gained a refusal branch and its arm is unchanged at
627P, because that branch's coverage lives in `unit/planner` (the R12 §2 shape,
already fixed there).

**The SUBFAILED second pass re-measured, not assumed.** Prefix census on this base: `managers` 10 = **8 FAILED + 1 ERROR
+ 1 SUBFAILED**; `utils` 46 = **44 FAILED + 2 SUBFAILED**; `mem_cache` 940 =
902 FAILED + 38 SUBFAILED; `planner`'s 2 and `model_executor`'s 15 are all
plain `FAILED`. A `^FAILED`-only grep would find 8 of 10 and 44 of 46 — the
same two arms and the same two counts R10, R11 and R12 each recorded
independently.

## 5. REGISTER — untouched by this batch, and verified so

| check | result |
|---|---|
| `git diff --numstat` on `docs/dev/631/CONTRADICTIONS_REGISTER.md`, base→tip | **empty** — not in the batch at all |
| file length | **2856**, unchanged from R12's tip |
| row count | **39** rows, `## N.` form |
| duplicate row labels | **none** |
| the duplicate grep PROVEN able to fail | appending a duplicate `## 52.` to a copy makes it report `## 52.` |

The union check is trivially satisfied because the batch adds no register rows.
It was still run, and the detector was still proven able to fail first, per
R11 §1.

## 6. WHAT THE NEXT SHIFT OWES

1. **The other half of #584.** Card rates are done. `#363`'s flip targets are
   still **0**, and this shift measured exactly why: `build_stage_table` refuses
   a solved-but-unmeasured candidate, and with per-stage measurements present it
   builds 2 stages / 1 flip target. The missing quantities are
   `measured_gain_pct`, `measured_band_pct` and `flip_cost_s` — an A-vs-A run on
   each stage's own phase plus one instrumented flip. That is now the only thing
   between `#363` and an actuation.
2. **Explain the 1258 MiB excess.** `RUNSHEET` §6b is written so that explaining
   it is what narrows C2′'s reference class back to the cut's own class. Until
   then every cut is charged the pooled worst, and nothing certifies. This is
   the highest-value open item on `#485`, and §6b names it as its own falsifier.
3. **Do not re-run P2 hoping for a different census.** §6b pins the governing
   rule precisely so that a shift cannot solve against whichever census admits a
   cut. If the seam work lands, the gate will admit `29,19,16` — a third cut,
   owed its own certification and its own throughput measurement (§6a).
4. **`--rank-gpu-memory-mib` is spent on this rig.** Measured, not argued: the
   solve refuses at every card's full nameplate total. Nobody should spend a
   window on it.
