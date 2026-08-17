# #349 -- Is the standing bug net still armed at HEAD? DETERMINATION

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots.

**Verdict: the net is ARMED but its REACH is 16 days behind the product.**

The harness is healthy -- it imports, runs, and its unit tests are green. What
has moved is the world it points at. The net covers the cross it was built for
and is structurally blind to the phase-flip world that arrived after it, which
is where this week's crashes lived.

That is a different failure than the one this determination was sent to look
for. The feared shape was #380-class rot: arms quietly matching nothing while
reporting green. The audit did not find a disarmed net. It found a net aimed
at last month's target.

## (a) When did it last actually run, and against what?

**2026-08-01**, sweep 3, on CT999 -- 17 arms, Qwen3.6-27B-FP8, TP=3 uneven
(5090 + 2x 3080). Recorded in `TASK_349_SWEEP_3_RESULTS.txt` and its
follow-up; artifacts under
`/spinning/gpu-battery-results/2026-08-01_349_boot_matrix` (17 arm dirs).

Two facts follow from the dates, and both matter:

**The arm roster grew after the last run.** Arms were edited on 2026-08-06
(`75ddf9fe1c`, `c7ee3cb3eb`, #550/#552) and 2026-08-07 (`b1c66b3708`, #614).
Today's roster is 20. Three arms **have never run once**:

| arm | added | for |
|---|---|---|
| `N_resume_under_spec` | 2026-08-06 | resume x spec |
| `O_hicache_contention` | 2026-08-06 | kvso spill copies x HiCache on one link (#550) |
| `P_hicache_nospill_control` | 2026-08-06 | the declared control for O |

An unrun arm is not a bug net entry yet -- it is an untested assertion about
a boot. `O`'s verdict is a DELTA against `P`, so neither is interpretable
alone, and the pair has never been exercised together on a card.

**The tree identity of that run was never recorded.** This determination could
only establish which tree the last sweep ran against by reading `git log` and
directory mtimes. `summary.json` held verdicts and nothing else -- no head, no
branch, no dirty bit, no roster. For a STANDING net, whose only question is "is
this still true at today's HEAD", a result that cannot name its own tree cannot
answer it. **Fixed here** (see "What was repaired").

The last run also used **Qwen3.6-27B-FP8**, which the production model has
since moved off. Not itself rot -- but the net's last green is a green about a
model no longer served.

## (b) Which arms are stale at today's HEAD?

See the per-arm audit table below (`ARM AUDIT`). Summary of the sort:

- VALID -- flags, envs and markers all resolve at HEAD
- STALE-FIXABLE -- something no longer resolves, intent still testable
- OBSOLETE-BY-DESIGN -- the world the arm boots no longer exists by design

## (c) Do the coherence gates cover the new failure surface?

**No. The net has no phase-flip dimension at all, and the blindness is
structural rather than incidental.**

Three findings, each checkable:

**1. No arm can enter the flip world.** HEAD carries 33 distinct `SGLANG_*` switches
across the flip/seam/regime surface (`SGLANG_PHASE_FLIP_INSTANCE`,
`SGLANG_PHASE_FLIP_SEAM_RESERVE`, `SGLANG_PHASE_FLIP_SEAM_RESERVE_MIB`,
`SGLANG_FLIP_SEAM_WAVES`, `SGLANG_REGIME_OBSERVE`, ...) and 18 modules under
`managers/` implementing it (`phase_flip_*.py`, `regime_*.py`,
`kvso_flip_contract.py`). **Not one appears in `arms.py`.** The eight env vars
any arm sets are `KVSO_ALLOW_SPEC`, `KVSO_ALLOW_HICACHE`, `SGLANG_BARLINK`,
`SGLANG_BARLINK_GRAPH_ENABLE`, `SGLANG_BARLINK_TRANSPORT`,
`SGLANG_MAMBA_SSM_DTYPE`, `SGLANG_UNEVEN_DCP`, `SGLANG_UNEVEN_DCP_WEIGHTED` --
all pre-flip-era.

**2. The consequence is a guard that cannot fire under the matrix.** This is
the sharp end. `kvso_flip_contract.restore_permitted` (`:247`) decides whether
a spilled KV image may be copied back into device memory, and its first branch
is (`:268-269`):

```python
if not current_phase:
    return True
```

documented as "NO PHASE MEANS NO PHASE FLIP, WHICH MEANS ALWAYS PERMITTED" --
correct, and deliberately so, for a process that never enabled the flip. But
every boot-matrix arm is such a process. So across the whole net this function
returns `True` unconditionally, and the wrong-phase-restore guard -- a
CORRECTNESS guard, whose failure mode is right-looking output computed from KV
captured in the wrong layout -- is unreachable. The net cannot catch it, and
cannot report that it did not try.

This is the requested file:line for the absence claim: the refusing gate is
`managers/kvso_flip_contract.py:268-269`, and it refuses by short-circuit.

**3. The grader has no phase dimension.** `coherence.grade_probes` (`:139`)
takes a flat sequence of probe dicts (`name`, `tier`, `text`, `ref_text`,
`min_score`). There is no before/after pairing and no place to put one. A flip
that degrades output *after* the layout change is invisible by construction:
one text, one reference, one verdict. The gate's own design note is about GDN
reproducibility, written before flips existed, and nothing in it is wrong --
it simply predates the question.

**Named gaps, not built** (boot-gated, with deps):

| gap | needs | dep |
|---|---|---|
| G1 flip-arm: boot with `SGLANG_PHASE_FLIP_INSTANCE`, assert the flip occurs | a flip-capable arm + `EffectiveConfig` fields for phase | #706 run-card |
| G2 HiCache x flip: spill in phase A, attempt restore in phase B, assert refusal | `restore_permitted` reachable, i.e. G1 | #718/#719 semantics |
| G3 seam funding: assert the seam reserve is actually reserved at flip time | `phase_flip_seam_reserve` observables in the log | #656-era seam work |
| G4 coherence across a flip: probe before AND after, compare | a phase dimension in `grade_probes` -- a schema change, not a marker fix | G1 |

G4 is the only one that changes the gate's shape; G1-G3 are arm work. None is
desk-fundable: every one of them needs a real flip on a card, which is a boot.

## (d) Does the runner still start hermetically?

**Yes, cleanly.** Verified on today's tree with `CUDA_VISIBLE_DEVICES=""`:

- all six modules import with no traceback (`arms`, `check`, `coherence`,
  `effective`, `sweep`, package `__init__`)
- the card-less path exists and works exactly as documented:
  `python -m sglang.srt.boot_matrix.sweep --list` (exit 0, prints the 20-arm
  plan, ~97 min estimated card time) and `--dry-run <ARM>` (exit 0, composes
  env + argv, touches no card). Confirmed by inspection that only `--run`
  reaches `subprocess.Popen`.
- registered unit tests: **140 passed, 0 failed** before this change; **147
  passed** after.
- artifact path machinery intact; `--out` defaults to `/tmp/boot_matrix` and is
  created on demand.

The 2026-08-01 sweep's known false-FAIL -- the fatal detector matching the
benign torchcodec `Traceback` that sglang itself frames as "Ignore import error
when loading" -- **was repaired**; the exemption is live at `check.py:104`.

## What was repaired here

**Sweep provenance** (`sweep.py`: `collect_provenance`, `write_provenance`,
called from `_main`). Every run now writes `provenance.json` into its out-dir
carrying head, branch, dirty flag, arm count and the full arm roster.

Three deliberate choices:

- **Written BEFORE the first boot**, not after the last verdict. A sweep that
  is killed halfway still leaves artifacts someone will try to read, and those
  are precisely the ones whose tree is hardest to reconstruct later.
- **`git_dirty` is recorded**, because a green taken on a dirty tree is not a
  green at that commit and the difference is invisible afterwards. An unknown
  tree reports dirty rather than claiming a cleanliness it did not verify.
- **Never raises.** Missing git binary or an exported tree yields
  `git_head: None`; an hour of card time must not be lost to provenance.

`summary.json`'s shape is unchanged (it has no in-repo reader, but a sibling
file is the lower-risk seam).

Pinned by `test/registered/unit/boot_matrix/test_provenance_349.py` (7 tests),
including a pin that `_main` actually CALLS the writer -- a provenance writer
nothing calls would be the always-absent-marker defect this determination was
sent to find, wearing a different hat.

## Honest scope

The three unrun arms (N/O/P) and all four coherence gaps (G1-G4) need card
time. They are filed, not built. No boots were taken.
