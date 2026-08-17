# #349 -- Is the standing bug net still armed at HEAD? DETERMINATION

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots.

**Verdict: MOSTLY armed, with two real disarms found and fixed -- and a reach
that is 16 days behind the product.**

The harness itself is healthy: it imports, runs card-less, and its unit tests
are green. The arm definitions are in far better shape than feared -- no
renamed flags, no removed flags, no dead env switches, no missing model paths.

But the audit did find rot, in both directions the class allows:

1. **A false RED.** `reject_dcp_crossalgo` has been failing every sweep for
   over two weeks on a marker the refusal never prints -- an arm that cannot
   pass is as disarmed as one that cannot fail, because it trains readers to
   ignore the net's reds.
2. **A false GREEN, net-wide.** The `graphs` axis was confirmed from the DRAFT
   worker's capture line on all 15 boot arms; the TARGET model's graphs have
   never once been observed. A target-side fallback to eager would have gone
   green.

Both are fixed and pinned here. The third finding is not rot but range: the
net is structurally blind to the phase-flip world that arrived after it, which
is where this week's crashes lived. That part cannot be fixed at the desk --
it needs boots, and it is filed.

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

All 20 arms were audited flag-by-flag, env-by-env, marker-by-marker against
HEAD. The sort:

| verdict | count | arms |
|---|---|---|
| VALID | 17 | A, B, C, D, E, G, H, I, J, K, M, N, O, and the four sound reject arms |
| STALE-FIXABLE | 2 | `reject_dcp_crossalgo` (fixed here), `P_hicache_nospill_control` (comments only, fixed here) |
| BLOCKED, not stale | 1 | `L_video_cotenancy` -- filed, see below |
| OBSOLETE-BY-DESIGN | 0 | -- |

**Nothing the feared shape predicted was found.** Every CLI flag across all 20
arms still exists as a real `server_args.py` field -- no renames, no removals.
All eight env vars have live readers -- no dead switches. The one model path
(`DFLASH_DRAFT_MODEL`) exists on disk. The two centralised spill-marker
literals are still emitted verbatim. On the #382/#315 axes the net is clean.

### The one genuinely disarmed arm: `reject_dcp_crossalgo`

`reject_markers` demanded the literal `"--speculative-cross-algorithm"`. The
guard that actually fires words its refusal as prose --

> `--draft-kv-layout dcp is not supported together with cross-algorithm
> speculative serving: the active draft rung changes at runtime...`

-- and never prints the flag spelling. `first_refusal` requires ALL markers in
ONE refusal, so the arm reported **FAIL on every sweep**. It is right there in
the 2026-08-01 record: *"refused, but with an unexpected error rather than the
named guard"*. It has been red for over two weeks while the server refused for
exactly the right reason.

This is the disarm, in the direction that is easy to miss. A false RED is not
a stricter net -- it is a net people stop reading, which `arms.py`'s own
docstring already says about the deleted draft-extend refusal: *"A net that
keeps asserting a deleted refusal reports a defect every run and teaches
everyone to stop reading it."* The net grew the very defect it warns about.

**Fixed**: the marker is now `"cross-algorithm speculative serving"`, a
substring the refusal actually contains.

**Pinned, and this is the durable part**: a meta-test now asserts that every
reject arm's markers appear in a `raise` message in `server_args.py`, parsed
out by AST. Written first as a whole-file substring check, it PASSED while the
broken arm was still broken -- every flag spelling occurs in `server_args.py`
as its own argparse definition. The check has to look where the words are
EMITTED, not merely where they exist. It cannot prove a guard fires (that half
goes red loudly on the next sweep); it pins the half that rots silently.

### `L_video_cotenancy`: blocked, deliberately left red

`LANE_RANK_MIB = "19000,15000,15000"` is the exact vector
`_validate_dual_group_lane_card_budget` was built to reject -- the guard's own
docstring names arm L as its reason. The arm cannot boot on this rig: ~35.9
GiB required against a ~32.1 GiB card, and the arm's own comment already
carries that arithmetic and the conclusion that shrinking the vector does not
help (14.8 GiB of lane complement leaves rank 0 no serving KV).

**Not fixed, on purpose.** The tempting move is a `blocked_on` field so the
sweep reports BLOCKED instead of FAIL. That was considered and rejected: a
skip mechanism is itself a way to disarm an arm silently, which is the thing
this determination exists to catch. A visible red with a documented cause is
safer than a green-looking skip. Filed instead, with its dep named by the arm:
the GGUF vehicle of the 4.11 recipe, or the two-card lane of EVAL_272 slice
C/D.

### `P_hicache_nospill_control`: three drifted pointers, no functional defect

Cited `kv_session_offload.py:1588-1589`, `server_args.py:6762`,
`kv_session_offload.py:2395`; real locations at HEAD are `:1788`, `:7308`,
`:2690-2692`. Comments only -- the arm behaves as described. Corrected.

### Net-wide: the `graphs` axis was never actually observed

Not an arm defect -- a defect in the shared resolver, and the most consequential
finding of the audit.

`report_effective` resolved `graphs` with
`r"Capture (draft )?(decode|extend|prefill) CUDA graph"`, while
`model_runner.py:3907` builds the line as
`role = "draft" if self.is_draft_worker else "target"`. The alternation covered
the DRAFT role and missed the TARGET role entirely -- and missed the `verify`
phase too. A real 2026-08-01 arm log carries exactly this:

```
Capture draft decode CUDA graph begin      <- matched
Capture draft extend CUDA graph begin      <- matched
Capture target verify CUDA graph begin     <- matched nothing (x3)
```

Every boot arm runs with speculation, so the draft line was always present and
`graphs` always resolved True **on the strength of the draft model alone**. The
matrix has never once observed whether the TARGET model captured its graphs. A
boot where target capture silently fell back to eager while draft capture
succeeded reported `graphs=True` and went green -- a FALSE GREEN, worse than
the STOP an always-absent marker usually yields, because `BASE_EXPECT` means
"full CUDA graphs, not eager" about the SERVED model.

Confirmed against real data rather than inferred: the three arms that PASSed on
2026-08-01 all carry `"graphs": true` in `summary.json`; the eight `null`s are
arms whose boot died before any capture, not the regex.

**Fixed**: `graphs` is now resolved from the target line only
(`(?:target )?(?:decode|extend|prefill|verify)`), with a roleless line read as
target so pre-role-prefix artifacts stay readable. Draft capture is no longer
treated as evidence about the target.

Why CI never caught it: all three existing fixtures used
`"Capture draft decode CUDA graph begin."` and nothing else, so the suite only
ever exercised the accidentally-matching branch. Those three fixtures now carry
both roles, as a real spec boot prints them; a new test pins that a draft-only
log resolves `None`, never `True`.

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
