# AUDIT — stale gates: guards whose JUSTIFICATION expired

Desk audit, read-only except for the one write class named below. Base commit
`4a16043d1a` (`train/0817-control`), branch `audit/stale-gates`. No GPU, no
boot, nothing executed against a server.

## Why this axis is new

A guard can fail in ways the existing sweeps do not look for. `AUDIT_421`
asked *is it wired?*; `AUDIT_500` asked *how far does it reach?*; `AUDIT_505`
asked *can it be silently wrong?*; `AUDIT_506` enumerated overflow, auth-reach,
key-agreement and test-gate strength. **None of them asks whether the REASON a
guard gives is still true.**

That is the failure this sweep looks for, and it has bitten repeatedly:

* the #630 parse-time blocker refused a boot whose blocker had been fixed —
  found and lifted by #703, in two places that had to move together;
* the #335 compat-surface inventory carried a "blocked on #333" that was no
  longer true, which is why the sdapi surface looked unbuildable for a week;
* catalog anchors drifted (`schedule_policy.py:1368` → `:1614`), so a reader
  following a citation landed on unrelated code.

The shape is always the same: **the behaviour may be right, but the stated
reason has expired**, and the next reader inherits a false model of the system.

## Classification

| class | meaning | action |
| --- | --- | --- |
| **STILL-VALID** | the cited blocker is genuinely unmet | none |
| **LIFT-CANDIDATE** | the cited blocker appears delivered | none here — a lift is its own decision with its own evidence (see below) |
| **STALE-TEXT** | behaviour correct, stated reason outdated | text fixed in place, provenance cited — the ONLY write this sweep makes |
| **DANGLING** | cites something with no trace it ever existed | filed |

**Nothing is lifted by this sweep, on purpose.** The #630 lift is the standard:
it needed a rendezvous proof and a named protecting suite
(`test_hicache_bounded_waits_630.py`), not merely "the fix landed". Each
LIFT-CANDIDATE below therefore carries *what evidence a lift would require*,
which is the useful half of finding it.

## Scope, honestly

Union of two greps over `python/sglang/srt/` (tests excluded): refusal-shaped
lines citing a `#NNN` ticket, and lines using expiry language (`blocked
on/until/by`, `not yet supported/implemented/built/wired`, `once #`, `when #`,
`until #`, `awaiting #`).

| set | lines |
| --- | --- |
| total candidates | 181 |
| in vendor/upstream backends (NPU, Ascend, `models/`, multimodal processors) | 8 |
| in `planner/rejected.py` (the rejection register) | 3 |
| fork-owned remainder | 170 |

The 8 vendor-backend lines are counted and not analysed in depth: they are
upstream capability statements about hardware this fork does not own, and are
not ours to lift.

**A candidate line is not a gate.** A large share of `#NNN` mentions are
PROVENANCE ("until #695 this counted X", "#631 introduced this") — historical
narration whose truth does not expire. Those are counted, not listed; only
lines whose stated REASON could expire are classified below.

## Verified rows

### STALE-TEXT — fixed in place

| # | file:line | what it said | what is true | evidence |
| --- | --- | --- | --- | --- |
| S1 | `managers/phase_flip_runtime.py:1127` (docstring of `flip_blocking_guards`) | "Mirrors the #297 Stage-A guard shape, **plus the #630 PP x disk-HiCache wedge**" | the #630 clause is GONE from that function — its own body comment says so: "#703 stage 2: the #630 clause is GONE, not narrowed further" | lift done by #703 after `9da9dfd025` fixed the root; `test_hicache_bounded_waits_630.py` is the active protection |

Why it matters beyond tidiness: a docstring listing a guard the function does
not have sends the next reader hunting for absent code, and — worse — could
persuade someone the combination is still refused when it is in fact served.
Fixed text-only; 267 tests across the phase-flip and #630 suites pass unchanged.

### LIFT-CANDIDATE / STALE — the rejection register (`planner/rejected.py`)

The register is the highest-yield surface on this axis, and by design: its own
docstring says "verdicts are re-checked before every feature order — several
rows are days old and some may be reopened". This sweep is that re-check. 27
entries, 19 BLOCKED. Every row below was re-verified at code by me, not taken
from the sweep alone.

| # | row (file:line) | what it claims | what is true now | class |
| --- | --- | --- | --- | --- |
| R1 | `planner/rejected.py:343` `pp_with_spec` | "BLOCKED by the engine: `pp_size > 1` asserts ... `speculative_algorithm is None`. It is a hard assert ... **Every PP number on this rig is therefore a no-spec number**" | the assert grew an exemption: `assert self.speculative_algorithm is None or self.enable_phase_flip` (`server_args.py:17791`, added `a16562c9ec` 2026-08-08, one day AFTER the row's own #625 re-verification). Read the surrounding comment precisely: the physical incompatibility is **not** waived — no draft worker exists in a PP phase — but the combination is now ALLOWED and speculation runs in the TP decode phase, "enforced by construction instead of by refusing the flag combination" | **STALE** |
| R2 | `planner/rejected.py:263` `gguf_on_sm75` | "the promised loud failure at GGUFConfig is **not wired** (open bug #269)" | it is wired: `GGUFConfig.get_min_capability` (`layers/quantization/gguf.py:121`) carries an explicit `#269:` comment and enforces the sm80 floor. #269 is closed | **STALE** |
| R3 | `planner/rejected.py:284` `gguf_moe_expert_offload` | "the expert-offload installer has no quantization guard ... Fail-fast **is task #268**" (future tense) | `assert_expert_offload_quant_supported` exists (`layers/moe/expert_offload.py:2194`). #268 is delivered, not open | **STALE** |
| R4 | `planner/rejected.py:321` `spill_with_pd` | evidence: "`server_args.py:4992`, hard reject at arg parse" | the GUARD still exists and the verdict holds in substance — but `:4992` today is `--dual-group-lane-share-min-windows` help text, i.e. **unrelated code**. The live guard is elsewhere in the same file | **STALE-TEXT (anchor drift)** |
| R5 | `planner/rejected.py:338` `spill_with_pp_dp` | evidence: "`server_args.py:5015`, hard reject at arg parse" | same shape as R4: verdict holds, `:5015` now points at unrelated help text | **STALE-TEXT (anchor drift)** |

**R1 has an unmerged fix.** `54ffc4d90a` ("#704b: correct the pp_with_spec
register row -- it was STALE, not mis-pointed") rewrites exactly this row and
lives on `feat/704-prefill-ladder` / `train/0817-desk*` /
`reconcile/cluster-b-seam-model`. It is **not** an ancestor of
`train/0817-control` (verified). So this is not a new lift decision at all —
it is a MERGE-TRAIN matter, and the register on the serving lineage will keep
misinforming the wizard until that commit lands. Flagged for the #706-Lane
rather than re-fixed here, because a second correction of the same row on a
different branch is how two versions start disagreeing.

**R4/R5 are NOT fixed in this sweep**, though they are text-only. The correct
repair is the one #625 already applied to `pp_with_spec` (`2872a398a8`): pin
the guard by its TEXT rather than by a line number, so the citation cannot
drift again. That is a small behaviour-neutral change to the register's
evidence convention, and it belongs with whoever lands `54ffc4d90a` so the
file is touched once rather than three times.

**The anchor-drift is structural, not incidental — measured.** The register
cites a `file.py:line` as evidence in exactly 4 rows. **Three of the four have
drifted:**

| cited anchor | what is there now | verdict |
| --- | --- | --- |
| `server_args.py:4992` (R4) | `--dual-group-lane-share-min-windows` help text | drifted |
| `server_args.py:5015` (R5) | help-text continuation | drifted |
| `server_args.py:16240` (R1) | a `transfer_engine` load-format fallback message | drifted; the real assert is at `:17792` |
| `gguf.py:40` (R2's `_has_sgl_gguf_kernels`) | `_has_sgl_gguf_kernels = False` | **accurate** |

The survivor is the one pointing at a MODULE-LEVEL CONSTANT; the three that
drifted all point INTO `server_args.py`, a file that grew ~1300 lines this
month. That is the lesson worth carrying past this register: a line number into
a growing file is not evidence, it is a decaying pointer, and the convention
should cite a SYMBOL or a quoted string. #625 already reached this conclusion
once for `pp_with_spec` (`2872a398a8` pins text rather than a line) — the
practice simply never spread to the other rows.

**Why R2/R3 matter more than they look.** Both rows tell the planner's wizard
that a combination is blocked because a guard is MISSING. Both guards now
exist. A row whose stated reason is "we have no protection here" outlives its
own fix in the most misleading possible direction: it keeps a working, guarded
path out of the wizard's proposals, and it tells the reader the engine is less
safe than it is.

### STILL-VALID — sampled

| # | file:line | cited blocker | why it stands |
| --- | --- | --- | --- |
| V1 | `planner/key_solver.py:382` (`session_max`) | "Blocked on one prior question — whether the scheduler can bind a session to a replicated state pool (affinity)" | an open DESIGN question, not a delivered ticket; the entry declares itself "Not implemented — no number is produced here", so it is an honest reserved interface rather than a gate with an expired reason |

## What a lift would require

For any LIFT-CANDIDATE, "the cited commit exists" is **not** sufficient
evidence. The #630 precedent sets the bar:

1. the root fix identified by commit, and shown to be an ancestor of the build
   being lifted;
2. an active protecting test named, so the guard is replaced by coverage rather
   than removed;
3. every TWIN moved together — #703 had to lift the parse-time and runtime
   clauses in one step, because lifting only one left the flag unusable while
   appearing fixed;
4. for anything on the serving path, a boot.
