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

## Part 2 — the tree-wide ticket-citing guards

Swept `python/sglang/srt/` excluding the register and tests. **The headline is
that the tree is healthy on this axis**: nearly every ticket-citing guard's
reason still holds, and **no DANGLING guard was found anywhere** — every
citation resolved to a real commit history or a checkable code artifact.

### The one verified STALE-TEXT — fixed

| # | file:line | said | true now |
| --- | --- | --- | --- |
| T1 | `server_args.py:2275` (help text) | park-tier routing happens at exhaustion sites "**once the #236 budget lands**" | #236 landed; `park_instead_of_demote` is live and called at `managers/kv_session_offload.py:3404`. Fixed: the conditional clause dropped, behaviour described as current |

### A classification I REJECTED after checking it myself

`distributed/device_communicators/barlink_path_rates.py:21` says "The pending
NCCL/system-RAM reference. **The measurement does not exist yet**". The sweep
classified this STALE-TEXT on the grounds that #278 closed and
`load_nccl_reference` is implemented.

**That is wrong, and the distinction is the whole point of this axis.** The
LOADER existing is not the MEASUREMENT existing. `scripts/gpu_battery/
s06_nccl_reference.{sh,py}` and `scripts/probe/nccl_reference.py` exist to
PRODUCE the data; no data file does, and the consumer at
`barlink_path_rates.py:448` explicitly handles that case — "rate source not
available ... its paths stay PLACEHOLDER". So the docstring is **correct** and
the row is STILL-VALID. Fixing it would have replaced a true statement with a
false one.

Recorded because it is the failure mode this sweep is most likely to commit:
"the ticket closed" is not the same claim as "the thing the guard waits for
exists".

### THE SHARPEST FIND — a lifted gate still pinned by a RED test

`test/registered/unit/server_args/test_phase_flip_args.py:88` asserts that
`enable_hierarchical_cache=True` raises a `ValueError` matching **`"#630"`**.
#703 stage 2 REMOVED that blocker. The test is therefore **failing on
`train/0817-control` right now** — verified failing at the untouched base
commit, so it is not this sweep's doing.

This is the audit's own axis turned on the test suite, and it upgrades a lesson
the #703 lift already knew. That lift was careful about TWINS: it moved the
parse-time and runtime clauses together, because "fixing only the runtime
clause is not enough ... both must move together or the flag is still
unusable". **There was a third twin.** The test that pinned the gate did not
move with them, so the lift left behind a red test asserting the presence of
something it had just deliberately removed.

NOT FIXED HERE: deleting or rewriting a test assertion is not behaviour-neutral
for the suite, so it falls outside this sweep's one permitted write. Filed as
its own item. The repair is small — drop the `#630` row from that loop and
point at the same protection the code comments already name
(`test_hicache_bounded_waits_630.py`) — but it is a decision about what the
suite guarantees, not a text fix.

**Generalised**: a lift has THREE twin classes, not two — parse-time guard,
runtime guard, and the tests pinning either. This belongs in the lift-evidence
bar below.

### STILL-VALID — the substantive set

Sampled from the sweep and spot-checked: the HiCache phase-rebind refusal
(`mem_cache/hicache_phase_binding.py:142`, waiting on a phase-matched host pool
that nothing in the tree ever sets), the #578 stage-table refusal
(`managers/regime_classifier.py:671` — its text reads liftable in isolation, but
#584's measurement canon fills what it can and this raise is the designed
fallback for what it cannot), the #452 CUDA-graph MoE-offload refutation
(`environ.py:1251`, reconfirmed by three later commits and never reversed), the
KVSO/HiCache opt-in gate (`server_args.py:7435`, its contention measurement
still outstanding with the boot-matrix arm that would produce it still
present), and the topk>1-under-uneven-DCP refusal (`server_args.py:8719`),
which matches the standing "do not re-attempt" record.

### Two loose citations worth naming

`layers/dcp/owner.py:127` and `layers/attention/triton_backend.py:432` both
cite **"#76"**, and in neither case does the real #76 (an unrelated PD-disagg
feature) match the claim. They are scale analogies written as if they were
pointers. Not DANGLING — the guards themselves are valid and independently
confirmed — but a reader following the citation lands somewhere unrelated,
which is the same reader-misdirection cost as the drifted anchors above.

### Counts, and an honest discrepancy

| quantity | count |
| --- | --- |
| candidate lines (my union grep) | 181 |
| fork-owned | 170 |
| vendor/upstream (counted, not analysed) | 19 ticketed + a larger un-ticketed bucket |
| STALE-TEXT found and fixed | 2 (S1 register docstring, T1 help text) |
| STALE (register rows, not fixed here) | 3 (R1-R3) |
| STALE-TEXT anchor drift (not fixed here) | 3 of 4 file:line citations |
| LIFT-CANDIDATE requiring a lift decision | 0 |
| DANGLING | 0 |
| red test pinning a lifted gate | 1 |

**The provenance count is where my number and the sweep's disagree, and I am
not reconciling them by picking one.** I counted 21 provenance-shaped lines
within my 181-line candidate union; the sweep counted 187 across the whole tree
without that pre-filter. Different denominators, both defensible, neither
checked against the other line by line. What both agree on is the shape: the
large majority of `#NNN` mentions in this tree are settled narration, and only a
handful are live conditions that could expire.

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
4. for anything on the serving path, a boot;
5. **the tests that pin the gate move with it.** #703 moved its parse-time and
   runtime clauses together and still left `test_phase_flip_args.py:88`
   asserting the removed blocker — red on the serving lineage ever since. Two
   twins were not enough; there are three.
