# HANDOFF MERGE-R8 — two merges, zero conflicts, and a fix whose own gate cannot run here

Shift `656-merge-r8`. Worktree `/spinning/wt-merge-r8`, branch `merge/r8-batch`,
based on `origin/feat/route-a-631` at `cd71ec34ce` (the MERGE-R7 tip both lines
carried). Frozen pre-merge baseline: `/spinning/wt-merge-r8-base`, detached at
`cd71ec34ce`, clean tree. Evidence and logs: `/spinning/evidence-631/merge-r8/`.

Content tip on both lines: **`1c92b12abe`** (merge steps 1-3). This handoff
cannot name the commit that contains it, so the actual branch tip is the single
docs-only commit sitting directly on top of `1c92b12abe` — same convention
MERGE-R7 used. `ls-remote` verified against `git rev-parse HEAD` after every
push.

ERRORS FIRST.

---

## 1. The mamba-floor fix ships with a gate that cannot run under the merge protocol

`test/registered/unit/mem_cache/test_gdn_resident_cap_floor_656.py` — the file
that pins the instance-killing fix of merge step 2 — **fails 7 of 7** under the
canonical CPU-only desk protocol with

```
RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform
plugin is available.
```

Every case constructs a real `ServerArgs`, and `__post_init__` resolves a
device before any assertion in the file is reached.

**Inherited, not introduced, and proven rather than assumed**: the identical arm
run against the untouched source worktree `/spinning/wt-656-mambafloor` at
`81e9e3c071` returns the same `7 failed`. The branch's own test record was
necessarily measured with a device visible; the two numbers are not comparable
and neither is wrong.

This is MERGE-R7 §2 repeating one branch later, and it lands worse this time,
for two reasons that compound:

* R7's device-requiring file gated a claim that other tests also touched. **This
  one is the only test of a fix that killed a live instance.**
* The file sits in `test/registered/unit/mem_cache/`, which **no canonical arm
  and no entry in `scripts/run_631_flip_family.sh` covers** — see §2. So it is
  not merely red, it is red in a place nothing looks.

### The gate was verified anyway, without a device

The *validator* needs no device even though *constructing* `ServerArgs` does, so
it was driven directly against a stand-in carrying only the attributes the floor
arithmetic reads. Script and log:
`/spinning/evidence-631/merge-r8/gdnfloor_desk_proof.{sh,log}`.

| case (`--max-mamba-cache-size` / `--max-running-requests` / `--gdn-resident-state-slots`) | result |
|---|---|
| **12 / 4 / 4** — the exact 2026-08-13 acceptance specimen | **REFUSED** |
| 12 / 4 / 11 — one slot below the floor | REFUSED |
| 12 / 4 / 12 — exactly at the floor | ACCEPTED |
| 12 / 4 / unset — flag not given | ACCEPTED |
| 12 / unset / 4 — concurrency not pinned | ACCEPTED |

The refusal message was checked against all four strings the branch's own test
asserts, each present: `--gdn-resident-state-slots 4`, `floor of 12 slots`,
`--max-running-requests to at most 1`, `--gdn-resident-state-slots to at least
12`. Refusing at floor−1 and accepting at the floor is the can-fail boundary, so
this is a proof and not a smoke test. The floor arithmetic reproduces on both
sides of the ping-pong term: `enable_mamba_extra_buffer()` False gives
floor(bs=4) = 12, True gives 16.

**What this does NOT establish**: that the fix behaves on a live boot. That
still needs a device arm in someone's GPU window. A desk proof of the predicate
is not a proof of the boot.

## 2. `test/registered/unit/mem_cache/` cannot be collected at all

Running the directory as an arm — the obvious way to reach the new file —
terminates before a single test executes:

```
ERROR test/registered/unit/mem_cache/test_hicache_nixl_storage.py
!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!
ImportError: Please install NIXL by following the instructions at ...
```

One unimportable file takes the whole directory down at collection time. That is
why R7's arm list omits `mem_cache`, and it means **any test added to that
directory is invisible by default**. With that file ignored the directory does
run, and is itself heavily red on CPU: **937 failed, 730 passed, 695 skipped**
at `cd71ec34ce` — recorded here as a baseline so a future shift can see whether
a number grew, not as a gate anyone should read as meaningful.

Recorded and not fixed: an unconditional `ImportError` at module scope in a
third-party-optional backend is a defect in that file, not something a merge
shift rewrites.

## 3. The register collision did NOT recur — R7's rename held

`docs/dev/631/CONTRADICTIONS_REGISTER.md` is touched by the #605 branch only
this round. The incumbent carries `C605-1`…`C605-8` (MERGE-R7 §1's renamed
rows); the arriving branch appends `C605-9`…`C605-17`. **Nine lines added, zero
deleted, no label used twice.** Verified by counting every `C605-n` occurrence
rather than by eyeballing headings — the check R7 had to learn:

| check | result |
|---|---|
| deleted lines in the register | **0** (`git diff --numstat` reports `9  0`) |
| added lines | **9**, one row each for `C605-9`…`C605-17` |
| duplicate row labels | **none** |
| file length | 2408 → **2417** |

`C605-1`…`C605-8` each appear twice (their row plus R7's mapping note) and
`C605-13` twice (row plus one body citation); the rest appear once. This is what
the `C605-*` namespace was created for in R7, and it worked exactly as intended:
a branch appending nine new contradictions collided with nothing.

## 4. The stale local ref trap is STILL armed — now FOUR rounds behind

| ref | SHA at shift start |
|---|---|
| `origin/feat/route-a-631` | `cd71ec34ce` |
| `origin/integration/r2` | `cd71ec34ce` |
| **local** `feat/route-a-631` | **`0ae49fafb4`** — the MERGE-**R4** base, now FOUR rounds stale |
| local `integration/r2` | `481411ac6b` — the R7 base, three rounds stale |

`feat/route-a-631` is checked out in `/spinning/wt-631-routea`, which belongs to
another strand, so git will not move it and forcing it would move another
session's working tree out from under it. This shift did what R6 and R7 did:
**every merge source named as `origin/<branch>`, every push as `git push origin
HEAD:refs/heads/<line>`, `ls-remote` compared against `git rev-parse HEAD` after
each.** The stale refs were never an input to anything. Do not "fix" them from a
merge shift.

## 5. Nothing on GPU, on purpose

Every suite run used `CUDA_VISIBLE_DEVICES=99`. Serving on 30030 was not
touched, no GPU arbitration window was claimed, port 30099 was not touched, no
`pkill` was used, and `git stash` was never invoked. The cost of that discipline
is §1, and it is the right trade — but §1 is the second round running in which
that cost has been paid on the branch whose claim mattered most.

---

## 6. WHAT MERGED

A path-overlap check ran **before** the first merge. The two sources share
exactly one file, `python/sglang/srt/server_args.py` (21 files and 2 files
touched respectively). Their hunks land far apart — the #605 changes at lines
~6504 and ~11857, the #656 validator at ~13616 and ~13709 — and both merged
without conflict. Both sides were then verified present by marker rather than by
trusting a clean merge (§8).

| step | source | at | resulting tip | conflicts |
|---|---|---|---|---|
| 1 | `feat/ledger-reconcile-605` | `20e2e3714e` | `1e060038c1` (`--no-ff`) | **0** |
| 2 | `fix/mamba-floor-resident-cap-656` | `81e9e3c071` | `974015fe77` (`--no-ff`) | **0** |
| 3 | flip-family test-list patch + proof-gate skipif (§9) | — | **`1c92b12abe`** | — |

Each step's suite was green and pushed to **both lines** before the next was
started; nothing was batched. Author on every commit `efschu
<efschu@users.noreply.github.com>`, no trailers.

### Step 1 — `feat/ledger-reconcile-605`, the reconcile loop closes

The headline is not a number, it is that **the instrument had been wrong before
the model was**, and one defect meant the tool had reconciled *nothing at all*
for an entire release: `attribute_flight.py` passed `read_marks`'s **PID-keyed**
dict into a **rank-keyed** lookup, so every card matched nothing and the empty
result printed as *"The ledger names no card whose rank left marks"* — read by
two shifts as a fact about the boot. The silent-skip path was a **pinned
contract**: a test asserted an unmatched rank is skipped rather than invented,
which sounds like conservatism and is what made a total keying mismatch look
like a quiet edge case.

Also carried: the pin-path weight completeness check (the ship config pins
`--rank-gpu-memory-mib`, which takes the pin path, which skips the planner — so
the ledger's largest post shipped as **0 MiB on every pinned boot**;
`_observation_weight_mib_per_rank()` now returns **None, never zeros**, and
`reconcile.completeness_failures()` names it loudly, where a zero would have
passed every check by looking like a price); the load-transient boundary with
its constant **refused** rather than recalibrated; the NCCL boundary
(`nccl_init_begin`/`nccl_init_end` around the whole group-building block, the
end mark deliberately **after** the phase-flip secondary groups because those
are communicators too) with `N/A(barlink)` rendering; hardware-residual
recalibration from 472 boots of recorded history under a **refuse-wide** rule
(a band wider than 50 % of its own high water mark yields a refusal naming the
distribution instead of a number) charging the band **HIGH, never the mean**;
measured-pool subtraction with loud refusal instead of a silent fallback to the
modelled budget; `_field_bytes` max-across-runners; and the card `uuid` in the
ledger dump so future joins are identity rather than inference.

**`corridor_trace.py` now has a production call site** —
`Scheduler._corridor_trace_tick()`, beside `_census_tick()`, arming once and
never retrying. This is the first half of the fix for the thing the acceptance
run exposed: the runtime could not see its own corridor breach. Verified OFF by
default independently of its own tests, in the merged tree:

```
TRACE_ENV = SGLANG_CORRIDOR_TRACE_MS   env set? = False
corridor_trace.start()                 -> None
Scheduler._corridor_trace              -> None      (class-level)
Scheduler._corridor_trace_armed        -> False     (class-level)
```

The attributes are class-level on purpose, so a scheduler that never arms the
sampler carries no per-instance attribute at all — the strictest reading of
"byte-identical when off".

### Step 2 — `fix/mamba-floor-resident-cap-656`, the acceptance-run killer

`--gdn-resident-state-slots` boot-sizes the GDN/Mamba state pool to **its own**
value, overriding `--max-mamba-cache-size` — which means it overrides the very
quantity `_validate_max_mamba_cache_size` had just checked, and **nothing
re-checked the number that actually wins**. The acceptance config
(`--max-mamba-cache-size 12 --max-running-requests 4
--gdn-resident-state-slots 4`) cleared the floor of 12 exactly, then sized the
pool to 4 — the floor for **one** running request, not four. After ~4 minutes of
mixed load the fail-loud `alloc_req_slots` RuntimeError took the whole instance
down with SIGQUIT (`mamba_available=0, mamba_schedulable=0`).

`PrefillAdder` budgets `available_size() + mamba_evictable_size()`, so it admits
against slots it believes are evictable; when they are pinned it is already too
late. That is a runtime race, and the floor is arithmetic knowable at parse
time, which is where `_validate_gdn_resident_state_slots` now puts it. **Refused,
not clamped** — silently lowering `--max-running-requests` would change the
serving shape the operator asked for, and silently raising the cap would give
back the KV bytes the cap exists to take; the message names both exits.

---

## 7. SUITE — every failure count identical to baseline, at every step

Baseline measured in `/spinning/wt-merge-r8` before the first merge, with the
frozen `/spinning/wt-merge-r8-base` detached at `cd71ec34ce` kept for diffing.
Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation so a truncation is isolated to its directory.

**Comparison base, stated explicitly**: `/spinning/wt-merge-r7-base` is at
`481411ac6b`, which is R7's *pre*-merge tree and therefore the wrong tree to
diff R8 against. The continuity check used instead is that **this shift's BASE
column reproduces R7's post-step-2 column exactly**, which it does on every arm —
so the chain back through R7's frozen base to R6's `/spinning/wt-merge-r6-base`
@ `598f570ba4` is unbroken and this shift is comparable to the previous four.

| suite | BASE `cd71ec34ce` | after step 1 | after step 2 |
|---|---|---|---|
| #631 flip family (canonical script) | **1116 passed** | 1116 passed | **1116 passed** |
| `unit/managers` | 9F 1357P 18S | 9F 1357P 18S | 9F 1357P 18S |
| `unit/mem_ledger` | 1F 359P→**379P** | 1F **437P** | 1F 437P |
| `unit/model_executor` | 15F 594P | 15F 594P | 15F 594P |
| `unit/server_args` | 615P | 615P | 615P |
| `unit/turnkey` | 116P | 116P | 116P |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | 46F 348P 4S |
| `unit/docker` | 4P | 4P | 4P |
| new-656 arm (6 files, R7's private arm) | 4F 56P | 4F 56P | 4F 56P |
| `mem_cache/test_gdn_resident_cap_floor_656.py` | *(file absent)* | *(absent)* | **7F** — §1 |
| `unit/mem_cache/` (nixl ignored) | 937F 730P 695S | — | — |

**Every failure count is identical on both sides at every step**, and the
failure *sets* were diffed **by name**, not merely counted, for `managers`,
`model_executor` and `utils`: byte-identical. The single `mem_ledger` failure is
the same inherited `test_communicator_group_contract_612` case throughout,
checked by name. Inherited failure sets carried unchanged: 46 `unit/utils`, 15
`unit/model_executor`, 9 `unit/managers`, 1 `unit/mem_ledger` — none grew.

**+58 tests collected, 0 new failures**: `mem_ledger` 379 → 437 at step 1,
reproducing the #605 branch's own claimed 437/1 exactly in the merged tree.

## 8. BOTH SIDES PRESENT IN THE SHARED FILE

`server_args.py` is the one file both branches touch, so a clean merge was not
taken as evidence. Verified by marker in the merged tree, and the file compiles:

| side | marker | line |
|---|---|---|
| #605 | `weight_mib_per_rank=self._observation_weight_mib_per_rank()` | 6510 |
| #605 | `history=load_boot_history()` | 6511 |
| #605 | `def _build_card_ledgers(self, *, weight_mib_per_rank=None, history=None)` | 11860 |
| #656 | `self._validate_gdn_resident_state_slots(view)` (call site) | 13709 |
| #656 | `def _validate_gdn_resident_state_slots(self, view)` | 13755 |

Cross-module dependencies each branch assumes were also checked to exist on the
line before merging, rather than discovered at runtime: `mamba_pool_floor.
{mamba_hard_floor,describe_mamba_floor}` and
`uneven_perf.PerfCostModel.per_rank_weight_bytes` were already present;
`mem_ledger/boot_history.py` is added by the #605 branch itself.

## 9. THE TEST-LIST PATCH, AND WHAT IT COSTS

MERGE-R7 §7 found six #656 test files that the explicit family list in
`scripts/run_631_flip_family.sh` did not name — collected by nothing, while the
family total sat at 1116 looking green. R7 covered them with a private arm and
left the list unchanged. **A private arm is not a canonical list**, so this
shift folded them in. The list runs 60 → **66** entries:

* after `test_phase_flip_resident_carry.py`:
  `test_phase_flip_seam_reserve.py`, `test_phase_policy_arm_outcome.py`,
  `test_seam_fingerprint_and_margin_656.py` (all `test/registered/scheduler/`)
* at the end: `test/srt/test_phase_flip_serving_proof_gate.py`,
  `test/srt/test_rope_lazy_cache.py`, `test/srt/test_yarn_rope_cache_growth.py`

The runner `cd`s into `test/registered/scheduler` and prefixes every entry with
`$WT/`, so the `test/srt/` paths resolve absolutely; both trees are reached by
the same `test/conftest.py`, so the runner's cwd does not change how they
collect. Checked, not assumed.

### The skipif, and the three tests it costs

`test/srt/test_phase_flip_serving_proof_gate.py` could not simply be added: it
needs a visible device (**measured: 4 failed / 3 passed** under
`CUDA_VISIBLE_DEVICES=99`, 7/7 with a device), and R7's entire 4-failure arm was
this one file. It carries a module-level
`pytest.mark.skipif(not torch.cuda.is_available(), ...)` so the CPU protocol
reports it as **7 skipped** — an honest "not measured" — instead of red.

**Stated plainly because it is a real cost: three cases that DID pass without a
device now skip too.** A module-level marker makes the device requirement one
declared fact about the file instead of four scattered ones, and the file's
subject is not meaningfully gated by three cases in isolation; converting to
per-test markers recovers them if a future shift wants them. **A skip is not a
pass** — the gate asserting the quarantine constant is gone is still discharged
only by a device-visible arm, which no merge shift takes.

### The runner stays green CPU-only

Verified after the patch, same protocol, exit code **0**:

```
1169 passed, 7 skipped, 18 warnings in 119.70s
```

**1116 → 1169 passed, +53**, and the arithmetic closes exactly: R7's private arm
was 60 tests over the six files (4 failed + 56 passed), of which the proof gate
is 7 (now the 7 skipped), leaving **53** that had been running nowhere and now
run in the canonical sweep. No new failures, and the +53 are tests the family
had never collected — which is the point of the patch, and the measure of how
much the explicit-list defect was actually hiding.

## 10. LINT DELTAS — recorded, not acted on

| tool | BASE | FINAL | note |
|---|---|---|---|
| `black` 26.1.0 | **3 dirty** of the 11 pre-existing touched `.py` | **0 dirty** of all 19 touched `.py` | this merge *fixes* three |
| `ruff` | 456 errors (11 files) | 456 (same 11), 456 (all 19) | **all 8 new files ruff-clean** |
| `codespell` | 3 hits | **3 hits, identical** | none new |

The three files black fixed are `mem_ledger/reconcile.py`,
`managers/scheduler.py` and `test_per_runner_reconcile_605.py` — the #605 branch
ran the pinned hook. `scheduler.py` was one of **R7 §3's six persistently dirty
files**; the standing defect is otherwise unchanged, and the remaining five are
still dirty on the line and untouched by this shift:

```
DIRTY  phase_flip_boot.py   phase_flip_runtime.py   phase_policy.py
DIRTY  model_runner_kv_cache_mixin.py               uneven_perf.py
clean  scheduler.py   (fixed by this merge)
```

The pinned pre-commit `black` is evidently still not running on this line. All
three codespell hits are pre-existing at base with only line numbers shifted;
the `schedul` hit in `CONTRADICTIONS_REGISTER.md` is the deliberate 15-character
`TASK_COMM_LEN` truncation from MERGE-R5 and **must not be "fixed"**.

## 11. STATE AT HANDOVER

- **Both lines at the same SHA** — content `1c92b12abe` plus this handoff
  commit on top, `ls-remote`-verified against
  local `HEAD` after every push. Pushed to **`origin` = the efschu fork only**;
  `upstream` was never a push target.
- Working branch `merge/r8-batch` in `/spinning/wt-merge-r8` — same SHA, kept.
- New frozen baseline `/spinning/wt-merge-r8-base` at `cd71ec34ce`, detached,
  clean — **kept deliberately** so R9 can diff against the tree R8 measured
  against. R7's and R6's frozen bases are also still there.
- **Serving, GPUs, arbitration, port 30099: untouched.** No boot, no window, no
  `pkill`, no `git stash`. Nothing under `/etc` modified.
- Local refs untouched, including the stale ones (§4).

## 12. REMAINING UNMERGED BRANCHES

**79** local branches are not merged into the tip (78 at R7; two new branches
appeared this round and one of them merged). The ones recent enough to be live
work, newest first:

| date | SHA | branch |
|---|---|---|
| 2026-08-12 | `d38bb6df32` | `trial/cumulative` |
| 2026-08-09 | `982b6434ce` | `feat/route-a-631-resume-gate` |
| 2026-08-09 | `00a1c50fcb` | `feat/gguf-q4-bringup-651` |
| 2026-08-08 | `18370879e3` | `integration/r3-probe-next2` |
| 2026-08-08 | `27f3bf7996` | `fix/collective-stream-622` |
| 2026-08-07 | `b851df7626` | `feat/dual-group-631` |
| 2026-08-06 | `cc2e03da59` | `serving/530-plus-603b` |

(`backup/pre-email-fix-s13` and `backup/pre-deps-strip-s13` are backups, not
merge candidates.) The three standing Claude strands own
`feat/gguf-q4-bringup-651` (#651), `fix/collective-stream-622` (#622/#649) and
the Route-A line; none was asked to be merged this shift, and none was.

## 13. CARRIED FOLLOW-UPS

Carried, none of it addressed by this shift.

1. **C22 / `KvReshardError` is now ACTUATED and instance-fatal.** #657's
   corridor steering decides correctly and group-uniformly — UUID permutation
   `[1,0,2]` agreed over the group, promoted counts identical on all three ranks
   — but its **application** re-sorts on a rank-local 1 s clock, so three ranks
   re-sorted at three different instants and at t+18 a `pp->tp` cutover died on
   `KvReshardError: payload checksum mismatch -- refusing to scatter`, while the
   steer's own replication check disarmed it in the same second on all three
   ranks. **A pure function applied on a private clock is not a group-uniform
   mutation of replicated state.** The lead worth chasing is the **checksum
   magnitude** — how far apart the payloads were tells you whether this is a
   torn re-sort or a different payload entirely. Register entry `C22` carries
   the full derivation.
2. **The corridor breach on GPU0 at 886 MiB is unexplained**, and it has two
   candidate terms that must be separated before either is believed: the
   **margin term**, and a **gate-threshold mismatch** — the arming floor runs
   `SGLANG_CORRIDOR_FLOOR_MIB=1536` while **the law itself stays 1024**, so a
   breach of the law and a failure to arm are two different events that have
   been read as one. §6's new `corridor_trace` call site is the instrument that
   can finally tell them apart; it has never run on a boot.
3. **Idle-vacate emitted 0 lines on the acceptance boot.** #364 engagement
   shipped in the kv-universe branch and the boot produced no evidence it ran.
   Silence is not proof of a no-op.
4. **Empty-completion characterization is open.** The livelock class where the
   server holds the corridor, answers `/health` 200 and emits no tokens is what
   the seam reserve was built against; the failure signature itself is still not
   characterized well enough to assert on.
5. **`test_communicator_group_contract_612` — the one inherited `mem_ledger`
   failure — is now LOAD-BEARING.** `parallel_state` builds `flip_dcp`,
   `flip_pp` and `flip_tp`; `RUNTIME_COMMUNICATOR_GROUPS` does not declare them.
   That was cosmetic until this merge: those undeclared groups allocate
   communicator buffers **inside the new `nccl_init_begin`/`nccl_init_end`
   gap**, so the first boot to measure `TERM_NCCL_BUFFERS` will measure them
   while the term's signature does not know they exist. Fixing the declaration
   is now a prerequisite for trusting the NCCL term, not a tidy-up.
6. **The weights row does not close under PP**, and was not faked: the split is
   a LAYER split on a hybrid checkpoint, and solving the measured three-card
   posts for a uniform per-layer cost gives 488 / 416 / 581 MiB — a 39 % spread.
   `_observation_weight_mib_per_rank()` returns None under `pp_size > 1` and the
   completeness check names it. What is needed is per-layer bytes **by layer
   type** plus embedding/lm_head placement at the stage ends; the system of
   equations to validate a candidate is already in `RECONCILE_SECOND_RUN.md`.
7. **`reconcile.py` has now RUN against live boots** (that is where step 1's
   four defects came from) but the residuum stays large — 23603 / 13560 / 15652
   MiB — for a nameable reason: the ledger sums PEAK and STEADY-STATE terms into
   one total. Fixing it is a taxonomy change (terms need a peak/resident kind),
   not a mapping fix.
8. Carried unchanged from R7 §11–12 and still open: **lazy RoPE's root cause**
   (ships default OFF; do not enable before the read-back guard exists); the
   **1M pool cost is unattributed**; **L3's 1128 MiB corridor is
   un-re-measured**; the driver-unattributed residual band **164–276 MiB per
   card**; #363's stage actuator is desk code on the line; the first real
   `docker build` is the #384 gate's first test; the #695 census lines still
   need a PP-unique rank identity; `route_a_631_prod_boot.sh` still diverges
   from the ship capture in seven flags.

## 14. NEXT, IN ORDER

1. **Give `test_gdn_resident_cap_floor_656.py` a runnable home** (§1–2): a
   `skipif` so it reports honestly, an entry in a canonical list, and a device
   arm to actually discharge it. Right now the instance-killing fix's only gate
   is red in a directory nothing collects.
2. **Fix the `mem_cache` collection break** (§2) — one unimportable NIXL module
   hides an entire directory.
3. **Declare `flip_dcp`/`flip_pp`/`flip_tp` in `RUNTIME_COMMUNICATOR_GROUPS`**
   (§13.5) *before* the first boot that measures the NCCL term, or that term's
   first number will be wrong in a way nothing flags.
4. **Boot with `SGLANG_CORRIDOR_TRACE_MS` set** and separate the 886 MiB
   breach's margin term from the 1536-vs-1024 threshold mismatch (§13.2). Both
   halves of the fix now exist; neither has produced data.
5. **Run the pinned pre-commit `black` over the five still-dirty files** (§10)
   as a standalone formatting commit, outside a merge shift.
