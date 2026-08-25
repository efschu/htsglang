# Task #861 — the draft KV half is an unregistered cutover participant

Branch `fix/861-draftkv-hicache`, based on `e3469633bd` (the W36 pin).
Investigation: R4, 2026-08-24/25. Fixes (0) and (b) are BUILT here; items (a)
and (d) are FILED below and deliberately not built.

## The defect, in one line

`Scheduler.__init__` registers the HiCache draft pool with
`draft_worker=self.draft_worker` / `spec_algorithm=self.spec_algorithm`
(`managers/scheduler.py`), and on a phase-flip boot #631 has DELIBERATELY
nulled both — the boot phase is PP, PP has no drafter, and the configured
algorithm is parked in `flip_spec_algorithm` while the real drafter is built on
`phase_flip_stacks`. So `get_draft_kv_pool` returned None,
`set_draft_kv_pool` was never called, `cache_controller.has_draft` stayed False
for process life, and every HiCache read-through in the TP phase restored a
TARGET-ONLY prefix whose draft rows held the previous occupants' bytes.

Nothing raises. The target verifies every proposed token, so the ANSWER is
unaffected — only acceptance collapses, which reads as "this rig is slow".

Measured before/after, same script against both trees
(`has_draft` in the TP phase, flip-boot shape):

| tree | after boot | cutover route | in the TP phase |
|---|---|---|---|
| `e3469633bd` (base) | False | ABSENT | **False** |
| `fix/861-draftkv-hicache` | False | present | **True** |

## Why it was invisible

The draft half is an unregistered participant in FIVE movers at once, and each
one had a good local reason:

1. **The KV mover** (`phase_flip_runtime.py:3232-3271`) views only the target
   pools' `full_kv_pool` buffers, 16 layers. `kv_reshard.py` contains zero
   occurrences of "draft".
2. **The #703 flip writeback fence** (`hicache_flip_writeback.py`) — no draft
   handling at all.
3. **The #706 canonical store** excludes draft keys BY NAME
   (`hicache_storage.py:877-886`, `:1038-1047`) — "the draft pool starts cold
   after a flip or a reboot — the designed shape".
4. **The #847 phase host pool** (`phase_flip_boot.py:2019-2033`) carries a
   single `PoolEntry(name=PoolName.KV, ...)`.
5. **The HiCache controller itself** — not merely unregistered but
   *structurally unregisterable*, because its one registration site reads the
   boot-phase `scheduler.draft_worker`. That is what this branch fixes.

And the radix tree covers draft KV implicitly with **no guard**: slot ids are
shared (`eagle_worker_v2.py:441-442`,
`phase_flip_draft_bootstrap.py:33-38`), so the tree asserts that TARGET row `s`
is valid and says nothing about draft row `s`. Contrast the GDN precedent,
where `batch_exists_v2` takes the MINIMUM across pools and a missing mamba blob
truncates the KV prefix to zero (`hicache_storage.py:143-148`). `PoolName.DRAFT`
appears in no `PoolTransfer` handed to an exists check.

## What is built here

### Fix (0) — register the draft half against the CURRENT binding generation

`mem_cache/kv_cache_builder.py`
* `resolve_draft_registration(scheduler, phase)` — reaches
  `phase_flip_stacks.draft_worker` only when the scheduler's own is the nulled
  boot-phase value, so a non-flip deployment reads exactly what it reads today.
* `rebind_hicache_draft_for_phase(scheduler, phase)` — called from
  `hicache_phase_binding.rebind_for_cutover` AFTER `coherence_check`, so the
  draft half can never be armed against a target binding that itself failed to
  move. Host pool allocated ONCE and re-stamped thereafter (a pinned pool per
  flip would charge the host budget every time, and on this box that budget
  binds — DESIGN_706 C1). A change in the 1-to-1 host-index invariant is
  refused loudly (the #345 right-token/wrong-slot class).
* `drafter_identity_hash(server_args)` — see (a) below.

`managers/cache_controller.py`
* `draft_tier_armed(direction)` — **THE ONE GATE**, three terms: registered at
  all / the active phase owns the drafter / the binding has not moved. Same
  shape as `2bf0f53498` ("give all four consume points one gate"), applied to
  the participant that never had one.
* All **six** consume points rewired through it — four here and two in
  `HybridCacheController`'s overrides, which is the lane this rig actually
  runs (the W31/W32/W33 shape: a correct mechanism a second copy overrides is a
  mechanism that never runs). A test pins by `ast` that no transfer site reads
  `has_draft` directly.
* `disarm_draft_kv_pool(reason)` — the tp→pp leg. Pools kept, not torn down.

**THE ORDERING TRAP, caught at the desk.** `rebind_for_cutover` runs AFTER the
active stack swap (`phase_flip_runtime.py`: `scheduler.draft_worker =
want_draft` at :2717, the rebind at :3026), so on the pp→tp leg the flip's
drafter IS reachable through `scheduler.draft_worker`. A resolution that
derived "is this a flip instance" from "did I have to fall back to the stacks"
would therefore answer NO on exactly the leg that needs the phase term most,
arming the draft half for BOTH phases. Ownership is read from the stacks'
EXISTENCE; the handle is looked up wherever it currently lives.

**Why the generation matters and a boot-time snapshot would not do:** draft
host indices are 1-to-1 with the TARGET host pool's, and
`hicache_phase_binding._stamp` re-points `mem_pool_host` at every rebind. A
registration minted at generation `g` indexes generation `g`'s slot space;
consumed at `g+1` it addresses a different pool. The stamp rides the SAME #719
authority the releases and the write-backs already use — one more consumer, not
a second scheme.

### Fix (b) — never speculate over draft rows nothing wrote

`managers/phase_flip_draft_bootstrap.py`
* `BOOTSTRAP_ATTR` widened from a flag to a **debt counter**, monotonic,
  truthiness-compatible with the pre-#861 `True`, bounded by
  `MAX_DRAFT_COLD_ROUNDS = 8` and refused ABOVE it at the SET site.
  `clear_bootstrap` decrements instead of clearing; with
  `DEFAULT_DRAFT_COLD_ROUNDS = 1` that is byte-identical to the old clear, which
  is why the `eagle_worker_v2` call site is untouched.
* `arm_draft_cold_for_admission(scheduler, batch)` — scrub, then mark, then
  stamp armed. Called from the ONE funnel every batch passes through
  (`Scheduler.run_batch`), because the #856 seam-transport re-admission is
  built by a purity exemption rather than by the ordinary prefill path.
* Two triggers: `SEAM_READMIT_ATTR` (stamped only by the #856 cutover's own
  retract closure, never by decode-OOM preemption) and "the draft tier is
  disarmed" — the cheap sound over-approximation, whose false positive costs
  one non-drafting round and whose false negative is a request speculating over
  rows nothing wrote.
* A request with NO cached prefix is never cold: `_draft_extend_for_prefill`
  wrote every draft row in that very batch.

**Why this had to exist even with (0):** #631's cutover leg
(`arm_draft_bootstrap_all_reachable`) is still correct and still armed, and
since #856 it finds an EMPTY resident set on every flip — the residents are
retracted at the seam and come back as a read-through prefill. The scrub-and-seed
had to move to where the request now enters, which is admission.

## FILED, NOT BUILT

### (a) Drafter identity in `compute_model_identity_hash`

`{hash}.draft` pages are keyed by the TARGET's identity alone.
`compute_model_identity_hash` (`hicache_storage.py:30-96`) covers
`model_path | revision | dtype | quantization | kv_cache_dtype` plus the
uneven-TP vectors, and **nothing about the drafter**. Two boots agreeing on the
target and differing in drafter — another NEXTN checkpoint, MTP↔EAGLE, the #156
cross-algorithm switch, #765 DFlash2 — read each other's draft KV AS VALID, with
blob length the only accidental guard and equal geometry the common case.

Fix (0) is what makes this reachable: before it, a flip boot wrote no draft
pages at all.

**The guard shipped in this branch instead**, and its two halves:
* GENERIC BACKENDS (the file backend this rig runs): the drafter hash goes into
  the component name, `{hash}.draft-{drafter}`
  (`cache_controller._draft_component_name`). A page written by another drafter
  simply does not exist — a clean MISS, the same argument `HiCacheFile` already
  makes for the identity hash it does carry.
* MOONCAKE v2: **refused by name.** A v2 page is keyed by the POOL NAME
  (`register_mem_host_pool_v2(pool, PoolName.DRAFT)`), so the suffix cannot ride
  along without changing the registered pool identity. Refused rather than left
  write-only: a page nobody may read is still a page the NEXT boot may read.
  The L2 host tier is unaffected. Not installed on this rig.

`draft_kv_layout` belongs in the DRAFT key and must NOT enter the target key —
DESIGN_631b records it as a parallelism decision rather than a weights one, and
it decides the draft pool's row space and per-row byte length.

**(a) proper** folds the drafter into `compute_model_identity_hash` so every
backend and both key routes pick it up, and lifts the mooncake refusal.

### (d) A canonical draft-page form

`hicache_storage._is_draft_key` (`:877-886`) excludes draft keys from every
neutralisation rule, so a `.draft` page keeps the full geometry suffix and
cannot cross a geometry change at L3. `mem_cache/draft_migrate.py` already
holds the umsharder (`DraftBlobSpec`, capability registry) that (d) would use.

Sizing, corrected from the scope note's guess: the draft pool is **1 layer**
against the target's **16** full-attention layers
(`model_runner_kv_cache_mixin.py:4288`, `canonical_page_store.py:30`), so the
draft half is **≈ +6.3 %** of host and disk bytes, not ~1.5 %.

## Test results (hermetic, `CUDA_VISIBLE_DEVICES=""`)

* New: 42 passed across three files
  (`test_draft_hicache_binding_861.py`, `test_draft_tier_gate_861.py`,
  `test_draft_cold_admission_861.py`).
* RED on base `e3469633bd`: all three files error at collection (the symbols do
  not exist), and the behavioural probe above shows `has_draft` False in the TP
  phase.
* CAN-FAIL by mutation, seven mutants, each killing ≥1 test; restoring returns green:

  | mutant | reds |
  |---|---|
  | gate loses the phase + generation terms | 3 |
  | resolution stops refusing the pp leg | 2 |
  | mark loses its ceiling | 1 |
  | discharge goes back to a clear | 1 |
  | scrub is not bounded to the cached prefix | 2 |
  | drafter identity leaves the key | 1 |
  | `owner_phase` derived from "did I fall back to the stacks" | 5 |

* Scoped gate (touched modules + consumers: hicache / rebind / binding /
  cache_controller / draft / #703 / #718 / #719 / #760 / #847 / #856 / W35 /
  black-ratchet): **679 passed, 13 skipped, 69 subtests**.
* `ruff`: zero delta on every touched file. `codespell`: one pre-existing hit
  (`retuned`, a deliberate word in `retune_carried_batches_for_phase`).

## What the W36 boot should show

* `grep -c "HiCache draft KV registered"` — **0 on base**, ≥1 per pp→tp cutover
  on this branch, with `owner_phase=tp` and a rising `binding_generation`.
* `grep "#861 draft-half HiCache .* DISARMED"` — expected once per tp→pp leg.
* `grep "PHASE-FLIP-DRAFT ADMISSION draft-cold"` — one line per re-admitted
  batch, naming rows scrubbed and the reason.
* Per-request `meta_info.spec_accept_length` (never `spec_ema_accept_len`)
  correlated with `#cached-token` on the seam-transport prefill.
