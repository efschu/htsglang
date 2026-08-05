# DESIGN 552 — kvso x speculative decoding: the resume path and the instrument that could not see it

Branch `feat/kvso-spec-resume`, off `integration/r3-probe-next2`.

## 0. What this document corrects

The task framing for this strand was "kvso x spec is REFUSED today; lift the
refusal and build the resume path". That framing is **stale against this
branch**. The recon below establishes what is actually in tree, and the work
that remained is a different and smaller thing than a lift: the mechanism is
built and the *instrument that was supposed to prove it* was structurally
incapable of doing so.

Recording the correction rather than quietly building the already-built is the
point — a second implementation of `_seed_resumed_draft_state` would have been
the expensive failure here.

## 1. Recon: the refusal sites, and which of them are still refusals

### 1.1 Arg-parse refusals (hard, named, never silent)

`python/sglang/srt/server_args.py:6566-6691`, in `_handle_kv_session_offload`.
`--kv-session-offload-resume-under-spec` is rejected at boot when:

* `--enable-kv-session-offload` is absent (`:6669`) — nothing spills, so the
  flag would silently do nothing;
* `--speculative-algorithm` is absent (`:6676`) — no live spec batch to rejoin;
* `--kv-session-offload-prefill` (PS2) is set (`:6691`) — a placement fact, not
  a policy: a born-spilled prompt never wrote the draft KV that the rejoined
  session's drafter attends.

The env twin `SGLANG_KVSO_RESUME` (legacy alias `KVSO_RESUME`) is OR-ed into
the flag at `:6569`, once, before the feature-disabled return.

### 1.2 The PS2 placement refusal

`kv_session_offload.py:257-345` — `prefill_spill_deep_reject_reason` /
`prefill_spill_deep_gate`. Three named blocks (spec-in-tick armed,
resume-under-spec armed, DFLASH-family prefill append), each returning its own
sentence. Pure over replicated config plus a rank-0-broadcast rung id, so
rank-uniform by construction.

### 1.3 The one remaining runtime decline — and it is a default, not a refusal

`kv_session_offload.py`, `_maybe_restore_flow`, the DEVICE-RESUME UNDER SPEC
gate. Spec active and the flag off ⇒ the session keeps decoding on host through
completion (the validated path). Spec active and the flag on ⇒ it proceeds to
restore.

This is the site the task called "the B10 refusal". It is not an unsupported
configuration; it is a default-off policy, and everything genuinely unsupported
is refused earlier and by name (§1.1, §1.2).

### 1.4 The resume path, end to end

```
pre_schedule (:4262)
  -> step 3 loop over self.spills            FIFO by spill time
  -> _maybe_restore_flow                     spec gate, fast-lane yield, quiescence
  -> _restore / _finalize_restore
       -> batch.spec_algorithm = server algo   (spill batch was built NONE)
       -> _seed_resumed_draft_state            draft-only backfill, no target re-forward
            -> _publish_resume_seed            FutureMap stash+publish, "MTP RESUME seed published"
       -> sentinel-clean assert over row [0,L) (half-restored pool cannot reach verify)
       -> _close_slot(..., "restored to device")
       -> "RESTORE complete ... rejoining device batch spec=N"
```

Two segments are made gap-free before the rejoin: `[0, L_spill)` from the
device-resident draft-KV bundle (Stage 1), and `[L_spill, L)` — the host-grown
positions — from the accumulated per-tick target hiddens.

### 1.5 The blockers the task named, against what is in tree

| named blocker | actual state |
|---|---|
| (a) resume must reconstruct GDN/Mamba state | **Does not arise.** GDN/Mamba state stays **device-resident** across a spill (`kv_session_offload.py:12`); the session keeps its Mamba slot. The target is never re-forwarded on resume, so there is no second SSM advance. The one way the state *could* be lost is the #364 GDN ladder vacating a parked session's blob — fixed in #551 half A via `live_offload_reqs()`. |
| (b) NEXTN draft state + 2x alloc reserve | Handled: `try_spill` frees the draft overhang and sets `kv_overallocated_freed`; `_finalize_restore` clears it so the resumed session re-allocates a fresh overhang and the finish path frees it exactly once. Spec state for the resumed request starts **fresh** — no stale draft tokens travel. |
| (c) CUDA graphs hold pool addresses | Does not bind. Graphs capture the pool **base** and index into it; restore hands back *slot indices*, never addresses, and never reallocates a pool. The #93/#286 remap machinery is not needed on this path. |
| (d) FCFS under spec, retract/resume ordering | Spill victim: fixed in `9e9bb07cb6` (back-most offered instead of declining). Resume order: FIFO by spill time, implemented **only** by `dict` insertion order and previously unpinned — pinned here (§2.3). |

## 2. What this branch changes

### 2.1 The defect: a restore cell that could not be non-zero

`scripts/dev/spill_matrix/drive.py` keyed H4 and H7 on the literal
`restored to device`. That text is not a log message — it is the `why`
**argument** `_close_slot` substitutes into a `logger.debug` template
(`kv_session_offload.py`, `_close_slot`). Every boot in the matrix runs at the
default `--log-level info` (`server_args.py:2810`), so the cell read zero
whether or not a session restored.

Boot K2 (`/spinning/spill-night-20260804/results/RESULTS_K2_spill.md:39,78-83`)
read that zero as "the resume SEED was published, but the on-device rejoin is
NOT corroborated", and #552's default flip stayed blocked on it.

Two aggravating details, both worth naming because they are the reason it
survived:

* `drive.py`'s own header comment says these signals exist so that "a renamed
  message is a test failure and not a silent always-green", and H3/H4 had
  *already* been corrected once after K1 for the same class of error;
* `smoke.sh`'s fixture line was `[TP0] closing slot 3: restored to device` — a
  **paraphrase**. The server emits
  `kv-session-offload: spill slot closed (restored to device, rpi=3, region=0)`.
  The smoke test was green against a string the code never produces.

### 2.2 Fix: an attributing signal at a level operators can see

`_finalize_restore` now logs, through `_log` (info on rank 0, debug elsewhere):

```
kv-session-offload RESTORE complete: rid=%s L=%d (rank %d) rejoining device batch spec=%d
```

`spec=1` is the half K2 could not establish. The seed line proves the
*republish*; `spec=1` proves the session went on to *rejoin a live spec batch*.
Both inputs are replicated, so the line is rank-uniform in content.

Harness: H4 → the info line (spec-agnostic, "did anything restore"); H7 → the
#552 pair, seed line **and** `spec=1`, so a plain restore cannot green it.
`smoke.sh` fixture lines copied verbatim from the format strings.

### 2.3 The pins

`test/registered/unit/test_kvso_restore_signal_552.py`:

* **the general falsifier** — no matrix signal may be text the manager owns but
  no info-level log emits. Deliberately catches the *argument* case, not only
  the format-string case, because the format-string-only version of this check
  would have passed the historical defect;
* the restore-complete format is bound to the harness regexes (rename ⇒ red);
* H7 discriminates `spec=1` from `spec=0`;
* **FIFO restore order** — `pre_schedule` walks `self.spills` in spill order; a
  re-spilled session goes to the back; a `park_pending` slot is skipped without
  reordering the rest. Nothing but `dict` insertion order implements the module
  docstring's "FIFO restore", and restores compete for the same freed space, so
  order *is* the queue discipline: reordering silently starves the eldest
  spilled session — under spec, exactly the long-context session the feature
  exists for;
* **rank-uniformity + neutrality of the capability gate** — the verdict is a
  pure function of (server spec algorithm, flag) and is identical on 8 synthetic
  ranks; with spec off the flag is indistinguishable in either position.

### 2.3a The desync falsifier

Added after production's 4th crash of the day: an NCCL watchdog timeout with
the DCP group at seq 15780 on one rank and 15781 on another — one rank entered
an **extra** collective.

That is the failure mode of a conditional collective whose condition reads
rank-local state, and it is silent until it hangs. Every collective body in
`kv_session_offload.py` already carries a rank-uniformity argument in its
docstring; what was unguarded is the **call site**, where the guard controls the
per-iteration collective *count*.

`TestNoCollectiveIsEnteredOnRankLocalState` pins three things:

* the set of methods entering a `torch.distributed` collective is a declared
  list — a new collective fails the test until someone states where it is
  entered from;
* no `if` enclosing a collective reads a rank-local quantity
  (`available_size`, `_free_regions`, `_tree_evictable_size`, `mem_get_info`,
  …). Under uneven DCP these differ **by design**, so such a guard splits the
  ranks across branches with mismatched collective counts;
* the one collective inside the per-session spill-tick loop
  (`_min_reduce_avail`) is guarded by `batch.spec_algorithm` and nothing else —
  notably not by the local `available_size()` whose value it is about to
  reduce. This site is the highest-risk one in the file because it is entered
  once *per spilled session that ticks under spec*, so its guard controls a
  count rather than a verdict.

These are structural (AST) tests, deliberately: a hermetic test cannot enter a
real process group, but the property that matters is decidable from the source.

### 2.4 The comment invariant

`_maybe_restore_flow`'s guard docstring said on-device MTP resume was "the
follow-up" — eight lines above the escape that runs it. Every auditor reading
that guard learned the mechanism did not exist. Rewritten to state the
mechanism, why the default is still off, and where the unsupported combinations
are refused; pinned by a test so it cannot drift back.

## 3. Honest GPU-gate list — what only a live boot can prove

Nothing below is claimed by this branch.

1. **The rejoin itself.** That a spilled session under a live NEXTN worker
   actually merges back and produces a spec decode. Cite:
   `RESTORE complete ... spec=1` on rank 0. This branch makes the signal
   *emittable*; it does not make it *true*.
2. **Correctness of the resumed session's output.** Blocked on an instrument,
   not on this code: `RESULTS_AVA_determinism_floor.md` shows two identical
   loads against one process with **zero spills** already diverge 6/6, so
   byte-equality of generated text is not a valid instrument on this rig. A
   correctness verdict needs deterministic inference first.
3. **The draft-KV bundle round-trip byte-exactness on real pools.** `KVSO_S1_VERIFY=1`
   exercises it; hermetic tests use mocked pools and cannot.
4. **That the sentinel-clean assert never fires** on a real restore under spec.
5. **H9** — a spill landing in the same round as a drafter-in-tick step. Still
   unobserved after K2; the spec gate exists for exactly that round.
6. **Rank-uniformity under the real TP=3 uneven-DCP geometry.** The tests pin
   that the *predicates* read only replicated state and that no collective is
   entered under a rank-local quantity. They cannot prove the collective COUNTS
   actually match at runtime — that needs a boot that does not hang, or a
   per-rank PG sequence-number probe. §2.3a narrows the source of a desync; it
   does not eliminate it.
7. **No NCCL hang / illegal access after a wave-back.**
8. **That the production desync is not on this path at all.** The observed
   crash was during hicache storage-prefetch, a different feature; §2.3a
   hardens the kvso path against the same class but says nothing about the
   crash's actual origin.

## 4. Not done, and why

* **The default does not flip.** It was bound to a K2 citation, and K2's
  restore cell was not evidence. The next boot can produce the citation now
  that the signal exists; flipping on this branch would repeat the mistake this
  branch documents.
* **PS2 x resume-under-spec** stays refused at arg-parse. Lifting it is #108
  (`--draft-kv-layout`), not this strand.
