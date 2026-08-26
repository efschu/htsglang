# DETERMINATION #898 — the standing test-failure bodies, named and rooted

**Determined 2026-08-26.** A determination is a SNAPSHOT. Every verdict below
is dated, and every verdict below is dated *2026-08-26* unless it says
otherwise. Before acting on any line here, check its age against the pin: a
sentence that was true on this date stops being true the moment the commit it
talks about is merged, reverted or superseded.

Base of every measurement in this document: **`be56c4ee1a`** (`#895` tip,
itself stacked on the shipping pin `0cd27d957d`), worktree
`/spinning/wt-898-failures`, branch `fix/898-standing-determination`.

Every run: `CUDA_VISIBLE_DEVICES=""`, `PYTHONPATH=<worktree>/python`,
interpreter `/spinning/htsglang-gpu/.venv/bin/python3` (`GATE_PY`). No boot, no
card, no serving touched.

## 0. INTERPRETER STAMP — and why the 2026-08-26 15:32 install does not reach

The `#897` strand installed `datasets 5.0.1` + `pandas 3.0.5` system-wide with
`--break-system-packages` at **15:32 UTC today**. Before that install a
`CustomTestCase` import collapsed at collect time, so a suite that used to be
uncollectable now collects and runs — which would make any A/B across that
instant compare two different collection worlds.

**It does not reach these measurements, and that is checked rather than
assumed:**

| | interpreter | datasets | pandas | package dir mtime |
|---|---|---|---|---|
| system | `/usr/bin/python3` | 5.0.1 | 3.0.5 | 2026-08-26 15:32 |
| **used here** | `/spinning/htsglang-gpu/.venv/bin/python3` | **5.0.0** | **3.0.3** | **2026-07-13 08:52** |

    $ .venv/bin/python3 -c "import sys;print([p for p in sys.path if 'dist-packages' in p] or 'NO dist-packages on venv sys.path')"
    NO dist-packages on venv sys.path

The venv does not have the system `dist-packages` on its path, so the install is
invisible to every run behind this document. The reproduction is the proof
rather than the argument: the first run here was taken at **15:58Z**, i.e.
*after* the install, and produced **exactly the 71** that `#894` reported at
~15:3x, i.e. *before* it. Had the install been effective, that is the number
that would have moved.

## 1. THE FOUR BODIES AT A GLANCE

| body | as reported | measured 2026-08-26 | overlap with the others | verdict |
|---|---|---|---|---|
| **#898** `test/registered/scheduler` | 71 on `2b13ba92d1`, both sides identical | **71 reproduced** (46 failed + 25 **errors**) | **disjoint** — different directory, no name in common | ONE regression (32) + one exclusion class (36) + 1 device + 2 genuine |
| **#815** `test/registered/unit/{managers,planner,server_args,mem_cache}` | 61 on `587e4c2e39` (2026-08-22) | see §4 | disjoint from #898 by directory | superseded: 53 of 61 fixed on the line, remainder re-rooted |
| **#864** `test_unified_radix_cache_unittest.py -k tp_swa_prefetch` | 16 standing, fail only WITH cards | **38 skipped hermetically (8+8+11+11), tally 38 == 38** | disjoint | ROOT KNOWN AND FIXED — but the fix is **ORPHANED** |
| **#899** pp-proxy / wraparound intermittents | 5 modules | already recorded at the base | 5 modules, all in `unit/managers` | RECORDED (#895); the shared-multiplier hypothesis is **falsified** |

**There is no set intersection to compute.** The four bodies live in four
different directories and share no test name. The interesting relation between
them is not overlap but **shape**, and three of the four turn out to be the
same shape: *production moved, the test that measured it did not, and nothing
in the gate world was watching that directory.*

## 2. THE COUNT ITSELF WAS WRONG-SHAPED — 46 failed + 25 errors

`71 = 46 failed + 25 errors`. The 25 are **setUpClass** failures, which pytest
reports in the `errors` column, not the `failed` one:

    46 failed, 530 passed, 1 skipped, 17 warnings, 25 errors, 2 subtests passed in 50.52s

Any extraction that greps `^FAILED` reports **46** here and reads a 25-test hole
as absence. This is the Extraktions-Zaehlprobe in its second form: the first
form was `SUBFAILED` breaking `^FAILED`, this one is `ERROR` never matching it
at all. `scripts/gate_partition_lib.py` already tallies **both axes**
(`names failed == summary failed` AND `names error == summary error`); hand
counts in circulation do not. Every number in this document was taken through
that two-axis gate.

## 3. #898 — `test/registered/scheduler`, 71 rooted

Reproduced at the base: **46 failed + 25 errors = 71**, in 50.5 s. Root causes,
extracted from the **chained** cause (`raise ... from e`) rather than from the
wrapper message — `CustomTestCase._callTestMethod` wraps every test in `retry()`
and 45 of the 46 read `Exception: retry() exceed maximum number of retries`,
which names nothing:

| # | class | root | verdict |
|---|---|---|---|
| **36** | **A — needs a live server** | `FileNotFoundError: No such file or directory: 'sglang'` at setUpClass | **EXCLUDED**, recorded (§3.1) |
| **32** | **B — `#856` seam contract** | `SeamOrderError("#856: no scheduler bound at the seam...")` | **REGRESSION, dated to one commit** (§3.2) |
| **1** | C — needs a device | `RuntimeError: No CUDA GPUs are available` | known desk class (#860 §0.7) |
| **2** | D — genuine test debt | source pin drift; a deleted premise | **FIXED here** (§3.3) |

36 + 32 + 1 + 2 = 71. Tally: names 71 == summary 46 failed + 25 errors.

### 3.1 Class A (36) — the launcher family, and why it is its OWN exclusion class

Seven modules call `popen_launch_server`: they start a **real server process**
out of process, load a model, and talk HTTP to it. At the desk that is refused
twice over — the hermetic run has no card, and the `sglang` console script is
not on the gate's PATH — so they die in `setUpClass` before one assertion of
their own runs.

    test_scheduler_control.py        11    test_retract_decode.py         6
    test_priority_scheduling.py       7    test_prefill_delayer.py        5
    test_load_snapshot_server.py      4    test_mixed_chunked_prefill.py  2
    test_routing_key_scheduling.py    1                                  ---
                                                                          36

Folding these into the existing `NEEDS_DEVICE` set would have been wrong in the
one direction that matters: a `NEEDS_DEVICE` module fails a **test**, a launcher
module fails at **setup**, and setup failures land in the other column. The
class that produces the 25 errors of §2 is exactly this class, so naming it also
repairs the count.

**Do not run these at the desk with the venv's `bin` on PATH.** Tried once
during this work: with `sglang` resolvable the module stops failing fast and
starts genuinely booting a server, and the box killed both that run and an
unrelated background battery. The gate's probe script deliberately does not
extend PATH; that is load-bearing, not incidental.

### 3.2 Class B (32) — `#856` deleted a premise and blinded the flip contract suite

Root, at `python/sglang/srt/managers/phase_flip_runtime.py:8452-8456`:

    scheduler = getattr(self, "_census_scheduler", None)
    if scheduler is None:
        raise SeamOrderError(
            "#856: no scheduler bound at the seam, so the residents cannot "
            "be retracted and the prefix tree cannot be dropped"
        )

The production guard is **correct and deliberate** (`_release_residents_for_cutover`
documents at length why a flip that cannot honour the no-carry contract must not
happen at all; `_census_scheduler` is bound at
`phase_flip_runtime.py:3507` on the boot wiring path, so production always has
one). What is defective is that the contract suite constructs
`PhaseFlipRuntime` directly — `_build_runtimes`,
`test_phase_flip_runtime.py:212-253` — and never binds a census scheduler,
because before `#856` there was nothing to bind.

**Dated by bisect, not inferred.** Same five files, same interpreter, hermetic:

    9fab2cc62e^  (8eed2a5aa7)   130 passed,  0 failed
    9fab2cc62e                   98 passed, 32 failed

`9fab2cc62e` = `[#856] The flip carries no KV: retract, drop the tree, rebuild
the plan empty`, **2026-08-24**. The five test files were last touched
2026-08-20 (`dd71976b442d`) and earlier. The commit did not touch them.

**What this cost.** The 32 are not incidental coverage. They are the `#631`/`#656`
flip contract itself:

    TestByteIdentity               pp->tp is byte-exact; the reverse round-trips
    TestWavedSeamOrdering    (11)  restore-before-release, waves do not interleave,
                                   reordering the seam changes no byte
    TestSeamWavesAreByteIdentical  one wave == the default split, byte for byte
    TestSharedArenaReadsPrecedeWrites, TestPreWriteSeamOrdering,
    TestConsensusDiscipline, TestAbortDeferral, TestEmptyLiveSetFlip
    + test_flip_frame_agreement_656 (5), test_flip_live_slot_agreement_656 (3),
      test_phase_flip_protocol (1), test_step6_harness (1)

From 2026-08-24 onward, every one of those properties has been unverified — and
`#856`'s own W27/W30/W31 history is a sequence of root defects found **on the
rig** (a retracted request still live in `resident_mamba_slots`; 78 released
requests dropped on the floor with every client timing out at 600 s). The suite
that existed to catch exactly that shape went dark in the same commit that
needed it.

**NOT a cheap fix, and named rather than smuggled.** `_build_runtimes` would
need a census-scheduler double carrying `tree_cache` (with a real `reset`),
`server_args`, `req_to_token_pool`, `token_to_kv_pool_allocator`, and it must be
FAITHFUL rather than convenient (#630): the seam calls `retract_all`, stamps
`seam_readmit_epoch`, consumes retracted refs out of the live universe,
re-admits into `waiting_queue` ordered by `kv_arrival_seq`, and reads
`_writeback_fence_ms`. A double that satisfies the `is None` check and nothing
else turns 32 reds into 32 greens that measure nothing — the exact failure mode
`#630` is named for.

**Estimate: 1 focused desk session (4-6 h).** One shared faithful double in the
test module, red-first per test class, plus a falsifier proving the double
actually exercises the retract/reset/re-admit order rather than short-circuiting
on an empty resident set. Owner should be whoever owns the `#856` seam, because
the double is a statement about that contract.

### 3.3 Class D (2) — genuine, cheap, FIXED here

**(a) `test_admission_limiter.py::TestThrottleBeforeRetractOrdering::test_source_calls_throttle_before_retract_decode`**

`ValueError: substring not found`. The test pins the *source text* of
`Scheduler.update_running_batch` to prove the admission throttle runs before the
retraction fallback. It searched for `batch.retract_decode(`. `82ba7e2c10`
`[#679]` (**2026-08-16**) moved that body behind
`self._retract_decode_and_requeue(`; the literal has not existed since. The test
file was last touched 2026-07-31.

The ordering it guards **still holds** in production (`throttle_before_retract(`
at offset 10487 of the block, `self._retract_decode_and_requeue(` at 10689). So
this was never a product failure — it was a pin that drifted and then reported
its own drift as a bare `ValueError` naming neither the literal nor the file.

Fixed two ways: the pin follows the call, and `str.index` is replaced by a
`self.fail` that names the missing literal, the file and the block size. Guard
falsified by mutation (pin set to a non-existent call ⇒ the new message fires).
49 passed / 1 failed → **50 passed**.

**(b) `test_phase_flip_runtime.py::TestSchedulerSideHelpers::test_quiescence_false_on_inflight_state`**

`AssertionError: True is not false : {'last_batch': namespace(reqs=[...])}`.
One row of the "these states block a flip" table asserted that a request
reachable only through `last_batch` — "not yet merged into the resident set the
carry harvests" — blocks the flip. `c2e69c22cd` `[#858]` (**2026-08-25**)
removed that gate from `build_flip_quiescence_fn` in as many words:

    # #858: THE ORPHAN GATE IS GONE, NOT NARROWED. [...] There is no
    # harvest: #856 retracts residents instead of carrying them, and
    # de4f541b41 made `_live_reqs` enumerate running_mbs, last_mbs,
    # running_batch AND last_batch -- the identical population.

The production commit did not touch the test. The replacement contract is
already pinned one test below, in the positive direction
(`test_last_batch_mirroring_the_resident_set_does_not_block_the_flip`). The
stale row is removed with its provenance written at the site so a later merge
does not blindly reinstate it. The file goes 23 → 22 failures; the remaining 22
are Class B.

### 3.4 VERZEICHNUNG — `test/registered/scheduler` had no gate at all

That is the structural reason 71 failures stood unrecorded: the partitioned
tier-2 gate's table covers `test/registered/unit/managers` and nothing else.
The runner already accepts `--table` and `--gate-path`; only a table was
missing.

**Rule added to `scripts/gate_partition_build.py`** (never a hand-edit of a
`.tsv` — the header forbids it and the runner re-checks each sha256):

    SERVER_LAUNCH_MARKERS = ("popen_launch_server",)
    def needs_live_server(source) -> str | None   ->  EXCLUDED
    reason: needs_server:launches_real_server:popen_launch_server

By **source probe**, not by name, for the reason the `RANKS` lane already gives:
a hand-kept list goes stale the moment somebody adds a module. Precision
measured before the rule was written — the marker fires on **7 of 34** modules
in `test/registered/scheduler` and on **0 of 668** in
`test/registered/unit/{managers,planner,server_args,mem_cache}`, so it **cannot
move a row in the existing managers table**.

**Table built:** `scripts/gate_partition_scheduler.tsv`, from a measured serial
reference plus 27 solo probes (the 7 launcher modules are excluded *before* the
solo lookup, so they are never executed).

    totals   {'PARALLEL': 20, 'RANKS': 7, 'SERIAL': 0, 'EXCLUDED': 7}
    verify   OK -- every module carries a verdict, every fast-lane verdict
             matches the bytes it was proved against

`SERIAL: 0` is itself a result: **every** non-launcher module produces solo
exactly the failure set it produces in the full serial run. None of the 33
remaining failures is crowding, order-dependence or a poisoned neighbour. They
are all real, and they all reproduce alone.

    gate run (post-fix)   wide 21.0s (20 mod, -n 8) | narrow 27.5s (7 mod, -n 4)
                          serial 0.0s | total 48.6s | 33 failing, all RECORDED
                          0 unrecorded, 0 partition violations

**This gate is RED by construction until §3.2 is fixed, and it must not be
promoted to a blocking gate before then.** Saying so is the point rather than an
excuse: what it buys today is not a green/red bit but the *discrimination* —
every one of the 33 is a RECORDED failure, so the moment a 34th appears the
runner classifies it (unrecorded ⇒ solo re-run ⇒ GENUINE or exit 6) instead of
letting it disappear into a number nobody owns. That is exactly the hole this
ticket exists because of. When the census-scheduler double lands, the table is
rebuilt from a fresh serial reference and the remaining `ref_failures` should be
one device row and nothing else.

## 4. #815 — the 61 of 2026-08-22, re-rooted at today's pin

*(measured 2026-08-26; the reference set of 2026-08-22 no longer exists as a
file — `/tmp/s16e/815_base_failed.txt` is gone — so the 61 are reconstructed by
name from the two COORD artefacts and then re-measured.)*

### 4.1 The 61 reconstructed by file (from `COORD-strand17b-merge-result.md` §NACHTRAG and `COORD-strand16e-820-815.md` §B)

| group | files × count | status on the line |
|---|---|---|
| 17b cluster 1 | `session_branch_rewind` 7 | FIXED `fe53663e1c` — **ancestor of the base** |
| 17b cluster 2 | `kv_arena_handle_retention_631` 7, `kv_arena_span_ops_631` 1 | FIXED `885162fbe4` — **ancestor** |
| 17b cluster 5 | `rejected_evidence_pins` 1 | FIXED `78d27da51d` — **ancestor** |
| #817 HOLD wrapper | `pp_progress_exit_677` 3, `phase_purity_631` 2, `drain_mode_tp_bundle_677` 2, `flip_reachability` 1 | FIXED `2c46ee972f` — **ancestor** |
| 16e stub drift | `scheduler_chunked_req_gate` 3, `collective_family_siblings_610` 3, `evict_rung_floor_invariant_717` 3, `acceptance_emitters_758` 3, `mamba_anchor_seams_747` 4 | FIXED `94b8b58aee` — **ancestor** |
| 16e stale tests | `vacuous_decode_exit_730` 6, `kvso_worker_stop_673` 1, `localslot_family_756` 1 | FIXED `94b8b58aee` — **ancestor** |
| 16e pp pair | `pp_slot_last_batch_631` 3, `pp_admission_wraparound_never_blocks` 2 | FIXED (16e §B.5) |
| deliberately open | `pp_flip_slot_hold_631` 7 | left open with reason (#791 test surface) |

Sum 7+7+1+1+8+16+8+5+7 = **61**. All five fix commits verified as ancestors of
`be56c4ee1a` by `git merge-base --is-ancestor`, individually.

### 4.2 The sub-item the brief flagged: `test_pp_presence_withholding_deadlock_800.py`

Reported as failing on the base while an inventory led the suite green.
**Measured today, 3 consecutive solo runs, hermetic: 27 passed, 27 passed, 27
passed.** Its `sha256` still matches the `#868` proof (`881563b75486a3af`) and
the gate table records it `PARALLEL` with an **empty** `ref_failures` — i.e. it
also produced zero failures in the `#868` serial reference. The inventory was
right; the 2026-08-22 red is superseded and is not a standing failure.

### 4.3 The battery re-measured — 61 → 18, and 17 of the 18 are one class

Same four directories, same interpreter, hermetic, one process, 2026-08-26:

    587e4c2e39   2026-08-22    61 failed / 8245 passed / 1852 skipped   960 s
    be56c4ee1a   2026-08-26    18 failed / 9958 passed / 1852 skipped  1008 s
                               12959 collected, 0 errors, 1131 subtests passed

Tally: extracted names 18 == summary failed 18, extracted errors 0 == summary
errors 0. (The +1713 passed is four days of new tests, not healing.)

The 18, rooted:

| root | count | modules |
|---|---|---|
| `RuntimeError: No CUDA GPUs are available` | **17** | `test_arena_high_water_631` 7, `test_phase_flip_rotation_wiring_809` 4, `test_restore_never_rebuild_677` 4, `test_acceptance_emitters_758::RefillTiming` 2 |
| evidence cite drifted | **1** | `test_rejected_evidence_pins` (§4.4) |

**The device 17 are not new and 15 of them are already recorded**: the first
three modules are the `#868` `NEEDS_DEVICE` set, `EXCLUDED` in
`scripts/gate_partition.tsv` with those exact test names as `ref_failures`.
This is the known desk baseline (`NOTE #860 §0.7`), and it is the same baseline
`#887` measured as "5 pre-existing" over its narrower family run.

**Two of the 17 are NOT recorded, and could not be**:
`test_acceptance_emitters_758::RefillTiming` (2) lives in
`test/registered/unit/mem_cache/`, which — like `test/registered/scheduler`
before §3.4 — **has no partition table at all**. Same structural hole, third
directory. Naming it is the honest action here; building a third table is a
separate measurement campaign (`mem_cache` alone is 1774 passed / 1658 skipped
and a `#862`-shaped darkness question sits on top of it).

**So: of the 61, 53 are gone (all five fix commits verified ancestors of the
base), 17 device failures were never test debt at all, and exactly ONE
addressable failure remains** — and it is the one the register's own drift
detector was built to catch.

### 4.4 The one genuine #815 survivor — a cite that has now drifted FIVE times

`test_rejected_evidence_pins::PpWithSpecEvidenceTest::test_evidence_cites_land_on_BOTH_halves_of_the_guard`:

    AssertionError: 'pp_size > 1' not found in
      '            and getattr(self, "_declarations_materialized", False)...'

The `pp_with_spec` row in `python/sglang/srt/planner/rejected.py:367` cited
`server_args.py:19269` / `:19284`. The guard actually sits at **`:19436`** (`if
self.pp_size > 1:`) and **`:19451`** (`assert self.speculative_algorithm is None
or self.enable_phase_flip`) — verified by reading the block, not by search.

**The re-pin ledger, because the count is the finding:**

    #625              :11214           -> :16240-16245
    #815 78d27da51d   :16240-16245     -> :18958/:18973
    #810 a09e71f4a4   re-pinned again
    #837 ddf009c43f   -> :19269/:19284
    #898 (here)       :19269/:19284    -> :19436/:19451

Five drifts, five hand re-pins, of ONE row. `#815` fixed this exact test four
days ago and it is red again. **The instance fix is not the class fix**, and
saying so is the point of writing it down.

* **INSTANCE, fixed here:** the cite now names `:19436`/`:19451`. 1 failed / 5
  passed → **6 passed**.
* **CLASS, named:** an absolute line number kept in prose about a file that
  grows above it will drift again, on a schedule set by how often
  `server_args.py` is edited. The real fix is a landmark-based citation format
  (cite the symbol or the assert text, resolve the line at read time), which
  touches `_LINE_REF`, every code-site row in the register, and every reader of
  the register. **Estimate: half a desk session** for the format plus a
  migration of the code-site rows, and it removes a recurring red permanently.
* **What was made free in the meantime:** the test that notices the drift
  already knows how to find the guard, so it now *computes and prints the
  correct pair* in its failure message instead of leaving the next reader to
  hunt through 19 000 lines. Falsified: with the cite wrong the message reads
  "the guard now sits at server_args.py:19436 ... and server_args.py:19451".
  That turns each future re-pin from a hunt into a copy.

## 5. #864 — root known, fix written, **fix orphaned**

`#864`'s 16 are not a new investigation: `NOTE_864_862_invocation_decides.md`
(2026-08-26 00:13) already determined them, and the determination holds under
re-measurement.

**Reproduced today, exactly:**

    pytest test_unified_radix_cache_unittest.py -k tp_swa_prefetch   (CVD="")
    -> 38 skipped, 1464 deselected, 0 passed, 0 failed, 7.07s
       SKIPPED [8] :2481 requires a real accelerator
       SKIPPED [8] :2457 requires a real accelerator
       SKIPPED [11] :2481 SWA-only fixture required
       SKIPPED [11] :2457 SWA-only fixture required
    tally 8+8+11+11 = 38 == summary 38

The 16 are the two 8s. They are the **inverse** of `#868`'s 15: hermetic they
SKIP, with cards they FAIL — a green standing over a red, which is the direction
nobody sees.

**Cause, re-verified in this tree rather than quoted:**

    production  python/sglang/srt/mem_cache/hiradix_cache.py:266-278
                torch.distributed.all_reduce(tensor, op=op, group=group, async_op=True)
    the double  test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py:2382
                def fake(tensor, op=None, group=None):        <- no async_op

`grep -n "async_op" test_unified_radix_cache_unittest.py` → **no hits at the
base.** The double cannot accept the production call; all 16 die in `mock`'s
dispatch before their first assertion. TEST-ONLY, zero live defects, and `#783`
gets nothing from them in either direction.

**THE FINDING OF THIS SECTION IS THAT THE FIX IS NOT ON THE LINE.**
`a428907a43` `[#864/#862]` fixes it — one keyword — and adds
`scripts/gate_double_drift.py` (an AST gate that decides drift by parsing, no
device) plus `test_double_signature_drift_864.py`. Verified at the base:

    scripts/gate_double_drift.py                     ABSENT
    the double at :2382                              still one keyword short
    branch containing a428907a43                     probe/864-862-desk only
    on the shipping line                             NO -- orphaned

Textbook PRESENT-ABER-UNVERDRAHTET, on a leaf. The remedy is a **merge order,
not a re-fix** — writing a second version of the same one-line change here would
create exactly the divergent-fork error `#820` recorded (two versions of the
same twenty lines).

**The merge order is proven actionable, not merely proposed.** Test cherry-pick
onto `be56c4ee1a` in a scratch worktree, then reset:

    git cherry-pick --no-commit a428907a43     -> CLEAN, no conflicts
      (its scripts/gate_partition_lib.py is byte-identical to #868's, 4371 B,
       so that half of the commit drops out silently)
    pytest test_double_signature_drift_864.py  -> 2 passed, hermetic

**Still owed after the merge, and it is `#864`'s own reservation, not a new
one:** the 16 have never been run WITH cards. The fix is proved to remove the
drift; it is *not* proved to make them pass. One card run settles it.

## 6. #899 — recorded already; the cheap class rule is FALSIFIED

The recording half is **done at the base**: `#895` (`be56c4ee1a`) put all five
modules in `NOT_CROWDING_PROVABLE` with the reason
`simultaneity:NOTE868_2.5_not_solo_provable;895_observed_2026-08-26`, and they
carry that verdict in `gate_partition.tsv` (lines 199, 221-224). Nothing to
record here.

The open question was whether the *class* rule is cheap — "e.g. one shared
deadline multiplier from the environment". **It is not, and that is measured
rather than estimated:**

| module | deadline literals |
|---|---|
| `test_pp_proxy_readiness_contract_789.py` | 13 |
| `test_pp_admission_wraparound_never_blocks.py` | 8 |
| `test_pp_proxy_readiness_rendezvous_789.py` | 1 |
| **`test_pp_proxy_cross_epoch_mispair_795.py`** | **0** |
| **`test_pp_proxy_retracted_pass_mispair_791c.py`** | **0** |

Two of the five have **no deadline of their own to multiply**, and the observed
failure of `795` was inside the gloo rendezvous (`connectFullMesh: Connection
closed by peer`) *before* any assert of its own — a multiplier cannot reach it.
And for `789` the short budget is **the specimen**: `SHORT_READINESS_BUDGET_S =
0.4` exists so the timeout FIRES, and the RED case asserts `"#789 PROXY
READINESS TIMEOUT"` appears. Scaling it up would disarm the test it is meant to
protect. There is also no shared harness — each module carries its own
module-level constants — so "one multiplier" would be five edits with three
different meanings.

### 6.1 NEW OBSERVATION — the class is NOT confined to the narrow lane

The two desk-gate runs taken to accept this branch both came back **exit 6**,
each with a **different** unrecorded failure, neither reproducing alone, while
the box carried foreign load from other strands:

| run | failing test | lane | solo re-run | box load (1/5/15 min) |
|---|---|---|---|---|
| 1 | `test_pp_void_send_contract_801.py::TheRingSurvivesAVoidWithNoRetraction::test_the_ring_completes` | **narrow** | rc=0, 14 passed, 56.1 s | 6.8 / 15.9 / 15.9 |
| 2 | `test_load_snapshot_backends.py::TestZmqRoundTrip::test_read_returns_latest` | **wide** | rc=0, not reproduced | 41.9 / 40.3 / 28.8 |

`0 genuine, 1 not reproduced, 0 inconclusive` on both runs. Neither module is
touched by this branch.

This is `#895`'s pattern to the letter — two consecutive runs, a different
member each time, external load on a shared box — **with one extension that
matters: run 2's member is in the WIDE lane.** `#895` recorded the class inside
the rank-spawning family and reasoned about it as a rank-starvation hazard.
`TestZmqRoundTrip::test_read_returns_latest` spawns no ranks; it is a ZMQ round
trip with a read deadline. So the independent variable is the load on the box,
not the presence of ranks, and the narrow lane's worker bound cannot be the
general answer.

**Deliberately NOT demoted.** A different member each run is the signature of
the box, not of a module; pinning the two that happened to lose would be
treating the symptom, and `#895` says so in the builder itself ("this is a rate
reduction, not the class fix"). Demoting wide-lane members on a single
observation would erode the 269-module lane that is the gate's reason to exist.
What the runs did do is exactly what `#895` built them to do: classify by
machine, refuse to forgive, and hand back a named exit instead of a number.

**Verdict: named, not built.** What is actually owed is the marker `NOTE #868
§6` already names — a per-module declaration of its wall-clock margin, so the
narrow lane can refuse a module whose margin is smaller than the lane's own
scheduling jitter, instead of discovering it a failure at a time. That is a real
piece of design, ~1 desk session, and it subsumes the multiplier idea properly.
Until then the `#895` runner classifies each new member by machine on the run it
happens (solo re-run, exit 6), which is the rate-reduction the base already
ships.

## 7. WHAT WAS DONE, IN ONE BLOCK

    FIXED (red-first, each falsified)
      test_admission_limiter.py            pin follows #679's rename + loud failure
      test_phase_flip_runtime.py           #858's deleted premise removed from the table
      planner/rejected.py + its pin test   5th cite drift re-pinned; the detector
                                           now PRINTS the correct pair

    RECORDED (by rule in the builder, never by hand-editing a .tsv)
      gate_partition_build.py              needs_live_server() source probe -> EXCLUDED
      gate_partition_scheduler.tsv         NEW table for test/registered/scheduler
                                           20 PARALLEL / 7 RANKS / 0 SERIAL / 7 EXCLUDED

    NAMED, WITH AN ESTIMATE
      #898 class B   32 flip-contract tests dark since 9fab2cc62e (2026-08-24)
                     -> faithful census-scheduler double, 1 desk session (4-6 h)
      #864           a428907a43 is orphaned on probe/864-862-desk; cherry-pick
                     onto the line is CLEAN and its guard is green -> merge order
      #899           the shared-multiplier rule is falsified (2 of 5 have no
                     deadline); the owed piece is NOTE #868 §6's margin marker.
                     NEW: the class also hit the WIDE lane (no ranks involved),
                     so the narrow lane's worker bound is not the general answer

    ACCEPTANCE
      desk gate      2 runs, exit 6 both, 0 genuine / 1 not-reproduced each,
                     different member each time, foreign load 15.9 -> 40.3.
                     Neither module touched by this branch.
      scheduler gate 33 failing, ALL recorded, 0 unrecorded, 0 violations
      ruff F401,F821,UP037   clean on every changed file
      black                  exact parity with the base (no new finding)
      codespell              no hits

      #815 class     absolute line numbers in register prose drift on a schedule
                     -> landmark-based citation format, ~half a desk session
      mem_cache      third directory with no partition table (2 unrecorded
                     device failures found there) -> own campaign, #862-shaped

    SUPERSEDED
      #815           61 -> 18; 53 fixed, all five fix commits ancestors of the
                     base; 17 of the remaining 18 are the known device baseline
      #800 deadlock  test_pp_presence_withholding_deadlock_800.py: 27 passed 3/3

## 8. THE ONE SENTENCE

Three of the four bodies are the same defect wearing different clothes: a
directory nobody's gate was watching. `test/registered/scheduler` had no table
(71), `unit/mem_cache` has none (2 of the 18), and `#864`'s finished fix sits on
a leaf branch nobody's ancestry check covers (16). The 32-test hole from `#856`
is what that costs when it is a contract suite that goes dark, and the four days
between 2026-08-24 and today are the interval in which the seam's own W27/W30/W31
defects were found on the rig instead.
