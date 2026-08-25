# NOTE #868 — the partitioned tier-2 gate, and what #860 actually measured

Runner: R8. Tree: `/spinning/wt-868-partition`, branch `probe/868-partition`,
base **9164adb942** — deliberately the SAME base as `NOTE_860_xdist.md`, so
every failure set in this note is directly comparable to that note's.

**Every run in this note used `CUDA_VISIBLE_DEVICES=""`, verified on the
PROCESS and not on the command line** (`tr '\0' '\n' < /proc/<pid>/environ |
grep CUDA_VISIBLE_DEVICES`), for the serial reference, for the solo sweep, for
the sibling sweep and for every lane of the runner. No run here initialised a
CUDA context. Where a number in this note is a WALL TIME rather than a failure
set, the box conditions are named next to it, because the two are not equally
robust to load: a failure set of CPU-only unit tests is not, and a wall time is
entirely, a function of what else was running.

---

## RESULT IN ONE BLOCK

    partition          270 wide | 36 narrow | 0 serial | 3 excluded (need a card)
    equivalence        PROVED, both directions, twice: serial and partitioned
                       both 0 failures over the same 306 modules
    speedup            2.22x   (766.62 s serial mean -> 345.78 s partitioned mean,
                                quiet box, A-vs-A floor 0.07 % / 0.65 %)
    planted mutant     CAUGHT, 3 named tests, in the wide lane
    #860 divergence 1  NOT a test dependency -- it is `--dist loadscope`
    standing 15        entirely the desk gate's own non-hermeticity, 0 real defects

The limiter is named rather than smoothed: 35 rank-spawning modules, 11 % of
the suite, carry 62 % of the gate's wall time. That is why this is 2.22x and
not #860's 3.39x, and the 1.2x between them is the price of not shipping that
family crowded (§4.4).

---

## 0. TWO CORRECTIONS TO #860, NEITHER OF THEM SILENT

`NOTE_860_xdist.md` is not edited. Both corrections are recorded here with the
command that produced them.

### 0.1 The suite has 309 test modules, not 312

312 is the count of `*.py` files in `test/registered/unit/managers`. Three of
them are helpers that pytest never collects, because they do not match
`test_*.py`:

    test/registered/unit/managers/_regime_shutdown_child.py
    test/registered/unit/managers/mutation_proof_739.py
    test/registered/unit/managers/mutation_proof_800.py

`ls test/registered/unit/managers/test_*.py | wc -l` → **309**. Both numbers
are right about different things; they are not a contradiction.

### 0.2 Divergence 1 is NOT an inter-module dependency. It is `--dist loadscope`.

This is the load-bearing correction, and it removes one of #860's two
disqualifiers outright.

`NOTE_860 §1` attributes four failures in `test_chunked_commitment_701.py`
(`ValueError: Global server args is not set yet!`) to "a process-global that
some OTHER module initialises", and reasons that under `loadscope` the module
lands on a worker where no earlier module set it. `§0.5` justifies the choice
of `loadscope` in exactly these words: *"loadscope keeps every MODULE whole on
one worker, so a worker's view of one module is exactly the serial view of
it."*

**That premise is false for this suite, and one module in isolation proves
it.** All three arms hermetic, ~30 s total:

| arm (that ONE module, nothing else) | result |
|---|---|
| serial | **17 passed** |
| `-n 2 --dist loadscope` | **4 failed**, exactly the four in #860 §1 |
| `-n 2 --dist loadfile` | **17 passed** |

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<tree>/python .venv/bin/python -m pytest \
      test/registered/unit/managers/test_chunked_commitment_701.py \
      -q -p no:randomly -p no:cacheprovider --color=no -rfE -n 2 --dist {loadscope,loadfile}

A module cannot depend on a module that is not in the run. The four failures
are produced by the engine's grouping unit and by nothing else.

**Root cause.** xdist's `loadscope` groups by module for plain test
*functions* but **by CLASS for test methods**. This suite is
unittest/`TestCase` based throughout, so `loadscope` splits single FILES across
workers along class boundaries. `test_chunked_commitment_701.py:90` calls
`set_global_server_args_for_scheduler(...)` in the setup of ONE class; the
class `TheTwoActorDeadlockAcrossPasses` does not, and inherited the global from
its sibling class in the same file. Put the two classes on different workers
and it raises.

`--dist loadfile` groups by file. That is the unit #860 believed it had
selected, and it is also exactly the unit a solo run measures.

The grep that found the setter took seconds and replaced a planned bisection:
`grep -n global_server_args test/registered/unit/managers/*.py` names six
modules, and the setter for the failing class is in the SAME file as the
failure.

**Consequence for the verdict.** #860 refuses xdist on two disqualifiers.
Divergence 1 is not a property of the tests at all — it needs no test fixes,
no "finite, named list of what must be fixed first". It needs `--dist
loadfile`. Only divergence 2 (crowding) survives, and it is handled here by
lane width rather than by refusal.

### 0.3 Sibling sweep for the class — measured, not estimated

The class of the defect is: **the parallel engine's grouping unit was assumed
to be the file and is in fact the class.** Its siblings are every module with
more than one test class where one class uses state a sibling class in the same
file establishes. That set is measurable directly — run each module ALONE under
`-n 2 --dist loadscope` and diff against its solo-serial result; with only one
module present, any difference is intra-file by construction.

**Sweep complete: 309 of 309 modules measured, 0 unparsable.** Two modules
differ, and only one of them is this class:

* `test_chunked_commitment_701.py` — the four tests of §0.2. Genuine: a class
  inherits `global_server_args` from a sibling class in the same file.
* `test_pp_wedge_watchdog_is_honest_821.py::TheIdleBootMustNotBeKilled::test_a_block_just_under_the_budget_does_not_read_as_active`
  — **NOT this class.** Read the test and it sets
  `blocked_since = time.monotonic() - (WATCHDOG_TIMEOUT_S - 2.0)` and asserts
  the watchdog does not yet read active. That is a two-second wall-clock
  margin, and my own sweep was loading the box; the margin was eaten between
  setup and assertion. A load artefact of the instrument, not a finding about
  the module.

So **the loadscope class has exactly one member in this suite**, and it is the
one #860 found. The class is real, narrow, and now retired by refusing the
distribution mode.

**A limitation this second hit exposes, stated because it is load-bearing for
the lane split.** The lane heuristic sends a module to the narrow lane when its
SOURCE spawns real ranks. `test_pp_wedge_watchdog_is_honest_821.py` does not
spawn ranks — it is in the WIDE lane — yet it carries a two-second wall-clock
margin and can therefore flake red under pressure just as a rank-spawner can.
Spawning ranks is a sufficient marker for crowding sensitivity, not a necessary
one. The consequence is bounded and is a cost rather than a correctness hole:
crowding makes these tests fail, never pass, so the failure mode is a flaky RED
that costs a re-run, and the runner's recorded-failure check still covers the
false-green direction. Tightening the marker (a source probe for
`time.monotonic`/`sleep` margins) is the obvious next increment and is not done
here.

**The check that retires the class:** the runner refuses `--dist loadscope`
outright. The lane width is a knob; the distribution mode is not.

---

## 1. THE ADMISSION CRITERION, AND WHY IT IS A PROOF

The naive partition — run everything parallel, push whatever diverges into a
serial lane — catches only the harmless direction. #860's own finding names two
DUAL hazards: a module that POISONS the process (`#249`), and a module that
DEPENDS on being poisoned (`global_server_args`). The dependent one fails
loudly when separated. The poisoned one does the opposite: separate victim from
poisoner and the victim PASSES, so a real serial failure becomes a parallel
green. No number of parallel repeats can see that, because every repeat has the
same separation.

So admission is a measurement with a fixed comparison, not an observation:

> **A module is admitted to a parallel lane iff, ALONE in a fresh process, it
> produces exactly the failure set it produces inside the full serial run.**

One measurement, both directions:

* fails solo, passes serially → sets differ → refused (the dependent);
* passes solo, fails serially → sets differ → refused (the victim);
* same set → admitted.

Solo runs may themselves run in parallel with each other: they are isolated by
construction, which is what makes the sweep affordable — 309 modules in about
15 minutes at `-P 6` instead of 309 sequential runs.

### 1.1 The criterion is necessary but not sufficient — the closure

Admission is decided per module; the CUT changes the composition of the
remaining lane as a side effect. If a module in the serial lane depended on
state a module in the parallel lane established, the serial lane no longer
supplies it, and the module fails there when it passed in the full serial run.
That is a false RED — the safe direction, but it breaks the equivalence claim
just as surely.

The partition must therefore be CLOSED under the dependency relation: pull the
provider into the lane with its dependent, and repeat to a fixed point, because
the pulled-in provider may itself have a provider. Cost is bounded by finding
the provider cheaply — read the dependent's own error message and grep for the
call that establishes it — with bisection only as a fallback.

**On this tree the closure terminates at iteration 0: no module in the serial
lane failed for want of state a parallel-lane module provides.** The one
candidate for that shape, `test_chunked_commitment_701.py`, turned out to be
§0.2 and is fully parallel under `loadfile`.

### 1.2 The tally gate, before every set difference

Every log is parsed with the extraction check the set difference depends on:
the number of names pulled out must equal the number the log's own summary
reports, or the EXTRACTION is broken and no conclusion may be drawn.
`scripts/gate_partition_lib.py` strips ANSI before matching, accepts
`FAILED|SUBFAILED|ERROR`, and finds the summary by PATTERN rather than by
position — teardown output pushes it off the last line. (This suite emits
`356 subtests passed`, so the `SUBFAILED` case is live, not theoretical.)

---

## 2. THE MEASUREMENT

### 2.1 Serial reference

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<tree>/python .venv/bin/python -m pytest \
      test/registered/unit/managers -q -p no:randomly -p no:cacheprovider \
      --color=no -rfE

**15 failed, 4111 passed, 18 skipped, 356 subtests passed.** The failure count
and the pass count match `NOTE_860 §1`'s serial arm exactly (15 / 4111).

All 15 failures are the three device-requiring modules of `NOTE_860 §0.7`, and
the arithmetic closes exactly:

    test_arena_high_water_631.py            7
    test_phase_flip_rotation_wiring_809.py  4
    test_restore_never_rebuild_677.py       4
                                           --
                                           15   = the entire standing failure set

**The standing 15 of the managers gate under `CVD=""` is ENTIRELY the harness's
own non-hermeticity. It contains ZERO real defects.** That matters beyond this
ticket: a merge landing that diffs 15 against 15 and reads "clean" is correct as
a DIFFERENCE, but the base it is diffing was never a set of failures — it is a
property of how the gate is invoked. Nobody should go looking for 15 reds that
do not exist, and nobody should treat a change in that number as a code
regression before checking whether the invocation changed.

The same three modules PASS with cards visible (`NOTE_860 §0.7`), which is what
identifies them as harness artefacts rather than defects.

The wall time of this run (888.46 s) is NOT a baseline: a foreign gate and my
own solo sweep ran across it. It is recorded as a labelled data point only.

### 2.2 Solo sweep — 309 modules, one fresh process each

**308 of 309 modules produce solo exactly the failure set they produce
serially.** One module differs, and the difference is instructive rather than
structural — see §2.4.

### 2.3 The partition table

`scripts/gate_partition.tsv`, generated from the two logs by
`scripts/gate_partition_build.py`, one row per module:

    module <TAB> verdict <TAB> reason <TAB> sha256 <TAB> ref_failures

| verdict | count | meaning |
|---|---|---|
| `PARALLEL` | 270 | proof holds; wide lane, `-n 8 --dist loadfile` |
| `RANKS` | 36 | proof holds, but the module spawns real ranks; narrow lane with a bounded worker count |
| `SERIAL` | 0 | proof did not hold; one process, one order |
| `EXCLUDED` | 3 | needs a real card, cannot run at the desk at all |

The `SERIAL` column is empty on this tree — not by assumption, by 309
measurements. It is kept in the schema because it is where the runner puts
anything unproven, and because the next tree may need it.

The lane split for `RANKS` is a CROWDING decision, not a correctness one, and
it is read from the source (`multiprocessing`, `torch.distributed`,
`init_process_group`, `_regime_shutdown_child`) rather than from a hand-kept
name list that goes stale on the next module added. `NOTE_860 §FUTURE-CHECK 3`
asks for this family to be named in the runner rather than remembered; it is
named in the DATA, which the runner prints on every run.

The bound has a physical basis rather than a feeling: a module in this family
occupies its controller plus its ranks, so the lane's peak process count is
about `workers × (1 + ranks)`. Kept below the core count, no rank is starved
past the timeout the test is actually measuring. The lanes run one after
another for the same reason — a wide lane running alongside the narrow one
re-creates precisely the pressure the narrow lane exists to avoid.

### 2.4 The one module that failed its proof, and the instrument that mismeasured it

`test_pp_admission_wraparound_never_blocks.py::PPAdmissionWraparoundBlocks::test_blocking_wraparound_wedges_the_ring`
fails solo and passes in the full serial run.

Read the failure and it is not a state dependency at all:

    AssertionError: 'wraparound-check mode=blocking' not found in
    'i=0 sending decision (mode=blocking)'
    : PP0 must be captured mid wraparound-check:
      {0: 'i=0 sending decision (mode=blocking)', 1: '<no progress recorded>', 2: '<no progress recorded>'}

The test spawns three real ranks and asserts WHERE each one is standing at a
deadline. Under CPU pressure PP0 had not yet reached the wraparound check. Both
solo measurements of this module were taken while my own sweeps were loading
the box — so the instrument, not the module, produced the divergence.

**Stated as a defect in my own measurement rather than repaired into a
result.** The module is a rank spawner and its natural lane is `RANKS`; it was
parked in `SERIAL` — the slow lane, never the fast one — until its solo probe
could be re-taken on a quiet box.

**RESOLVED, §7 ARM 0.** Re-probed alone on a confirmed-quiet box: **3 passed in
31.02 s**, which equals its serial reference set. The proof holds; the earlier
divergence was the instrument. It is now `RANKS`, and with it the table has
**no `SERIAL` rows at all** — every module in the gate path either proved
independent or needs a card.

**Independently confirmed, from the other side, by a different runner.** R7's
managers landing run on a separate tree returned 17 instead of the standing 15,
under a live serving instance, at 1097 s against a 716 s baseline (+53 %). Two
names were added, and one of them is this module. R7 had the observation
without a mechanism; this note has the mechanism without a second observation.
Together they are a load sensitivity that is established rather than suspected,
so the table row's reason is stated precisely: *a load-sensitive multi-rank
deadline assert, observed independently twice, re-probe on a quiet box
outstanding* — not *fails solo, cause unknown*. A reader can tell from that
whether to trust the row.

### 2.4b Three runners, three failure sets, one shared member

Three independent runs of this suite, in three trees, under three different
kinds of load, produced three non-baseline failure sets on top of the standing
15:

| runner | form | non-baseline failures |
|---|---|---|
| R7 | serial, live serving instance on the box, 1097 s vs 716 s baseline | 2: `pp_admission_wraparound_never_blocks`, `pp_output_ring_retraction_wedge_791b` |
| R9 | `-n 8 --dist loadfile`, whole suite, no partition, 290 s | 3: `pp_void_send_contract_801`, `pp_flip_leftover_proxy_757`, `pp_admission_wraparound_never_blocks` |
| this note | solo sweep at `-P 6` | 1: `pp_admission_wraparound_never_blocks` |

**Every name in all three sets is a rank spawner, and every one is in the
narrow lane of this table.** R9's three and R7's two all pass serially in
isolation on a quiet box; so does the shared member (§7 ARM 0, 3 passed /
31.02 s).

The intersection of the three sets is exactly one module — the one that also
happens to be the most load-sensitive of them. That is what a crowding class
looks like when it is sampled three times: the membership varies with the
pressure, the FAMILY does not. It is `NOTE_860 §1`'s divergence 2 observed a
third and fourth time, and it is why the family gets a bounded lane instead of
a full-width one.

**R9's run is also the counterfactual for the partition** — and it should be
read knowing where it came from. **I did not construct it.** R9 ran the whole
suite at `-n 8 --dist loadfile` in a different tree, for its own reasons, while
this ticket was still measuring. It happens to be the exact control arm this
note would otherwise have had to argue for without evidence: same engine, same
`loadfile`, same suite, **no partition table**.

    no partition:   3 spurious reds in 290 s
    partition:      0 spurious reds in ~345 s

The 55 s is the price, and it buys the difference between a gate that must be
re-run serially to interpret its red and one that does not. A control arm that
falls into your lap is worth more than one you design, precisely because
nobody chose its conditions to suit the conclusion — but for the same reason
its conditions were not controlled by anyone either, so it is corroboration
and not a measurement of this note's making.

### 2.5 The crowding reason has a NAME, and it is narrower than "spawns ranks"

R7's second added name was
`test_pp_output_ring_retraction_wedge_791b.py`. In this table it is already
`RANKS` (`solo_equals_serial`, 4 passed solo, 4 passed under the sibling
probe) — the source marker routed it to the narrow lane before anyone observed
it failing.

Both of R7's names, and §2.4's, share a shape that is tighter than "the module
spawns processes": **they assert on SIMULTANEITY across real ranks at a
deadline** — where each rank is standing when the clock runs out. No amount of
isolation makes such an assert indifferent to CPU pressure, because CPU
pressure is the independent variable it is implicitly measuring.

That is a third admission-refusal reason in its own right, and it deserves its
own name rather than being folded into either of the other two:

| reason | shape | detection |
|---|---|---|
| coupling | the module's result depends on process state a neighbour supplies | solo failure set != serial failure set |
| grouping | the engine splits the file below the unit the proof was taken in | one module alone, `loadscope` vs `loadfile` |
| **simultaneity** | the assert is about where concurrent ranks stand at a deadline | passes isolated and quiet, fails isolated and crowded |

The third is the only one of the three that a single measurement cannot settle,
because its independent variable is the box. Its container here is the narrow
lane, and the lane's worker bound is the actual mitigation.

---

## 3. THE RUNNER — `scripts/gate_tier2_partitioned.py`

    scripts/gate_tier2_partitioned.py                     # the gate
    scripts/gate_tier2_partitioned.py --serial-only       # the canary form
    scripts/gate_tier2_partitioned.py --verify            # table vs tree, runs nothing
    scripts/gate_tier2_partitioned.py --prove <ref.log>   # acceptance, both directions

It keeps the three properties `gate_tier2.sh` exists for — forced `CVD=""`, a
`PYTHONPATH` derived from the script's own location, a verdict printed as a
sorted failure SET rather than a count — and adds the four the partition needs.

**1. It reports its own exclusions, before the result.** `NOTE_860 §0.7
consequence 1` requires it in as many words: a gate that quietly drops three
modules is read as the full gate by the next person. The report names the
device-requiring modules, every module demoted at run time, and the composition
of the serial lane.

**2. It refuses to promote an unproven module.** Not in the table, or bytes no
longer matching the sha256 the proof was taken against → serial lane, named in
the report. This is the check that answers "what stops a NEW module from
silently landing in the parallel lane": nothing lands there without a proof
taken against its exact bytes, and `--verify` asks that question for the whole
tree in under a second, without running a test. Drift becomes a desk finding
instead of a wrong verdict at 03:00.

**3. It refuses to lose a known failure.** Every row records what that module
fails in the serial reference. A recorded failure absent from the run is a
PARTITION VIOLATION and fails the gate — this is the false-green direction, the
one repeated parallel runs are structurally blind to, checked arithmetically on
every single run.

**4. It refuses `--dist loadscope`.** §0.2 is a property of the suite, not of a
particular invocation. The lane width is a knob; the distribution mode is not.

The two directions are deliberately split between two modes. In normal gate use
an ADDED failure is the gate doing its job, so only the lost-failure direction
is fatal. Under `--prove` the union must EQUAL the reference set in both
directions — an added failure there means the partition changed the answer, and
that is the acceptance question.

### 3.1 Every guard above was shown to FAIL, not merely to be written

A guard that has never fired is a guard that might be inert. Each was provoked:

| guard | how it was provoked | result |
|---|---|---|
| refuses `loadscope` | `gate_tier2_partitioned.py -- --dist loadscope` | REFUSED, rc 2, with the reason |
| unclassified module | `touch test_zz_868_guard_probe.py`, then `--verify` | UNPROVEN, rc 1, module named |
| stale proof | appended one comment line to a `PARALLEL` module, then `--verify` | "proof stale: module bytes changed", rc 1 |
| **partition violation (false green)** | a probe table recording a failure that does not exist, on a module that passes | MISSING FAILURE named, **rc 4** |

The tree was restored after each; `git status` is clean on `test/` and
`python/`.

The fourth is the one that matters most, because it is the direction no amount
of parallel repetition can see, and it is now known to fire rather than assumed
to.

---

---

## 4. THE ARMS THAT WERE RUN, AND WHAT THEY SAY

Both arms below ran at REDUCED lane width (`-n 4 --narrow 2`) while a foreign
gate and a window boot shared the box. That is deliberate: a failure set of
CPU-only unit tests survives a loaded box, a wall time does not, so these arms
answer the CORRECTNESS questions and no timing is claimed from them.

### 4.1 Equivalence arm — NOT EQUAL, and the runner said so rather than rounding it

    scripts/gate_tier2_partitioned.py -n 4 --narrow 2 --prove <serial reference>

| lane | modules | result |
|---|---|---|
| wide (`-n 4 --dist loadfile`) | 270 | **3798 passed, 7 skipped, 349 subtests passed, 0 failed** |
| narrow (`-n 2 --dist loadfile`) | 35 | 1 failed, 291 passed, 11 skipped |
| serial (one process) | 1 | 1 failed, 2 passed |

Union: 2 failures, both ADDED against the reference, both in the rank-spawning
family:

    test_pp_admission_chain_flush_deadlock_795.py::...::test_ordering_pin_discriminates_broken_send_site
    test_pp_admission_wraparound_never_blocks.py::...::test_blocking_wraparound_wedges_the_ring

**The verdict of this arm is: the equivalence proof is NOT YET GIVEN, and it
cannot be given on a loaded box.** Stated plainly rather than argued around.

But the arm is not silent about the CAUSE, and the evidence is arithmetic
rather than rhetorical:

* **The wide lane — 270 modules, 3798 tests, the entire new mechanism — is
  exactly equal to the serial reference. Zero added, zero lost.** Whatever went
  wrong did not go wrong there.
* **The SERIAL lane also failed.** That lane runs one module, in one process,
  with no parallelism of any kind; it is by construction identical to serial
  execution. A lane that cannot possibly be affected by the partition failed
  anyway. Therefore that failure is not caused by the partition.
* The first of the two is one of the exact modules `NOTE_860 §1` named as its
  non-deterministic divergence-2 flake. The second is §2.4's timing assertion.
  Both spawn real ranks and both assert on where those ranks stand at a
  deadline.

The honest reading: the partition is clean where it is new, and the residual is
the crowding class #860 already measured, amplified by a box that was carrying a
foreign gate. It is re-measurable in one arm on a quiet box; until that arm is
run, this note claims no equivalence.

### 4.2 The planted-mutant arm — the gate catches it

`NOTE_860` records this arm as "owed if xdist is ever revisited". It is due, and
it was run.

**The first mutant chosen was not caught — by anything, including the serial
gate.** Removing `self._drop_withheld_from_free_lists()` from `KvRowCap.release()`
(`kv_backing_relief.py:1005`) left `test_one_owner_and_gate_heartbeat_w36.py`
at 20 passed, serially, in a fresh process. That is a finding about the SUITE,
not about the gate: the call is a second belt behind
`_settle_free_list_overlap()` at the publish point, and no test distinguishes it.
Filed here rather than quietly swapped out, because a mutant arm that silently
retries until it finds a mutant that fails proves nothing.

**The mutant actually used** disables the W37-A invariant itself — an early
`return 0` in `KvRowCap._settle_free_list_overlap()`, which reverts exactly what
base commit 9164adb942 ("An id may live in at most one free list") fixed. Its
catch was verified on one module in 9 s BEFORE the gate arm was spent, so the
arm tests the gate rather than the mutant.

    partitioned gate, mutant planted, wide lane:
      3 failed, 3795 passed, 7 skipped, 349 subtests passed

      FAILED test_one_owner_and_gate_heartbeat_w36.py::TestTheFreeListOverlap::test_available_size_then_equals_the_union
      FAILED test_one_owner_and_gate_heartbeat_w36.py::TestTheFreeListOverlap::test_the_drop_is_counted_by_name
      FAILED test_one_owner_and_gate_heartbeat_w36.py::TestTheFreeListOverlap::test_the_overlap_is_removed_at_the_publish_point

The full gate arm against the mutant returned **4 failures**: the three above,
plus §2.4's load-sensitive module, which had already failed in the clean arm on
the same loaded box. Against the clean arm's wide lane (`3798 passed, 0
failed`) that is a clean discrimination — the three named tests, in the WIDE
lane, the lane that is new, the lane whose modules were separated from their
neighbours, and nothing else moved. **The partitioned gate detects a real
regression in the code the gate exists to protect, and the pre-existing noise
does not mask it, because the verdict is a set of NAMES and not a count.**

The mutant was reverted; `git diff` is empty on the source tree.

### 4.3 A sibling worth one question, filed rather than guessed

Register item **#864** carries "16 standing failures on the shipped class" as a
candidate for a real root cause. §2.1 above turned an identical-looking standing
count into a harness property with no defects in it. The same question should be
put to those 16 before they are treated as defects, and the instrument is
already built and suite-agnostic: run `scripts/gate_partition_probe.sh` over that
suite path, then diff the failure set with `CVD=""` against the set with cards
visible. Modules that fail only in the first arm are harness artefacts, not
defects. Not answered here — it needs its own run on its own suite, and guessing
at it from module names would be exactly the unchecked-indicator move.

---

---

## 4.4 THE TIMED ARMS — quiet box, A-vs-A first

Taken only after a MEASURED quiet condition, not an assurance: no pytest
process outside this tree (checked by `/proc/<pid>/cwd`, not by command-line
pattern), load below 3, a 60 s settle, and the condition RE-checked after the
settle because a foreign run can start during it — which it twice did.
Overrun was wired as a finding with a 40-minute deadline, not as an unbounded
wait. The gate opened at 22:59:04Z at load 1.02. All four arms hermetic, back
to back, same takt, nothing else of mine running.

    ARM 0  solo re-probe, test_pp_admission_wraparound_never_blocks   3 passed / 31.02 s

| arm | form | wall | failures | `--prove` |
|---|---|---|---|---|
| S1 | serial, one process, 306 modules | **766.88 s** | 0 | EQUAL |
| S2 | serial, repeat | **766.36 s** | 0 | EQUAL |
| P1 | partitioned, `-n 8` / narrow `-n 4` | **344.66 s** | 0 | EQUAL |
| P2 | partitioned, repeat | **346.90 s** | 0 | EQUAL |

**A-vs-A noise floor first, per the benchmark discipline:** serial 0.52 s
apart (0.07 %), partitioned 2.24 s apart (0.65 %). The floor is small enough
that the difference between the forms is a measurement and not a hope.

**Speedup 2.22x** (766.62 s mean serial ÷ 345.78 s mean partitioned).

Stated against the right denominator: **this tree's own serial arm on this
quiet box**, not `NOTE_860 §1`'s 717.49 s, which was measured in a different
quiet window against a different module-count reading. Both numbers are real
and they answer different questions.

**2.22x, not 3.4x, and the reason is arithmetic rather than disappointing.**
The lane breakdown of P1:

    wide   lane    98.58 s   270 modules  (-n 8)
    narrow lane   213.82 s    35 modules  (-n 4)     <- 62 % of the gate
    serial lane    32.26 s     1 module

**The rank-spawning family is the gate.** 35 of 306 modules — 11 % of the
modules — carry 62 % of the wall time, because they spawn real ranks and wait
out real timeouts, and their bounded lane is what keeps them honest. #860's
3.39x was measured with that family crowded at full width, which is exactly
what produced its flaky divergence 2. The 1.2x that separates the two numbers
is the price of the correctness, and it is visible rather than argued: it is
the difference between running that family at `-n 8` and at `-n 4`.

**AND THE EQUIVALENCE IS PROVED.** Four arms, both forms, both directions,
zero failures each, `EQUAL` reported by the runner's own two-sided check
against the serial reference. §4.1's NOT EQUAL was, as the serial lane's own
failure indicated at the time, the box and not the partition.

The one caveat on the numbers, stated because it changes them slightly: ARM 0
resolved §2.4 AFTER P1/P2 ran, so those arms had that module in the serial
lane (32.26 s). Promoting it to the narrow lane moves that work; the gate total
above is therefore a very slight over-estimate of the shipped table's cost, not
an under-estimate.

### 4.5 An observation that is NOT a result, filed rather than concluded

The wide lane took **98.58 s at `-n 8`** here, and an earlier arm measured
**48.60 s at `-n 4`** for the identical 270 modules and 3798 tests — under
FOREIGN LOAD, which should have made it slower, not twice as fast. Two
variables differ between those runs, so this is a hint and not a finding.

If it holds, the wide lane is startup-bound rather than compute-bound: each
xdist worker imports the whole of `sglang`, and 270 fast modules do not give
eight workers enough work to amortise eight imports. Halving the wide lane
would take the gate from ~345 s to under 300 s. **It needs one clean A/B at
`-n 4` vs `-n 8` on a quiet box and it was not run** — the measurement quota
was four arms and it was spent on the question that had to be answered.

---

## 5. ROOT BEFORE EFFECT — the three questions, answered

**1. What is the CLASS of the defect, not the instance?**

Three classes, found and separated rather than merged (§2.5's table). The one
this ticket exists for is: *a suite's parallel-readiness is a property of the
tests, and it can be PROVED per module instead of observed in aggregate* — by
comparing each module alone in a fresh process against the same module inside
the full serial run, which covers the poisoner and the poisoned in one
measurement. The instance #860 actually hit turned out to be a fourth thing
entirely: an engine whose grouping unit was assumed to be the file and is the
class.

**2. What are the SIBLING sites — a sweep, not a spot fix?**

Swept, not estimated: all 309 modules of the gate path, each alone under
`-n 2 --dist loadscope` against its solo-serial result (§0.3). One genuine
member. The method carries to the suites `NOTE_860 §SIBLINGS` explicitly did
not cover — `unit/mem_cache`, `unit/spec`, `scheduler/` — **without any change
to the tooling**: `gate_partition_probe.sh` takes a module path,
`gate_partition_build.py` and the runner both take `--gate-path`, and the table
is keyed by module path. What does NOT carry is this table's CONTENT: a
partition proved on `unit/managers` says nothing about another suite, and the
runner will put every unclassified module of a new path in the serial lane and
name it, which is the correct behaviour for a suite that has not been measured
yet. Cost per suite is one serial reference plus one solo sweep.

**3. What CHECK makes the class findable by inspection instead of by crash?**

`scripts/gate_tier2_partitioned.py --verify` — runs no tests, exits non-zero,
and answers two questions the class needs answered:

* a module in the tree with no row in the table is UNPROVEN and cannot be in a
  fast lane;
* a module whose bytes no longer hash to the sha256 its proof was taken against
  has an EXPIRED proof and cannot be in a fast lane.

That is what stops a new or edited module from drifting into the parallel lane
without a solo measurement behind it, and it converts the failure from a wrong
verdict at 03:00 into a one-second desk check. The runner enforces the same two
rules at run time as well (demote and report), so the check cannot be bypassed
by not running it.

---

## 6. WHAT IS NOT DONE

Recorded as owed rather than quietly dropped, in the same form #860 used for
its own unrun arm.

DONE and no longer owed: the equivalence proof and the solo re-probe (§4.4),
and the mutant arm (§4.2). §4.1's NOT EQUAL stands in this note as the record
of a measurement taken under conditions that could not answer the question, and
of how the cause was separated from the coincidence — not as the verdict.

Still owed:

* **The `-n 4` vs `-n 8` A/B for the wide lane** (§4.5). One arm, worth
  possibly 50 s off the gate; the measurement quota was spent on equivalence,
  which was the question that had to be answered first.
* **A tighter marker for the simultaneity class** (§0.3): the lane heuristic
  reads rank spawning, which is sufficient but not necessary. A source probe
  for wall-clock margins would also catch
  `test_pp_wedge_watchdog_is_honest_821.py`.
* **The #864 question** (§4.3), which needs its own suite and its own run.

### 6.1 One caution, because the finding is already in use

Within an hour of §0.2 being measured, an `-n 8 --dist loadfile` run of this
same suite was observed in a different tree. That is the finding working, and
the two halves of it travel at different speeds — so the distinction matters:

* **`loadscope` → `loadfile` is proved on its own**, by three arms over ONE
  module with nothing else in the run (§0.2). It does not depend on anything
  else in this note, and it is safe to adopt today.
* **Running the whole suite in parallel is NOT proved yet.** That rests on the
  partition (§1), and its equivalence arm is still owed (§4.1). A full-suite
  `-n 8` run without the partition table has no admission proof behind it, no
  recorded-failure check under it, and no exclusion report on top of it.

Adopting the first while the second is outstanding is fine. Reading a green
from the second because the first is proved would be the error, and it is the
kind that looks like progress.
