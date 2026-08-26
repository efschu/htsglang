# NOTE #864 + #862 — when the INVOCATION decides the verdict

Runner: R8. Tree: `/spinning/wt-864-desk`, branch `probe/864-862-desk`, base
**d7be60817c** (`fix/861c-admission-term` tip, the shipped integration line).
`/spinning/wt-861c` NOT touched — R9 owns it and the cards.

**Every run in this note used `CUDA_VISIBLE_DEVICES=""`, verified on the
PROCESS** (`tr '\0' '\n' < /proc/<pid>/environ`) and cross-checked against
`nvidia-smi --query-compute-apps`, whose only entries throughout were R9's
serving ranks (PIDs 851455/851456/851457). No process of mine appears on a
card. **No arm in this note was run with cards visible**, which bounds what it
can conclude — said once here and again at each number that depends on it.

---

## RESULT IN ONE BLOCK

    #864  the 16 are NOT the same form as #868's 15. They are its INVERSE,
          and the inverse is the dangerous direction.
          Cause found and proved WITHOUT a card: a test double whose signature
          drifted one keyword behind its production call site. TEST-ONLY.
          Fixed (one line) + a CPU-lane guard that catches the next one.

    #862  mem_cache hermetic: 2 failed / 1774 passed / 1658 skipped.
          Of the 1658:  1008 DARK (hardware/dependency)
                         645 NOT DARK (wrong parametrisation, never runnable here)
                           0 BY ACCIDENT
                           5 unclassified
          The darkness is by DESIGN. The defect is that nothing REPORTS it.
          Both remaining failures are device-caused; REAL defects: 0.

    Corrections to prior art: the "16 passed" in NOTE_861fg is a numeric
    coincidence, not the same 16 (they are SKIPPED hermetically). And
    mem_cache DOES run directory-wide on CPU on this tree — 3396 collected,
    no abort.

---

## 0. PRIOR ART — read before building, and it changed what I built

* `/spinning/gpu-arb/NOTE_861fg.md:3058-3201` already carries the measurements
  behind both items: the 1506/16 split, the 16 failures named, the
  pre-existence check at `f8079a09c8`, and the verdict "the mechanism is not
  the defect — the defect is that NOTHING REPORTS THE ASYMMETRY". **This note
  does not re-derive any of that.** What was missing: the CAUSE of the 16 (the
  note reaches "test-only" but not the mechanism), a card-free way to prove it,
  the sibling sweep, and the check.
* `test/registered/unit/mem_cache/conftest.py:1-64` — the skip mechanism, with
  a docstring that already explains itself (944 hermetic failures → skips,
  commit `8ac8e7e60a`, #585). Not a defect; not to be re-litigated.
* `python/sglang/test/ci/ci_register.py:62` — `register_cpu_ci` and its
  per-backend siblings. There is no `register_gpu_ci`; the GPU lanes are
  `register_cuda_ci` / `register_amd_ci` / etc.
* **Does not exist**, searched for and not found: any shared
  `requires_cuda` / `skip_without_device` helper; any `#862`/`#864` ticket file;
  any commit tagged either number.

---

## 1. #864 — the 16 are the INVERSE of #868's 15, and that is the whole point

The operator's question was whether these 16 are the same hermeticity form as
the 15 that #868 resolved. **They are not. They are its mirror image, and the
mirror image is the one that hurts.**

| | #868's 15 | #864's 16 |
|---|---|---|
| hermetic (`CVD=""`) | **FAIL** | **SKIP** |
| cards visible | pass | **FAIL** |
| what the desk gate shows | a red it cannot act on | **a green over a standing red** |
| direction | false RED | **false GREEN** |

Both are the same class — *the invocation decides the verdict* — but only one
of them is invisible in the direction that matters. #868's 15 announce
themselves on every desk run. These 16 announce themselves nowhere anybody
looks: the machine that can see them is the one with cards, and the gate people
read is the one without.

### 1.1 First, a correction to the prior art, because it misleads on re-reading

`NOTE_861fg.md:3134-3144` records:

    Full file, cards VISIBLE:  838 passed, 648 skipped, 16 FAILED, 29 s
                    hermetic:   16 passed, 1506 skipped, 0 failed
    [...] the shipped class has SIXTEEN STANDING FAILURES that every hermetic
    run reports as "16 passed".

The two 16s are **a coincidence of arithmetic, not the same tests.** Read
quickly, that last sentence says the sixteen failures are being counted as
passes. They are not: hermetically they are SKIPPED, inside the 1506. Measured
directly rather than inferred:

    pytest test_unified_radix_cache_unittest.py -k tp_swa_prefetch    (CVD="")
    -> 38 skipped, 1464 deselected, 0 passed, 0 failed

    SKIPPED [8] :2481: requires a real accelerator: ...
    SKIPPED [8] :2457: requires a real accelerator: ...
    SKIPPED [11] :2481: SWA-only fixture required
    SKIPPED [11] :2457: SWA-only fixture required

Tally probe: 8+8+11+11 = 38 = the summary's 38. The 16 that fail with cards are
the two 8s that skip for want of an accelerator; the two 11s are the non-SWA
parametrisations, which do not apply at all.

The coincidence is worth naming because a reader who takes it as identity
concludes the harness mis-reports skips as passes, and goes looking for a bug in
pytest. (A subagent of mine did exactly that, from exactly this passage, which
is how I noticed.) The note's numbers are right; only the sentence is ambiguous.

### 1.2 The cause, proved without a card

Production, `python/sglang/srt/mem_cache/hiradix_cache.py:266-278` — both call
sites, unconditionally:

    torch.distributed.all_reduce(tensor, op=op, group=group, async_op=True)

The double, `test_unified_radix_cache_unittest.py:2382`:

    def fake(tensor, op=None, group=None):

`scripts/gate_double_drift.py` decides this by parsing rather than running —
no device, no ranks, no import of a module that would need them:

    stub          fake(tensor, op=None, group=None)
    production call: 1 positional + ['op', 'group', 'async_op']
    VERDICT: DRIFTED -- the double CANNOT accept the production call.
             got an unexpected keyword argument 'async_op'

So every one of the 16 dies with a `TypeError` inside `mock`'s dispatch
**before its own first assertion runs**. They are not evidence about SWA
prefetch adoption in either direction — they never reach the behaviour they
were written to check.

**And the second half of the call was checked too, rather than assumed.**
Production hands the result to `_wait_bounded`, and the double returns `None`.
That is fine: `hicache_collective.py:190-196` documents `None` as a completed
no-op and returns immediately. So the drift is exactly one keyword wide, and a
one-line fix is the whole fix — worth establishing, because a fix that restored
`async_op` and left a second incompatibility would have produced 16 fresh reds
on the next card run and looked like a new defect.

**Verdict on #864: TEST-ONLY. Zero live defects. #783 gets nothing from these
16, in either direction** — which agrees with `NOTE_861fg`'s reading and now
has a mechanism under it instead of an inference.

### 1.3 What I did NOT measure

I never ran these 16 with cards visible, so "16 red with cards" remains prior
art, not my measurement, and **my fix is unverified against the tests it
repairs**. What is verified card-free: the drift exists, it is one keyword, the
double now binds the production call, and the guard catches its removal. The
behavioural assertions still need one card run. Owed, named, not smuggled in.

---

## 2. #862 — the darkness is real, chosen, and unreported

Full hermetic run of the directory, `scripts/gate_darkness.py`:

    2 failed, 1774 passed, 1658 skipped, 361 subtests passed, 143.9 s

    tally gate:  failures  names=2  summary=2      OK
                 skips     extracted=1658  summary=1658  OK

**Failures: 2, both DEVICE-caused, REAL defects 0.**

**Skips, classified by reason:**

| bucket | count | meaning |
|---|---|---|
| HARDWARE | 996 | needs an accelerator. Dark, on purpose. |
| DEPENDENCY | 12 | NIXL not installed. Dark, on purpose, curable without a card. |
| **CONFIG (not darkness)** | **645** | the body does not apply to this parametrisation |
| BY ACCIDENT | **0** | nothing is dark by mistake |
| UNCLASSIFIED | 5 | reported as its own bucket, never folded into another |

**Answer to the operator's question: skip-by-design, not skip-by-accident.**
`BY ACCIDENT` is zero. The mechanism (`conftest.py`) is correct engineering and
says so in its own docstring.

**But the ticket is still right, and the reason is the 645.** A reader told
"1658 skipped" cannot tell that 1008 of those would run on a card and 645 would
never run anywhere, because they belong to other cache configurations. Those
are different facts with the same appearance, and the summary line renders them
identically — which is the #862 complaint exactly, one level finer than it was
filed.

### 2.1 My own classifier was wrong first, and the data said so

The first version of `gate_darkness.py` had two buckets: BY DESIGN and BY
ACCIDENT, on the assumption that a skip either names hardware or is a mistake.
The first real run put **636 of 1658 skips in neither** — "requires SWA",
"requires Mamba component", "page_size > 1 only".

Folding those into "dark" would have inflated the darkness figure by 62 % and
aimed the ticket at a problem that does not exist. I read all 36 that survived
the second pass individually and encoded what they said; 5 remain unclassified
and are reported as such rather than tuned away. **A bucket that empties because
the pattern was widened until it did is not a measurement** — the same
Indikator-Gesetz that caught me in #868, on a different axis.

### 2.2 Two corrections to the prior art

**(a) mem_cache DOES run directory-wide on CPU.** `NOTE_861fg.md:3150-3158`
records a hermetic sweep aborting at COLLECTION with `RuntimeError: No CUDA GPUs
are available`, and concludes "`test/registered/unit/mem_cache/` CANNOT BE RUN
DIRECTORY-WIDE ON CPU AT ALL". On d7be60817c it can:

    pytest test/registered/unit/mem_cache/ --collect-only   (CVD="")
    -> 3396 tests collected in 10.33s, no errors

Either it was fixed between the two trees or the abort needed a condition my
run did not reproduce. Reported, not resolved — I did not bisect it.

**(b) The escape is real, but it is at RUNTIME and it is two tests.** The
substance behind that note's finding survives its collection-time framing, and
it is the more interesting half:

    test_acceptance_emitters_758.py::RefillTiming::test_the_refill_is_timed_and_reported
    test_acceptance_emitters_758.py::RefillTiming::test_the_mode_is_named_so_the_baseline_cannot_be_misread
      RuntimeError: No CUDA GPUs are available

    phase_flip_boot.py:1000  -> rotation_executor.py:344 -> read_buffer_pool.py:85
      factory=lambda: torch.empty(int(chunk_bytes), dtype=torch.uint8, pin_memory=True)

**`pin_memory=True` is a device door that does not go through `get_device()`.**
The conftest guard patches `get_device`; this path never calls it. So these two
tests FAIL where their neighbours SKIP — and a device-caused failure is
indistinguishable, in a summary line, from a defect.

---

## 3. ROOT BEFORE EFFECT — both items, three questions

### (1) The CLASS, not the instance

**One class, three faces: the verdict depends on the invocation, and the report
does not say which invocation produced it.**

| face | hermetic | cards | desk gate shows | seen in |
|---|---|---|---|---|
| false red | FAIL | pass | a red nobody can act on | #868's 15 |
| **false green** | **SKIP** | **FAIL** | **green over a standing red** | **#864's 16** |
| dark | SKIP | pass | green over untested code | #862's 1008 |

And one narrower class underneath, which is what produces the third face's
ragged edge: **a skip guard covers ONE device door while device access has
many.** `get_device()` is guarded; `pin_memory=True` is not.

### (2) SIBLING SITES — swept, not spot-fixed

* **Doubles that cannot accept their production call.**
  `gate_double_drift.py --sweep test/registered/unit/mem_cache` enumerates
  every patched double with a named stub: **6 found**, of which one had
  drifted (the `all_reduce` fake). The others are `*args, **kwargs` stubs or
  match their call sites. The sweep is a REPORT, deliberately: deciding
  automatically which production call site a double stands in for needs
  import-time resolution this tool refuses to do. What it removes is "nobody
  knew the double was there".
* **Device doors that bypass `get_device()`**, in `mem_cache/`,
  `model_executor/`, `managers/`: `pin_memory=True` in **11 files**,
  `torch.cuda.*` in **57**. Reachability from a guarded test is what decides
  whether each matters, and only 2 fired in this run — so the door list is an
  upper bound on exposure, not a defect count. Named so the next one is looked
  up rather than rediscovered.
* **Suites with the same shape as mem_cache**: the classifier takes any path,
  so `unit/spec`, `scheduler/`, `unit/managers` can be illuminated at one
  command each. Not run here.

### (3) The CHECK that turns the class from crash into inspection

* **`test/registered/unit/mem_cache/test_double_signature_drift_864.py`** — new,
  **CPU lane** (`register_cpu_ci`), 2 tests, **0.11 s**. It parses the double
  and parses the production call sites and asserts the first can bind the
  second. Deliberately in the CPU lane: the failure it guards against is
  GPU-only, so the guard must live where people actually look. It also asserts
  that production still passes `async_op`, so if that ever changes the guard
  reports it instead of silently testing a call that no longer exists.
  **Can-fail proven**: re-introducing the drift turns it red with the exact
  diagnostic; the tree was restored afterwards.
* **`scripts/gate_darkness.py`** — the reporter #862 says does not exist.
  Prints, for any suite, the failure set split into DEVICE vs REAL and the skip
  set split into HARDWARE / DEPENDENCY / CONFIG / BY ACCIDENT / UNCLASSIFIED,
  behind a two-axis tally gate (extracted names == summary failures; extracted
  skip counts == summary skips) that refuses to report buckets at all if the
  extraction is broken. `--from-log` re-classifies an existing run, so improving
  the reading never costs another run of the tests.

---

## 4. WHAT IS NOT DONE

* **The 16 with cards, before and after the fix.** One card run. Until then the
  fix is proved to remove the drift and not proved to make the tests pass.
* **Why prior art saw a collection abort and I see none** (§2.2a). Not bisected.
* **The two `pin_memory` failures are reported, not fixed.** The fix is a
  decision about where the guard belongs — widen the conftest patch, or make
  the ring lazy — and that is a design call on somebody else's module.
* **The other suites.** The instrument is suite-agnostic; nothing outside
  `unit/mem_cache` was measured, and this note claims nothing about them.
