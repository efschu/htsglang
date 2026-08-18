# NOTE 749 — the ~50-test order dependence: root, fix, and the re-run protocol

## Root

`test/registered/unit/managers/test_collective_family_siblings_610.py` entered
three `unittest.mock.patch` context managers **from inside worker threads**.
`run_ranks` (`:120-141`) spawns one thread per rank; each `body(rank)` opened
the same three patches on process globals:

    unittest.mock.patch.object(torch.distributed, "all_reduce", ...)
    unittest.mock.patch.object(torch.distributed, "get_world_size", ...)
    unittest.mock.patch("sglang.srt.distributed.utils.uneven_dcp_active",
                        lambda *a: True)

`mock.patch` mutates a module attribute and restores it on exit. It is not
thread-safe: N threads entering and exiting the same patch race on the restore
stack, and one lost restore leaves the global patched for the rest of the
process.

## Why it looked like a code regression

The damage was in another directory and had nothing to do with the leak's
subject. With `uneven_dcp_active` stuck at `True`, every later scheduler test
took `scheduler.py:4615`'s pinned-admission branch and at `:4622` asked its
tree-cache fake for `evictable_size()` — which the minimal fakes in the
`distributed/` collective-floor suites do not implement:

    AttributeError: '_FakeTreeCache' object has no attribute 'evictable_size'

~50 failures across six error signatures, none near the cause. And
**intermittent**, because whether a restore is lost depends on thread
interleaving: commit `3c49cbc0c2` produced 0 and 50 failures on two runs of one
command. That same-commit contradiction is what proved it was ordering rather
than code, and it is what nearly got #739 blamed.

## Fix

**1. The leak.** All three patch sites in 610 hoisted out of `body` into the
main thread, wrapping `run_ranks(body)`. The threads need the globals patched
*while* they run, not to own the patching.

**2. The guard** (`test/conftest.py`). Sentinels snapshotted at
`pytest_runtest_setup`, compared at `pytest_runtest_teardown`; a mismatch names
the test, the attribute and the likely cause, and fails the session.

Three design points, each of which a naive version got wrong first:

* **Snapshot at setup, not lazily at teardown.** A lazy baseline records the
  *first* test's leaked value as the baseline. The guard's own can-fail test
  caught that.
* **Record and repair; do not raise in teardown.** Raising there aborts the
  teardown chain and pytest errors the *next* item ("previous item was not torn
  down properly") — one leak, two failures, the cascade in miniature. The guard
  restores the baseline and fails the session at the end instead.
* **`trylast=True` is load-bearing.** pytest runs fixture finalizers from its
  own `pytest_runtest_teardown`. Without `trylast` the guard runs first, sees
  `monkeypatch`'s value still in place, and accuses honest tests — it reported
  `test_barlink_port.py`, which was innocent. The `with`-block clean case
  cannot catch this (it restores synchronously inside the test); only a
  fixture-restored patch distinguishes the orderings, and that case is now
  pinned.

It **asserts rather than silently cleans**: a reset that kept the suite green
would leave the next leaker undiscovered.

## Re-run protocol — the combined sweep's gating power is RESTORED

Isolation is **not** required. Measured after the fix, hermetically
(`CUDA_VISIBLE_DEVICES=99`, `-p no:randomly`):

| run | failed |
|---|---|
| `managers/` alone | 25 |
| `distributed/` alone | 28 |
| *sum of isolated* | *53* |
| combined, managers→distributed | 54 |
| combined, distributed→managers | 52 |

The combined sweep now equals the sum of its isolated parts within ±1, and the
guard is silent in both orders. Before the fix the same combination produced
~50 extra failures on top of that sum, intermittently.

So the standing protocol is unchanged — run the combined sweep and compare
per-suite — with one addition: **a `#749 PROCESS-GLOBAL LEAK GUARD` block in
the output means the run's failure list is not trustworthy until the named
leaker is fixed.** The guard reports the cause; it does not have to be inferred
from the victims any more.

**Residual, stated rather than rounded away:** ±1 between the two orders. That
is at the level of the single order-dependent test already recorded in
`MERGE_TRAIN_PASS2_2026-08-17.md` (`test_phase_flip_mover_streaming_631`, which
passes 3/3 in isolation on two different commits). It is not covered by the
three sentinels and is not claimed to be fixed here.

**Sentinel coverage, named:** `uneven_dcp_active`, `torch.distributed.all_reduce`,
`torch.distributed.get_world_size`. These are the globals this class of leak
actually reached. The list is meant to grow when a new one is found — adding a
sentinel is how a fixed leak stays fixed.
