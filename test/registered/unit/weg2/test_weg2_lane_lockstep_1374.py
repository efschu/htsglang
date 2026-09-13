# SPDX-License-Identifier: Apache-2.0
"""#1374 F2 -- the two-process ordering test that would have caught xsn30.

THE WALL (boot weg2xsn30, 9423f16ba5, the lane's first real flip). A circular
wait on the co-located 5090:

  D0 (source) `_weg2_xchg_deposit_before_sleep` (weight_updater.py:3953) runs
  BEFORE the pause loop that publishes a VRAM credit per tag (:3974). It
  deposits its first band, posts `full`, and waits on `empty` for the second --
  i.e. on PP0's collect.
  PP0 (destination, same card) sits in its resume waiting for D0's credit:
  `W35 Weg2VramCreditRefused card=GPU-31d7ef41 tag=weights_3 credit=0
  published=0 consumed=0 requested=2916 MiB peer_leg_complete=False
  budget=120s EXPIRED`.
  D0 cannot deposit because PP0 cannot collect, PP0 cannot collect because it
  cannot resume, it cannot resume because D0 has not paused, and D0 has not
  paused because it is still in the deposit. Both budgets ran their full 120 s:
  D0 13:48:17->13:50:17 (W68 "still full"), PP0 13:48:13->13:50:13 (W35).

WHY EVERY DESK TOOL WAS BLIND: chain_smoke_1342, the replay and --shadow-leg
all run deposit and collect with NO pause, NO credit and NO resume between
them, so the ordering that deadlocks cannot occur there. The order
deposit-before-pause vs credit-before-resume was nowhere pinned as an
invariant, and the 120 s budgets turn a deadlock into a "timeout".

THIS FILE PINS THE ORDERING, in two processes, on the REAL handshake
(`xr.create_semaphores` + `tp.SemSet` + `CrossSlotRendezvous`) with a SHORT
budget. The bytes are irrelevant to the ordering and are not moved: what
deadlocks is the handshake, and a test that needed CUDA to show it could not
run at the desk at all.
"""

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp
from sglang.srt.weg2.weight_exchange_bounce import CrossSlotRendezvous
from sglang.test.test_utils import CustomTestCase

BUDGET = 1.5          # the 120 s budget, scaled so a deadlock costs seconds
PAIR = 0


def _rv(nonce, root, budget=BUDGET):
    """The REAL rendezvous: real semaphores AND the real shm slot record, which
    has to be the real one because `wait_full` checks the SEQ out of it and a
    python stub is not shared across the two processes."""
    from sglang.srt.weg2.weight_exchange_bounce import BounceSlots

    return CrossSlotRendezvous(tp.SemSet(nonce),
                               BounceSlots(nonce, shm_root=root),
                               pair=PAIR, budget_s=budget)


def _source(nonce, root, batches, slots, credit, per_tag, out):
    """The sleeping group's leg: deposit, then pause, then publish the credit.

    `per_tag` is the CREDIT GRANULARITY and it is the whole subject:
      "never" -- TODAY'S ORDER: every band of every tag, then one credit.
      "tag"   -- F1 AS ORDERED: the credit follows the LAST band of the tag,
                 because `memory_saver_adapter.pause(tag)` cannot pause half a
                 tag and the credit may not precede the pause.
      "band"  -- a finer granularity than the pause allows; kept because it is
                 the control that shows WHY "tag" is the binding constraint.
    """
    rv = _rv(nonce, root)
    try:
        for b in range(batches):
            if not rv.wait_empty(slot=b % slots, seq=b):
                out.put(("source", f"W68 blocked at band {b} of {batches}"))
                return
            rv.post_full(slot=b % slots, seq=b, nbytes=1)
            if per_tag == "band":
                credit.set()
        # "tag" and "never" both land here; with ONE tag they are the same
        # instant, which is exactly the worst case F1 has to survive.
        credit.set()
        out.put(("source", "deposited"))
    except Exception as exc:  # pragma: no cover
        out.put(("source", f"{type(exc).__name__}: {exc}"))


def _destination(nonce, root, batches, slots, credit, out):
    """The waking group's leg: it may not collect before the credit exists."""
    rv = _rv(nonce, root)
    try:
        if not credit.wait(timeout=BUDGET):
            out.put(("dest", "W35 credit never published"))
            return
        for b in range(batches):
            if rv.wait_full(slot=b % slots, seq=b) is None:
                out.put(("dest", f"W68 no full band at {b}"))
                return
            rv.post_empty(slot=b % slots, seq=b)
        out.put(("dest", "collected"))
    except Exception as exc:  # pragma: no cover
        out.put(("dest", f"{type(exc).__name__}: {exc}"))


def _run(nonce, *, batches, slots, per_tag):  # per_tag: never|tag|band
    from sglang.srt.weg2.weight_exchange_bounce import BounceSlots

    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    # A tmpfs root of our OWN, never /dev/shm/weg2-*: that tree belongs to a
    # live boot and is only ever cleaned by epoch/holder (shm-residue rule).
    root = tempfile.mkdtemp(prefix="weg2-1374-")
    os.makedirs(xr.region_dir(nonce, root), exist_ok=True)
    BounceSlots(nonce, shm_root=root, create=True)
    try:
        ctx = mp.get_context("fork")
        credit, out = ctx.Event(), ctx.Queue()
        ps = [ctx.Process(target=_source,
                          args=(nonce, root, batches, slots, credit, per_tag,
                                out)),
              ctx.Process(target=_destination,
                          args=(nonce, root, batches, slots, credit, out))]
        for p in ps:
            p.start()
        for p in ps:
            p.join(timeout=4 * BUDGET)
            if p.is_alive():           # a real hang, not a named refusal
                p.terminate()
                p.join()
        got = {}
        while not out.empty():
            who, what = out.get()
            got[who] = what
        return got
    finally:
        xr.unlink_semaphores(nonce)
        shutil.rmtree(root, ignore_errors=True)


class TodaysOrderDeadlocks(CustomTestCase):
    """RED-FIRST: this is xsn30, in two processes, in seconds."""

    def test_deposit_before_credit_deadlocks_when_the_tag_exceeds_the_slots(self):
        got = _run("desk-1374-today", batches=4, slots=2, per_tag="never")
        self.assertIn("source", got)
        self.assertIn("blocked at band 2", got["source"],
                      f"the source must block once the slots are full: {got}")
        self.assertIn("dest", got)
        self.assertIn("credit never published", got["dest"],
                      f"and the destination must starve on the credit: {got}")


class LockstepRunsThrough(CustomTestCase):
    def test_a_tag_that_fits_the_slots_completes_at_depth_one(self):
        """depth=1 -> assemble_slots(1, comparing=True) = 2 usable slots."""
        got = _run("desk-1374-fits", batches=2, slots=2, per_tag="tag")
        self.assertEqual(got.get("source"), "deposited", got)
        self.assertEqual(got.get("dest"), "collected", got)


class TheLockstepUnitMustFitTheBuffer(CustomTestCase):
    """THE SIZING CONSTRAINT, executable. Per-tag lockstep is deadlock-free
    only where a tag's bands FIT the deposit buffer -- and xsn30's own
    `weights_3` does not: 2916 MiB against slots x slot_bytes = 2 x 128 MiB =
    256 MiB (published terms: slot_bytes=134217728, depth=1), so 23 bands
    against 2 slots. The pause granularity is the TAG, so no credit can be
    published mid-tag to let the collector in, and the cycle closes INSIDE the
    one tag.

    The `band` control below is what proves the constraint is the GRANULARITY
    and not the buffer as such: publish a credit per band -- which the pause
    cannot do -- and the same 4 bands through 2 slots complete. That is the
    measurement that decides between sizing the buffer per tag and pipelining
    the collect one tag behind the deposit."""

    def test_one_tag_larger_than_the_buffer_deadlocks_under_per_tag_credit(self):
        got = _run("desk-1374-bigtag", batches=4, slots=2, per_tag="tag")
        self.assertIn("blocked at band 2", str(got.get("source")), got)
        self.assertIn("credit never published", str(got.get("dest")), got)

    def test_a_finer_credit_would_drain_it_which_is_why_granularity_is_the_issue(self):
        got = _run("desk-1374-band", batches=4, slots=2, per_tag="band")
        self.assertEqual(got.get("source"), "deposited", got)
        self.assertEqual(got.get("dest"), "collected", got)

    def test_the_arithmetic_is_stated_from_the_published_terms(self):
        """The numbers that decide the sizing, so the next reader re-derives
        them instead of trusting this docstring."""
        from sglang.srt.weg2 import xchg_bounce as xb

        slot_bytes = 134217728                      # published terms, xsn30
        slots = xb.assemble_slots(1, comparing=True)
        self.assertEqual(slots, 2)
        capacity_mib = slots * slot_bytes // (1 << 20)
        self.assertEqual(capacity_mib, 256)
        self.assertGreater(2916, capacity_mib,
                           "weights_3 (2916 MiB) must exceed the buffer, or "
                           "xsn30's deadlock had another cause")


if __name__ == "__main__":
    unittest.main()
