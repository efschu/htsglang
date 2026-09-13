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


def _rv(nonce, root, rows, budget=BUDGET):
    """The REAL rendezvous: real semaphores AND the real shm record, which has
    to be real because `wait_full` reads the SEQ out of it and a python stub is
    not shared across the two processes."""
    from sglang.srt.weg2.weight_exchange_bounce import BounceSlots

    return CrossSlotRendezvous(tp.SemSet(nonce),
                               BounceSlots(nonce, shm_root=root,
                                           rows_per_pair=rows),
                               pair=PAIR, budget_s=budget)


# --------------------------------------------------------------------------
# ARM 1: THE OLD CONTRACT, kept as a MODEL because the code no longer offers
# it. The per-band claim is what boot weg2xsn30 died on, so it stays testable
# after being deleted -- otherwise the regression it guards is unobservable.
# --------------------------------------------------------------------------
def _old_source(nonce, root, batches, slots, credit, out):
    sems = tp.SemSet(nonce)
    try:
        for b in range(batches):
            if not sems.timedwait(PAIR, b % slots, "empty", BUDGET):
                out.put(("source", f"W68 blocked at band {b} of {batches}"))
                return
            sems.post(PAIR, b % slots, "full")
        credit.set()                      # only after the WHOLE deposit
        out.put(("source", "deposited"))
    except Exception as exc:  # pragma: no cover
        out.put(("source", f"{type(exc).__name__}: {exc}"))


def _old_destination(nonce, root, batches, slots, credit, out):
    sems = tp.SemSet(nonce)
    try:
        if not credit.wait(timeout=BUDGET):
            out.put(("dest", "W35 credit never published"))
            return
        for b in range(batches):
            if not sems.timedwait(PAIR, b % slots, "full", BUDGET):
                out.put(("dest", f"W68 no full band at {b}"))
                return
            sems.post(PAIR, b % slots, "empty")
        out.put(("dest", "collected"))
    except Exception as exc:  # pragma: no cover
        out.put(("dest", f"{type(exc).__name__}: {exc}"))


# --------------------------------------------------------------------------
# ARM 2: THE #1374 CONTRACT. No per-band wait; `full` counts; `drained` once
# per tag, taken AFTER the credit.
# --------------------------------------------------------------------------
def _source(nonce, root, tags, bands, credit, out, drain_between=True):
    rv = _rv(nonce, root, bands)
    try:
        # The drain counter starts at 0: at tag 0 there is no previous tag.
        rv.prime_drain()
        for t in range(tags):
            for b in range(bands):
                # fill -> ONE sync -> publish -> post. The per-band order rule
                # is unchanged; only the claim in front of it is gone.
                rv.post_full(slot=b, seq=t * bands + b, nbytes=1)
            credit.set()                        # pause(t) then credit(t)
            if t + 1 < tags:
                if drain_between and not rv.wait_drained(tag=f"weights_{t}"):
                    out.put(("source", f"blocked before tag {t + 1}"))
                    return
        out.put(("source", "deposited"))
    except Exception as exc:
        out.put(("source", f"{type(exc).__name__}: {exc}"))


def _destination(nonce, root, tags, bands, credit, out, post_drain=True):
    rv = _rv(nonce, root, bands)
    try:
        if not credit.wait(timeout=BUDGET):
            out.put(("dest", "W35 credit never published"))
            return
        last = -1
        for t in range(tags):
            for b in range(bands):
                got = rv.wait_full(slot=b, seq=t * bands + b)
                if got is None:
                    out.put(("dest", f"W68 no full band at {t}.{b}"))
                    return
                if t * bands + b <= last:
                    out.put(("dest", "W68 seq went backwards"))
                    return
                last = t * bands + b
            if post_drain:
                rv.post_drained(tag=f"weights_{t}")
        out.put(("dest", "collected"))
    except Exception as exc:
        out.put(("dest", f"{type(exc).__name__}: {exc}"))


def _run(target_pair, nonce, **kw):
    from sglang.srt.weg2.weight_exchange_bounce import BounceSlots

    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    # A tmpfs root of our OWN, never /dev/shm/weg2-*: that tree belongs to a
    # live boot and is only ever cleaned by epoch/holder (shm-residue rule).
    root = tempfile.mkdtemp(prefix="weg2-1374-")
    os.makedirs(xr.region_dir(nonce, root), exist_ok=True)
    BounceSlots(nonce, shm_root=root, create=True,
                rows_per_pair=int(kw.pop("rows", 8)))
    try:
        ctx = mp.get_context("fork")
        credit, out = ctx.Event(), ctx.Queue()
        src, dst = target_pair
        ps = [ctx.Process(target=src, args=(nonce, root), kwargs=dict(
                  credit=credit, out=out, **kw.get("src", {}))),
              ctx.Process(target=dst, args=(nonce, root), kwargs=dict(
                  credit=credit, out=out, **kw.get("dst", {})))]
        for p in ps:
            p.start()
        for p in ps:
            p.join(timeout=4 * BUDGET)
            if p.is_alive():
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
    """RED-FIRST: xsn30, in two processes, in seconds."""

    def test_the_old_per_band_claim_deadlocks_when_the_tag_exceeds_the_slots(self):
        got = _run((_old_source, _old_destination), "desk-1374-today",
                   rows=8, src={"batches": 4, "slots": 2},
                   dst={"batches": 4, "slots": 2})
        self.assertIn("blocked at band 2", str(got.get("source")), got)
        self.assertIn("credit never published", str(got.get("dest")), got)

    def test_the_per_band_claim_is_gone_from_the_code(self):
        """DELETED, not deprecated (operator wording: "der Wait wird
        geloescht"). A method kept as a named refusal would be a new UNWIRED
        raiser and would have to be carried in the #1335 frozen debt list --
        a recorded decision for a call nobody may make. Absence is cheaper and
        just as checkable."""
        rv = CrossSlotRendezvous(None, None, pair=PAIR)
        self.assertFalse(hasattr(rv, "wait_empty"),
                         "the per-band claim is back; it is the xsn30 cycle")
        self.assertTrue(hasattr(rv, "wait_drained"))
        self.assertTrue(hasattr(rv, "post_drained"))


class TheLockstepDrains(CustomTestCase):
    def test_one_tag_larger_than_the_old_slots_drains_under_the_new_contract(self):
        """OPTION 1: the buffer holds the whole tag, so the deposit needs no
        collector -- the same 4 bands that deadlocked above run through."""
        got = _run((_source, _destination), "desk-1374-sized", rows=8,
                   src={"tags": 1, "bands": 4}, dst={"tags": 1, "bands": 4})
        self.assertEqual(got.get("source"), "deposited", got)
        self.assertEqual(got.get("dest"), "collected", got)

    def test_two_tags_run_in_lockstep_and_seq_never_goes_backwards(self):
        got = _run((_source, _destination), "desk-1374-two", rows=4,
                   src={"tags": 2, "bands": 3}, dst={"tags": 2, "bands": 3})
        self.assertEqual(got.get("source"), "deposited", got)
        self.assertEqual(got.get("dest"), "collected", got)

    def test_the_next_tag_may_not_overwrite_a_tag_still_being_collected(self):
        """The one wait that remains. Without the collector's `drained`, the
        source must STOP before tag t+1 rather than overwrite the buffer."""
        got = _run((_source, _destination), "desk-1374-nodrain", rows=4,
                   src={"tags": 2, "bands": 3},
                   dst={"tags": 2, "bands": 3, "post_drain": False})
        self.assertIn("blocked before tag 1", str(got.get("source")), got)


class TheRecordRowsAreNotFolded(CustomTestCase):
    def test_a_band_beyond_the_table_refuses_instead_of_aliasing(self):
        """#1374 (c): `_row` used `% SLOTS_PER_PAIR`, so band 2 shared band
        0's (seq, bytes) row. At depth=2 (slots=3 under shadow) that was one
        boot away; the #1358 depth=1 default is the only reason it never bit."""
        from sglang.srt.weg2.weight_exchange_bounce import (
            BounceSlots,
            Weg2XchgBouncePhaseUnordered,
        )

        root = tempfile.mkdtemp(prefix="weg2-1374-row-")
        try:
            nonce = "desk-1374-rows"
            os.makedirs(xr.region_dir(nonce, root), exist_ok=True)
            slots = BounceSlots(nonce, shm_root=root, create=True,
                                rows_per_pair=2)
            slots.publish(slot=0, seq=7, nbytes=11, pair=PAIR)
            with self.assertRaises(Weg2XchgBouncePhaseUnordered):
                slots.publish(slot=2, seq=9, nbytes=13, pair=PAIR)
            self.assertEqual(slots.read(slot=0, pair=PAIR), (7, 11),
                             "band 0's row was overwritten by the fold")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TheSizingIsStatedFromTheTerms(CustomTestCase):
    def test_the_arithmetic_is_stated_from_the_published_terms(self):
        from sglang.srt.weg2 import xchg_bounce as xb

        slot_bytes = 134217728                      # published terms, xsn30
        self.assertEqual(xb.assemble_slots(1, comparing=True), 2)
        self.assertEqual(2 * slot_bytes // (1 << 20), 256)
        self.assertGreater(2916, 256,
                           "weights_3 (2916 MiB) must exceed the old buffer, "
                           "or xsn30's deadlock had another cause")


if __name__ == "__main__":
    unittest.main()
