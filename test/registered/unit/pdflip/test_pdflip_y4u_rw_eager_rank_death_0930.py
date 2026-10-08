# SPDX-License-Identifier: Apache-2.0
"""y4u (f50f51020f, Profil -tse, ohne REWAKE), D TP1 (3080, Form-A-Worker),
15:18:30Z: ``IndexError: index 2147483647 is out of bounds for dimension 0 with
size 128`` in ``expert_pool_device.sync_tables`` (``if old >= 0 and
int(hot[old]) == r``) <- ``sync_pool_from_host`` <- ``_run_eager_host_plan``.
15:18:28 TP1: ``RW WAKE-FIRST-TOKEN ... warm_layers=8 ... finish_layers=40
finish_rows=1120 finish_landed=0 finish=first_decode`` -- 40 layers' warm
copies in flight on the side stream, their rows RESERVED (row_key =
SEAT_OFF_KEY = 0x7FFFFFFF = 2147483647).

(A) WURZEL. ``ResumeWarm.eager_reached`` left an in-flight layer reserved
("one in flight stays reserved until its event completes"); the host eager
plan then wrote its scratch rows over the reserved ones (two writers on one
bank row: the warm copy and the eager fetch) and ``sync_tables`` read the
reserved key of a WRITTEN row as an expert id. ``old >= 0`` holds for
SEAT_OFF_KEY by design (an OFF row is "occupied"); the loop only knew the
OFF SEAT block, not the RW reservation. Now the writer: an eager pass
reaching an in-flight layer lands it first, ordered on the device (the
current stream waits for its event -- no host wait -- and commits the rows
to their experts). The reader: a key outside [-1, E) on a row the sync
touches, outside the OFF seat block, stops by name (W-POOL-KEY).

(B) WÄCHTER-LÜCKE. The rank died at 15:18:30; state.json lifecycle stayed
"serving" until DEADMAN_CRASH at 15:19:48 (78 s). Now the scheduler's
exception handler writes lifecycle ``dead`` (origin rank, code
RANK_EXCEPTION) through the one state writer, at once.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.layers.moe import expert_pool_device as epd  # noqa: E402
from flliper.srt.pdflip import state_file as sf  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_rw_finish_y4u", os.path.join(os.path.dirname(__file__), "test_pdflip_resume_warm_finish_287_0930.py"))
rwt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rwt)


class _EagerOps(rwt._Ops):
    """The eager pass (an extend, never a decode round) may order its stream
    behind an UNFINISHED warm event -- a device wait, still no host wait."""

    def current_waits(self, ev):
        self.log.append(("current_waits",) if ev.done else ("current_waits_inflight",))


class WriterEagerLandsTheInflightWarm(rwt._Base):
    def _inflight(self):
        os.environ[rwt.FINISH_ENV] = "1"
        rw, caches = self._armed(3)
        ops = rw._ops = _EagerOps()
        rw.tick(layers=1)                       # layer 0 warmed now
        rw.cancel("first_decode")               # layers 1, 2 on the side stream, not landed
        self.assertEqual(int(caches[1]._pool_tables.row_key[4]), epd.SEAT_OFF_KEY)
        return rw, caches, ops

    def test_y4u_an_eager_pass_on_an_inflight_layer_no_longer_meets_a_reservation(self):
        """RED on f50f51020f/4ffd70b9d4: the row stays SEAT_OFF_KEY, the eager
        pass writes it and the sync raises IndexError index 2147483647."""
        rw, caches, ops = self._inflight()
        c = caches[1]
        rw.eager_reached(c)
        self.assertIn(("current_waits_inflight",), ops.log)  # device order, no host wait
        self.assertFalse(any(getattr(e, "synced", False) for e in ops.events))
        self.assertNotIn(c, [x for x, _ev in rw._inflight])
        self.assertEqual(int(c._pool_tables.row_key[4]), 12)  # committed to its expert
        # the host eager plan writes its scratch row 4 with expert 5, then syncs
        epd.sync_tables(c._pool_tables, {4: 5}, keep_unwritten=True)
        self.assertEqual(int(c._pool_tables.row_key[4]), 5)
        self.assertEqual(int(c._pool_tables.hot_phys[12]), -1)
        self.assertEqual(epd.bijection_breaks(c._pool_tables), 0)
        # the other in-flight layer is untouched: it lands by its own event
        self.assertEqual(int(caches[2]._pool_tables.row_key[4]), epd.SEAT_OFF_KEY)
        ops.events[-1].done = True
        self.assertEqual(rw.promote_ready(), 1)
        self.assertEqual(rw._inflight, [])


class ReaderStopsByName(unittest.TestCase):
    def _t(self):
        return rwt._tables()

    def test_a_written_reserved_row_stops_by_name_not_with_an_index_error(self):
        t = self._t()
        epd.reserve_warm_rows(t, torch.tensor([4]))
        with self.assertRaises(epd.PoolRowKeyOutOfRange) as cm:
            epd.sync_tables(t, {4: 5}, keep_unwritten=True)
        msg = str(cm.exception)
        self.assertIn("W-POOL-KEY", msg)
        self.assertIn("row 4 holds key 2147483647", msg)
        self.assertIn("two writers", msg)
        self.assertEqual(int(t.row_key[4]), epd.SEAT_OFF_KEY)      # nothing moved

    def test_an_unwritten_reservation_under_keep_stays(self):
        t = self._t()
        epd.reserve_warm_rows(t, torch.tensor([5]))
        epd.sync_tables(t, {4: 5}, keep_unwritten=True)
        self.assertEqual(int(t.row_key[5]), epd.SEAT_OFF_KEY)
        self.assertEqual(int(t.row_key[4]), 5)

    def test_without_keep_a_reservation_stops_by_name(self):
        t = self._t()
        epd.reserve_warm_rows(t, torch.tensor([5]))
        with self.assertRaises(epd.PoolRowKeyOutOfRange):
            epd.sync_tables(t, {4: 5}, keep_unwritten=False)


class RankDeathReachesTheLifecycle(unittest.TestCase):
    def _serving(self, root):
        d = sf.init(root, "b1", "boot", {})
        sf.transition(d, "launching")
        sf.transition(d, "serving")
        return d

    def test_the_dying_rank_writes_dead_at_once(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._serving(root)
            exc = IndexError("index 2147483647 is out of bounds for dimension 0 with size 128")
            self.assertTrue(sf.note_rank_death(d, group="D", tp_rank=1, pp_rank=0, exc=exc))
            st = sf.read(d)
            self.assertEqual(st["lifecycle"]["state"], "dead")
            c = st["cause"]
            self.assertEqual((c["code"], c["origin"], c["group"], c["rank"], c["exception_type"], c["rc"]),
                             ("RANK_EXCEPTION", "rank", "D", "tp1pp0", "IndexError", sf.RC_DEAD_AFTER_SERVING))
            self.assertEqual(sf.health(d)[0], sf.HEALTH_DEAD)
            self.assertTrue(any(e.get("type") == "lifecycle" and (e.get("data") or {}).get("state") == "dead"
                                for e in sf.events(d)))
            # the first cause wins: a second rank's death changes nothing
            self.assertFalse(sf.note_rank_death(d, group="D", tp_rank=2, pp_rank=0, exc=RuntimeError("x")))
            self.assertEqual(sf.read(d)["cause"]["rank"], "tp1pp0")

    def test_no_state_dir_or_state_nothing_and_never_raises(self):
        self.assertFalse(sf.note_rank_death(None, group="D", tp_rank=0, pp_rank=0, exc=RuntimeError()))
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(sf.note_rank_death(root, group="D", tp_rank=0, pp_rank=0, exc=RuntimeError()))

    def test_the_rank_writer_owns_only_its_death(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._serving(root)
            with self.assertRaises(sf.StateFileError):
                sf.transition(d, "stopping", writer="rank")
            with self.assertRaises(sf.StateFileError):
                sf.transition(d, "dead", cause=sf.make_cause("X", "front"), writer="rank")

    def test_the_scheduler_writes_it_after_the_hold_before_the_sigquit(self):
        from flliper.srt.managers import scheduler

        src = inspect.getsource(scheduler.run_scheduler_process)
        i = src.index("Scheduler hit an exception")
        hold = src.index("debug_hold.maybe_hold(", i)
        death = src.index("note_rank_death(", i)
        sigquit = src.index("SIGQUIT", death)
        self.assertLess(hold, death)
        self.assertLess(death, sigquit)


if __name__ == "__main__":
    unittest.main()
