# SPDX-License-Identifier: Apache-2.0
"""SP (partial park, KV trigger; note tmp/pa/TEILPARKEN_notiz_0927.md). A seat is free, but an OLDER
waiting request did not fit D's KV (adder NO_TOKEN) while a younger one runs -> the youngest pauses
(retain: its KV leaves the device only as far as the older needs, LRU eviction, tail last). The
NO_TOKEN verdict is read through the group MIN; at most one displacement per pass, never an older.
_partial_keep calls NF's #248 keep_role through a guarded import."""

import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_park_runtime as DPR  # noqa: E402
from sglang.srt.weg2 import d_seats as DS  # noqa: E402

MOD = "sglang.srt.weg2.handoff_pending"


def R(n):
    return f"weg2-12-{n}"


def _req(rid, span=1000, prefix=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * span, output_ids=[],
                                 prefix_indices=[0] * prefix)


def _sched(running, waiting, cap=6, no_token=None, group_min=None, avail=100):
    batch = types.SimpleNamespace(reqs=list(running), released=[], spec_algorithm=None)
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices])
    sched = types.SimpleNamespace(
        waiting_queue=list(waiting), server_args=types.SimpleNamespace(max_running_requests=cap),
        _weg2_sa_no_token=no_token, calls=[],
        tree_cache=types.SimpleNamespace(page_size=64),
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: avail))
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)

    def gm(flags):
        sched.calls.append(list(flags))
        return group_min(flags) if group_min else flags

    sched._weg2_group_min_flags = gm
    return sched, batch


def _run(fn, env=None):
    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DPR, "seat_cap", lambda s: None), \
         mock.patch.dict(os.environ, env or {}):
        return fn()


class KvTrigger(unittest.TestCase):
    def test_older_refused_for_kv_displaces_the_youngest_with_a_free_seat(self):
        sched, batch = _sched([_req(R(5)), _req(R(9))], [_req(R(3), span=5000)], no_token=R(3))
        with self.assertLogs(DPR.logger, level="WARNING") as cap:
            got = _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(got, R(9))
        self.assertEqual(batch.released, [(R(9), True)])
        self.assertEqual(DS.park_site(sched.waiting_queue[-1]), DS.SITE_PRESSURE)
        self.assertIn("trigger=kv", cap.output[0])
        # with #248 present keep_role answers a page count, without it the requested window is logged
        self.assertRegex(cap.output[0], r"pages_out=(window=\d+-\d+(\(retain\))?|\d+)")
        self.assertIsNone(sched._weg2_sa_no_token, "the signal is consumed")
        self.assertEqual(sched.calls, [[True]], "the verdict went through the group MIN")

    def test_ranks_disagree_no_displacement(self):
        sched, batch = _sched([_req(R(5)), _req(R(9))], [_req(R(3))], no_token=R(3),
                              group_min=lambda f: [False])
        self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(batch.released, [])

    def test_no_refusal_no_displacement_but_the_collective_is_entered(self):
        sched, batch = _sched([_req(R(5)), _req(R(9))], [_req(R(3))], no_token=None)
        self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(sched.calls, [[False]], "every rank enters: the precondition is replicated")

    def test_a_younger_refused_never_displaces_an_older(self):
        sched, batch = _sched([_req(R(2)), _req(R(4))], [_req(R(9))], no_token=R(9))
        self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(sched.calls, [], "no candidate: no collective (replicated)")

    def test_sub_switch_off_seat_trigger_only(self):
        sched, batch = _sched([_req(R(5)), _req(R(9))], [_req(R(3))], no_token=R(3))
        self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch),
                               {"SGLANG_WEG2_SEAT_AGE_KV_DISPLACE": "0"}))

    def test_seat_trigger_unchanged(self):
        sched, batch = _sched([_req(R(5)), _req(R(9))], [_req(R(3))], cap=2)
        with self.assertLogs(DPR.logger, level="WARNING") as cap:
            self.assertEqual(_run(lambda: DPR.displace_for_age(sched, batch)), R(9))
        self.assertIn("trigger=seat", cap.output[0])
        self.assertIn("pages_out=0", cap.output[0])


class PartialKeep(unittest.TestCase):
    def test_window_from_the_shortfall(self):
        # older needs 5000, 100 free -> shortfall 4900 -> 77 pages of 64; victim span 1000 -> 16 pages
        sched, _ = _sched([], [], avail=100)
        victim, older = _req(R(9), span=10000), _req(R(3), span=5000)
        # "no module": once #248 is imported the package attribute answers ``from ... import``,
        # so the attribute is hidden as well as the sys.modules entry
        import sglang.srt.weg2 as _w2
        with mock.patch.dict(sys.modules, {MOD: None}), \
             mock.patch.object(_w2, "handoff_pending", None, create=True):
            self.assertEqual(DPR._partial_keep(sched, victim, older), "window=80-157(retain)")

    def test_keep_role_called_with_the_window_and_its_return_logged(self):
        m = types.ModuleType(MOD)
        m.calls = []
        m.keep_role = lambda rid, role, page_range=None, *, page_keys=None, page_size=0: (
            m.calls.append((rid, role, page_range, page_size)) or 42)
        sched, _ = _sched([], [], avail=100)
        import sglang.srt.weg2 as pkg

        with mock.patch.dict(sys.modules, {MOD: m}), mock.patch.object(pkg, "handoff_pending", m, create=True):
            got = DPR._partial_keep(sched, _req(R(9), span=10000), _req(R(3), span=5000))
        self.assertEqual(got, "42")
        self.assertEqual(m.calls, [(R(9), "park", (80, 157), 64)])

    def test_a_raising_keep_role_falls_back_to_retain(self):
        m = types.ModuleType(MOD)

        def boom(*a, **k):
            raise RuntimeError("ledger")

        m.keep_role = boom
        sched, _ = _sched([], [], avail=100)
        import sglang.srt.weg2 as pkg

        with mock.patch.dict(sys.modules, {MOD: m}), mock.patch.object(pkg, "handoff_pending", m, create=True):
            self.assertIn("(retain)", DPR._partial_keep(sched, _req(R(9), span=10000), _req(R(3), span=5000)))

    def test_seat_trigger_is_a_pure_pause(self):
        sched, _ = _sched([], [])
        self.assertEqual(DPR._partial_keep(sched, _req(R(9)), None), "0")


class Wiring(unittest.TestCase):
    def test_scheduler_sets_the_signal_on_no_token(self):
        from sglang.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i = src.index("if res == AddReqResult.NO_TOKEN:")
        self.assertIn("self._weg2_sa_no_token = str(req.rid)", src[i:i + 600])


if __name__ == "__main__":
    unittest.main()


class TensorFields(unittest.TestCase):
    """Review 28.09. (the 7e227e5f76 class, 27B rc12z9 D 08:23:24): on D
    ``prefix_indices`` is a torch tensor; ``x or ()`` asks its truth value,
    which raises for an empty AND for a multi-element tensor. _partial_keep
    swallowed it (``except: return "?"``) -- the KV window was never computed
    and keep_role never called: partial parking was silently off on the metal."""

    def test_window_with_tensor_prefix_indices(self):
        import torch

        sched, _ = _sched([], [], avail=100)
        victim, older = _req(R(9), span=10000), _req(R(3), span=5000)
        for prefix in (torch.zeros(0, dtype=torch.int64), torch.zeros(64, dtype=torch.int64)):
            older.prefix_indices = prefix
            victim.prefix_indices = prefix
            import sglang.srt.weg2 as pkg

            stub = types.ModuleType(MOD)  # NF's module without keep_role: the window is logged
            with mock.patch.dict(sys.modules, {MOD: stub}), \
                    mock.patch.object(pkg, "handoff_pending", stub, create=True):
                got = DPR._partial_keep(sched, victim, older)
            need = 5000 - len(prefix)
            n_pages = -(-(need - 100) // 64)
            b = -(-10000 // 64)
            self.assertEqual(got, f"window={b - n_pages}-{b}(retain)", f"prefix={len(prefix)}")
