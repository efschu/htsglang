"""H105: on a Form A D group the attention host's ADMISSION verdict is the
group's; the expert workers take it, and a split is a named stop.

THE DEATH (dpr, rc12j 3e97ef0c8f, boot dkrnfh91dprbar1dauer09270832, D log
203833-203988). weg2-14-70 arrived at D with a host-backed hit: KV key match
81856, mamba anchor 75264, uncached 6817. Every rank matched 75264 (#1042),
every rank passed the X gate ("uncached=6817 ... verdict=admit") and the #794
group chunk (3712). Then:

* TP1/TP2 built the extend -- ``#969 EXTENT n=68 ... ('weg2-14-', 75264,
  78976, 75264, 3712)`` -- and entered the forward;
* TP0 printed nothing for the rid: no #969, no ``#988 LOADBACK`` (logged
  unconditionally for every host load-back), no ``WEG2-ARENA-LOAD``, no
  ``WEG2-LOADBACK-WAIT`` (never printed in the whole boot, and it prints its
  first three). So TP0 left ``add_one_req`` BEFORE ``init_load_back`` -- the
  only rank-local exits there are the ``total_tokens >= rem_total_tokens``
  gates (NO_TOKEN; ``batch_is_full`` then keeps the rid in the queue). The
  watchdog's batch on TP0 at 08:56:59 is the 5-request DECODE batch
  (12-55, 14-73, 14-62, 14-65, 14-71), without 14-70; TP0's pool then:
  ``available=99008, evictable=704``.

WHY TP0 ALONE. The gate prices ``fill - len(prefix_indices)``. On TP0 (the only
rank with KV and arena bytes) the 75264-token hit is host-backed, 0 device
rows until the load-back AFTER the gate: 82081 rows + reservation against its
pool. The workers entered with ``prefix_indices`` = 75264 (#969, and no #988
on them either) and priced 6817. Same rid, same budget arithmetic, two
verdicts -- the arena pin census (pinned=2436) was flat from 08:50:59 on and
is never reached before ``init_load_back``.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL
``Scheduler._form_a_admission_follow_fn`` / ``_form_a_extend_set_riegel``
(bound to a stand-in scheduler; the TP broadcast is a mailbox the host fills
first). RED on 9417507cd2: the host refuses, the worker admits. GREEN with H105.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers import tp_match_floor as m
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

#: dpr weg2-14-70 geometry.
FILL = 82081
HOST_HIT = 75264
TP0_AVAILABLE = 80000  # below the host's price (82081 + 64 + 1), far above a worker's (6817 + 65)


def _tree_cache():
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _running_batch():
    b = MagicMock()
    b.reqs = []
    return b


def _adder(available):
    return PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(available),
        running_batch=_running_batch(),
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=4096,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req(*, device_prefix: int):
    """weg2-14-70 as a rank sees it at ``add_one_req``: TP0 with the hit on
    the host (0 device rows), a worker with it as device rows."""
    req = MagicMock(spec=Req)
    req.rid = "weg2-14-70"
    req.priority = 0
    req.prefix_indices = list(range(device_prefix))
    req.full_untruncated_fill_ids = list(range(FILL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


class _Group:
    """Three ranks of one Form A group. ``exchange`` is the TP broadcast: the
    host posts, every rank reads the host's post (the host runs first)."""

    def __init__(self):
        self.mailbox = {}

    def scheduler(self, tp_rank):
        s = SimpleNamespace(
            ps=SimpleNamespace(tp_size=3, pp_size=1),
            tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
            tp_cpu_group=None,
        )

        def _exchange(site, payload):
            if payload is not None:
                self.mailbox[site] = payload
            return self.mailbox.get(site)

        s._form_a_tp_exchange = _exchange
        for name in (
            "_form_a_is_host",
            "_form_a_admission_follow_fn",
            "_form_a_extend_set_riegel",
        ):
            fn = getattr(Scheduler, name, None)
            if fn is not None:
                setattr(s, name, types.MethodType(fn, s))
        return s


def _install(adder, sched, tp_rank):
    """What ``_get_new_batch_prefill_raw`` does right after building the
    adder (absent on the base: the gate stays rank-local, which is the red)."""
    fn = getattr(sched, "_form_a_admission_follow_fn", None)
    if fn is None:
        return
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=tp_rank != 0
    ):
        adder.form_a_admission_follow = fn()


class DprAdmissionIsTheHostsTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _run_dpr(self):
        group = _Group()
        results, adders, reqs = {}, {}, {}
        # the host first: its verdict is on the wire before a worker reads it
        for tp_rank, device_prefix in ((0, 0), (1, HOST_HIT), (2, HOST_HIT)):
            adder = _adder(TP0_AVAILABLE)
            _install(adder, group.scheduler(tp_rank), tp_rank)
            req = _req(device_prefix=device_prefix)
            results[tp_rank] = adder.add_one_req(req, truncation_align_size=None)
            adders[tp_rank], reqs[tp_rank] = adder, req
        return results, adders, reqs

    def test_dpr_every_rank_takes_the_same_verdict(self):
        """THE DEATH: host NO_TOKEN, workers built the extend."""
        results, adders, _ = self._run_dpr()
        self.assertEqual(results[0], AddReqResult.NO_TOKEN)
        self.assertEqual(
            {results[1], results[2]},
            {AddReqResult.NO_TOKEN},
            f"the workers admitted weg2-14-70 ({results[1]}, {results[2]}) while the "
            "attention host refused it -- the dpr split: workers in the extend's "
            "collectives, the host in a decode pass",
        )
        for r in (1, 2):
            self.assertEqual(adders[r].can_run_list, [])

    def test_refused_request_is_parked_not_lost(self):
        """A host refusal leaves the request whole on every rank: nothing in a
        batch, the prefix untouched -- the waiting queue keeps it for the next
        pass (the scheduler's NO_TOKEN branch never pops it)."""
        results, adders, reqs = self._run_dpr()
        for r in (0, 1, 2):
            self.assertEqual(adders[r].can_run_list, [])
        self.assertEqual(len(reqs[0].prefix_indices), 0)
        self.assertEqual(len(reqs[1].prefix_indices), HOST_HIT)

    def test_host_admit_is_followed_by_a_worker_that_would_refuse(self):
        """The other direction: the host admits, a worker's bookkeeping pool
        says NO_TOKEN -- the worker follows the host."""
        self.assertTrue(hasattr(m, "form_a_admission_verdict"), "H105 absent")
        box = {}

        def ex(p):
            if p is not None:
                box["v"] = p
            return box["v"]

        self.assertEqual(
            m.form_a_admission_verdict("r1", "ADMIT", is_host=True, exchange=ex), "ADMIT"
        )
        self.assertEqual(
            m.form_a_admission_verdict("r1", "NO_TOKEN", is_host=False, exchange=ex),
            "ADMIT",
        )

    def test_rid_split_is_a_named_stop(self):
        self.assertTrue(hasattr(m, "form_a_admission_verdict"), "H105 absent")
        with self.assertRaises(m.FormAAdmissionSplit) as cm:
            m.form_a_admission_verdict(
                "weg2-14-71",
                "ADMIT",
                is_host=False,
                exchange=lambda p: ("weg2-14-70", "ADMIT", 1, 2),
            )
        self.assertIn("H105 RU FORM-A ADMISSION SPLIT", str(cm.exception))

    def test_extend_set_split_is_a_named_stop(self):
        """The riegel after the loop: the dpr batch shapes (host nothing,
        worker 14-70 [75264, 78976)) stop by name before the forward."""
        group = _Group()
        host = group.scheduler(0)
        worker = group.scheduler(1)
        self.assertTrue(hasattr(worker, "_form_a_extend_set_riegel"), "H105 absent")
        built = MagicMock()
        built.rid = "weg2-14-70"
        built.extend_range = SimpleNamespace(start=75264, end=78976)
        with patch.object(m, "form_a_follow_active", return_value=True):
            with patch.object(m, "this_rank_follows", return_value=False):
                host._form_a_extend_set_riegel([])
            with patch.object(m, "this_rank_follows", return_value=True):
                with self.assertRaises(m.FormAAdmissionSplit) as cm:
                    worker._form_a_extend_set_riegel([built])
        self.assertIn("EXTEND-SET SPLIT", str(cm.exception))

    def test_extend_set_agreement_passes(self):
        group = _Group()
        host, worker = group.scheduler(0), group.scheduler(2)
        self.assertTrue(hasattr(worker, "_form_a_extend_set_riegel"), "H105 absent")
        b = MagicMock()
        b.rid = "weg2-14-70"
        b.extend_range = SimpleNamespace(start=75264, end=78976)
        with patch.object(m, "form_a_follow_active", return_value=True):
            with patch.object(m, "this_rank_follows", return_value=False):
                host._form_a_extend_set_riegel([b])
            with patch.object(m, "this_rank_follows", return_value=True):
                worker._form_a_extend_set_riegel([b])

    def test_off_a_form_a_group_nothing_is_installed(self):
        """qwen27b / classic boots: no follow callable, no broadcast -- the
        gates in add_one_req stay rank-local and byte-identical."""
        group = _Group()
        s = group.scheduler(0)
        self.assertTrue(hasattr(s, "_form_a_admission_follow_fn"), "H105 absent")
        with patch.object(m, "form_a_follow_active", return_value=False):
            self.assertIsNone(s._form_a_admission_follow_fn())
            s._form_a_extend_set_riegel([MagicMock()])  # no exchange, no raise
        self.assertEqual(group.mailbox, {})
        s.ps.tp_size = 1
        self.assertIsNone(s._form_a_admission_follow_fn())
        s.ps.tp_size, s.ps.pp_size = 3, 3
        self.assertIsNone(s._form_a_admission_follow_fn())

    def test_unfollowed_adder_returns_the_first_gate_unchanged(self):
        """Without a follow callable the pre-lock gate returns before the lock,
        exactly as on the base (no lock taken)."""
        adder = _adder(TP0_AVAILABLE)
        req = _req(device_prefix=0)
        self.assertEqual(adder.add_one_req(req, truncation_align_size=None), AddReqResult.NO_TOKEN)
        adder.tree_cache.inc_lock_ref.assert_not_called()


class _Channel:
    """The TP broadcast as it really is: one ordered stream from the host;
    every worker consumes the host's posts in order, whatever call site it is
    at (a verdict call and a riegel call are the same broadcast_pyobj pair)."""

    def __init__(self):
        self.posts = []
        self.read = {1: 0, 2: 0}

    def exchange(self, tp_rank):
        def _ex(payload):
            if tp_rank == 0:
                self.posts.append(payload)
                return payload
            i = self.read[tp_rank]
            self.read[tp_rank] += 1
            return self.posts[i]

        return _ex


class GateCallCountTest(unittest.TestCase):
    """Coordinator point 1: the number of gate calls is the collective count.
    A rank whose loop makes a different number of gate calls than the host
    stops by NAME at the next broadcast -- never a hang, never a guess."""

    def setUp(self):
        self.assertTrue(hasattr(m, "form_a_extend_set_check"), "H105 absent")

    def test_worker_skips_the_rid_the_host_gates(self):
        """The host gates weg2-14-70 then ends its loop; the worker skipped
        14-70 rank-locally and gates weg2-14-71 -> named SPLIT."""
        ch = _Channel()
        m.form_a_admission_verdict("weg2-14-70", "NO_TOKEN", is_host=True, exchange=ch.exchange(0))
        m.form_a_extend_set_check([], is_host=True, exchange=ch.exchange(0))
        with self.assertRaises(m.FormAAdmissionSplit) as cm:
            m.form_a_admission_verdict("weg2-14-71", "ADMIT", is_host=False, exchange=ch.exchange(1))
        self.assertIn("ADMISSION SPLIT", str(cm.exception))

    def test_worker_loop_ends_while_the_host_is_at_a_gate(self):
        """The host made one gate call, the worker none (its loop broke
        earlier): the worker's riegel reads the host's verdict -> named."""
        ch = _Channel()
        m.form_a_admission_verdict("weg2-14-70", "ADMIT", is_host=True, exchange=ch.exchange(0))
        with self.assertRaises(m.FormAAdmissionSplit) as cm:
            m.form_a_extend_set_check([], is_host=False, exchange=ch.exchange(1))
        self.assertIn("EXTEND-SET MALFORMED", str(cm.exception))

    def test_worker_at_a_gate_while_the_host_is_in_the_riegel(self):
        """The mirror: the host's loop ended, the worker is still at a gate."""
        ch = _Channel()
        m.form_a_extend_set_check([], is_host=True, exchange=ch.exchange(0))
        with self.assertRaises(m.FormAAdmissionSplit) as cm:
            m.form_a_admission_verdict("weg2-14-70", "ADMIT", is_host=False, exchange=ch.exchange(1))
        self.assertIn("ADMISSION MALFORMED", str(cm.exception))

    def test_same_calls_same_stream(self):
        """Equal loops consume exactly the host's stream -- nothing left over,
        nothing short: the collective count matches by construction."""
        ch = _Channel()
        for rid, code in (("a", "ADMIT"), ("b", "NO_TOKEN")):
            m.form_a_admission_verdict(rid, code, is_host=True, exchange=ch.exchange(0))
        m.form_a_extend_set_check([("a", 0, 64)], is_host=True, exchange=ch.exchange(0))
        for r in (1, 2):
            self.assertEqual(
                m.form_a_admission_verdict("a", "NO_TOKEN", is_host=False, exchange=ch.exchange(r)),
                "ADMIT",
            )
            self.assertEqual(
                m.form_a_admission_verdict("b", "ADMIT", is_host=False, exchange=ch.exchange(r)),
                "NO_TOKEN",
            )
            m.form_a_extend_set_check([("a", 0, 64)], is_host=False, exchange=ch.exchange(r))
            self.assertEqual(ch.read[r], len(ch.posts))


class NoStarvationTest(unittest.TestCase):
    """Coordinator point 2: refused in pass n (the full host-backed span does
    not fit TP0's budget), admitted in pass n+k once room frees -- the ranks
    agree in every pass, and the wait is on the log with both prices."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        m._ADMISSION_WAIT.clear() if hasattr(m, "_ADMISSION_WAIT") else None

    def _pass(self, available):
        group = _Group()
        out = {}
        for tp_rank, device_prefix in ((0, 0), (1, HOST_HIT), (2, HOST_HIT)):
            adder = _adder(available)
            _install(adder, group.scheduler(tp_rank), tp_rank)
            req = _req(device_prefix=device_prefix)
            out[tp_rank] = (adder.add_one_req(req, truncation_align_size=None), adder)
        return out

    def test_refused_then_admitted_ranks_agree(self):
        self.assertTrue(hasattr(m, "_ADMISSION_WAIT"), "H105 absent")
        with self.assertLogs(m.logger, level="INFO") as logs:
            n1 = self._pass(TP0_AVAILABLE)  # pass n: the host refuses
            n2 = self._pass(TP0_AVAILABLE)  # pass n+1: still no room
            n3 = self._pass(200_000)  # pass n+k: room freed
        for r in (0, 1, 2):
            self.assertEqual(n1[r][0], AddReqResult.NO_TOKEN, r)
            self.assertEqual(n2[r][0], AddReqResult.NO_TOKEN, r)
            self.assertNotEqual(n3[r][0], AddReqResult.NO_TOKEN, r)
            self.assertEqual(len(n3[r][1].can_run_list), 1, r)
        text = "\n".join(logs.output)
        self.assertIn("H105 RU FORM-A ADMISSION WAIT rid=weg2-14-70 host=NO_TOKEN", text)
        # both prices on the line: the host's full span, the worker's uncached rest
        self.assertIn(f"host_price={FILL + 64 + 1}", text)
        self.assertIn(f"local_price={FILL - HOST_HIT + 64 + 1}", text)
        self.assertIn("refusals=2", text)
        # both worker ranks share this test process (and its wait ledger), so
        # the admission line counts their refusals together: 2 passes x 2 ranks
        self.assertIn("H105 RU FORM-A ADMISSION AFTER-WAIT rid=weg2-14-70 refusals=4", text)
        self.assertIn("host_budget=200000", text)


if __name__ == "__main__":
    unittest.main()
