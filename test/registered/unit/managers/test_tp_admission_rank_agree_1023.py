"""RA (23.10., SOLL v2 R11): on a plain TP group the admission verdict is the group's, and the built set is checked
before any branch reads it -- a split is a named stop, never two ranks in two different collectives.

THE DEATH (27B NVFP4 duo rc35n4duo5 = 77bdedf7b4, boot ...duo10231145_..._114625, D log 641479-641536, WEDGE-STACK
12:27:04Z, W17 12:33:12Z). Group D: TP2 uneven (5090 + 3080, DCP), pp 1, no Form A. pdflip-36-127 (N=113072,
d_uncached=1, ``price=117169``) was refused 186x by the ``lifetime`` gate on both ranks; at 12:26:33Z TP0's local
budget crossed the price and TP0 admitted it (``#988 LOADBACK``, ``D-ADMIT-AFTER-REFUSALS refusals=186``), TP1's did
not. TP1 took the D-HOL head and entered PW-R's all_reduce MIN (``pick_unservable_head`` ->
``_pdflip_group_min_flags``), TP0 entered ``prepare_for_extend`` -> ``assert_prefix_lens_rank_uniform``'s all_reduce:
FORWARD-HANG forward_ct 11140 on both ranks.

Two simulated ranks run in two threads through the REAL ``PrefillAdder.add_one_req`` (TP0 with more free rows than
TP1, the same request), the REAL ``tp_admission_agree`` wiring the scheduler installs, and the scheduler's post-loop
order (S35 result, riegel, PW-R with the REAL ``pick_unservable_head``, prepare_for_extend's collective). The fake TP
group pairs collectives by sequence number like gloo and records each rank's collective order; two different
collectives at one sequence number (or a peer that never arrives) is the hang. RED on 93eae72472 (the module does not
exist, every gate is rank-local: TP0 = [dcp_prefix_lens], TP1 = [pw_r_group_min]). GREEN with RA.
"""

from __future__ import annotations

import os
import re
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flliper.srt.managers import schedule_policy as sp
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.pdflip import d_wall_head
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

try:
    from flliper.srt.managers import tp_admission_agree as ra
except ImportError:  # the base: no agreement, every gate rank-local (the red)
    ra = None

#: boot 15 geometry: pdflip-36-127, N=113072, max_new 4096 -> price 113072 + 4096 + 1 = 117169
FILL = 113072
MAX_NEW = 4096
PRICE = FILL + MAX_NEW + 1
TP0_AVAILABLE = 130000  # the 5090's view at 12:26:33Z: the budget crossed the price
TP1_AVAILABLE = 116000  # the 3080's view: still below it
RID = "pdflip-36-127"


class _Hang(RuntimeError):
    pass


class _Mismatch(RuntimeError):
    pass


class _TPGroup:
    """N ranks; collective #i of every rank meets at one barrier. ``all_gather_object`` calls pair with each other
    whatever their call site (one kind on the TP cpu group); an ``all_reduce`` at a different call SITE is a
    different collective (other group / shape: PW-R's int64[1] on the cpu group vs the DCP prefix check) -- in the
    real group that is the hang, here it is :class:`_Mismatch`. A peer that never arrives is :class:`_Hang`."""

    def __init__(self, n=2, timeout=3.0):
        self.n = n
        self.timeout = timeout
        self.lock = threading.Lock()
        self.seq = [0] * n
        self.slots = {}
        self.barriers = {}
        self.logs = [[] for _ in range(n)]

    def _arrive(self, rank, kind, site, payload):
        with self.lock:
            i = self.seq[rank]
            self.seq[rank] += 1
            slot = self.slots.setdefault(i, [None] * self.n)
            slot[rank] = (kind, site, payload)
            bar = self.barriers.setdefault(i, threading.Barrier(self.n, timeout=self.timeout))
        self.logs[rank].append(f"{kind}:{site}")
        try:
            bar.wait()
        except threading.BrokenBarrierError:
            raise _Hang(f"tp{rank} blocked in {kind}:{site} at collective #{i}: no peer arrives") from None
        slot = self.slots[i]
        kinds = {k for k, _, _ in slot}
        sites = {s for k, s, _ in slot if k == "all_reduce"}
        if len(kinds) > 1 or len(sites) > 1:
            raise _Mismatch(f"collective #{i}: " + ", ".join(f"tp{r}={k}:{s}" for r, (k, s, _) in enumerate(slot)))
        return slot

    def gather(self, rank, site, payload):
        return [p for _, _, p in self._arrive(rank, "all_gather_object", "gather", payload)]

    def all_reduce_min(self, rank, site, vals):
        slot = self._arrive(rank, "all_reduce", site, list(vals))
        return [min(col) for col in zip(*[p for _, _, p in slot])]


def _isolate(testcase):
    env = patch.dict(os.environ)
    env.start()
    testcase.addCleanup(env.stop)
    os.environ.pop("FLLIPER_PDFLIP_GROUP", None)
    os.environ.pop("FLLIPER_TP_ADMISSION_AGREE", None)
    sp._PDFLIP_CHUNK_ADMIT = None
    sp._PDFLIP_PARK_ON = None

    def _reset():
        sp._PDFLIP_CHUNK_ADMIT = None
        sp._PDFLIP_PARK_ON = None

    testcase.addCleanup(_reset)


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


def _adder(available):
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(available),
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=4096,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req(rid=RID):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(FILL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=MAX_NEW, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


def _sched(group, tp_rank, tp_size=2, pp_size=1):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=tp_size, pp_size=pp_size),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=list(range(tp_size))),
        tp_cpu_group=None,
        server_args=SimpleNamespace(enable_dp_attention=False),
    )
    s._form_a_tp_gather = lambda site, payload: group.gather(tp_rank, site, payload)
    s._form_a_is_host = types.MethodType(Scheduler._form_a_is_host, s)  # real: None off a Form A plan
    return s


def _rank_pass(group, tp_rank, available, out):
    """One rank's admission pass for the one queued request, in the scheduler's order (scheduler.py
    ``_get_new_batch_prefill_raw``): build adder + install, add_one_req, S35 / RA result, the riegels, PW-R under
    ``_hol.head``, then -- only with a non-empty extend -- prepare_for_extend's DCP prefix check."""
    try:
        sched = _sched(group, tp_rank)
        adder = _adder(available)
        adder.form_a_admission_follow = None
        if ra is not None:
            adder.tp_admission_agree = ra.make_agree(sched)
        req = _req()
        waiting = [req]
        res = adder.add_one_req(req, truncation_align_size=None)
        added = bool(adder.can_run_list) and adder.can_run_list[-1] is req
        if ra is not None:
            res = ra.uniform_result(res, added, adder.tp_admission_agree)
        hol_head = str(req.rid) if res == AddReqResult.NO_TOKEN else None  # HolPass.may_overtake
        if ra is not None:
            ra.riegel(sched, adder.tp_admission_agree, adder.can_run_list, hol_head)
        if hol_head is not None:
            d_wall_head.pick_unservable_head(
                head_rid=hol_head,
                waiting=waiting,
                tree=None,
                available=available,
                group_min=lambda flags: group.all_reduce_min(tp_rank, "pw_r_group_min", flags),
            )
        if adder.can_run_list:
            group.all_reduce_min(tp_rank, "dcp_prefix_lens", [len(adder.can_run_list[0].prefix_indices)])
        out[tp_rank] = ("ok", res, list(adder.can_run_list))
    except Exception as exc:  # noqa: BLE001 -- recorded per rank, asserted by the test
        out[tp_rank] = ("raise", exc, None)


def _run(available_by_rank, target=_rank_pass):
    group = _TPGroup(n=len(available_by_rank))
    out = {}
    threads = [
        threading.Thread(target=target, args=(group, r, a, out), daemon=True)
        for r, a in enumerate(available_by_rank)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    return group, out


class Boot15DeadlockReproTest(unittest.TestCase):
    def setUp(self):
        _isolate(self)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_boot15_ranks_enter_the_same_collectives(self):
        """THE DEATH: TP0 admits, TP1 refuses -> TP0 in the DCP prefix check, TP1 in PW-R's MIN."""
        group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE])
        failures = {r: v[1] for r, v in out.items() if v[0] == "raise"}
        self.assertEqual(
            failures, {},
            f"collective order per rank: tp0={group.logs[0]} tp1={group.logs[1]} -- {failures}",
        )
        self.assertEqual(group.logs[0], group.logs[1], "the ranks entered different collectives")

    def test_boot15_no_rank_admits_against_a_peer_refusal(self):
        """The danger direction: the group verdict is the MIN -- TP1's pool cannot fund 117169 rows, so NO rank
        admits (a MAX / local verdict would make TP0 run an extend TP1 never builds, or TP1 admit past its pool)."""
        group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE])
        self.assertTrue(all(v[0] == "ok" for v in out.values()), out)
        self.assertEqual(out[0][2], [], "TP0 admitted while TP1's pool refused")
        self.assertEqual(out[1][2], [])
        self.assertEqual({out[0][1], out[1][1]}, {AddReqResult.NO_TOKEN})

    def test_both_pools_fund_it_both_admit(self):
        group, out = _run([TP0_AVAILABLE, TP0_AVAILABLE])
        self.assertTrue(all(v[0] == "ok" for v in out.values()), out)
        self.assertEqual([r.rid for r in out[0][2]], [RID])
        self.assertEqual([r.rid for r in out[1][2]], [RID])
        self.assertEqual(group.logs[0], group.logs[1])
        self.assertEqual(group.logs[0][-1], "all_reduce:dcp_prefix_lens")

    def test_both_refuse_both_enter_pw_r(self):
        group, out = _run([TP1_AVAILABLE, TP1_AVAILABLE])
        self.assertTrue(all(v[0] == "ok" for v in out.values()), out)
        self.assertEqual(group.logs[0], group.logs[1])
        self.assertEqual(group.logs[0][-1], "all_reduce:pw_r_group_min")


class TpAdmissionAgreeUnitTest(unittest.TestCase):
    def setUp(self):
        _isolate(self)
        self.assertIsNotNone(ra, "RA absent (managers/tp_admission_agree.py)")

    def _gate_pair(self, payloads):
        """Both ranks call gate_verdict at once with (rid, gate); returns per-rank outcome."""
        group = _TPGroup(n=len(payloads))
        out = {}

        def run(r):
            rid, gate = payloads[r]
            try:
                out[r] = ra.gate_verdict(rid, gate, gather=lambda p: group.gather(r, "g", p), tp_rank=r,
                                         price=r, budget=10 * r)
            except Exception as exc:  # noqa: BLE001
                out[r] = exc

        ts = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(len(payloads))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        return out

    def test_min_is_the_first_refusal_in_tp_order(self):
        out = self._gate_pair([("r", None), ("r", AddReqResult.NO_TOKEN), ("r", AddReqResult.OTHER)])
        self.assertEqual(out, {0: "NO_TOKEN", 1: "NO_TOKEN", 2: "NO_TOKEN"})

    def test_all_admit_admits(self):
        self.assertEqual(self._gate_pair([("r", None), ("r", None)]), {0: "ADMIT", 1: "ADMIT"})

    def test_rid_split_is_a_named_stop(self):
        out = self._gate_pair([("pdflip-36-127", None), ("pdflip-38-128", None)])
        for r in (0, 1):
            self.assertIsInstance(out[r], ra.RankAgreeSplit)
            self.assertIn("RID SPLIT", str(out[r]))

    def test_gate_against_riegel_is_a_named_stop_on_both(self):
        """One rank's loop ended (riegel) while the peer is still at a gate: the two gathers pair and both name it."""
        group = _TPGroup(n=2)
        out = {}

        def gate_rank():
            try:
                out[0] = ra.gate_verdict("r", None, gather=lambda p: group.gather(0, "g", p), tp_rank=0)
            except Exception as exc:  # noqa: BLE001
                out[0] = exc

        def riegel_rank():
            try:
                out[1] = ra.extend_set_riegel([], None, 0, gather=lambda p: group.gather(1, "s", p), tp_rank=1)
            except Exception as exc:  # noqa: BLE001
                out[1] = exc

        ts = [threading.Thread(target=gate_rank, daemon=True), threading.Thread(target=riegel_rank, daemon=True)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        for r in (0, 1):
            self.assertIsInstance(out[r], ra.RankAgreeSplit, out)
            self.assertIn("CALL-COUNT SPLIT", str(out[r]))

    def _riegel_pair(self, a, b):
        group = _TPGroup(n=2)
        out = {}

        def run(r, args):
            try:
                out[r] = ra.extend_set_riegel(*args, gather=lambda p: group.gather(r, "s", p), tp_rank=r)
            except Exception as exc:  # noqa: BLE001
                out[r] = exc

        ts = [threading.Thread(target=run, args=(0, a), daemon=True),
              threading.Thread(target=run, args=(1, b), daemon=True)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        return out

    def test_extend_set_split_is_a_named_stop(self):
        built = SimpleNamespace(rid=RID)
        out = self._riegel_pair(([built], None, 1), ([], RID, 1))
        for r in (0, 1):
            self.assertIsInstance(out[r], ra.RankAgreeSplit)
            self.assertIn("EXTEND-SET SPLIT", str(out[r]))

    def test_hol_head_split_is_a_named_stop(self):
        """Both admitted nothing, but only one rank holds a HOL head (a rank-local NO_TOKEN behind the gate:
        AgreedRoomShort / chain mismatch) -- PW-R's MIN would be entered by one rank: named stop instead."""
        out = self._riegel_pair(([], RID, 1), ([], None, 1))
        for r in (0, 1):
            self.assertIsInstance(out[r], ra.RankAgreeSplit)
            self.assertIn("EXTEND-SET SPLIT", str(out[r]))

    def test_agreement_passes(self):
        built = SimpleNamespace(rid=RID)
        self.assertEqual(self._riegel_pair(([built], None, 1), ([built], None, 1)), {0: None, 1: None})

    def test_uniform_result(self):
        self.assertEqual(ra.uniform_result(AddReqResult.NO_TOKEN, True, object()), AddReqResult.CONTINUE)
        self.assertEqual(ra.uniform_result(AddReqResult.NO_TOKEN, False, object()), AddReqResult.NO_TOKEN)
        self.assertEqual(ra.uniform_result(AddReqResult.NO_TOKEN, True, None), AddReqResult.NO_TOKEN)
        self.assertEqual(ra.uniform_result(AddReqResult.OTHER, True, object()), AddReqResult.OTHER)

    def test_applies_only_on_a_plain_tp_group(self):
        g = _TPGroup(n=2)
        self.assertTrue(ra.applies(_sched(g, 0)))
        self.assertFalse(ra.applies(_sched(g, 0, tp_size=1)))
        self.assertFalse(ra.applies(_sched(g, 0, pp_size=3)))
        s = _sched(g, 0)
        s.server_args = SimpleNamespace(enable_dp_attention=True)
        self.assertFalse(ra.applies(s))
        s = _sched(g, 0)
        s._form_a_is_host = lambda: True  # Form A: H105 owns the verdict
        self.assertFalse(ra.applies(s))
        self.assertFalse(ra.applies(_sched(g, 0), env={"FLLIPER_TP_ADMISSION_AGREE": "0"}))
        self.assertTrue(ra.applies(_sched(g, 0), env={}))  # default ON

    def test_off_the_group_add_one_req_is_unchanged(self):
        """tp 1: nothing installed, the local gate returns as before (no collective)."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        g = _TPGroup(n=1)
        self.assertIsNone(ra.make_agree(_sched(g, 0, tp_size=1)))
        adder = _adder(TP1_AVAILABLE)
        self.assertIsNone(adder.tp_admission_agree)
        self.assertEqual(adder.add_one_req(_req(), truncation_align_size=None), AddReqResult.NO_TOKEN)
        self.assertEqual(g.logs[0], [])


class SchedulerWiringTest(unittest.TestCase):
    """The scheduler installs the agreement with the adder and runs the riegel AFTER the loop and BEFORE PW-R."""

    def test_install_riegel_and_pw_r_order(self):
        import sys

        src = Path(sys.modules[Scheduler.__module__].__file__).read_text()  # the scheduler actually imported
        body = src[src.index("def _get_new_batch_prefill_raw("):]
        body = body[: body.index("\n    def ", 10)]
        i_install = body.index("adder.tp_admission_agree = (")
        i_loop = body.index("# Get requests from the waiting queue to a new prefill batch")
        i_riegel = body.index("_tp_admission_agree.riegel(")
        i_pw_r = body.index("_pdflip_d_wall_head.pick_unservable_head(")
        i_extend = body.index("new_batch.prepare_for_extend()")
        self.assertLess(i_install, i_loop)
        self.assertLess(i_loop, i_riegel)
        self.assertLess(i_riegel, i_pw_r)
        self.assertLess(i_riegel, i_extend)
        self.assertRegex(body, re.compile(r"_tp_admission_agree\.uniform_result\("))


if __name__ == "__main__":
    unittest.main()
