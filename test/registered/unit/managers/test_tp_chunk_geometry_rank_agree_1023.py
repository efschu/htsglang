"""CHUNKGEOM (23.10., SOLL v2 R11; review deskq/done/qwen-chunkgeom-r1-1023.md): on a plain TP group the CHUNK LENGTH of
the carried continuation is the group's, not each rank's -- the second R11 hole of the RA class.

THE HOLE. ``PrefillAdder.add_chunked_req`` runs before the admission loop and passes no gate: it cuts the continuation at
``min(rem_chunk_tokens, rem_total_tokens)``, and ``rem_total_tokens`` is THIS rank's allocator + tree (scheduler.py's
#996 capture names it "exactly the 254-vs-301 shape"; the #681 floor caps it only where a group floor was published).
Same rid, same prefix, two chunk lengths: the RA riegel compared rids / HOL head only, ``assert_prefix_lens_rank_uniform``
(layers/dcp/prefix_lens_check.py) compares PREFIX lengths only and only on a DCP group, and the forward's per-token
all_reduces run with two token counts -> hang to the watchdog.

Two simulated ranks run the REAL ``PrefillAdder.add_chunked_req`` with the REAL ``tp_admission_agree`` wiring, then the
riegel, then a forward whose collective is keyed by its token count (a different count is a different collective, as in
NCCL). RED on 80268e7d04 (TP0 cuts 4096, TP1 3000: the forward pairs two token counts). GREEN: both cut at the MIN.

MUTANTS. (M1) the group MAX instead of the MIN -- a rank extends past its own pool (the danger direction): caught by the
per-rank budget assertion. (M2) no chunk MIN at all (only the riegel): the riegel's extent check stops both ranks by
name (``EXTEND-LEN SPLIT``) instead of a forward with two token counts.
"""

from __future__ import annotations

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flliper.srt.managers import schedule_policy as sp
from flliper.srt.managers import tp_admission_agree as ra
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import PrefillAdder
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.utils.common import Range

RID = "pdflip-9-77"
PREFIX = 8192
FILL = 20000
CHUNK = 4096
TP0_AVAILABLE = 100000  # the larger card: its own pool funds the whole chunk
TP1_AVAILABLE = 3000    # the smaller card: its own pool funds 3000


class _Hang(RuntimeError):
    pass


class _Mismatch(RuntimeError):
    pass


class _TPGroup:
    """Collective #i of every rank meets at one barrier; gathers pair whatever their site, an all_reduce at a different
    site (here: a different token count) is a different collective -> :class:`_Mismatch` (the real hang)."""

    def __init__(self, n=2, timeout=3.0):
        self.n, self.timeout = n, timeout
        self.lock = threading.Lock()
        self.seq = [0] * n
        self.slots, self.barriers = {}, {}
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
            raise _Hang(f"tp{rank} blocked in {kind}:{site} at collective #{i}") from None
        slot = self.slots[i]
        kinds = {k for k, _, _ in slot}
        sites = {s for k, s, _ in slot if k == "all_reduce"}
        if len(kinds) > 1 or len(sites) > 1:
            raise _Mismatch(f"collective #{i}: " + ", ".join(f"tp{r}={k}:{s}" for r, (k, s, _) in enumerate(slot)))
        return slot

    def gather(self, rank, site, payload):
        return [p for _, _, p in self._arrive(rank, "all_gather_object", "gather", payload)]

    def all_reduce(self, rank, site):
        self._arrive(rank, "all_reduce", site, None)


class _Alloc:
    """A desk allocator (no ``get_kvcache``: not a QSA pool, grain 1)."""

    def __init__(self, available):
        self.available = int(available)

    def available_size(self):
        return self.available

    def full_available_size(self):
        return self.available

    def swa_available_size(self):
        return 0


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


def _adder(available):
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_Alloc(available),
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=CHUNK,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _continuation():
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = list(range(PREFIX))
    req.full_untruncated_fill_ids = list(range(FILL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=256, ignore_eos=False)
    req.retracted_stain = False
    req.pdflip_lane_parked = False
    req.multimodal_inputs = None
    req.extend_range = Range(PREFIX, PREFIX)

    def _set(start, end):
        req.extend_range = Range(int(start), int(end))

    req.set_extend_range.side_effect = _set
    return req


def _sched(group, tp_rank, tp_size=2):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=tp_size, pp_size=1),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=list(range(tp_size))),
        tp_cpu_group=None,
        server_args=SimpleNamespace(enable_dp_attention=False),
        _form_a_is_host=lambda: None,
    )
    s._form_a_tp_gather = lambda site, payload: group.gather(tp_rank, site, payload)
    return s


def _rank_pass(group, tp_rank, available, out, chunk_min_override=None):
    """The scheduler's order (``_get_new_batch_prefill_raw``): adder + RA install, the carried continuation
    (``add_chunked_req``), the riegel, the forward (its collectives keyed by the pass's token count)."""
    try:
        sched = _sched(group, tp_rank)
        adder = _adder(available)
        adder.form_a_admission_follow = None
        adder.tp_admission_agree = ra.make_agree(sched)
        if chunk_min_override is not None:
            adder.tp_admission_agree.chunk_min = chunk_min_override(sched, tp_rank)
        req = _continuation()
        local_budget = min(CHUNK, int(adder.rem_total_tokens))
        adder.add_chunked_req(req)
        ra.riegel(sched, adder.tp_admission_agree, adder.can_run_list, None)
        tokens = sum(r.extend_range.length for r in adder.can_run_list)
        group.all_reduce(tp_rank, f"forward[{tokens}]")
        out[tp_rank] = ("ok", tokens, local_budget)
    except Exception as exc:  # noqa: BLE001
        out[tp_rank] = ("raise", exc, None)


def _run(available_by_rank, **kw):
    group = _TPGroup(n=len(available_by_rank))
    out = {}
    ts = [threading.Thread(target=_rank_pass, args=(group, r, a, out), kwargs=kw, daemon=True)
          for r, a in enumerate(available_by_rank)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    return group, out


class ChunkGeometryRankAgreeTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("FLLIPER_TP_ADMISSION_AGREE", "FLLIPER_PDFLIP_GROUP"):
            os.environ.pop(k, None)
        sp._PDFLIP_CHUNK_ADMIT = None
        sp._PDFLIP_PARK_ON = None
        self.addCleanup(lambda: (setattr(sp, "_PDFLIP_CHUNK_ADMIT", None), setattr(sp, "_PDFLIP_PARK_ON", None)))
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_same_prefix_two_pools_one_chunk_length(self):
        """THE HOLE, red on 80268e7d04: TP0 cuts 4096, TP1 3000 -> the forward pairs two token counts."""
        group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE])
        fails = {r: v[1] for r, v in out.items() if v[0] == "raise"}
        self.assertEqual(fails, {}, f"tp0={group.logs[0]} tp1={group.logs[1]} -- {fails}")
        self.assertEqual(out[0][1], out[1][1])
        self.assertEqual(out[0][1], TP1_AVAILABLE)  # the group's MIN
        self.assertEqual(group.logs[0], group.logs[1])

    def test_no_rank_extends_past_its_own_pool(self):
        """The danger direction: the agreed length is never above any rank's own budget."""
        _group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE])
        for r, v in out.items():
            self.assertEqual(v[0], "ok", v)
            self.assertLessEqual(v[1], v[2], f"tp{r} extends {v[1]} rows past its own budget {v[2]}")

    def test_mutant_group_max_is_caught(self):
        """M1: a MAX instead of the MIN makes TP1 extend 4096 rows on a 3000-row pool."""

        def _max(sched, tp_rank):
            def f(rid, local):
                got = sched._form_a_tp_gather(ra.SITE_CHUNK, ("chunk", str(rid), int(local)))
                return max(int(v[2]) for v in got)
            return f

        _group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE], chunk_min_override=_max)
        over = [r for r, v in out.items() if v[0] == "ok" and v[1] > v[2]]
        self.assertEqual(over, [1], "the MAX mutant must be the one that over-extends a rank")

    def test_mutant_without_the_chunk_min_the_riegel_stops_by_name(self):
        """M2: the continuation cut rank-locally -> the riegel's extent check names it on BOTH ranks before the forward."""
        _group, out = _run([TP0_AVAILABLE, TP1_AVAILABLE],
                           chunk_min_override=lambda sched, tp_rank: (lambda rid, local: int(local)))
        for r in (0, 1):
            self.assertEqual(out[r][0], "raise", out[r])
            self.assertIsInstance(out[r][1], ra.RankAgreeSplit)
            self.assertIn("EXTEND-LEN SPLIT", str(out[r][1]))

    def test_equal_pools_unchanged(self):
        group, out = _run([TP0_AVAILABLE, TP0_AVAILABLE])
        self.assertEqual({v[0] for v in out.values()}, {"ok"})
        self.assertEqual(out[0][1], CHUNK)
        self.assertEqual(group.logs[0], group.logs[1])

    def test_continuation_on_one_rank_only_is_a_named_stop(self):
        """A rank carrying a continuation its peer does not (it is at the riegel) pairs its chunk gather with the peer's
        riegel gather: both name it, nobody hangs."""
        group = _TPGroup(n=2)
        out = {}

        def chunk_rank():
            try:
                out[0] = ra.chunk_min(RID, 3000, gather=lambda p: group.gather(0, "c", p), tp_rank=0)
            except Exception as exc:  # noqa: BLE001
                out[0] = exc

        def riegel_rank():
            try:
                out[1] = ra.extend_set_riegel([], None, 0, gather=lambda p: group.gather(1, "s", p), tp_rank=1)
            except Exception as exc:  # noqa: BLE001
                out[1] = exc

        ts = [threading.Thread(target=chunk_rank, daemon=True), threading.Thread(target=riegel_rank, daemon=True)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.assertIsInstance(out[0], ra.RankAgreeSplit)
        self.assertIn("CALL-COUNT SPLIT", str(out[0]))
        self.assertIsInstance(out[1], ra.RankAgreeSplit)

    def test_riegel_extent_unit(self):
        """Same rids, different lengths -> EXTEND-LEN SPLIT; same lengths -> nothing."""
        a = SimpleNamespace(rid="r", extend_range=Range(10, 4106))
        b = SimpleNamespace(rid="r", extend_range=Range(10, 3010))
        for pair, split in (((a, a), False), ((a, b), True)):
            group = _TPGroup(n=2)
            out = {}

            def run(r, req):
                try:
                    out[r] = ra.extend_set_riegel([req], None, 1, gather=lambda p: group.gather(r, "s", p), tp_rank=r)
                except Exception as exc:  # noqa: BLE001
                    out[r] = exc

            ts = [threading.Thread(target=run, args=(r, q), daemon=True) for r, q in enumerate(pair)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(10)
            for r in (0, 1):
                if split:
                    self.assertIsInstance(out[r], ra.RankAgreeSplit)
                    self.assertIn("EXTEND-LEN SPLIT", str(out[r]))
                else:
                    self.assertIsNone(out[r])


if __name__ == "__main__":
    unittest.main()
