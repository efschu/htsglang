"""S59 (19.10.): D reads at the admission what the store holds.

Metal: NF rc15 (``boot_weg2_...dauer10190714_489909d138_1019_071429``) and rc14
(``...dauer10190107_92dd662831_1019_010742``), D log. S56 counted 13 R6/REROUTE_P
flips after which P prefilled less than X; these are their D refusals.

(A) CLAIM-END. The prefetch thread's claim reduce MINs page COUNTS while the
    Form A host (TP0) and the expert workers start their spans at different
    depths (the host at its mamba anchor 17856 / 16832 / 79808 / 73024, the
    workers at 0). The workers took the host's count from 0, the completion END
    MIN cut the group there, and D refused the difference -- always the host's
    start: ``X-GATE-TERMS`` extents 17984 / 18936 / 18080 / 16888 / 18028 /
    18207 / 18809, W123 ``covered=704``, ``store_short_after_flip`` 79660.
(B) VISION-FLOOR. The W123 vision guard priced the head arm (a worker's
    mamba-refused walk = 0) while the X gate admitted the Form A floor:
    pdflip-4-9 ``H98x X-FLOOR-CREDIT head=0 floor=95744`` -> ``W123 covered=0``.
(C) STORE-REREAD. A W31 on a floor that fell after the read while the group
    store arm held more (16-211 floor 0 / store 27200, 72-525 floor 0 / store
    68608): D refused without re-reading.

RED on 489909d138 (desk/flliper-nf-rc15-1019), GREEN on the S59 branch.
"""

from __future__ import annotations

import inspect
import queue
import threading
import types

import pytest
import torch

from flliper.srt.managers import cache_controller as cc
from flliper.srt.managers.cache_controller import HiCacheController
from flliper.srt.mem_cache import unified_radix_cache as urc

try:  # S59 module; absent on the base -> the old length reduce runs (RED)
    from flliper.srt.managers.claim_end_frame import note_claim_end_frame
except ImportError:  # pragma: no cover - the RED run
    def note_claim_end_frame(*_a, **_k):
        return None

PAGE = 64
WORLD = 3

# (boot, rid, total ids on D, host span start, host claim (tokens from its
#  start), worker span start, worker claim (null tier: its whole key),
#  metal extent D refused, metal P-computed after the flip, P cached)
CASES = [
    ("rc15", "pdflip-24-95", 84928, 17856, 66944, 0, 84864, 17984, 128, 84800),
    ("rc15", "pdflip-26-100", 120888, 17856, 101952, 0, 120832, 18936, 1080, 119808),
    ("rc15", "pdflip-26-102", 122464, 17856, 104384, 0, 122432, 18080, 224, 122240),
    ("rc15", "pdflip-30-117", 88952, 16832, 71040, 0, 87872, None, None, None),
    ("rc15", "pdflip-48-336", 80044, 79808, 192, 0, 80000, None, None, None),
    ("rc14", "pdflip-36-560", 56300, 17856, 38272, 0, 56256, 18028, 172, 56128),
    ("rc14", "pdflip-47-708", 115039, 17856, 96832, 0, 115008, 18207, 351, 114688),
    ("rc14", "pdflip-52-774", 74849, 73024, 704, 0, 74816, None, None, None),
    ("rc14", "pdflip-56-800", 116409, 17856, 97600, 0, 116352, 18809, None, 115456),
]
IDS = [c[1] for c in CASES]


class _Group:
    """Three ranks in threads; a real MIN/MAX over their tensors; a rank left
    alone in a collective is a named hang (barrier timeout)."""

    def __init__(self, timeout=3.0):
        self.barrier = threading.Barrier(WORLD, timeout=timeout)
        self.slots = {}
        self.calls = {r: 0 for r in range(WORLD)}
        self.lock = threading.Lock()

    def all_reduce(self, rank, tensor, op):
        self.calls[rank] += 1
        with self.lock:
            self.slots[rank] = tensor.clone()
        self.barrier.wait()
        stack = torch.stack([self.slots[r].reshape(-1) for r in range(WORLD)])
        red = stack.max(0).values if op == torch.distributed.ReduceOp.MAX else stack.min(0).values
        self.barrier.wait()
        tensor.copy_(red.reshape(tensor.shape).to(tensor.dtype))


def _ctl(rank, group, claim, threshold):
    ctl = types.SimpleNamespace()
    ctl.prefetch_sync_groups = [object()]
    ctl._agree_claim_form = lambda op: False  # H51 off: the live rc15 form (scalar)
    ctl._all_reduce_prefetch_groups = lambda t, op: group.all_reduce(rank, t, op)
    ctl._storage_hit_query = lambda op: (["h%d" % i for i in range(claim // PAGE)], claim)
    ctl.storage_stop_event = threading.Event()
    ctl.storage_stop_event.set()  # drain, then leave the loop
    ctl.prefetch_queue = queue.Queue()
    ctl.prefetch_io_aux_func = lambda: None
    ctl._prefetch_drained_after_stop = 0
    ctl.stops = []
    ctl._stop_group_from_thread = lambda exc: ctl.stops.append(exc)
    ctl.draft_cold_spans = {}
    ctl.prefetch_revoke_queue = queue.Queue()
    ctl.released = []
    ctl.append_host_mem_release = lambda idx, generation=None, **kw: ctl.released.append(len(idx))
    ctl.prefetch_threshold = threshold
    ctl.page_size = PAGE
    return ctl


def _claim_round(rid, bases, claims, host_base, threshold=0, frame=True, packed=False):
    """One store read through the REAL ``prefetch_thread_func`` on three ranks."""
    group = _Group()
    ctls, ops, errors = {}, {}, {}

    def _rank(r):
        try:
            ctl = ctls[r] = _ctl(r, group, claims[r], threshold)
            if packed:  # fnFL2x22 packed vote: the byteless workers abstain
                ctl._agree_claim_form = lambda op: True
                ctl.storage_backend = types.SimpleNamespace(abstains_from_claim_vote=(r != 0))
            if frame:
                note_claim_end_frame(ctl, rid, bases[r], host_base)
            op = ops[r] = types.SimpleNamespace(
                request_id=rid, host_indices=torch.arange(claims[r]),
                binding_generation=None, mark_terminate=lambda: None,
                is_terminated=lambda: False,
            )
            ctl.prefetch_queue.put(op)
            HiCacheController.prefetch_thread_func(ctl)
        except BaseException as e:  # noqa: BLE001
            errors[r] = e

    ts = [threading.Thread(target=_rank, args=(r,)) for r in range(WORLD)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(15)
    return ctls, ops, errors, group


def _group_end(bases, own):
    """The completion END MIN (HP1 ``_hp1_end_pack``) over the three ranks."""
    packs = [urc._hp1_end_pack(bases[r], own[r], own[r], urc._ANCHOR_ABSTAIN) for r in range(WORLD)]
    return min(p[0] for p in packs)


def _case(c):
    _boot, rid, total, hb, hc, wb, wc = c[:7]
    return rid, total, {0: hb, 1: wb, 2: wb}, {0: hc, 1: wc, 2: wc}, hb


# --------------------------------------------------------------------------- (A)


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_A_group_end_is_the_hosts_store_end(c):
    rid, total, bases, claims, hb = _case(c)
    ctls, ops, errors, group = _claim_round(rid, bases, claims, hb)
    assert errors == {} and all(not x.stops for x in ctls.values()), (errors, [x.stops for x in ctls.values()])
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    host_end = bases[0] + claims[0]
    # the host keeps its own claim; each worker reads to the host's END from ITS start
    assert own[0] == claims[0]
    assert own[1] == own[2] == host_end - bases[1], (rid, own)
    end = _group_end(bases, own)
    assert end == host_end, (rid, end, host_end)
    # every rank holds exactly its span up to the END (no tail kept past it)
    for r in range(WORLD):
        assert len(ops[r].host_indices) == own[r]
        assert len(ops[r].hash_value) == own[r] // PAGE
    metal_extent, p_computed, p_cached = c[7], c[8], c[9]
    if p_cached is not None:
        assert end == p_cached, "D now holds what P read from the same store"
    if metal_extent is not None:
        old_end = min(bases[r] + min(claims.values()) for r in range(WORLD))
        assert total - old_end == metal_extent, "the old reduce reproduces the metal W31 extent"
        assert total - end < metal_extent
    if p_computed is not None:
        assert total - end == p_computed, "D's extent is what P computed after the flip"


def test_A_48_336_no_store_short_after_flip():
    """rc15 08:04:22: TP0 ``completed_local=192 completed_synced=0`` -> #1068
    store_prefix_short -> D-NORECOMPUTE would_compute=79660 -> RESUME-VIA-P."""
    rid, total, bases, claims, hb = _case(CASES[4])
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb)
    assert errors == {}
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    end = _group_end(bases, own)
    assert urc._hp1_end_unpack(bases[0], end) == 192, "the host keeps its 3 pages"
    assert end == 80000 and total - end == 44  # D extends 44 (P later: cached 80000)


def test_A_52_774_w123_covered_is_the_hosts_end():
    """rc14 06:35:29: ``W123 ... covered=704`` (= the host's 704-token claim
    applied from the workers' start 0); the host's END was 73728."""
    rid, _total, bases, claims, hb = _case(CASES[7])
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb)
    assert errors == {}
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    assert _group_end(bases, own) == 73728


def test_A_revoke_is_one_verdict_on_every_rank():
    """48-336's host claim (192) under a threshold of 256 without a floor: every
    rank revokes -- the threshold reads the HOST's length of the group END,
    never a rank's own count (a worker's 80000 would transfer while the host
    revoked: the ranks split)."""
    rid, _t, bases, claims, hb = _case(CASES[4])
    ctls, ops, errors, _g = _claim_round(rid, bases, claims, hb, threshold=256)
    assert errors == {}
    revoked = {r: ctls[r].prefetch_revoke_queue.qsize() for r in range(WORLD)}
    assert revoked == {0: 1, 1: 1, 2: 1}, revoked


def test_A_one_claim_collective_per_read_on_every_rank():
    rid, _t, bases, claims, hb = _case(CASES[1])
    _c, _o, errors, group = _claim_round(rid, bases, claims, hb)
    assert errors == {}
    assert group.calls == {0: 1, 1: 1, 2: 1}, group.calls


def test_A_a_rank_never_reads_past_its_own_store_answer():
    """Wrong key = wrong KV: a rank whose own probe answered fewer pages than
    the group END (here a KV worker at 100000 of the host's 119808) reads only
    what its probe answered -- the completion MIN then takes the group there."""
    rid, _t, bases, _claims, hb = _case(CASES[1])
    claims = {0: 101952, 1: 100032, 2: 120832}
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb)
    assert errors == {}
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    assert own[1] <= claims[1] and own[2] <= claims[2]
    assert _group_end(bases, own) == 100032


def _packed_env(monkeypatch):
    from flliper.srt import rank_role

    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: False)
    monkeypatch.setattr(rank_role, "form_a_token_cut_active", lambda: False)


def test_A_packed_form_abstaining_workers_take_the_hosts_end(monkeypatch):
    _packed_env(monkeypatch)
    rid, total, bases, claims, hb = _case(CASES[1])
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb, packed=True)
    assert errors == {}, errors
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    assert own == {0: 101952, 1: 119808, 2: 119808}, own
    assert total - _group_end(bases, own) == 1080


def test_A_packed_abstainer_never_reads_past_its_own_probe(monkeypatch):
    """Wrong key = wrong KV, the abstaining form: a worker whose own probe was
    cut (terminated mid-walk: 100032 of its 120832) is not handed the host's
    END past it -- it would claim pages its own key list never answered."""
    _packed_env(monkeypatch)
    rid, _t, bases, _claims, hb = _case(CASES[1])
    claims = {0: 101952, 1: 100032, 2: 120832}
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb, packed=True)
    assert errors == {}, errors
    own = {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)}
    assert own[1] == 100032 and len(ops[1].hash_value) == 100032 // PAGE, own
    assert _group_end(bases, own) == 100032


def test_A_without_a_frame_the_reduce_is_unchanged():
    """Every non-Form-A read: the plain length MIN, byte for byte."""
    rid, _t, bases, claims, hb = _case(CASES[1])
    _c, ops, errors, _g = _claim_round(rid, bases, claims, hb, frame=False)
    assert errors == {}
    assert {r: int(ops[r].probed_hit_tokens) for r in range(WORLD)} == {0: 101952, 1: 101952, 2: 101952}


def test_A_the_tree_registers_the_frame_from_the_end_vote():
    """Through the REAL ``prefetch_from_storage`` (HP1/H99 harness): rc15
    26-100's bases -- host 17856, workers 0 -- land as each rank's claim frame
    (its own start, the reduced host start)."""
    # S59r: the HP1 harness lives in ../managers -- import it by path, so the
    # file runs alone (PYTHONPATH=python:<this dir>) as well as in a batch.
    import importlib
    import os
    import sys

    _mgr = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "managers")
    if _mgr not in sys.path:
        sys.path.insert(0, _mgr)
    hp1 = importlib.import_module("test_nf_form_a_end_vote_hp1")

    bases = {0: 17856, 1: 0, 2: 0}
    caches, _r, errors, _g = hp1._intake(bases, "pdflip-26-100", end=120832)
    assert errors == {}, errors
    frames = {r: tuple(getattr(caches[r].cache_controller, "pdflip_claim_end_frame", {})
                       .get("pdflip-26-100", ())) for r in caches}
    assert frames == {0: (17856, 17856), 1: (0, 17856), 2: (0, 17856)}, frames


def test_A_frame_pieces():
    from flliper.srt.managers import claim_end_frame as f

    fr = f.ClaimEndFrame(0, 17856)
    assert f.local_claim_vote(fr, 120832) == 120832
    assert f.local_claim_vote(f.ClaimEndFrame(17856, 17856), 101952) == 119808
    assert f.adopt_group_claim(fr, 119808, 120832) == (119808, 101952)
    assert f.adopt_group_claim(None, 101952, 120832) == (101952, 101952)
    ctl = types.SimpleNamespace()
    f.note_claim_end_frame(ctl, "r", 0, 17856)
    f.note_claim_end_frame(ctl, "r", None, None)  # a read without an END vote clears it
    assert f.take_claim_end_frame(ctl, types.SimpleNamespace(request_id="r")) is None


def test_A_the_loop_uses_the_end_reduce():
    src = inspect.getsource(HiCacheController.prefetch_thread_func)
    assert "HiCacheController._reduce_store_claim(" in src
    assert "_claim_threshold_count < revoke_threshold(self, operation)" in src


# --------------------------------------------------------------------------- (B)


def _vreq(image_end, total):
    it = types.SimpleNamespace(offsets=[(1000, image_end - 1)])
    return types.SimpleNamespace(
        rid="r", multimodal_inputs=types.SimpleNamespace(mm_items=[it]),
        full_untruncated_fill_ids=list(range(total)), output_ids=[],
        prefix_indices=[], host_hit_length=0,
    )


@pytest.mark.parametrize("rid,head,floor,total,image_end", [
    ("pdflip-4-9", 0, 95744, 95758, 95000),     # rc15 07:19:09 W123 covered=0
    ("pdflip-28-111", 0, 88704, 88952, 88000),  # rc15 07:31:05 W123 covered=0
])
def test_B_vision_guard_prices_the_admitted_floor(monkeypatch, rid, head, floor, total, image_end):
    from flliper.srt.managers import scheduler as sch
    from flliper.srt.managers import tp_match_floor as tmf
    from flliper.srt.pdflip import vision_d_guard as vdg

    monkeypatch.setattr(sch.tp_head_congruence, "group_match_for", lambda hi, r: head)
    monkeypatch.setattr(tmf, "group_usable_for", lambda tc, r: floor)
    monkeypatch.setattr(sch, "_pdflip_x_floor_credit_on", lambda: True)
    s = types.SimpleNamespace(tree_cache=object(), ps=types.SimpleNamespace(tp_size=3))
    req = _vreq(image_end, total)
    covered = sch.Scheduler._pdflip_vision_d_covered(s, req, object())
    assert covered == floor, (rid, covered)
    assert vdg.verdict(req, covered) == vdg.ADMIT
    assert vdg.verdict(req, head) == vdg.REFUSE  # the metal W123


def test_B_a_floor_below_the_head_is_what_is_admitted():
    from flliper.srt.pdflip import vision_d_guard as vdg

    assert vdg.admitted_covered(90000, 60000, True) == 60000
    assert vdg.admitted_covered(90000, 60000, False) == 60000
    assert vdg.admitted_covered(0, 95744, False) == 0  # credit off: the head, as before
    assert vdg.admitted_covered(5000, None, True) == 5000


# --------------------------------------------------------------------------- (C)


class _Req(types.SimpleNamespace):
    pass


def _sched(terms):
    from flliper.srt.managers import scheduler as sch

    s = types.SimpleNamespace(enable_hicache_storage=True, _pdflip_x_terms=terms)
    s._prefetch_deferral_refusal_reason = lambda reason="rate_limited": None
    return s, sch


@pytest.mark.parametrize("rid,terms", [
    ("pdflip-16-211", (38649, 0, 27200, 0)),    # rc14 05:41:25 W31 uncached=38649
    ("pdflip-72-525", (98592, 0, 68608, 0)),    # rc15 08:31:34 W31 uncached=98592
])
def test_C_store_reread_instead_of_the_refusal(rid, terms):
    s, sch = _sched(terms)
    req = _Req(rid=rid, prefetch_deferred=None)
    assert sch.Scheduler._pdflip_s59_reread_before_refusal(s, req) is True
    assert req.prefetch_deferred == sch._DEFER_REASON_STORE_SHORT
    # the read landed and brought nothing new: same terms -> the W31 stands
    req.prefetch_deferred = None
    assert sch.Scheduler._pdflip_s59_reread_before_refusal(s, req) is False


def test_C_bounded_and_store_not_above_the_floor():
    from flliper.srt.pdflip import d_store_reread as rr

    assert rr.reread_due((120888, 101952, 101888, 101952)) == (False, None)  # 26-100 pre-S59
    done = ((0, 27200), (10000, 30000))
    assert rr.reread_due((38649, 0, 40000, 20000), done)[0] is False  # MAX_REREADS
    assert rr.reread_due((38649, 0, 27200, 0), (), on=False) == (False, None)
    assert rr.reread_due((38649, None, 27200, 0)) == (False, None)  # no group opinion


def test_C_same_terms_same_verdict_on_every_rank():
    terms = (98592, 0, 68608, 0)
    out = []
    for _rank in range(WORLD):
        s, sch = _sched(terms)
        req = _Req(rid="pdflip-72-525", prefetch_deferred=None)
        out.append((sch.Scheduler._pdflip_s59_reread_before_refusal(s, req),
                    req.prefetch_deferred, req._s59_reread_keys))
    assert len(set(out)) == 1, out


def test_C_admission_loop_asks_before_it_refuses():
    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler)
    i = src.index("if self._pdflip_x_refuses(req, _head_inputs):")
    j = src.index("_x_refused.append(req)", i)
    assert "self._pdflip_s59_reread_before_refusal(req)" in src[i:j]
