# SPDX-License-Identifier: Apache-2.0
"""L15-INT: hermetic CPU integration test of the whole L1.5 chain.

Every AP (A wake restore, B fallback/defer, C2 l2_of, D refill, E1
group_check, E2 cap-0 gate, F2 keep-arm, F7 shared prefix) was tested
against its own fakes. This file drives the seams between them on REAL CPU
pool objects and the fake-self wake style:

  sleep (retain_at_sleep per rank, F7 shared-prefix shape, manifests per
  (group, rank)) -> manifest content asserts -> wake per rank (cap>0 keeps
  the hold; cap-0 votes None -> group verdict "fallback" -> fallback drop
  frees everything) -> failing keep-arm (F2) drops the whole round.

Geometry: 3-rank D group, uneven prefix (0, 1, 2, 3) -> S = 3; rank r owns
global slots with L % 3 == r. Two requests share their first three slots
(10, 11, 12) -- one per class -- and the same mamba anchor; unique tails
13 (rank 1) and 14 (rank 2). need = [1, 2, 2] -> blocks 2 -> L_H = 6.
"""

from __future__ import annotations

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.weg2 import l15_manifest, l15_retain
from sglang.srt.weg2.l15_policy import Candidate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
MAMBA_SIZE = 6
MAX_CONTEXT_LEN = 128

PREFIX = (0, 1, 2, 3)
EPOCH = 777
PID = 4242

# F7 shape: sa/sb share (10, 11, 12); tails 13 (rank1) / 14 (rank2); both
# end on the same radix node -> the SAME mamba anchor 5.
SLOTS_OF = {"sa": (10, 11, 12, 13), "sb": (10, 11, 12, 14)}
ANCHOR_SLOT_OF = {"sa": 5, "sb": 5}

CANDIDATES = [
    Candidate(rid="sa", kind="served", last_active=9.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
    Candidate(rid="sb", kind="served", last_active=8.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
]


def _build_pool() -> HybridReqToTokenPool:
    """Real CPU HybridReqToTokenPool (scaffold from retain_cpu_pool_1001)."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=512, n_groups=4,
            num_heads=8, head_dim=64, state_size=32, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SIZE, mamba_spec_state_size=10,
        max_context_len=MAX_CONTEXT_LEN, device="cpu",
        enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=mamba_layers, enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )


def _mamba_views(pool: HybridReqToTokenPool):
    cache = pool.mamba_pool.mamba_cache
    views = []
    for c in cache.conv:
        views.extend(c[i] for i in range(int(c.shape[0])))
    views.extend(cache.temporal[i] for i in range(int(cache.temporal.shape[0])))
    return views


class FakeAllocator:
    def __init__(self, size=16):
        self.size = size
        self.free_pages = torch.empty(0, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)


class FakeNode:
    pass


def _recorder(node, kv_map, anchor_map, visited):
    node.rewritten = (dict(kv_map), dict(anchor_map))


def _retain_rank(tmp_path, rank, caps=(4, 4, 4), manifest_name=None, mamba=None,
                 pid=PID):
    """One rank's sleep: retain_at_sleep against fresh per-rank fakes."""
    pool = mamba if mamba is not None else _build_pool()
    alloc = FakeAllocator()
    alloc.clear()
    nodes = {"sa": FakeNode(), "sb": FakeNode()}
    res = l15_retain.retain_at_sleep(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=lambda rid: SLOTS_OF[rid],
        anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        rewrite_tree=_recorder,
        caps_rows_by_rank=caps,
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=rank,
        epoch=EPOCH,
        pid=pid,
        kv_buffers=[torch.zeros(8, 3), torch.zeros(8, 3)],
        mamba_buffers=_mamba_views(pool),
        allocator=alloc,
        mamba_allocator=pool.mamba_allocator,
        reset_keep=lambda ns: None,
        set_keep=lambda buf, spans: None,
        manifest_path=str(tmp_path / (manifest_name or "l15_manifest.json")),
        log=lambda line: None,
    )
    return res, pool, alloc, nodes


def test_sleep_writes_per_rank_manifests_f7_shape(tmp_path):
    # STEP 1+2 (brief): sleep on ranks 1 and 2 (cap>0); manifests written;
    # same fingerprint; remapped slots; shared slots once per holder; l2 set.
    res1, _pool1, _a1, nodes1 = _retain_rank(tmp_path, 1,
                                             manifest_name="m_rank1.json")
    res2, _pool2, _a2, nodes2 = _retain_rank(tmp_path, 2,
                                             manifest_name="m_rank2.json")
    assert res1 is not None and res2 is not None, "F7 shape must retain"
    m1 = l15_manifest.from_bytes((tmp_path / "m_rank1.json").read_bytes())
    m2 = l15_manifest.from_bytes((tmp_path / "m_rank2.json").read_bytes())
    # The fingerprint covers the hold, not the rank: equal inputs -> equal fp.
    assert l15_manifest.fingerprint(m1) == l15_manifest.fingerprint(m2)
    assert m1.epoch == EPOCH and m2.epoch == EPOCH
    for m in (m1, m2):
        spans = {s.rid: s for s in m.spans}
        assert set(spans) == {"sa", "sb"}
        sa, sb = spans["sa"], spans["sb"]
        # L_H = 6; 10, 11, 12, 13, 14 are ALL >= L_H and move to the free
        # class rows: 10 -> 1, 11 -> 2, 12 -> 3, 13 -> 4, 14 -> 5 (slot 0 is
        # the reserved padding and never a target).
        assert sa.slots == (1, 2, 3, 4)
        assert sb.slots == (1, 2, 3, 5)
        # Shared images identical at the same indices, once per holder.
        assert sa.slots[:3] == sb.slots[:3]
        assert len(set(sa.slots)) == 4 == len(set(sb.slots))
        # Both rids share the ONE anchor image; A_H = 2 -> anchor 5 -> 1.
        assert sa.anchor_slot == sb.anchor_slot == 1
        # l2 columns present (C2: arena page slots recorded per span).
        assert sa.l2_slots == (201, 202) and sb.l2_slots == (201, 202)
        assert sa.l2_gens == (5, 6)
    # The tree rewrite saw the per-rid maps through the same seam.
    kv_map_a, anc_a = nodes1["sa"].rewritten
    kv_map_b, _ = nodes1["sb"].rewritten
    for shared in (10, 11, 12):
        assert kv_map_a[shared] == kv_map_b[shared]
    assert anc_a == {5: 1}  # the shared anchor moves once, both rids see it
    # Mamba allocator: the two compact anchor rows are carved out.
    free = set(_pool1.mamba_allocator.free_slots.tolist())
    assert 1 not in free


# ---------------------------------------------------------------------------
# PART 2: the wake half -- hold signal, group verdict, fallback drop.
# Fake-self style from test_weg2_l15_wake_restore_1001 / cap0_wake; the
# manifests are the REAL ones written by the sleep step above.
# ---------------------------------------------------------------------------

import os  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import l15_restore  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager
ALLOC_SIZE = 16
HELD_SLOTS = {1, 2, 3, 4, 5}  # span union of the part-1 geometry (no pad 0)
A_H = 2                       # anchor region: shared anchor image 1 + pad 0


class _Pool:
    def __init__(self):
        self.clear_calls = []

    def clear(self, *a, **k):
        self.clear_calls.append((a, k))


class _Alloc:
    def __init__(self):
        self.size = ALLOC_SIZE
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.clears = 0

    def clear(self):
        self.clears += 1
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)


class _Tree:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _sched():
    # tp_worker.model_runner.token_to_kv_pool must exist: the hold signal
    # derives the planner cap via l15_shadow.cell_bytes_from(pool) and
    # returns "no signal" (None) when the pool is missing.
    cell_t = torch.zeros(4, 2, 2)
    sched = SimpleNamespace(
        tp_size=3,
        server_args=SimpleNamespace(tp_size=3, rank_gpu_id=None),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(k_buffer=[cell_t],
                                                 v_buffer=[cell_t]))),
        req_to_token_pool=_Pool(),
        token_to_kv_pool_allocator=_Alloc(),
        tree_cache=_Tree(),
        memory_saver_adapter=SimpleNamespace(set_keep_byte_spans=lambda b, s: None),
    )
    return sched


def _fake_self(sched, rank):
    fs = SimpleNamespace()
    fs.scheduler = sched
    fs._l15_wake_manifest = None
    fs._l15_wake_refill = False
    fs._weg2_group_name = lambda: "D"
    fs._weg2_rank = lambda: rank
    fs._l15_wake_hold_signal = lambda: WU._l15_wake_hold_signal(fs)
    fs._l15_flush_zero_kv_bounded = lambda s, k: None
    fs._l15_clear_tms_keep_spans = lambda s: None
    fs._l15_fallback_drop = lambda s: WU._l15_fallback_drop(fs, s)
    fs._l15_wake_act = lambda s, v, **kw: WU._l15_wake_act(fs, s, v, **kw)
    fs.flushed = []
    fs.flush_cache = lambda: fs.flushed.append(1) or False
    return fs


def _env(monkeypatch, tmp_path, mib):
    monkeypatch.setenv("SGLANG_WEG2_L15", "1")
    monkeypatch.setenv("SGLANG_WEG2_L15_MIB", mib)
    monkeypatch.setenv("SGLANG_WEG2_L15_MANIFEST", str(tmp_path) + os.sep)


def _sleep_all(tmp_path, ranks, pid):
    """Sleep on the given ranks; real manifests land at manifest_path(D, r)."""
    for r in ranks:
        path = l15_manifest.manifest_path("D", r, os.environ)
        res, _p, _a, _n = _retain_rank(tmp_path, r, manifest_name=path, pid=pid)
        assert res is not None, f"sleep skipped on rank {r}"


def _hold_votes(fs):
    m = fs._l15_wake_manifest
    if m is None:
        return None
    fp = l15_manifest.fingerprint(m)
    return l15_restore.check_vote(fp, len(m.spans), 0, 0, ())


def test_wake_cap0_votes_none_group_fallback_frees_every_pool(monkeypatch, tmp_path):
    # STEP 3 (brief): sleep on all 3 ranks; at wake rank 0 has planner cap 0
    # (no c0 in the MIB env) and its manifest lacks anchor identity -> the
    # anchor gate closes and the hold signal votes None; the cap>0 ranks keep
    # (free_pages lacks the held slots, mamba rows [1, A_H) stay); the group
    # sees a mixed fp -> verdict "fallback" -> the drop frees EVERY pool.
    _env(monkeypatch, tmp_path, mib="c1=64,c2=64")
    pid = os.getpid()
    _sleep_all(tmp_path, (0, 1, 2), pid)

    # cap>0 ranks 1 and 2: the hold path survives the restore.
    keep_ranks = {}
    for r in (1, 2):
        sched = _sched()
        fs = _fake_self(sched, r)
        assert WU._weg2_wake_restore_pools(fs) is True
        assert fs.flushed == []
        # mamba anchors [0, A_H) survive: keep_mamba_rows == anchor_slots.
        assert sched.req_to_token_pool.clear_calls == [
            ((), {"keep_mamba_rows": A_H})
        ]
        free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
        assert not free & HELD_SLOTS, "held slots must stay out of free_pages"
        keep_ranks[r] = (sched, fs)

    # cap-0 rank 0: anchor gate closed -> votes None, pools untouched.
    sched0 = _sched()
    fs0 = _fake_self(sched0, 0)
    m, rank, keep, master_on = WU._l15_wake_hold_signal(fs0)
    assert (m, rank, keep, master_on) == (None, 0, 0, True)
    assert sched0.token_to_kv_pool_allocator.clears == 0

    # The group decision on the real votes: mixed (None vs fp) -> fallback.
    votes = [None, _hold_votes(keep_ranks[1][1]), _hold_votes(keep_ranks[2][1])]
    gc = l15_restore.group_check(votes)
    assert gc.verdict == "fallback"

    # The act on EVERY rank: fallback drop, then all pools fully free.
    assert WU._l15_wake_act(fs0, sched0, "fallback", group_ok=True,
                            master_on=True) == 0
    for r in (1, 2):
        sched, fs = keep_ranks[r]
        dropped = WU._l15_wake_act(fs, sched, "fallback", group_ok=True,
                                   master_on=True)
        assert dropped == len(HELD_SLOTS)
    for sched in (sched0, keep_ranks[1][0], keep_ranks[2][0]):
        free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
        assert free == set(range(1, ALLOC_SIZE + 1)), "every pool must be free"
        assert sched.tree_cache.resets == 1
        # the last req clear ran WITHOUT a keep (the hold was dropped).
        assert sched.req_to_token_pool.clear_calls[-1] == ((), {})


def test_f2_failing_keep_arm_manifest_gone_group_fallback(monkeypatch, tmp_path):
    # STEP 4 (brief): rank 2's keep-arm failed at sleep (F2) -> its manifest
    # is gone; rank 0 is cap-0. Nobody keeps anything at the end.
    _env(monkeypatch, tmp_path, mib="c1=64")
    pid = os.getpid()
    _sleep_all(tmp_path, (0, 1), pid)  # rank 2 never published (keep-arm failed)

    sched1 = _sched()
    fs1 = _fake_self(sched1, 1)
    assert WU._weg2_wake_restore_pools(fs1) is True
    free = {int(x) for x in sched1.token_to_kv_pool_allocator.free_pages.tolist()}
    assert not free & HELD_SLOTS

    fs_missing = _fake_self(_sched(), 2)
    m2 = WU._l15_wake_hold_signal(fs_missing)[0]
    assert m2 is None, "a rank whose keep-arm failed has no manifest to vote"

    gc = l15_restore.group_check([None, _hold_votes(fs1), None])
    assert gc.verdict == "fallback"

    WU._l15_wake_act(fs_missing, fs_missing.scheduler, "fallback",
                     group_ok=True, master_on=True)
    assert WU._l15_wake_act(fs1, sched1, "fallback", group_ok=True,
                            master_on=True) == len(HELD_SLOTS)
    free = {int(x) for x in sched1.token_to_kv_pool_allocator.free_pages.tolist()}
    assert free == set(range(1, ALLOC_SIZE + 1))
    assert sched1.req_to_token_pool.clear_calls[-1] == ((), {})


# ---------------------------------------------------------------------------
# PART 3 (L15-INT3): the E1X decide wiring on the REAL env the next boot
# uses -- SGLANG_WEG2_L15=1, SGLANG_WEG2_L15_MIB="c1=7616,c2=1792", REFILL
# unset. The 27B cell (32768 B/row) makes caps_from_env (0, 243712, 57344):
# rank 0 (card 0, unnamed) holds nothing; ranks 1/2 sleep their manifests.
# At wake EVERY rank runs _l15_decide_wake_verdict (ONE all_gather_object
# over scheduler.world_group.cpu_group, the same group object everywhere,
# xsn410); mixed votes -> "fallback" on every rank -> the drop frees all
# pools. Unanimous fingerprints -> "hold".
# ---------------------------------------------------------------------------

from sglang.srt.weg2 import l15_shadow  # noqa: E402

CELL_27B = 32768                       # bytes per token row, 27B geometry
REAL_MIB = "c1=7616,c2=1792"           # the boot-1926 card plan
CAPS_27B = (0, 243712, 57344)          # 7616/32768, 1792/32768 rows


def _env_real(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, mib=REAL_MIB)
    monkeypatch.delenv("SGLANG_WEG2_L15_REFILL", raising=False)


def _sleep_cap_ranks(tmp_path):
    """Sleep on the two cap>0 ranks with the REAL caps; rank 0 holds nothing
    (cap 0 -> it never retains, no manifest exists at its path)."""
    for r in (1, 2):
        path = l15_manifest.manifest_path("D", r, os.environ)
        res, _p, _a, _n = _retain_rank(tmp_path, r, caps=CAPS_27B,
                                       manifest_name=path, pid=os.getpid())
        assert res is not None, f"sleep skipped on rank {r}"


def _wake_ranks():
    """Three fake D ranks on the SAME cpu group object: rank 0 only runs the
    hold signal (no manifest -> votes None); ranks 1/2 restore and hold."""
    grp = object()  # one identity: the group every gather must see
    fss, scheds = [], []
    for r in range(3):
        sched = _sched()
        # the 27B cell: one row of K and of V at 16384 B each.
        t = torch.zeros(1, 8, 1024, dtype=torch.bfloat16)
        sched.tp_worker.model_runner.token_to_kv_pool = SimpleNamespace(
            k_buffer=[t], v_buffer=[t])
        sched.world_group = SimpleNamespace(cpu_group=grp)
        fs = _fake_self(sched, r)
        if r == 0:
            assert WU._l15_wake_hold_signal(fs) == (None, 0, 0, True)
        else:
            assert WU._weg2_wake_restore_pools(fs) is True
            assert fs.flushed == []
        fss.append(fs)
        scheds.append(sched)
    return grp, fss, scheds


def _fake_gather(monkeypatch, votes, calls):
    import torch.distributed as dist

    def _fake(gathered, obj, group=None, **kw):
        calls.append((group, obj))
        gathered[:] = list(votes)

    monkeypatch.setattr(dist, "all_gather_object", _fake)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: len(votes))


def test_real_env_caps_and_mixed_votes_fallback_frees_all_pools(
        monkeypatch, tmp_path):
    _env_real(monkeypatch, tmp_path)
    # The env the next boot ships with, on the 27B cell: rank 0 caps to 0.
    assert l15_shadow.caps_from_env(os.environ, 3, [CELL_27B] * 3,
                                    [0, 1, 2]) == CAPS_27B
    _sleep_cap_ranks(tmp_path)
    grp, fss, scheds = _wake_ranks()
    assert l15_shadow.cell_bytes_from(
        scheds[1].tp_worker.model_runner.token_to_kv_pool) == CELL_27B

    fps = [None,
           l15_manifest.fingerprint(fss[1]._l15_wake_manifest),
           l15_manifest.fingerprint(fss[2]._l15_wake_manifest)]
    assert fps[1] == fps[2] is not None  # F7: identical holds
    held = [None, _hold_votes(fss[1]), _hold_votes(fss[2])]
    assert held[1] is not None and held[2] is not None
    assert held[1][1] == held[2][1] == 2  # both spans of the F7 hold
    # The vote each rank sends: decide builds it from the fp (missing=0).
    votes = [None] + [l15_restore.check_vote(fps[r], 0, 0, 0, ())
                      for r in (1, 2)]

    calls = []
    _fake_gather(monkeypatch, votes, calls)
    verdicts = [WU._l15_decide_wake_verdict(fss[r], True, fps[r], epoch=EPOCH)
                for r in range(3)]
    assert verdicts == ["fallback", "fallback", "fallback"]
    # xsn410: exactly one gather per rank, each on the SAME group object,
    # each carrying that rank's OWN vote.
    assert [g for g, _o in calls] == [grp, grp, grp]
    assert [o for _g, o in calls] == votes

    for r in range(3):
        dropped = WU._l15_wake_act(fss[r], scheds[r], "fallback",
                                   group_ok=True, master_on=True)
        assert dropped == (0 if r == 0 else len(HELD_SLOTS))
    for sched in scheds:
        free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
        assert free == set(range(1, ALLOC_SIZE + 1)), "every pool must be free"
        assert sched.tree_cache.resets == 1
        assert sched.req_to_token_pool.clear_calls[-1] == ((), {})


def test_unanimous_fingerprints_every_rank_holds(monkeypatch, tmp_path):
    # REFILL-on shape simulated at the vote: rank 0 carries the same
    # fingerprint -> no rank falls back, the group holds together.
    _env_real(monkeypatch, tmp_path)
    _sleep_cap_ranks(tmp_path)
    grp, fss, scheds = _wake_ranks()
    fp = l15_manifest.fingerprint(fss[1]._l15_wake_manifest)
    votes = [l15_restore.check_vote(fp, 0, 0, 0, ())] * 3
    calls = []
    _fake_gather(monkeypatch, votes, calls)
    for r in range(3):
        assert WU._l15_decide_wake_verdict(fss[r], True, fp, epoch=EPOCH) \
            == "hold"
    assert [g for g, _o in calls] == [grp, grp, grp]


# ---------------------------------------------------------------------------
# PART 4 (L15-INT4): the chain through the REAL optimistic refill with
# SGLANG_WEG2_L15_REFILL=1. Same boot env and caps as PART 3, but now ALL
# THREE ranks sleep, and the manifests carry the E2a/E2b record: full l2
# columns per token and the anchor's L2 identity (anchor_l2_slot 31, gen 3).
# Rank 0 (cap 0) therefore opens the refill gate at its hold signal; wake
# runs the production order: hold-aware restore -> optimistic refill ->
# sample check (stubbed clean) -> decide -> act. The fake host pools record
# every load, so "the act does not refill a second time" is a load census.
# ---------------------------------------------------------------------------

import logging  # noqa: E402

from sglang.srt.weg2 import l15_check as l15_check_mod  # noqa: E402

# Shared prefix rows carry the IDENTICAL l2 identity in both spans -- that
# is what chain_host_rows actually records: both walks pass the same radix
# nodes, so the same node's host slot/generation appears in both spans
# (L15-DEDUPE; the pre-dedupe fixture here gave the shared rows per-rid
# distinct sources, a shape no real manifest can have, solely to dodge the
# old duplicate-slot guard). Only the spans' own tail rows differ.
REFILL_L2 = {
    "sa": ((201, 202, 203, 204), (5, 6, 7, 8)),
    "sb": ((201, 202, 203, 208), (5, 6, 7, 8)),
}
REFILL_ANCHOR_L2 = (31, 3)
REFILL_KV_GENS = {201: 5, 202: 6, 203: 7, 204: 8, 208: 8}
# rank 0 owns slot 3 (index 2 of both spans): the shared row is refilled
# ONCE (L15-DEDUPE) -> the KV load is exactly:
REFILL_KV_CALL = ([203], [1])
REFILL_ANCHOR_CALL = ([31, 31], [1, 1])   # both spans share anchor 31 -> row 1


class _HostKV:
    """Fake L2 KV host pool: generation census + recording page loader."""

    _arena_page_tokens = 1

    def __init__(self, gens):
        self.gens = dict(gens)
        self.load_calls = []

    def slot_gens(self, slots):
        return [self.gens.get(int(s), -1) for s in slots]

    def _load_pages_all_layers(self, device_pool, slots, didx,
                               lanes=None, mode=None):
        self.load_calls.append(([int(x) for x in slots],
                                [int(x) for x in didx]))


class _HostMamba:
    """Fake L2 mamba host pool: anchor census + recording state loader."""

    def __init__(self, gens):
        self.gens = dict(gens)
        self.load_calls = []

    def slot_gens(self, slots):
        return [self.gens.get(int(s), -1) for s in slots]

    def _load_states_all_layers(self, dev_mamba, slots, didx):
        self.load_calls.append(([int(x) for x in slots],
                                [int(x) for x in didx]))


class _PoolM(_Pool):
    """req_to_token_pool stand-in with the mamba device pool attached."""

    def __init__(self):
        super().__init__()
        self.mamba_pool = object()


class _Tree2:
    """tree_cache whose cache_controller exposes the fake host pools
    exactly like the hybrid 27B (L15-FIX-HOSTGROUP resolver path)."""

    def __init__(self, kv_host, mamba_host):
        self.cache_controller = SimpleNamespace(
            mem_pool_host=kv_host, mamba_pool_host=mamba_host)
        self.resets = 0

    def reset(self):
        self.resets += 1


def _sleep_all_e2(tmp_path):
    """Sleep on all three ranks with the REAL caps and the E2a/E2b record
    (l2 columns per token, anchor_l2_slot 31 gen 3): every rank writes its
    manifest; the contents are rank-independent -> equal fingerprints."""
    for r in range(3):
        path = l15_manifest.manifest_path("D", r, os.environ)
        res = l15_retain.retain_at_sleep(
            candidates=list(CANDIDATES),
            node_of=lambda rid: FakeNode(),
            slots_of=lambda rid: SLOTS_OF[rid],
            anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
            l2_of=lambda rid: REFILL_L2[rid],
            anchor_l2_of=lambda rid: REFILL_ANCHOR_L2,
            rewrite_tree=_recorder,
            caps_rows_by_rank=CAPS_27B,
            cap_anchor_slots=4,
            prefix=PREFIX, rank=r, epoch=EPOCH, pid=os.getpid(),
            kv_buffers=[torch.zeros(8, 3), torch.zeros(8, 3)],
            mamba_buffers=_mamba_views(_build_pool()),
            allocator=_fresh_alloc(),
            mamba_allocator=_build_pool().mamba_allocator,
            reset_keep=lambda ns: None, set_keep=lambda buf, spans: None,
            manifest_path=path, log=lambda line: None,
        )
        assert res is not None, f"sleep skipped on rank {r}"


def _fresh_alloc():
    alloc = FakeAllocator()
    alloc.clear()
    return alloc


def _wake4(anchor_gen=3):
    """Three fake D ranks on one cpu group: real 27B cell, host pools with
    the given anchor generation, production wake order (the hold-aware
    restore runs first on EVERY rank, refill mark included)."""
    grp = object()
    kv_host = _HostKV(REFILL_KV_GENS)
    mb_host = _HostMamba({31: anchor_gen})
    fss, scheds = [], []
    for r in range(3):
        sched = _sched()
        t = torch.zeros(1, 8, 1024, dtype=torch.bfloat16)  # the 27B cell
        sched.tp_worker.model_runner.token_to_kv_pool = SimpleNamespace(
            k_buffer=[t], v_buffer=[t])
        sched.req_to_token_pool = _PoolM()
        sched.tree_cache = _Tree2(kv_host, mb_host)
        sched.world_group = SimpleNamespace(cpu_group=grp)
        fs = _fake_self(sched, r)
        # fs bound per iteration via default args: a plain closure would
        # re-read the loop variable and every fake would act as the LAST
        # rank.
        fs._l15_wake_sample_check = lambda fs=fs: WU._l15_wake_sample_check(fs)
        fs._l15_optimistic_refill = lambda fs=fs: WU._l15_optimistic_refill(fs)
        fs._l15_do_refill = (
            lambda s, optimistic=False, fs=fs:
            WU._l15_do_refill(fs, s, optimistic=optimistic))
        fs._l15_decide_wake_verdict = (
            lambda w, fp, check=None, *, epoch, fs=fs:
            WU._l15_decide_wake_verdict(fs, w, fp, check, epoch=epoch))
        assert WU._weg2_wake_restore_pools(fs) is True
        fss.append(fs)
        scheds.append(sched)
    return grp, fss, scheds, kv_host, mb_host


def _stub_clean_check(monkeypatch):
    """l15_check.sample_check -> (k, 0, 0); records one entry per call."""
    seen = []

    def _clean(plan, host_pool, device_pool, scratch, page_tokens, k=64):
        seen.append(len(plan))
        return (k, 0, 0)

    monkeypatch.setattr(l15_check_mod, "sample_check", _clean)
    return seen


def test_refill1_chain_holds_and_the_act_never_refills_again(
        monkeypatch, tmp_path):
    # AP L15-INT4 case 1: rank 0's hold signal opens the gate, the REAL
    # optimistic refill lands the KV + anchor loads ONCE before decide(),
    # all three ranks vote the same fingerprint -> "hold", and the act
    # after decide() does NOT refill a second time.
    _env_real(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_WEG2_L15_REFILL", "1")
    _sleep_all_e2(tmp_path)
    grp, fss, scheds, kv_host, mb_host = _wake4()
    # The gate opened exactly on the cap-0 rank and nowhere else.
    assert [fs._l15_wake_refill for fs in fss] == [True, False, False]
    fps = [l15_manifest.fingerprint(fs._l15_wake_manifest) for fs in fss]
    assert fps[0] == fps[1] == fps[2] is not None
    seen = _stub_clean_check(monkeypatch)
    # Optimistic refill BEFORE decide: one KV page load, one anchor load.
    assert [WU._l15_optimistic_refill(fs) for fs in fss] == [False] * 3
    assert kv_host.load_calls == [REFILL_KV_CALL]
    assert mb_host.load_calls == [REFILL_ANCHOR_CALL]
    v = l15_restore.check_vote(fps[0], 64, 0, 0, ())
    calls = []
    _fake_gather(monkeypatch, [v, v, v], calls)
    for r in range(3):
        assert WU._l15_wake_check_and_decide(fss[r], True, fps[r],
                                             epoch=EPOCH) == "hold"
    # Real sample plans: distinct owned-with-L2 rows per rank (slot%3
    # classes), shared prefix rows sampled ONCE (L15-DEDUPE): rank 1 rows
    # 10,13, rank 2 rows 11,14. L15-FIX-CAP0-CHECK: rank 0's rows were just
    # copied from L2 under the generation check -- no sample re-read; it
    # votes the same fingerprint with a clean (0, 0, 0) check.
    assert seen == [2, 2]
    v0 = l15_restore.check_vote(fps[0], 0, 0, 0, ())
    assert [o for _g, o in calls] == [v0, v, v]   # same fp, nobody bad
    for r in range(3):
        assert WU._l15_wake_act(fss[r], scheds[r], "hold",
                                group_ok=True, master_on=True) == 0
    # No second refill (the load census is still one call each) and the
    # hold keeps every held row reserved on every rank.
    assert kv_host.load_calls == [REFILL_KV_CALL]
    assert mb_host.load_calls == [REFILL_ANCHOR_CALL]
    for sched in scheds:
        free = {int(x) for x in
                sched.token_to_kv_pool_allocator.free_pages.tolist()}
        assert free == set(range(6, ALLOC_SIZE + 1))  # HELD_SLOTS stay out
        assert sched.tree_cache.resets == 0


def _census(scheds):
    """Pool state census per rank: free KV rows, tree resets, req clears."""
    return [({int(x) for x in
              s.token_to_kv_pool_allocator.free_pages.tolist()},
             s.tree_cache.resets, len(s.req_to_token_pool.clear_calls))
            for s in scheds]


def test_refill1_anchor_gen_mismatch_falls_back_like_refill0(
        monkeypatch, tmp_path):
    # AP L15-INT4 case 2: ONE anchor generation mismatch (arena slot 31
    # moved to gen 4, manifest recorded 3) -> rank 0's optimistic refill
    # copies NOTHING and votes None -> every rank "fallback" -> the act
    # frees every pool; the census equals the REFILL=0 scenario.
    _env_real(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_WEG2_L15_REFILL", "1")
    _sleep_all_e2(tmp_path)
    grp, fss, scheds, kv_host, mb_host = _wake4(anchor_gen=4)
    assert fss[0]._l15_wake_refill is True
    _stub_clean_check(monkeypatch)
    assert WU._l15_optimistic_refill(fss[0]) is True    # -> vote None
    assert [WU._l15_optimistic_refill(fs) for fs in fss[1:]] == [False, False]
    assert kv_host.load_calls == []           # all-or-nothing: nothing moved
    assert mb_host.load_calls == []
    fps = [l15_manifest.fingerprint(fs._l15_wake_manifest) for fs in fss]
    v = l15_restore.check_vote(fps[1], 64, 0, 0, ())
    calls = []
    _fake_gather(monkeypatch, [None, v, v], calls)
    verdicts = [WU._l15_wake_check_and_decide(
        fss[r], True, None if r == 0 else fps[r], epoch=EPOCH)
        for r in range(3)]
    assert verdicts == ["fallback"] * 3
    assert [o for _g, o in calls] == [None, v, v]
    dropped = [WU._l15_wake_act(fss[r], scheds[r], "fallback",
                                group_ok=True, master_on=True)
               for r in range(3)]
    assert dropped == [len(HELD_SLOTS)] * 3
    census_refill = _census(scheds)
    full = set(range(1, ALLOC_SIZE + 1))
    assert census_refill == [(full, 1, 2)] * 3   # every pool freed, once

    # REFILL=0 on the same sleep: rank 0 never had the gate, votes None
    # from the off line; the group still falls back -> SAME census.
    monkeypatch.delenv("SGLANG_WEG2_L15_REFILL")
    _sleep_all_e2(tmp_path)                        # manifests were consumed
    grp, fss0, scheds0, kv0, mb0 = _wake4(anchor_gen=4)
    assert fss0[0]._l15_wake_refill is False
    _stub_clean_check(monkeypatch)
    assert [WU._l15_optimistic_refill(fs) for fs in fss0] == [False] * 3
    assert kv0.load_calls == [] and mb0.load_calls == []
    verdicts0 = [WU._l15_wake_check_and_decide(
        fss0[r], True, None if r == 0 else fps[r], epoch=EPOCH)
        for r in range(3)]
    assert verdicts0 == ["fallback"] * 3
    for r in range(3):
        WU._l15_wake_act(fss0[r], scheds0[r], "fallback",
                         group_ok=True, master_on=True)
    assert _census(scheds0) == census_refill       # operator condition


def test_refill0_chain_cap0_rank_never_loads(monkeypatch, tmp_path, caplog):
    # AP L15-INT4 case 3: with REFILL unset the cap-0 rank votes None from
    # the OFF line, the gate never opens, and rank 0 loads nothing at all.
    _env_real(monkeypatch, tmp_path)
    _sleep_all_e2(tmp_path)
    with caplog.at_level(logging.INFO):
        grp, fss, scheds, kv_host, mb_host = _wake4()
        assert [fs._l15_wake_refill for fs in fss] == [False, False, False]
        assert [WU._l15_optimistic_refill(fs) for fs in fss] == [False] * 3
    assert kv_host.load_calls == [] and mb_host.load_calls == []
    assert "L15-REFILL rank=0 off" in caplog.text
