"""fnFL2 H63b: the tail-part store as a bounded buffer with D's consumption receipt.

Hermetic (no CUDA). Metal x163/x166 (burst 8x4.2k, P --max-running-requests
4): P published the 8 rids' parts within ~7 s, D read them one after the
other only after the flip -- and P's count rule (keep the capture_keep() = 4
newest rids) had removed the first four by then: D logged no_parts /
parts_partial:1/3 for weg2-8-10..13 and extended them (1.24-1.50 s D wall
each against 0.56-0.66 s for the adopted four; x163's first one also sat out
the 1.46 s H45 hold). What these cases pin:

* default (SGLANG_WEG2_TAIL_KEEP_MIB unset): the count rule, byte for byte --
  a burst of 8 keeps only its 4 newest rids (the loss, named);
* the budget keeps a whole burst (red on the pre-H63b tree), drops the OLDEST
  rids once the parts would exceed it, and never keeps fewer rids than the
  count rule (the rid just written always stays);
* the census ignores a part another rank is writing (``*.tmp``);
* D removes a rid's parts once its group agreed on it -- after every rank
  staged, never at a rank's own staging -- and the adoption is unaffected
  (the payload is in memory); without the budget D removes nothing.
"""

import contextlib
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

MIB = 1 << 20
PAGE, RATIO = 64, 4


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


@contextlib.contextmanager
def _keep_mib(mib):
    """The budget where it exists; None = leave it unset (the default)."""
    if mib is None:
        yield
        return
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(mib):
        yield


# ------------------------------------------------------------------ P: the prune
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(th, "capture_keep", lambda: 4)  # P --max-running-requests 4 (the burst form)
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True):
        yield tmp_path / "handoff"


def _rids_on_disk(d):
    return sorted({p.split(".tail.", 1)[0] for p in os.listdir(d) if ".tail." in p}) if d.exists() else []


def _publish(rid: str, t: float, mib: float = 1.0, part: str = "pp0-1", named: bool = True):
    """What one rank's publish thread does (write_part, then _prune), with the
    part's mtime pinned to ``t`` so the order is exact. ``named=False`` calls
    the prune the pre-H63b way (rid only)."""
    n = 241
    spec = th.spec_for(rid, list(range(n)), None, PAGE, RATIO)
    rows = int(mib * MIB) // (64 * 4)
    th.write_part(spec, part, {3: (torch.zeros(rows, 64),)}, {}, n_parts=1)
    for p in th.part_paths(rid, part):
        os.utime(p, (t, t))
    if named:
        th._prune(rid, part)
    else:
        th._prune(rid)


def _burst(n=8, mib=1.0, named=True):
    rids = [f"weg2-8-{10 + i}" for i in range(n)]
    for i, rid in enumerate(rids):
        _publish(rid, 1000.0 + i, mib, named=named)
    return rids


def test_default_count_rule_keeps_only_the_newest_rids(store):
    """Unchanged default: a burst deeper than capture_keep() loses its oldest
    parts before D reads them (x163/x166 weg2-8-10..13)."""
    with _keep_mib(0 if hasattr(envs, "SGLANG_WEG2_TAIL_KEEP_MIB") else None):
        rids = _burst(named=False)
    assert _rids_on_disk(store) == rids[4:]


def test_budget_keeps_a_whole_burst(store):
    """H63b: red on the pre-H63b tree (the count rule above), green here."""
    with _keep_mib(16):
        rids = _burst()
    assert _rids_on_disk(store) == rids
    newest, size = th.census(str(store))
    assert set(newest) == set(rids) and sum(size.values()) <= 16 * MIB


def test_budget_drops_the_oldest_beyond_the_bound(store):
    with _keep_mib(6):
        rids = _burst()
    kept = _rids_on_disk(store)
    assert kept == rids[3:]  # 5 rids of ~1 MiB fit 6 MiB, a sixth would not
    _newest, size = th.census(str(store))
    assert sum(size.values()) <= 6 * MIB


def test_budget_never_keeps_fewer_rids_than_the_count_rule(store):
    with _keep_mib(1):  # smaller than two parts
        rids = _burst()
    assert _rids_on_disk(store) == rids[4:]


def test_census_ignores_a_part_being_written(store):
    """A part another rank is writing (``*.tmp``) costs no bytes. H81
    (fnNV4f2): its rid is NEW, though -- the census names it with the temp
    file's mtime, so no prune takes a rid under write for an old one."""
    with _keep_mib(16):
        _publish("weg2-8-10", 1000.0)
    tmp = store / "weg2-8-11.tail.pp1-2.pt.99.tmp"
    tmp.write_bytes(b"x" * 4096)
    newest, size = th.census(str(store))
    assert set(size) == {"weg2-8-10"} and "weg2-8-11" not in size
    assert newest["weg2-8-11"] > newest["weg2-8-10"]


def test_prune_victims_is_newest_first_and_keeps_the_rid_just_written():
    newest = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    size = {r: 10 for r in newest}
    # "a" is the rid just written although its mtime is the oldest: it stays
    assert th.prune_victims(newest, size, "a", keep_min=1, budget=30) == ["b", "c"]
    assert th.prune_victims(newest, size, "e", keep_min=1, budget=1000) == []
    assert th.prune_victims(newest, size, "e", keep_min=3, budget=0) == ["a"]  # the floor wins over the budget


# ------------------------------------------------------------------ D: the receipt
RID = "weg2-0-4"
N = 241  # c = 240, page prefix 192
PREFIX, C = 192, 240
FIRST = 151645
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
SLOTS, SLOT = 5, 2
REQ_SLOTS, D_RPI = 6, 1
KV_ROWS = 6 * PAGE
D_PAGE = 3
FP8 = torch.float8_e4m3fn


def _publish_parts(ids=None):
    """E1 + END parts of every P rank (the H24 form)."""
    ids = ids or list(range(N))
    spec = th.spec_for(RID, ids, None, PAGE, RATIO)
    g = torch.Generator().manual_seed(7)
    rows = N - PREFIX
    for part, (fl, gl) in PARTS.items():
        fa1 = {gid: (torch.randn(C - PREFIX, 2, 8, generator=g).to(FP8), torch.randn(C - PREFIX, 2, 8, generator=g).to(FP8),
                     torch.randn((C - PREFIX) // RATIO, 1, 4, generator=g).to(torch.bfloat16)) for gid in fl}
        gdn1 = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
                for gid in gl}
        end = th.EndPayload(
            first_token=FIRST, key=th.tail_key(ids, N, None), rows=rows, groups=12, ring_rows=1,
            fa={gid: (torch.randn(rows, 2, 8, generator=g).to(FP8), torch.randn(rows, 2, 8, generator=g).to(FP8),
                      torch.randn(12, 1, 4, generator=g).to(torch.bfloat16)) for gid in fl},
            gdn={gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
                 for gid in gl},
            ring={gid: (torch.randn(1, 1, 4, generator=g).to(torch.bfloat16),) for gid in fl},
            rope=torch.full((1, 3), 240, dtype=torch.int64),
        )
        th.write_part(spec, part, fa1, gdn1, end=end, n_parts=len(PARTS))


def _d_pools(worker: bool):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
        v_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.zeros(KV_ROWS // RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_key_state_buffer_pool = [torch.zeros(REQ_SLOTS * RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_rope_position_buffer = torch.zeros(REQ_SLOTS * RATIO, 3, dtype=torch.int64)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {gid: len(GDN_GIDS) - 1 - i for i, gid in enumerate(GDN_GIDS)}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(len(GDN_GIDS), SLOTS, 0 if worker else 2, 4, 4),
        conv=[torch.zeros(len(GDN_GIDS), SLOTS, 0 if worker else 6, 3, dtype=torch.bfloat16)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, 512, dtype=torch.int32)
    return kv, rp


class Rank:
    def __init__(self, worker: bool):
        self.kv, self.rp = _d_pools(worker)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: self.kv),
                                    req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending, self.skips = {}, {}, [], {}

    @contextlib.contextmanager
    def active(self):
        saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
        ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.jobs, self.agreed, self.pending, self.skips
        try:
            yield
        finally:
            ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = saved


def _req():
    ids = list(range(N))
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                           prefix_indices=torch.arange(PAGE, PAGE + PREFIX, dtype=torch.int64),
                           mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=D_RPI, return_logprob=False,
                           return_hidden_states=False, grammar=None, sampling_params=sp)


@pytest.fixture
def d_group(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.common as common

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)  # group D
    monkeypatch.setattr(ta, "_SKIP_SERVER", [True])
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True), envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield [Rank(worker=False), Rank(worker=True), Rank(worker=True)], tmp_path / "handoff"


def _stage_and_vote(ranks):
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            _join("weg2-tail-stage")
            votes.append(ta.local_vote(RID))
    return votes


def _agree_and_admit(ranks, group):
    plans = []
    for r in ranks:
        req = _req()
        with r.active():
            ta.agree(RID, group)
            plan = ta.plan_adopt(req, len(req.prefix_indices), batch_empty=True)
            if plan is not None:
                ta.commit_adopt(req, plan, tree_cache=r.tree, page_size=PAGE)
            plans.append(plan)
    _join("weg2-tail-consumed")
    return plans


def test_d_removes_the_parts_once_the_group_agreed(d_group):
    """H63b: red on the pre-H63b tree (the parts stay), green here."""
    ranks, d = d_group
    _publish_parts()
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(512):
        votes = _stage_and_vote(ranks)
        assert _rids_on_disk(d) == [RID]  # a rank's own staging removes nothing
        plans = _agree_and_admit(ranks, min(votes))
    assert votes == [2, 2, 2] and all(p is not None and p.skip for p in plans)  # adoption unaffected
    assert _rids_on_disk(d) == []


def test_d_keeps_the_parts_by_default(d_group):
    """Unchanged default: D reads the parts and leaves them to P's prune."""
    ranks, d = d_group
    _publish_parts()
    with _keep_mib(0 if hasattr(envs, "SGLANG_WEG2_TAIL_KEEP_MIB") else None):
        votes = _stage_and_vote(ranks)
        plans = _agree_and_admit(ranks, min(votes))
    assert votes == [2, 2, 2] and all(p is not None and p.skip for p in plans)
    assert _rids_on_disk(d) == [RID]
