"""F4b: E2 (the END-state skip of P's hand-off) under the Form A token cut.

Metal z30r3 (-st-cut, ad95392095, 29.09., cut [0, 1, 1] = S 2, host share 0):
of 97 hand-offs with parts every one read "verdict=ready" on all ranks, but
the workers' END staging answered 'cut_ring_on_worker' (194 lines = 97 x 2).
The parts are the H63 fold (END-only, 'e1=absent(fold)'): no E1 level to fall
back on, so a worker votes 0 and the group MIN is 0 -- 91 x 'skipped:group_vote'
(the extend [page_prefix, N) ran on D, ~2 s on NF-D), 6 x E1 'done'
(unfolded parts), 0 x WEG2-TAIL-SKIP-EXTEND. Before the cut (x178, 25.09.):
36 x SKIP-EXTEND, 0 x cut_ring_on_worker, 0 x group_vote.

The refusal came from #239 S4b part 5 ("the QSA ring is the host's indexer
state"). A Form A worker runs no indexer but keeps the pending ring and the
RoPE row (qsa_kv_pool: "tail adopt and the flip carry name them per pool"),
and it owns K/V token rows at compact slots -- the same rows E1 already hands
it through ``owner_rows``. F4b (SGLANG_WEG2_ENABLE_CUT_WORKER_END): the
worker takes the END state like the host -- its owned K/V rows at their
compact slots, the ring and RoPE rows -- the group votes 2 and skips. What
these cases pin, in the metal geometry (S 2: TP1 owns slot offsets [0, 1),
TP2 [1, 2), the host none):

* every rank votes 2 on folded parts; the admission skips on all three;
* each worker's owned K/V rows land byte-exact at their compact slot, the
  host writes no K/V row but the groups, the ring and the GDN slot;
* each rank's readback of what it wrote equals the staged payload
  (digest=match per rank, as x178's single-rank digest);
* switch off: the refusal as before (red evidence of the metal path).
"""

import logging
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

RID = "weg2-0-5"
PAGE, RATIO = 64, 4
N = 241  # page_prefix 192, c = 240: 49 END rows, 12 groups, 1 ring row
PREFIX, C = 192, 240
FIRST = 9764
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
SLOTS, SLOT = 5, 2
REQ_SLOTS, D_RPI = 6, 1
GLOBAL_ROWS = 6 * PAGE
D_PAGE = 3  # D's allocator hands out global slots [192, 256)
FP8 = torch.float8_e4m3fn
# the metal cut [0, 1, 1]: (S, lo, hi, per_block)
HOST, TP1, TP2 = (2, 0, 0, 0), (2, 0, 1, 1), (2, 1, 2, 1)


def _publish_fold():
    """P's three PP parts in the fold form the metal ran (END-only)."""
    ids = list(range(N))
    spec = th.spec_for(RID, ids, None, PAGE, RATIO)
    g = torch.Generator().manual_seed(29)
    rows = N - PREFIX
    fa = {gid: (torch.randn(rows, 2, 8, generator=g).to(FP8), torch.randn(rows, 2, 8, generator=g).to(FP8),
                torch.randn(12, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    ring = {gid: (torch.randn(1, 1, 4, generator=g).to(torch.bfloat16),) for gid in FA_GIDS}
    rope = torch.full((1, 3), 240, dtype=torch.int64)
    for part, (fl, gl) in PARTS.items():
        end = th.EndPayload(first_token=FIRST, key=th.tail_key(ids, N, None), rows=rows, groups=12,
                            ring_rows=1, fa={x: fa[x] for x in fl}, gdn={x: gdn[x] for x in gl},
                            ring={x: ring[x] for x in fl}, rope=rope)
        th.write_part(spec, part, {}, {}, end=end, n_parts=3, e1=False)
    return fa, gdn, ring, rope


def _pools(owner):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    host = owner == HOST
    kv_rows = 8 if host else GLOBAL_ROWS // owner[0] * owner[3]  # compact per-rank pools
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.zeros(kv_rows, 2, 8, dtype=FP8) for _ in FA_GIDS],
        v_buffer=[torch.zeros(kv_rows, 2, 8, dtype=FP8) for _ in FA_GIDS],
    )
    # a Form A worker runs no indexer: no compressed rows, but the ring and RoPE row
    kv.qsa_compressed_k_buffer_pool = (
        [torch.zeros(GLOBAL_ROWS // RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS] if host else [])
    kv.qsa_key_state_buffer_pool = [torch.zeros(REQ_SLOTS * RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_rope_position_buffer = torch.zeros(REQ_SLOTS * RATIO, 3, dtype=torch.int64)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {gid: len(GDN_GIDS) - 1 - i for i, gid in enumerate(GDN_GIDS)}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(len(GDN_GIDS), SLOTS, 2 if host else 0, 4, 4),
        conv=[torch.zeros(len(GDN_GIDS), SLOTS, 6 if host else 0, 3, dtype=torch.bfloat16)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, 512, dtype=torch.int32)
    return kv, rp


class Rank:
    def __init__(self, owner):
        self.owner = owner
        self.kv, self.rp = _pools(owner)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: self.kv),
                                    req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending, self.skips = {}, {}, [], {}
        self.req = None

    def active(self, monkeypatch):
        rank = self

        class _Ctx:
            def __enter__(self):
                self.saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
                ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = (
                    rank.jobs, rank.agreed, rank.pending, rank.skips)
                monkeypatch.setattr(ta, "cut_owner", lambda: rank.owner)

            def __exit__(self, *exc):
                ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.saved
                return False

        return _Ctx()


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


def _req():
    ids = list(range(N))
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                           prefix_indices=torch.arange(0, PREFIX, dtype=torch.int64),
                           mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=D_RPI, return_logprob=False,
                           return_hidden_states=False, grammar=None, sampling_params=sp, output_ids=[])


@pytest.fixture
def group(tmp_path, monkeypatch):
    import sglang.srt.distributed.utils as du
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.common as common

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)  # group D
    monkeypatch.setattr(du, "uneven_dcp_active", lambda *a, **k: True)  # the token cut is uneven DCP
    monkeypatch.setattr(ta, "_SKIP_SERVER", [True])
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True), envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True), \
            envs.SGLANG_WEG2_ENABLE_P_TAIL_FOLD.override(True):
        yield [Rank(HOST), Rank(TP1), Rank(TP2)]


def _switch(on: bool):
    env = getattr(envs, "SGLANG_WEG2_ENABLE_CUT_WORKER_END", None)
    if env is None:  # a tree without the switch: the cases fail on the behaviour
        import contextlib

        return contextlib.nullcontext()
    return env.override(on)


def _run(ranks, monkeypatch):
    votes = []
    for r in ranks:
        with r.active(monkeypatch):
            ta.stage(RID, r.tree)
            _join("weg2-tail-stage")
            votes.append(ta.local_vote(RID))
    plans = []
    for r in ranks:
        r.req = _req()
        with r.active(monkeypatch):
            ta.agree(RID, min(votes))
            plan = ta.plan_adopt(r.req, len(r.req.prefix_indices), batch_empty=True)
            if plan is not None:
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
            plans.append(plan)
    return votes, plans


def _prepare_for_extend(rank):
    req = rank.req
    n_prefix = len(req.prefix_indices)
    rank.rp.req_to_token[D_RPI, :n_prefix] = req.prefix_indices.to(torch.int32)
    last = int(req.prefix_indices[-1])
    rank.rp.req_to_token[D_RPI, n_prefix:N] = torch.arange(last + 1, last + 1 + N - n_prefix, dtype=torch.int32)


def _skip(rank, monkeypatch):
    batch = SimpleNamespace(reqs=[rank.req], hicache_consumer_index=-1)
    with rank.active(monkeypatch):
        tokens = ta.skip_tokens(batch)
        ta.run_skip(batch, counter=None)
    _join("weg2-tail-verify")
    return tokens


def test_metal_path_every_rank_takes_the_end_state_and_skips(group, monkeypatch, caplog):
    fa, gdn, ring, rope = _publish_fold()
    caplog.set_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt")
    with _switch(True):
        votes, plans = _run(group, monkeypatch)
        assert votes == [2, 2, 2]  # z30r3: [2, 0, 0] -> MIN 0 -> 'skipped:group_vote'
        assert all(p is not None and p.skip for p in plans)
        for r in group:
            assert len(r.req.prefix_indices) == C
            _prepare_for_extend(r)
        tokens = [_skip(r, monkeypatch) for r in group]
    assert tokens == [[FIRST]] * 3
    slots = group[0].rp.req_to_token[D_RPI, PREFIX:N].to(torch.int64)  # global slots of [192, 241)
    for r in group[1:]:
        S, lo, hi, per = r.owner
        own = (slots % S >= lo) & (slots % S < hi)
        compact = (slots[own] // S) * per + (slots[own] % S - lo)
        for gid, local in r.kv.full_attention_layer_id_mapping.items():
            k, v, _c = fa[gid]
            full = r.kv.full_kv_pool
            assert torch.equal(full.k_buffer[local][compact].view(torch.uint8), k[own].view(torch.uint8))
            assert torch.equal(full.v_buffer[local][compact].view(torch.uint8), v[own].view(torch.uint8))
            # the ring and its RoPE row at the worker's own ring slot
            assert torch.equal(r.kv.qsa_key_state_buffer_pool[local][[D_RPI * RATIO]], ring[gid][0])
        assert torch.equal(r.kv.qsa_rope_position_buffer[[D_RPI * RATIO]], rope)
        assert int(own.sum()) in (24, 25)  # 49 END rows split by slot parity
    host = group[0]
    for gid, local in host.kv.full_attention_layer_id_mapping.items():
        assert int(host.kv.full_kv_pool.k_buffer[local].view(torch.uint8).count_nonzero()) == 0  # share 0
        assert torch.equal(host.kv.qsa_compressed_k_buffer_pool[local][slots[:48:RATIO] // RATIO], fa[gid][2])
        assert torch.equal(host.kv.qsa_key_state_buffer_pool[local][[D_RPI * RATIO]], ring[gid][0])
    for gid, local in host.rp.mamba_map.items():
        assert torch.equal(host.rp.mamba_pool.mamba_cache.temporal[local, SLOT], gdn[gid][0][0])
    text = caplog.text
    assert text.count(f"WEG2-TAIL-SKIP-EXTEND rid={RID} prefix={N} first_token={FIRST}") == 3
    assert text.count("digest=match") == 3  # every rank's readback equals what it staged
    assert "cut_ring_on_worker" not in text and "group_vote" not in text


def test_switch_off_is_the_metal_refusal(group, monkeypatch, caplog):
    _publish_fold()
    caplog.set_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt")
    with _switch(False):
        votes, plans = _run(group, monkeypatch)
    assert votes == [2, 0, 0] and plans == [None] * 3  # fold: no E1 level either
    assert caplog.text.count("end=cut_ring_on_worker") == 2
    assert caplog.text.count("adopt=skipped:group_vote") == 3



# ----------------------------------------------------------- F4 park, same path
PROMPT, OUT = 200, 43
NP = PROMPT + OUT - 1  # 242 consumed: c = 240, rows from 192 - 128 = 64
WIN, PFIRST, SRC_RPI, SRC_BASE = 64, 151645, 4, 64


def _park_ids():
    return list(range(1000, 1000 + PROMPT)), [2000 + i for i in range(OUT - 1)] + [PFIRST]


def _running(rpi, **kw):
    prompt, out = _park_ids()
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    base = dict(rid=RID, origin_input_ids=prompt, output_ids=list(out), full_untruncated_fill_ids=prompt + out,
                extra_key=None, mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=rpi, return_logprob=False,
                return_hidden_states=False, grammar=None, sampling_params=sp)
    base.update(kw)
    return SimpleNamespace(**base)


def _fill(rank, seed):
    g = torch.Generator().manual_seed(seed)
    for bufs in (rank.kv.full_kv_pool.k_buffer, rank.kv.full_kv_pool.v_buffer,
                 rank.kv.qsa_compressed_k_buffer_pool, rank.kv.qsa_key_state_buffer_pool):
        for b in bufs:
            b.copy_(torch.randn(b.shape, generator=g).to(b.dtype))
    rank.kv.qsa_rope_position_buffer.copy_(torch.arange(rank.kv.qsa_rope_position_buffer.numel()).view_as(
        rank.kv.qsa_rope_position_buffer))
    c = rank.rp.mamba_pool.mamba_cache
    c.temporal.copy_(torch.randn(c.temporal.shape, generator=g))


def test_the_park_under_the_cut_rides_the_same_path(group, monkeypatch, caplog):
    src = [Rank(HOST), Rank(TP1), Rank(TP2)]
    for i, r in enumerate(src):
        _fill(r, 40 + i)
    caplog.set_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt")
    with _switch(True), envs.SGLANG_WEG2_ENABLE_D_PARK_END.override(True):
        for i, r in enumerate(src):
            r.rp.req_to_token[SRC_RPI, :NP] = torch.arange(SRC_BASE, SRC_BASE + NP, dtype=torch.int32)
            alloc = SimpleNamespace(get_kvcache=lambda r=r: r.kv)
            with r.active(monkeypatch):
                why, _ev = th.publish_park_end(_running(SRC_RPI), r.rp, alloc, PAGE, f"dpark{i}-77", 3, 2 * PAGE)
            assert why == ""
        _join("weg2-park-end")
        dst = group
        votes = []
        for r in dst:
            with r.active(monkeypatch):
                ta.stage(RID, r.tree)
                _join("weg2-tail-stage")
                votes.append(ta.local_vote(RID))
        assert votes == [2, 2, 2]
        for r in dst:
            r.req = _running(D_RPI, prefix_indices=torch.arange(0, 192, dtype=torch.int64))
            with r.active(monkeypatch):
                ta.agree(RID, 2)
                plan = ta.plan_adopt(r.req, 192, batch_empty=True)
                assert plan is not None and plan.skip
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
            n = len(r.req.full_untruncated_fill_ids)
            r.rp.req_to_token[D_RPI, :n] = torch.arange(0, n, dtype=torch.int32)  # token t at global slot t
        tokens = [_skip(r, monkeypatch) for r in dst]
    assert tokens == [[PFIRST]] * 3
    t = torch.arange(WIN, NP)
    for s, d in zip(src[1:], dst[1:]):
        S, lo, hi, per = s.owner
        g_src, g_dst = SRC_BASE + t, t
        own = (g_dst % S >= lo) & (g_dst % S < hi)
        assert torch.equal(own, (g_src % S >= lo) & (g_src % S < hi))  # S | page: the owner is the token's
        c_src = (g_src[own] // S) * per + (g_src[own] % S - lo)
        c_dst = (g_dst[own] // S) * per + (g_dst[own] % S - lo)
        for gid, local in d.kv.full_attention_layer_id_mapping.items():
            for name in ("k_buffer", "v_buffer"):
                got = getattr(d.kv.full_kv_pool, name)[local][c_dst].view(torch.uint8)
                want = getattr(s.kv.full_kv_pool, name)[local][c_src].view(torch.uint8)
                assert torch.equal(got, want)
            # the ring came from the host's part (a worker's own ring is no indexer state)
            assert torch.equal(d.kv.qsa_key_state_buffer_pool[local][D_RPI * RATIO: D_RPI * RATIO + 2],
                               src[0].kv.qsa_key_state_buffer_pool[local][SRC_RPI * RATIO: SRC_RPI * RATIO + 2])
    h_s, h_d = src[0], dst[0]
    for gid, local in h_d.kv.full_attention_layer_id_mapping.items():
        n_groups = (NP - WIN) // RATIO
        assert torch.equal(h_d.kv.qsa_compressed_k_buffer_pool[local][t[: n_groups * RATIO: RATIO] // RATIO],
                           h_s.kv.qsa_compressed_k_buffer_pool[local][(SRC_BASE + t)[: n_groups * RATIO: RATIO] // RATIO])
    for gid, local in h_d.rp.mamba_map.items():
        assert torch.equal(h_d.rp.mamba_pool.mamba_cache.temporal[local, SLOT],
                           h_s.rp.mamba_pool.mamba_cache.temporal[local, SLOT])
    assert caplog.text.count("digest=match") == 3


def test_the_park_under_the_cut_is_refused_with_the_switch_off(group, monkeypatch):
    r = Rank(TP1)
    _fill(r, 7)
    r.rp.req_to_token[SRC_RPI, :NP] = torch.arange(SRC_BASE, SRC_BASE + NP, dtype=torch.int32)
    with _switch(False), envs.SGLANG_WEG2_ENABLE_D_PARK_END.override(True), r.active(monkeypatch):
        why, _ev = th.publish_park_end(_running(SRC_RPI), r.rp, SimpleNamespace(get_kvcache=lambda: r.kv),
                                       PAGE, "dpark1-77", 3, 2 * PAGE)
    assert why in ("cut_worker_end_off", "uneven_dcp") and th.headers_for(RID) == []
