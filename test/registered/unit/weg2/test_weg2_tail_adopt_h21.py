"""fnFL2 H21 (second half of E1): D adopts P's partial page + state at c.

Hermetic (no CUDA). Three simulated TP ranks of group D -- TP0 holding every
attention/GDN layer (Form A host), TP1/TP2 Form-A expert workers holding none
-- run the vote, the admission and the install against fake pools. What the
cases pin (derived properties / bookkeeping a later diff can silently break):
* the parts of PP0/PP1/PP2 land per GLOBAL layer id in D's own local layout:
  rows at the request's page slots, compressed groups at slot // ratio, state
  in the request's mamba slot -- byte-exact, fp8 included;
* every rank runs the SAME extend [c, N) (the Form-A forward's all-reduces
  only line up then), and every non-ready verdict on any rank puts every rank
  back on today's extend [floor_page(c), N);
* a worker holding no layer votes 'not_mine', never a shape refusal (metal
  fnFL2x133 TP1/TP2: 'fa_shape:3:[2, 256]!=[0, 256]');
* the partial page is request-owned: prefix = page prefix + the first
  c - floor_page(c) slots of ONE page, and the extend continues inside it;
* the post-write readback digests equal the publish digests byte for byte
  (typed tensors vs uint8 readback), and a changed byte is a MISMATCH.
"""

import contextlib
import logging
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

RID = "weg2-0-4"
N, PAGE, RATIO = 241, 64, 4  # c = 240, page prefix 192, 48 rows, extend 1
C, PREFIX, ROWS = 240, 192, 48
FP8 = torch.float8_e4m3fn
#: P's partition (PP0 -> PP1 -> PP2), global layer ids; GDN precedes FA
PARTS = {
    "pp0-1": ([3], [0, 1, 2]),
    "pp1-2": ([7], [4, 5, 6]),
    "pp2-3": ([11], [8, 9, 10]),
}
FA_GIDS = [3, 7, 11]
GDN_GIDS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
SLOTS, SLOT = 5, 2
KV_ROWS = 6 * PAGE
D_PAGE = 3  # the page D's allocator hands out: slots [192, 256)


def _p_payload(seed=21):
    g = torch.Generator().manual_seed(seed)
    fa = {gid: (torch.randn(ROWS, 2, 8, generator=g).to(FP8), torch.randn(ROWS, 2, 8, generator=g).to(FP8),
                torch.randn(ROWS // RATIO, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    return fa, gdn


def _publish(fa, gdn, ids=None):
    spec = th.spec_for(RID, ids or list(range(N)), None, PAGE, RATIO)
    for part, (fl, gl) in PARTS.items():
        th.write_part(spec, part, {g: fa[g] for g in fl}, {g: gdn[g] for g in gl})
    return spec


def _d_pools(worker: bool):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    # D's local order differs from any P rank's: 11 -> 0, 3 -> 1, 7 -> 2
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
        v_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.zeros(KV_ROWS // RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {gid: len(GDN_GIDS) - 1 - i for i, gid in enumerate(GDN_GIDS)}  # reversed locals
    h = 0 if worker else 2
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(len(GDN_GIDS), SLOTS, h, 4, 4),
        conv=[torch.zeros(len(GDN_GIDS), SLOTS, 6 if not worker else 0, 3, dtype=torch.bfloat16)],
    ))
    return kv, rp


class Rank:
    """One TP rank: its pools and its OWN copy of the module state."""

    def __init__(self, worker: bool):
        self.kv, self.rp = _d_pools(worker)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: self.kv),
                                    req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending = {}, {}, []
        self.req = None

    @contextlib.contextmanager
    def active(self):
        saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS)
        ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS = self.jobs, self.agreed, self.pending
        try:
            yield
        finally:
            ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS = saved


def _req(ids=None, prefix=PREFIX):
    ids = ids or list(range(N))
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                           prefix_indices=torch.arange(PAGE, PAGE + prefix, dtype=torch.int64),
                           mamba_pool_idx=torch.tensor(SLOT))


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


@pytest.fixture
def d_group(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.common as common

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)  # group D
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True):
        yield [Rank(worker=False), Rank(worker=True), Rank(worker=True)]


def _run_group(ranks, reqs=None, tamper=None):
    """stage + vote (MIN) + agree + admission commit on every rank; returns
    the group vote and each rank's prefix length after the admission."""
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            votes.append(ta.local_vote(RID))
    group = min(votes)
    out = []
    for i, r in enumerate(ranks):
        r.req = (reqs[i] if reqs else _req())
        with r.active():
            ta.agree(RID, group)
            plan = ta.plan_adopt(r.req, len(r.req.prefix_indices))
            if plan is not None:
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
        out.append(len(r.req.prefix_indices))
    return votes, out


def _forward_on(rank):
    """The extend forward's GDN layer reads, in layer order."""
    with rank.active():
        for gid in sorted(rank.rp.mamba_map):
            ta.install_layer(gid)
    _join("weg2-tail-verify")


# ------------------------------------------------------------------ the whole path
def test_parts_assemble_in_layer_order_into_d_layout(d_group, caplog):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    tp0 = d_group[0]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, prefix = _run_group(d_group)
        assert votes == [1, 1, 1]
        assert tp0.pending and not d_group[1].pending and not d_group[2].pending
        _forward_on(tp0)
    assert not tp0.pending
    page_rows = torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + ROWS)
    for gid, local in tp0.kv.full_attention_layer_id_mapping.items():
        k, v, c = fa[gid]
        full = tp0.kv.full_kv_pool
        assert torch.equal(full.k_buffer[local][page_rows].view(torch.uint8), k.view(torch.uint8))
        assert torch.equal(full.v_buffer[local][page_rows].view(torch.uint8), v.view(torch.uint8))
        groups = page_rows[::RATIO] // RATIO
        assert torch.equal(tp0.kv.qsa_compressed_k_buffer_pool[local][groups], c)
        # nothing outside the 48 rows / 12 groups was touched
        assert int(full.k_buffer[local].view(torch.uint8).count_nonzero()) == int(k.view(torch.uint8).count_nonzero())
    cache = tp0.rp.mamba_pool.mamba_cache
    for gid, local in tp0.rp.mamba_map.items():
        assert torch.equal(cache.temporal[local, SLOT], gdn[gid][0][0])
        assert torch.equal(cache.conv[0][local, SLOT], gdn[gid][1][0])
        assert int(cache.temporal[local].count_nonzero()) == int(gdn[gid][0].count_nonzero())
    text = caplog.text
    assert text.count("adopt=done") == 3
    assert "verdict=ready adopt=done" in text and text.count("verdict=not_mine adopt=done") == 2
    assert "WEG2-TAIL-ADOPT rid=weg2-0-4 page_prefix=192 tail_rows=48 state_at=240 extend=1 " \
           "fa_rows_written=48 fa_layers=3 gdn_layers=9 digest=match" in text


def test_every_rank_runs_the_same_extend(d_group):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    _votes, prefix = _run_group(d_group)
    assert prefix == [C, C, C]
    for r in d_group:
        assert r.req.full_untruncated_fill_ids[len(r.req.prefix_indices):] == [C]  # extend [c, N) = 1 token
        assert len(r.req.full_untruncated_fill_ids) - len(r.req.prefix_indices) == N - C


def test_partial_page_is_request_owned_and_the_extend_stays_in_it(d_group):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    _run_group(d_group)
    req = d_group[0].req
    tail = req.prefix_indices[PREFIX:]
    assert torch.equal(tail, torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + ROWS))
    assert int(tail[0]) % PAGE == 0  # the rows open a page of their own ...
    last = int(req.prefix_indices[-1])
    assert last % PAGE == ROWS - 1 < PAGE - 1  # ... and the extend's slot c is in it
    # the paged alloc_extend needs no new page for [c, N): ceil(N/64) == ceil(c/64)
    assert -(-N // PAGE) == -(-C // PAGE)
    # the tree part is untouched: the page is not a prefix anyone else matches
    assert torch.equal(req.prefix_indices[:PREFIX], torch.arange(PAGE, PAGE + PREFIX))


# ------------------------------------------------------------------ fallbacks
def _assert_fallback(d_group, caplog, adopt_reason):
    assert all(len(r.req.prefix_indices) == PREFIX for r in d_group)
    assert all(not r.pending for r in d_group)
    assert f"adopt=skipped:{adopt_reason}" in caplog.text
    assert "adopt=done" not in caplog.text


def test_digest_mismatch_on_tp0_puts_every_rank_back_on_the_page(d_group, caplog):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    _j, ppath = th.part_paths(RID, "pp1-2")
    bad = dict(fa)
    bad[7] = (fa[7][0], fa[7][1], fa[7][2] + 1)
    torch.save({"fa": {7: bad[7]}, "gdn": {g: gdn[g] for g in PARTS["pp1-2"][1]}}, ppath)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, prefix = _run_group(d_group)
    assert votes == [0, 1, 1]
    assert "verdict=digest_MISMATCH:pp1-2 adopt=skipped:group_vote" in caplog.text
    _assert_fallback(d_group, caplog, "group_vote")


def test_missing_part_on_tp0_puts_every_rank_back(d_group, caplog):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    th.remove(RID)
    spec = th.spec_for(RID, list(range(N)), None, PAGE, RATIO)
    for part in ("pp0-1", "pp2-3"):  # PP1's part never arrived
        fl, gl = PARTS[part]
        th.write_part(spec, part, {g: fa[g] for g in fl}, {g: gdn[g] for g in gl})
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, _ = _run_group(d_group)
    assert votes == [0, 1, 1]
    assert "verdict=fa_layer_missing:7" in caplog.text
    _assert_fallback(d_group, caplog, "group_vote")


@pytest.mark.parametrize("what, reason", [("prefix", "prefix:128!=192"), ("key", "key_mismatch"),
                                          ("retracted", "n_tokens:242!=241")])
def test_uniform_refusals_at_the_admission(d_group, caplog, what, reason):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    ids = list(range(N))
    if what == "key":
        ids[200] += 1  # a token of the handed-over partial page differs
    reqs = [_req(ids=list(ids), prefix=128 if what == "prefix" else PREFIX) for _ in d_group]
    if what == "retracted":
        for r in reqs:
            r.full_untruncated_fill_ids.append(7)  # origin + one output token
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, prefix = _run_group(d_group, reqs=reqs)
    assert votes == [1, 1, 1]  # the parts were fine ...
    assert len(set(prefix)) == 1  # ... and the refusal is the same on every rank
    assert caplog.text.count(f"adopt=skipped:{reason}") == 3
    assert all(not r.pending for r in d_group)
    # one-shot: a second admission of the same rid finds nothing to adopt
    with d_group[0].active():
        assert ta.plan_adopt(_req(), PREFIX) is None


def test_switch_off_is_todays_extend(d_group):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    with envs.SGLANG_WEG2_TAIL_ADOPT.override(False):
        votes, prefix = _run_group(d_group)
    assert votes == [0, 0, 0] and prefix == [PREFIX] * 3


# ------------------------------------------------------------------ worker verdict
def test_form_a_worker_holds_nothing_and_votes_not_mine(d_group):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    worker = d_group[1]
    held = ta.held_shapes(worker.kv, worker.rp)
    assert held.holds_nothing
    st = ta.stage_parts(th.headers_for(RID), held, check_digest=True)
    assert st.verdict == "not_mine" and st.ok and not st.fa and not st.gdn
    # the H18 probe compared the worker's 0-head rows and refused (x133)
    full = {int(g): list(worker.kv.full_kv_pool.k_buffer[l].shape[1:])
            for g, l in worker.kv.full_attention_layer_id_mapping.items()}
    assert th.local_readiness(st.spec, st.headers, full, {}).startswith("fa_shape:")
    tp0 = ta.held_shapes(d_group[0].kv, d_group[0].rp)
    assert sorted(tp0.fa) == FA_GIDS and sorted(tp0.gdn) == GDN_GIDS and tp0.qsa_ratio == RATIO


# ------------------------------------------------------------------ install bookkeeping
def test_readback_mismatch_is_named(d_group, caplog):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    tp0 = d_group[0]
    _run_group(d_group)
    inst = tp0.pending[0]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        _forward_on(tp0)
    assert "digest=match" in caplog.text
    assert ta.readback_digest(inst) == "match"
    inst.readback["gdn5"][0][0, 0, 0, 0] ^= 1  # one written byte differs
    assert ta.readback_digest(inst) == "MISMATCH:pp1-2"


def test_a_second_forward_abandons_an_incomplete_install(d_group, caplog):
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    tp0 = d_group[0]
    _run_group(d_group)
    with tp0.active(), caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        ta.install_layer(0)
        ta.install_layer(1)
        ta.install_layer(1)  # a repeated read of the same layer: still this forward
        assert tp0.pending  # 7 GDN layers still owed
        ta.install_layer(0)  # the next forward's first GDN layer
        assert not tp0.pending
    assert "INCOMPLETE" in caplog.text


def test_fa_rows_land_before_the_first_attention_layer(d_group):
    """Rows are written at the FIRST GDN read (layer 0), i.e. before layer 3
    -- the first attention layer -- reads them."""
    fa, gdn = _p_payload()
    _publish(fa, gdn)
    tp0 = d_group[0]
    _run_group(d_group)
    with tp0.active():
        ta.install_layer(0)
    local = tp0.kv.full_attention_layer_id_mapping[11]
    rows = torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + ROWS)
    assert torch.equal(tp0.kv.full_kv_pool.k_buffer[local][rows].view(torch.uint8), fa[11][0].view(torch.uint8))
    with tp0.active():
        tp0.pending.clear()
