"""fnFL2 H63: the END-ANCHOR tail folded into the last chunk (SGLANG_WEG2_ENABLE_P_TAIL_FOLD).

Hermetic (no CUDA). Metal x163-x166: every P prompt ran its tail [c, N) --
1-4 tokens -- as a forward of its own after the last chunk [X, c) (PP0 125-903
ms, PP1 52-133, PP2 31-76; in the 8x4.2k burst a lone tail of 1.5 s on PP0
while the next wave waited for seats). Under E2 (H24) D never reads the E1
state at c (x153b-x166: 52 skips, 0 skip refusals); it takes P's END state.
What these cases pin:

* P, fold armed: the last chunk is NOT split when N is not a page multiple
  (its extra_buffer track lands on floor_page(N) == floor_page(N-1), the
  anchor the cut gave), and IS split when N % page == 0 (the fold would
  anchor at N, one token deeper than any reader may claim);
* the fold needs E2 on P (TAIL_HANDOFF + ADOPT + SKIP_EXTEND): without it the
  cut stays;
* P arms an END-only capture for a request whose extend reaches N and
  publishes an END-only part at its finish: header e1=False, the layer lists
  and row shapes of the END section, an empty E1 payload whose digests
  verify, the END section = rows [floor_page(c), N), ring, GDN state AFTER N,
  P's token -- with no stash at c ever having happened;
* D serves END-only parts as E2 on every rank (vote 2) or not at all (vote
  0): never E1 (there is no state at c), and a skip refused at the admission
  is the page resume (prefix stays floor_page(c)), uniformly;
* a pre-H63 header (no ``e1`` field) still decodes as e1=True.
"""

import contextlib
import logging
import threading
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

RID = "weg2-0-4"
PAGE, RATIO = 64, 4
N = 241  # c = 240, page prefix 192; N % 4 == 1: one open-group member (240)
PREFIX, C = 192, 240
FIRST = 151645
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
SLOTS, SLOT = 5, 2  # mamba slots
REQ_SLOTS, P_RPI, D_RPI = 6, 4, 1
KV_ROWS = 6 * PAGE
D_PAGE = 3  # the page D's allocator hands out: slots [192, 256)
FP8 = torch.float8_e4m3fn


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


@contextlib.contextmanager
def _e2(fold: bool, skip: bool = True):
    with contextlib.ExitStack() as stack:
        for knob, value in (("SGLANG_WEG2_TAIL_HANDOFF", True), ("SGLANG_WEG2_TAIL_ADOPT", True),
                            ("SGLANG_WEG2_TAIL_SKIP_EXTEND", skip)):
            stack.enter_context(getattr(envs, knob).override(value))
        # the default cases touch the switch only where it exists, so they
        # also run -- and pass -- on a pre-H63 tree (the unchanged path)
        if fold or hasattr(envs, "SGLANG_WEG2_ENABLE_P_TAIL_FOLD"):
            stack.enter_context(envs.SGLANG_WEG2_ENABLE_P_TAIL_FOLD.override(fold))
        yield


# ------------------------------------------------------------------ P: the split
def _qsa_allocator():
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    return SimpleNamespace(get_kvcache=lambda: kv)


def _split(monkeypatch, start: int, length: int, total: int):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # group P
    adder = SimpleNamespace(rem_chunk_tokens=16384, page_size=PAGE, token_to_kv_pool_allocator=_qsa_allocator())
    req = SimpleNamespace(full_untruncated_fill_ids=list(range(total)), rid="rid")
    return sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, length)


def test_default_cuts_the_tail_off(monkeypatch):
    """Unchanged default (the H18/H24 form): the last chunk ends at c."""
    with _e2(fold=False):
        assert _split(monkeypatch, 0, N, N) == (C, True)
        assert _split(monkeypatch, 81920, 97841 - 81920, 97841) == (97840 - 81920, True)


def test_fold_keeps_the_tail_in_the_last_chunk(monkeypatch):
    """H63: red on the old path (the cut above), green with the switch."""
    with _e2(fold=True):
        assert _split(monkeypatch, 0, N, N) == (N, False)
        assert _split(monkeypatch, 81920, 97841 - 81920, 97841) == (97841 - 81920, False)
        # a chunk that does not reach the end is never touched
        assert _split(monkeypatch, 0, 16384, 97841) == (16384, False)


@pytest.mark.parametrize("n", [256, 97792])
def test_fold_keeps_the_cut_at_a_page_multiple(monkeypatch, n):
    """N % page == 0: the unsplit chunk would anchor at N (a reader claims at
    most N-1, i.e. the page before) -- the cut at c = N-4 stays."""
    with _e2(fold=True):
        start = n - 64 * 3
        assert _split(monkeypatch, start, n - start, n) == (n - 4 - start, True)


def test_fold_needs_e2(monkeypatch):
    with _e2(fold=True, skip=False):
        assert _split(monkeypatch, 0, N, N) == (C, True)
        assert not th.fold_enabled()


# ------------------------------------------------------------------ P: the publish
def _p_pools(seed=24):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    g = torch.Generator().manual_seed(seed)
    rows = N + 2 * PAGE
    fa = {3: 0, 7: 1}
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = dict(fa)
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in fa],
        v_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in fa],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.randn(rows // RATIO + 1, 1, 4, generator=g) for _ in fa]
    kv.qsa_key_state_buffer_pool = [torch.randn(REQ_SLOTS * RATIO, 1, 4, generator=g) for _ in fa]
    kv.qsa_rope_position_buffer = torch.arange(REQ_SLOTS * RATIO * 3, dtype=torch.int64).view(-1, 3)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {0: 0, 1: 1, 2: 2}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.randn(3, SLOTS, 2, 4, 4, generator=g),
        conv=[torch.randn(3, SLOTS, 6, 3, generator=g)],
    ))
    return kv, rp, SimpleNamespace(get_kvcache=lambda: kv)


def _p_req(ids, start, end, output_ids=(FIRST,), rid=RID):
    return SimpleNamespace(rid=rid, origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
                           extend_range=SimpleNamespace(start=start, end=end), mamba_pool_idx=torch.tensor(SLOT),
                           req_pool_idx=P_RPI, output_ids=list(output_ids), return_logprob=False,
                           return_hidden_states=False)


@pytest.fixture
def p_arena(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # group P
    th._CAPTURES.clear()
    yield tmp_path
    th._CAPTURES.clear()


def test_arm_fold_registers_only_last_chunks(p_arena):
    _kv, _rp, alloc = _p_pools()
    ids = list(range(N))
    last, mid = _p_req(ids, 128, N), _p_req(ids, 0, 128, rid="weg2-0-5")
    page_multiple = _p_req(list(range(256)), 192, 256, rid="weg2-0-6")
    with _e2(fold=False):
        assert th.arm_fold([last, mid, page_multiple], alloc, PAGE, None) == 0
        assert not th._CAPTURES
    with _e2(fold=True):
        assert th.arm_fold([last, mid, page_multiple], alloc, PAGE, None) == 1
    cap = th._CAPTURES[RID]
    assert cap.e1 is False and cap.gdn == {} and cap.event is None
    assert (cap.spec.page_prefix, cap.spec.cut, cap.spec.n_tokens) == (PREFIX, C, N)
    assert cap.spec.key == th.tail_key(ids, C, None)


def _fold_publish(kv, rp, alloc, ids, output_ids=(FIRST,)):
    req = _p_req(ids, 0, N, output_ids)
    assert th.arm_fold([req], alloc, PAGE, None) == 1  # before the last chunk's forward
    rp.mamba_pool.mamba_cache.temporal[:, SLOT] += 1.0  # the last chunk [0, N) runs, tail included
    kv_indices = torch.arange(N) + PAGE
    th.publish_rows(req, kv_indices, alloc, "pp0-1", req_to_token_pool=rp, n_parts=3)
    _join("weg2-tail-publish")
    return th.headers_for(RID), kv_indices


def test_fold_publishes_an_end_only_part(p_arena, caplog):
    kv, rp, alloc = _p_pools()
    ids = list(range(N))
    with _e2(fold=True), caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_handoff"):
        (h,), kv_indices = _fold_publish(kv, rp, alloc, ids)
    assert h.e1 is False and h.n_parts == 3
    assert (h.fa_layers, h.gdn_layers) == ([3, 7], [0, 1, 2])  # the END section's layers
    assert h.fa_row_shapes == {"3": [2, 8], "7": [2, 8]} and h.gdn_row_shapes["0"] == [2, 4, 4]
    e = h.end
    assert (e.first_token, e.rows, e.groups, e.ring_rows) == (FIRST, N - PREFIX, 12, 1)
    assert e.key == th.tail_key(ids, N, None)
    bundle = th.verify_part(h)  # the empty E1 sections verify against their digests
    assert bundle is not None and bundle["fa"] == {} and bundle["gdn"] == {}
    assert th.end_digest_refusal(h, bundle) == ""
    sec = bundle["end"]
    rows = kv_indices[PREFIX:N]
    for gid, local in kv.full_attention_layer_id_mapping.items():
        k, v, c = sec["fa"][gid]
        assert torch.equal(k, kv.full_kv_pool.k_buffer[local][rows])
        assert torch.equal(v, kv.full_kv_pool.v_buffer[local][rows])
        assert torch.equal(c, kv.qsa_compressed_k_buffer_pool[local][rows[:48:RATIO] // RATIO])
        assert torch.equal(sec["ring"][gid][0], kv.qsa_key_state_buffer_pool[local][[P_RPI * RATIO]])
    for gid, local in rp.mamba_map.items():
        # the state AFTER N (the last chunk advanced it), never one at c
        assert torch.equal(sec["gdn"][gid][0][0], rp.mamba_pool.mamba_cache.temporal[local, SLOT])
    assert "| E1 absent (fold): cut=240" in caplog.text
    assert not th._CAPTURES


def test_fold_without_a_token_publishes_nothing(p_arena, caplog):
    kv, rp, alloc = _p_pools()
    with _e2(fold=True), caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_handoff"):
        headers, _k = _fold_publish(kv, rp, alloc, list(range(N)), output_ids=())
    assert headers == []  # no state at c, no END: D resumes at the page anchor
    assert f"fold rid={RID}: END refused, no part (D resumes at page_prefix={PREFIX})" in caplog.text


def test_write_part_refuses_an_end_only_part_with_e1_rows(p_arena):
    spec = th.spec_for(RID, list(range(N)), None, PAGE, RATIO)
    with pytest.raises(ValueError, match="END-only"):
        th.write_part(spec, "pp0-1", {3: (torch.zeros(1),)}, {}, end=None, e1=False)


def test_a_pre_h63_header_decodes_as_e1():
    spec = th.spec_for(RID, list(range(N)), None, PAGE, RATIO)
    h = th.TailHeader(spec=spec, part="pp0-1", fa_layers=[], gdn_layers=[], fa_row_shapes={}, gdn_row_shapes={},
                      fa_digest="", gdn_digest="", nbytes=0)
    raw = msgspec.json.decode(msgspec.json.encode(h))
    del raw["e1"]
    assert msgspec.json.decode(msgspec.json.encode(raw), type=th.TailHeader).e1 is True


# ------------------------------------------------------------------ D side
def _payload(seed=24, rows=N - PREFIX, groups=12, ring_rows=1):
    g = torch.Generator().manual_seed(seed)
    fa = {gid: (torch.randn(rows, 2, 8, generator=g).to(FP8), torch.randn(rows, 2, 8, generator=g).to(FP8),
                torch.randn(groups, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    ring = {gid: (torch.randn(ring_rows, 1, 4, generator=g).to(torch.bfloat16),) for gid in FA_GIDS}
    rope = torch.full((ring_rows, 3), 240, dtype=torch.int64)
    return fa, gdn, ring, rope


def _publish_end_only(ids=None, first=FIRST):
    ids = ids or list(range(N))
    spec = th.spec_for(RID, ids, None, PAGE, RATIO)
    fa, gdn, ring, rope = _payload()
    for part, (fl, gl) in PARTS.items():
        end = th.EndPayload(first_token=first, key=th.tail_key(ids, N, None), rows=N - PREFIX, groups=12,
                            ring_rows=1, fa={g: fa[g] for g in fl}, gdn={g: gdn[g] for g in gl},
                            ring={g: ring[g] for g in fl}, rope=rope)
        th.write_part(spec, part, {}, {}, end=end, n_parts=len(PARTS), e1=False)
    return fa, gdn, ring, rope


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
        self.req = None

    @contextlib.contextmanager
    def active(self):
        saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
        ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.jobs, self.agreed, self.pending, self.skips
        try:
            yield
        finally:
            ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = saved


def _req(ids=None, **kw):
    ids = ids or list(range(N))
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    base = dict(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                prefix_indices=torch.arange(PAGE, PAGE + PREFIX, dtype=torch.int64),
                mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=D_RPI, return_logprob=False,
                return_hidden_states=False, grammar=None, sampling_params=sp)
    base.update(kw)
    return SimpleNamespace(**base)


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
        yield [Rank(worker=False), Rank(worker=True), Rank(worker=True)]


def _run_group(ranks, reqs=None, batch_empty=True, group_vote=None):
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            _join("weg2-tail-stage")
            votes.append(ta.local_vote(RID))
    group = min(votes) if group_vote is None else group_vote
    plans = []
    for i, r in enumerate(ranks):
        r.req = reqs[i] if reqs else _req()
        with r.active():
            ta.agree(RID, group)
            plan = ta.plan_adopt(r.req, len(r.req.prefix_indices), batch_empty=batch_empty)
            if plan is not None:
                start = plan.resume_at
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
                assert len(r.req.prefix_indices) == start
            plans.append(plan)
    return votes, plans


def _prepare_for_extend(rank):
    req = rank.req
    n_prefix = len(req.prefix_indices)
    rank.rp.req_to_token[D_RPI, :n_prefix] = req.prefix_indices.to(torch.int32)
    rank.rp.req_to_token[D_RPI, n_prefix:N] = torch.arange(
        int(req.prefix_indices[-1]) + 1, int(req.prefix_indices[-1]) + 1 + N - n_prefix, dtype=torch.int32)


def _skip_on(rank):
    batch = SimpleNamespace(reqs=[rank.req], hicache_consumer_index=-1)
    with rank.active():
        tokens = ta.skip_tokens(batch)
        ta.run_skip(batch, counter=None)
    _join("weg2-tail-verify")
    return tokens


def test_end_only_parts_skip_the_extend_on_every_rank(d_group, caplog):
    fa, gdn, ring, rope = _publish_end_only()
    tp0 = d_group[0]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group)
        assert votes == [2, 2, 2] and all(p is not None and p.skip for p in plans)
        for r in d_group:
            assert len(r.req.prefix_indices) == C  # E1's shape: prefix c inside ONE page
            assert not r.pending  # nothing rides a forward's GDN reads
            _prepare_for_extend(r)
        tokens = [_skip_on(r) for r in d_group]
    assert tokens == [[FIRST]] * 3
    slots = torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + N - PREFIX)
    full = tp0.kv.full_kv_pool
    for gid, local in tp0.kv.full_attention_layer_id_mapping.items():
        k, v, c = fa[gid]
        assert torch.equal(full.k_buffer[local][slots].view(torch.uint8), k.view(torch.uint8))
        assert torch.equal(full.v_buffer[local][slots].view(torch.uint8), v.view(torch.uint8))
        assert torch.equal(tp0.kv.qsa_compressed_k_buffer_pool[local][slots[:48:RATIO] // RATIO], c)
        assert torch.equal(tp0.kv.qsa_key_state_buffer_pool[local][[D_RPI * RATIO]], ring[gid][0])
    cache = tp0.rp.mamba_pool.mamba_cache
    for gid, local in tp0.rp.mamba_map.items():
        assert torch.equal(cache.temporal[local, SLOT], gdn[gid][0][0])
    text = caplog.text
    assert text.count("e1=absent(fold)") == 3
    assert f"WEG2-TAIL-ADOPT rid={RID} page_prefix=192 tail_rows=49 state_at={N} extend=0 fa_rows_written=49 " \
           "fa_layers=3 gdn_layers=9 digest=match" in text


def test_end_only_parts_never_offer_e1(d_group, monkeypatch):
    """A D that would run the extend anyway (no skip branch) has no E1 to
    fall back on for END-only parts: vote 0 on every rank, the page resume."""
    _publish_end_only()
    monkeypatch.setattr(ta, "_SKIP_SERVER", [False])
    votes, plans = _run_group(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert all(len(r.req.prefix_indices) == PREFIX for r in d_group)


def test_end_only_parts_under_a_level_one_group_are_not_agreed(d_group):
    _publish_end_only()
    _votes, plans = _run_group(d_group, group_vote=1)
    assert plans == [None] * 3
    assert all(len(r.req.prefix_indices) == PREFIX and not r.pending for r in d_group)


@pytest.mark.parametrize("what, reason", [
    ("batch", "batch_not_empty"),
    ("logprob", "logprob_or_hidden"),
])
def test_a_refused_skip_on_end_only_parts_is_the_page_resume(d_group, caplog, what, reason):
    _publish_end_only()
    kw = {"logprob": {"return_logprob": True}}.get(what, {})
    reqs = [_req(**kw) for _ in d_group]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group, reqs=reqs, batch_empty=what != "batch")
    assert votes == [2, 2, 2] and plans == [None] * 3
    assert all(len(r.req.prefix_indices) == PREFIX and not r.pending and not r.skips for r in d_group)
    assert caplog.text.count(f"adopt=skipped:end_only:{reason}") == 3
