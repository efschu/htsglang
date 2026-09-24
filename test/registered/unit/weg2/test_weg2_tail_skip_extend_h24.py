"""fnFL2 H24 (E2): P hands over its END state, D starts decoding without an extend.

Hermetic (no CUDA). Metal x137 (E1) left one forward on the flip's critical
path: D extended the single token [97840, 97841) -- 574 gpu-ms for 1 token
(compute 368 + 96 all-reduces 205) -- although P had computed that token as
its own final chunk and sampled the answer's first token. What these cases
pin (derived properties / bookkeeping a later diff can silently break):

* P's END section is the state AFTER N, not after c: rows [floor_page(c), N)
  (one more than E1 when N-c == 1), the complete QSA groups only, the open
  group's pending-ring rows read at P's ``req_pool_idx * r + j`` and P's
  sampled token;
* the vote is a LEVEL (2 = END servable, 1 = E1 only, 0 = page resume) inside
  the same MIN slot, so ONE rank short of the END state puts EVERY rank on E1
  -- never a split where one rank skips the forward its peers run (the Form-A
  all-reduces would wedge);
* the admission under SKIP grows the prefix to N-1 inside one page, closes the
  batch, and the (skipped) extend slot N-1 receives P's last row;
* the install lands byte-exact at D's own slots (rows via req_to_token, groups
  at slot // r, ring rows at D's ``req_pool_idx * r + j``, the GDN slot) and
  after the batch's HiCache load join;
* the worker returns P's token and a shape-valid draft seed with that token as
  the bonus -- the request enters the decode queue like a finished prefill.
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
PAGE, RATIO = 64, 4
N = 241  # c = 240, page prefix 192; N % 4 == 1: one open-group member (240)
PREFIX, C = 192, 240
FIRST = 151645
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
SLOTS, SLOT = 5, 2  # mamba slots
REQ_SLOTS, P_RPI, D_RPI = 6, 4, 1  # request slots: P's and D's req_pool_idx differ
KV_ROWS = 6 * PAGE
D_PAGE = 3  # the page D's allocator hands out: slots [192, 256)
FP8 = torch.float8_e4m3fn


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize(
    "n, rows, groups, ring",
    [
        (97841, 49, 12, 1),  # fnFL2x137 weg2-0-4: E1 extended [97840, 97841)
        (12663, 55, 13, 3),  # x137 weg2-2-6: N-c = 3, open group [12660, 12663)
        (12664, 56, 14, 0),  # N % 4 == 0: the group [c, N) is complete, ring empty
        (97856, 64, 16, 0),  # N % 64 == 0: the END rows fill the page exactly
    ],
)
def test_end_geometry(n, rows, groups, ring):
    spec = th.spec_for("r", list(range(n)), None, PAGE, RATIO)
    assert th.end_geometry(spec, RATIO) == (rows, groups, ring)
    assert spec.page_prefix + groups * RATIO + ring == n  # nothing counted twice or missed
    assert rows <= PAGE  # the END rows always fit the one page D allocates
    assert th.end_geometry(spec, 0) == (rows, 0, 0)


# ------------------------------------------------------------------ P side
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


def _p_req(ids, end, output_ids=(FIRST,)):
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
                           extend_range=SimpleNamespace(end=end), mamba_pool_idx=torch.tensor(SLOT),
                           req_pool_idx=P_RPI, output_ids=list(output_ids), return_logprob=False,
                           return_hidden_states=False)


@pytest.fixture
def p_arena(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # group P
    th._CAPTURES.clear()
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield tmp_path


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


def _p_publish(kv, rp, alloc, ids, output_ids=(FIRST,)):
    req = _p_req(ids, C, output_ids)
    assert th.capture_state(req, rp, alloc, PAGE, None)  # stash of the chunk ending at c
    state_at_c = rp.mamba_pool.mamba_cache.temporal[:, SLOT].clone()
    rp.mamba_pool.mamba_cache.temporal[:, SLOT] += 1.0  # the final chunk [c, N) advances the state
    kv_indices = torch.arange(N) + PAGE
    th.publish_rows(req, kv_indices, alloc, "pp0-1", req_to_token_pool=rp)
    _join("weg2-tail-publish")
    (h,) = th.headers_for(RID)
    return h, kv_indices, state_at_c


def test_p_publishes_the_state_after_n_and_its_token(p_arena, caplog):
    kv, rp, alloc = _p_pools()
    ids = list(range(N))
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_handoff"):
        h, kv_indices, state_at_c = _p_publish(kv, rp, alloc, ids)
    e = h.end
    assert (e.first_token, e.rows, e.groups, e.ring_rows) == (FIRST, 49, 12, 1)
    assert e.key == th.tail_key(ids, N, None) != h.spec.key  # END key covers [0, N), E1's [0, c)
    bundle = th.verify_part(h)
    assert th.end_digest_refusal(h, bundle) == ""
    sec = bundle["end"]
    rows = kv_indices[PREFIX:N]
    for gid, local in kv.full_attention_layer_id_mapping.items():
        k, v, c = sec["fa"][gid]
        assert torch.equal(k, kv.full_kv_pool.k_buffer[local][rows])  # 49 rows: 97840's row included
        assert torch.equal(v, kv.full_kv_pool.v_buffer[local][rows])
        assert torch.equal(c, kv.qsa_compressed_k_buffer_pool[local][rows[:48:RATIO] // RATIO])  # complete only
        # the open group's member 240 lives in P's ring row P_RPI * 4 + 0
        assert torch.equal(sec["ring"][gid][0], kv.qsa_key_state_buffer_pool[local][[P_RPI * RATIO]])
    assert torch.equal(sec["rope"], kv.qsa_rope_position_buffer[[P_RPI * RATIO]])
    for gid, local in rp.mamba_map.items():
        assert torch.equal(sec["gdn"][gid][0][0], rp.mamba_pool.mamba_cache.temporal[local, SLOT])  # after N
        assert not torch.equal(sec["gdn"][gid][0][0], state_at_c[local])
        assert torch.equal(bundle["gdn"][gid][0][0], state_at_c[local])  # E1 section unchanged: after c
    assert f"tail_rows=49 state_at={N} first_token={FIRST}" in caplog.text
    assert "| E1 tail_rows=48 state_at=240" in caplog.text


@pytest.mark.parametrize("why, kwargs", [("no_sampled_token", {"output_ids": ()})])
def test_p_without_a_token_publishes_e1_only(p_arena, caplog, why, kwargs):
    kv, rp, alloc = _p_pools()
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_handoff"):
        h, _k, _s = _p_publish(kv, rp, alloc, list(range(N)), **kwargs)
    assert h.end is None and th.verify_part(h) is not None
    assert f"end=none rid={RID} reason={why}" in caplog.text


def test_p_switch_off_publishes_e1_only(p_arena):
    kv, rp, alloc = _p_pools()
    with envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(False):
        h, _k, _s = _p_publish(kv, rp, alloc, list(range(N)))
    assert h.end is None


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


def _e1_payload(seed=21):
    g = torch.Generator().manual_seed(seed)
    fa = {gid: (torch.randn(C - PREFIX, 2, 8, generator=g).to(FP8), torch.randn(C - PREFIX, 2, 8, generator=g).to(FP8),
                torch.randn((C - PREFIX) // RATIO, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    return fa, gdn


def _publish(with_end=True, ids=None, first=FIRST):
    ids = ids or list(range(N))
    spec = th.spec_for(RID, ids, None, PAGE, RATIO)
    fa1, gdn1 = _e1_payload()
    fa, gdn, ring, rope = _payload()
    for part, (fl, gl) in PARTS.items():
        end = th.EndPayload(first_token=first, key=th.tail_key(ids, N, None), rows=N - PREFIX, groups=12,
                            ring_rows=1, fa={g: fa[g] for g in fl}, gdn={g: gdn[g] for g in gl},
                            ring={g: ring[g] for g in fl}, rope=rope) if with_end else None
        th.write_part(spec, part, {g: fa1[g] for g in fl}, {g: gdn1[g] for g in gl}, end=end)
    return fa, gdn, ring, rope


def _d_pools(worker: bool):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}  # D's own local order
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
    monkeypatch.setattr(ta, "_SKIP_SERVER", [True])  # EAGLEWorkerV2 registered its skip branch
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True), envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield [Rank(worker=False), Rank(worker=True), Rank(worker=True)]


def _run_group(ranks, reqs=None, batch_empty=True, stage_hook=None):
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            _join("weg2-tail-stage")
            if stage_hook is not None:
                stage_hook(r)
            votes.append(ta.local_vote(RID))
    group = min(votes)
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
    """What ScheduleBatch.prepare_for_extend writes: the prefix and the
    extend's slot (alloc_extend continues inside the adopted page)."""
    req = rank.req
    n_prefix = len(req.prefix_indices)
    rank.rp.req_to_token[D_RPI, :n_prefix] = req.prefix_indices.to(torch.int32)
    rank.rp.req_to_token[D_RPI, n_prefix:N] = torch.arange(
        int(req.prefix_indices[-1]) + 1, int(req.prefix_indices[-1]) + 1 + N - n_prefix, dtype=torch.int32)


class _Counter:
    num_layers = 48

    def __init__(self):
        self.calls = []

    def set_consumer(self, i):
        self.calls.append(("consumer", i))

    def wait_until(self, layer):
        self.calls.append(("wait", layer))


def _skip_on(rank, counter=None):
    batch = SimpleNamespace(reqs=[rank.req], hicache_consumer_index=1)
    with rank.active():
        tokens = ta.skip_tokens(batch)
        ta.run_skip(batch, counter=counter)
    _join("weg2-tail-verify")
    return tokens


def test_level_two_vote_skips_the_extend_on_every_rank(d_group, caplog):
    fa, gdn, ring, rope = _publish()
    tp0 = d_group[0]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group)
        assert votes == [2, 2, 2] and all(p.skip for p in plans)
        # every rank: prefix N-1 inside ONE page, the extend is the 1-token shape [N-1, N)
        for r in d_group:
            assert len(r.req.prefix_indices) == N - 1
            assert torch.equal(r.req.prefix_indices[PREFIX:], torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + N - 1 - PREFIX))
            assert not r.pending  # nothing rides the (absent) forward's GDN reads
            _prepare_for_extend(r)
        counter = _Counter()
        tokens = [_skip_on(r, counter if r is tp0 else None) for r in d_group]
    assert tokens == [[FIRST]] * 3  # uniform: every rank returns P's token, none runs a forward
    assert counter.calls == [("consumer", 1), ("wait", 47)]  # the load-back is joined first
    assert all(not r.skips for r in d_group)
    slots = torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + N - PREFIX)  # [192, 241) incl. the extend slot 240
    full = tp0.kv.full_kv_pool
    for gid, local in tp0.kv.full_attention_layer_id_mapping.items():
        k, v, c = fa[gid]
        assert torch.equal(full.k_buffer[local][slots].view(torch.uint8), k.view(torch.uint8))
        assert torch.equal(full.v_buffer[local][slots].view(torch.uint8), v.view(torch.uint8))
        assert torch.equal(tp0.kv.qsa_compressed_k_buffer_pool[local][slots[:48:RATIO] // RATIO], c)
        assert int(tp0.kv.qsa_compressed_k_buffer_pool[local].count_nonzero()) == int(c.count_nonzero())
        # the open group's member lands in D's OWN ring row, not P's
        assert torch.equal(tp0.kv.qsa_key_state_buffer_pool[local][[D_RPI * RATIO]], ring[gid][0])
        assert int(tp0.kv.qsa_key_state_buffer_pool[local].count_nonzero()) == int(ring[gid][0].count_nonzero())
    assert torch.equal(tp0.kv.qsa_rope_position_buffer[[D_RPI * RATIO]], rope)
    cache = tp0.rp.mamba_pool.mamba_cache
    for gid, local in tp0.rp.mamba_map.items():
        assert torch.equal(cache.temporal[local, SLOT], gdn[gid][0][0])
        assert torch.equal(cache.conv[0][local, SLOT], gdn[gid][1][0])
    text = caplog.text
    assert text.count(f"state_at={N} extend=0 parts=3") == 3
    assert f"WEG2-TAIL-ADOPT rid={RID} page_prefix=192 tail_rows=49 state_at={N} extend=0 fa_rows_written=49 " \
           "fa_layers=3 gdn_layers=9 digest=match" in text
    assert text.count(f"WEG2-TAIL-SKIP-EXTEND rid={RID} prefix={N} first_token={FIRST} draft=no") == 3


def _assert_e1(d_group, plans):
    assert all(p is not None and not p.skip for p in plans)
    assert all(len(r.req.prefix_indices) == C for r in d_group)  # E1: extend [c, N)
    assert all(not r.skips for r in d_group)
    assert d_group[0].pending and not d_group[1].pending  # E1 installs at the forward's GDN reads


def test_parts_without_end_are_e1(d_group):
    _publish(with_end=False)
    votes, plans = _run_group(d_group)
    assert votes == [1, 1, 1]  # the workers read the headers too: end_missing
    _assert_e1(d_group, plans)


def test_end_digest_mismatch_on_tp0_puts_every_rank_on_e1(d_group, caplog):
    _publish()
    _j, ppath = th.part_paths(RID, "pp1-2")
    bundle = torch.load(ppath)
    bundle["end"]["gdn"][5] = (bundle["end"]["gdn"][5][0] + 1, bundle["end"]["gdn"][5][1])
    torch.save(bundle, ppath)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group)
    assert votes == [1, 2, 2]  # workers hold nothing and could skip -- the MIN overrules them
    _assert_e1(d_group, plans)
    assert "skip_refused=level1:end_digest_MISMATCH:pp1-2" in caplog.text


def test_e1_refusal_is_todays_extend(d_group):
    _publish()
    th.remove(RID)  # no parts at all
    votes, plans = _run_group(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert all(len(r.req.prefix_indices) == PREFIX for r in d_group)


@pytest.mark.parametrize("what, reason", [
    ("batch", "batch_not_empty"),
    ("logprob", "logprob_or_hidden"),
    ("penalty", "penalty_or_min_new_tokens"),
    ("grammar", "grammar"),
])
def test_uniform_admission_refusal_is_e1_on_every_rank(d_group, caplog, what, reason):
    _publish()
    kw = {"logprob": {"return_logprob": True}, "grammar": {"grammar": object()},
          "penalty": {"sampling_params": SimpleNamespace(frequency_penalty=0.5, presence_penalty=0.0,
                                                         repetition_penalty=1.0, min_new_tokens=0)}}.get(what, {})
    reqs = [_req(**kw) for _ in d_group]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group, reqs=reqs, batch_empty=what != "batch")
    assert votes == [2, 2, 2]
    _assert_e1(d_group, plans)
    assert caplog.text.count(f"skip_refused=admission:{reason}") == 3


def test_a_process_without_the_skip_branch_votes_e1(d_group, monkeypatch):
    """A D whose model worker would run the extend forward anyway (no
    EAGLEWorkerV2) must never vote 2: the forward over [N-1, N) on a GDN
    state already at N would advance it twice."""
    _publish()
    monkeypatch.setattr(ta, "_SKIP_SERVER", [False])
    votes, plans = _run_group(d_group)
    assert votes == [1, 1, 1]
    _assert_e1(d_group, plans)


def test_switch_off_is_e1(d_group):
    _publish()
    with envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(False):
        votes, plans = _run_group(d_group)
    assert votes == [1, 1, 1]
    _assert_e1(d_group, plans)


def test_parts_disagreeing_on_the_token_refuse_the_skip(d_group):
    _publish()
    h = th.headers_for(RID)[1]
    jpath, _p = th.part_paths(RID, h.part)
    bad = th.TailHeader(**{f: getattr(h, f) for f in h.__struct_fields__ if f != "end"},
                        end=th.EndHeader(**{**{f: getattr(h.end, f) for f in h.end.__struct_fields__},
                                            "first_token": FIRST + 1}))
    import msgspec

    with open(jpath, "wb") as f:
        f.write(msgspec.json.encode(bad))
    votes, plans = _run_group(d_group)
    assert votes == [1, 1, 1]  # the workers read the token too: 'end_differs' everywhere
    _assert_e1(d_group, plans)


def test_mixed_batch_is_refused_loudly(d_group):
    _publish()
    _run_group(d_group)
    tp0 = d_group[0]
    other = SimpleNamespace(rid="weg2-9-9")
    with tp0.active():
        assert ta.skip_tokens(SimpleNamespace(reqs=[other])) is None
        with pytest.raises(RuntimeError, match="mixed batch"):
            ta.skip_tokens(SimpleNamespace(reqs=[tp0.req, other]))


def test_readback_mismatch_names_the_ring(d_group, caplog):
    _publish()
    tp0 = d_group[0]
    _run_group(d_group)
    _prepare_for_extend(tp0)
    inst = tp0.skips[RID].install
    _skip_on(tp0)
    assert ta.readback_digest(inst) == "match"
    inst.readback["ring7"][0][0, 0, 0] ^= 1
    assert ta.readback_digest(inst) == "MISMATCH:pp1-2:ring"


# ------------------------------------------------------------------ the worker's branch
def test_worker_returns_p_token_and_a_draft_seed_without_a_forward(d_group):
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    _publish()
    tp0 = d_group[0]
    _run_group(d_group)
    _prepare_for_extend(tp0)
    published = []
    seed = {}

    def stub(batch, next_token_ids):
        seed["bonus"] = next_token_ids
        return SimpleNamespace(bonus_tokens=next_token_ids)

    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(prefill_rank_timer=None),
                                      hicache_layer_transfer_counter=_Counter()),
        device="cpu", _solo_stub_draft_input=stub,
    )
    batch = SimpleNamespace(reqs=[tp0.req], hicache_consumer_index=-1, seq_lens=torch.tensor([N]))
    with tp0.active():
        out = EAGLEWorkerV2._forward_skip_extend(worker, batch, [FIRST], on_publish=published.append)
    _join("weg2-tail-verify")
    assert out.next_token_ids.tolist() == [FIRST]
    assert torch.equal(out.new_seq_lens, batch.seq_lens) and published == [batch.seq_lens]
    assert torch.equal(out.next_draft_input.bonus_tokens, out.next_token_ids)
    assert out.logits_output.hidden_states is None and not out.can_run_cuda_graph
    assert worker.target_worker.hicache_layer_transfer_counter.calls == []  # consumer -1: no load to join
    assert not tp0.skips
