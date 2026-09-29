"""F4 (#259 4c): D's flip park keeps the WHOLE decode tail -- the resume is a skip.

Hermetic (no CUDA). Metal z30r3 / FLIPZEIT-VERLAUF-0929: a request D parks
for the D->P flip keeps its KV and its GDN state only up to the last mamba
track point (#59b resumable depth); the resume extends the tail from there --
33-128 tokens, one eager expert pass of ~2 s on NF-D, paid by every parked
request, also behind the wake's first decode round (F3).

F4 writes, at the park and before the retaining retraction frees the slots,
the END state of every running request in P's own hand-off format (an
END-only part, H63's fold form): the KV (+ complete QSA groups) rows from a
window below the cut up to the last CONSUMED token, the open group's pending
ring, the GDN slot. The resume then takes E2's skip -- no target forward --
and the decode feeds the parked token next. What these cases pin:

* the park part is keyed on the tokens D had CONSUMED (prompt + all output
  but the last); its token is the last sampled one, never fed to the model;
* the resume re-enters anywhere inside the park's row window; the commit
  grows the prefix to c, pops the parked token off ``output_ids`` (the skip
  hands it back; the stream sent it before the park) and the install lands
  byte-exact at D's own new slots;
* P's prune/census never ages a park part, and a park only clears P's
  leftovers and its OWN rank's earlier part -- never a peer's fresh one;
* refused by name under uneven DCP / the token cut, and off by default.
"""

import contextlib
import logging
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.pdflip import tail_adopt as ta
from flliper.srt.pdflip import tail_handoff as th

RID = "pdflip-7-31"
PAGE, RATIO, INTERVAL = 64, 4, 64
PROMPT, OUT = 200, 43
N = PROMPT + OUT - 1  # 242 consumed tokens: c = 240, window from 192 - 128 = 64
C, WIN = 240, 64
FIRST = 151645  # the last sampled token (output_ids[-1]), not yet fed
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
SLOTS, SLOT = 5, 2
REQ_SLOTS, SRC_RPI, D_RPI = 6, 4, 1
KV_ROWS = 8 * PAGE
SRC_BASE = 128  # the parked request's token slots: [128, 128 + N)
D_PAGE = 5  # D's allocator after the wake hands out slots from 320
FP8 = torch.float8_e4m3fn


def _ids():
    prompt = [1000 + i for i in range(PROMPT)]
    out = [2000 + i for i in range(OUT - 1)] + [FIRST]
    return prompt, out


def _pools(worker: bool, seed=None):
    from flliper.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from flliper.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    make = (lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype)) if seed is None else None
    g = torch.Generator().manual_seed(seed or 0)

    def t(*shape, dtype=torch.float32):
        if make is not None:
            return make(*shape, dtype=dtype)
        return torch.randn(*shape, generator=g).to(dtype)

    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[t(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
        v_buffer=[t(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
    )
    kv.qsa_compressed_k_buffer_pool = [t(KV_ROWS // RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_key_state_buffer_pool = [t(REQ_SLOTS * RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_rope_position_buffer = (torch.arange(REQ_SLOTS * RATIO * 3, dtype=torch.int64).view(-1, 3)
                                   if seed is not None else torch.zeros(REQ_SLOTS * RATIO, 3, dtype=torch.int64))
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {gid: len(GDN_GIDS) - 1 - i for i, gid in enumerate(GDN_GIDS)}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=t(len(GDN_GIDS), SLOTS, 0 if worker else 2, 4, 4),
        conv=[t(len(GDN_GIDS), SLOTS, 0 if worker else 6, 3, dtype=torch.bfloat16)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, 512, dtype=torch.int32)
    return kv, rp


class Rank:
    def __init__(self, worker: bool, seed=None):
        self.kv, self.rp = _pools(worker, seed)
        self.alloc = SimpleNamespace(get_kvcache=lambda: self.kv)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=self.alloc, req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending, self.skips = {}, {}, [], {}
        self.req = None

    def active(self):
        rank = self

        class _Ctx:
            def __enter__(self):
                self.saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
                ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = (
                    rank.jobs, rank.agreed, rank.pending, rank.skips)

            def __exit__(self, *exc):
                ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.saved
                return False

        return _Ctx()


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


def _running_req(rpi=SRC_RPI, **kw):
    prompt, out = _ids()
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    base = dict(rid=RID, origin_input_ids=prompt, output_ids=list(out),
                full_untruncated_fill_ids=prompt + out, extra_key=None,
                mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=rpi, return_logprob=False,
                return_hidden_states=False, grammar=None, sampling_params=sp)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def arena(tmp_path, monkeypatch):
    import flliper.srt.distributed.utils as du
    import flliper.srt.managers.schedule_policy as sp
    import flliper.srt.mem_cache.common as common

    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_PDFLIP_END_ANCHOR", False)  # group D
    monkeypatch.setattr(du, "uneven_dcp_active", lambda *a, **k: False)
    monkeypatch.setattr(ta, "_SKIP_SERVER", [True])
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.FLLIPER_PDFLIP_TAIL_HANDOFF.override(True), envs.FLLIPER_PDFLIP_TAIL_ADOPT.override(True), \
            envs.FLLIPER_PDFLIP_TAIL_VERIFY.override(True), envs.FLLIPER_PDFLIP_TAIL_SKIP_EXTEND.override(True), \
            _park_switch(True):
        yield tmp_path


def _park_switch(on: bool):
    # tolerant of a tree without the switch: the cases then fail on the
    # behaviour they pin, not on the fixture
    env = getattr(envs, "FLLIPER_PDFLIP_ENABLE_D_PARK_END", None)
    return env.override(on) if env is not None else contextlib.nullcontext()


def _park(ranks):
    """park_running's F4 step on every rank of the group: the request ran at
    token slots [SRC_BASE, SRC_BASE + N) in request row SRC_RPI."""
    reqs = []
    for i, r in enumerate(ranks):
        r.rp.req_to_token[SRC_RPI, :N] = torch.arange(SRC_BASE, SRC_BASE + N, dtype=torch.int32)
        req = _running_req()
        why, _ev = th.publish_park_end(req, r.rp, r.alloc, PAGE, f"dpark{i}-77", len(ranks), 2 * INTERVAL)
        assert why == ""
        reqs.append(req)
    _join("pdflip-park-end")
    return reqs


def _resume(ranks, prefix_len=192):
    """The wake: every rank stages, votes, agrees and admits the parked
    request; its prefix came back from the store at [0, prefix_len)."""
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            _join("pdflip-tail-stage")
            votes.append(ta.local_vote(RID))
    group = min(votes)
    plans = []
    for r in ranks:
        r.req = _running_req(rpi=D_RPI, prefix_indices=torch.arange(0, prefix_len, dtype=torch.int64))
        with r.active():
            ta.agree(RID, group)
            plan = ta.plan_adopt(r.req, len(r.req.prefix_indices), batch_empty=True)
            if plan is not None:
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
            plans.append(plan)
    _join("pdflip-park-end-rm")
    return votes, plans


def _prepare_for_extend(rank):
    req = rank.req
    n_prefix = len(req.prefix_indices)
    n = len(req.full_untruncated_fill_ids)
    rank.rp.req_to_token[D_RPI, :n_prefix] = req.prefix_indices.to(torch.int32)
    last = int(req.prefix_indices[-1])
    rank.rp.req_to_token[D_RPI, n_prefix:n] = torch.arange(last + 1, last + 1 + n - n_prefix, dtype=torch.int32)


def _skip_on(rank):
    batch = SimpleNamespace(reqs=[rank.req], hicache_consumer_index=-1)
    with rank.active():
        tokens = ta.skip_tokens(batch)
        ta.run_skip(batch, counter=None)
    _join("pdflip-tail-verify")
    return tokens


def _group(seed=None):
    return [Rank(worker=False, seed=seed), Rank(worker=True), Rank(worker=True)]


# ----------------------------------------------------------------- the write
def test_park_writes_the_end_of_what_d_consumed(arena, caplog):
    src = _group(seed=4)
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.tail_handoff"):
        _park(src)
    headers = th.headers_for(RID)
    assert sorted(h.part for h in headers) == ["dpark0-77", "dpark1-77", "dpark2-77"]
    assert th.manifest_state(headers) == ("complete", 3, 3)
    prompt, out = _ids()
    consumed = prompt + out[:-1]
    for h in headers:
        assert (h.spec.n_tokens, h.spec.page_prefix, h.spec.cut) == (N, WIN, C)
        assert not h.e1  # END-only: the fold form, served as the skip or not at all
        assert h.end.first_token == FIRST
        assert h.end.key == th.tail_key(consumed, N, None)
        assert (h.end.rows, h.end.groups, h.end.ring_rows) == (N - WIN, (N - WIN) // RATIO, N % RATIO)
    tp0 = [h for h in headers if h.part == "dpark0-77"][0]
    bundle = th.verify_part(tp0)
    sec = bundle["end"]
    assert sorted(sec["fa"]) == sorted(FA_GIDS) and sorted(sec["gdn"]) == sorted(GDN_GIDS)
    for h in headers:
        if h.part != "dpark0-77":
            b = th.verify_part(h)
            assert not b["end"]["fa"] and not b["end"]["gdn"]  # a Form-A worker holds nothing
    assert caplog.text.count(f"F4 PARK-END rid={RID} n_tokens={N} rows_from={WIN} cut={C}") == 3


def test_switch_off_writes_nothing(arena):
    r = Rank(worker=False, seed=4)
    with _park_switch(False):
        why, ev = th.publish_park_end(_running_req(), r.rp, r.alloc, PAGE, "dpark0-77", 3, 128)
    assert (why, ev) == ("off", None) and th.headers_for(RID) == []


def test_uneven_dcp_is_refused_by_name(arena, monkeypatch):
    import flliper.srt.distributed.utils as du

    monkeypatch.setattr(du, "uneven_dcp_active", lambda *a, **k: True)
    r = Rank(worker=False, seed=4)
    why, _ev = th.publish_park_end(_running_req(), r.rp, r.alloc, PAGE, "dpark0-77", 3, 128)
    assert why == "uneven_dcp" and th.headers_for(RID) == []


# ---------------------------------------------------------------- the resume
def test_resume_is_a_skip_with_the_parked_token(arena, caplog):
    src = _group(seed=4)
    _park(src)
    dst = _group()
    caplog.set_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt")
    votes, plans = _resume(dst, prefix_len=192)
    assert votes == [2, 2, 2] and all(p is not None and p.skip and p.fill_drop == 1 for p in plans)
    prompt, out = _ids()
    for r in dst:
        assert len(r.req.prefix_indices) == C  # the prefix grew to c inside D's new page
        assert r.req.output_ids == out[:-1]  # the parked token comes back as the skip's
        assert r.req.full_untruncated_fill_ids == prompt + out[:-1]  # extend [c, N) over consumed tokens
        _prepare_for_extend(r)
    tokens = [_skip_on(r) for r in dst]
    assert tokens == [[FIRST]] * 3  # no rank runs a target forward
    # the install: token t's rows land at D's slot for t, byte-exact from the park
    s, d = src[0], dst[0]
    src_slots = s.rp.req_to_token[SRC_RPI, WIN:N].to(torch.int64)
    dst_slots = d.rp.req_to_token[D_RPI, WIN:N].to(torch.int64)
    for gid, local in d.kv.full_attention_layer_id_mapping.items():
        sl = s.kv.full_attention_layer_id_mapping[gid]
        for buf in ("k_buffer", "v_buffer"):
            got = getattr(d.kv.full_kv_pool, buf)[local][dst_slots].view(torch.uint8)
            want = getattr(s.kv.full_kv_pool, buf)[sl][src_slots].view(torch.uint8)
            assert torch.equal(got, want)
        n_groups = (N - WIN) // RATIO
        assert torch.equal(d.kv.qsa_compressed_k_buffer_pool[local][dst_slots[: n_groups * RATIO: RATIO] // RATIO],
                           s.kv.qsa_compressed_k_buffer_pool[sl][src_slots[: n_groups * RATIO: RATIO] // RATIO])
        ring = N % RATIO
        assert torch.equal(d.kv.qsa_key_state_buffer_pool[local][D_RPI * RATIO: D_RPI * RATIO + ring],
                           s.kv.qsa_key_state_buffer_pool[sl][SRC_RPI * RATIO: SRC_RPI * RATIO + ring])
    for gid, local in d.rp.mamba_map.items():
        assert torch.equal(d.rp.mamba_pool.mamba_cache.temporal[local, SLOT],
                           s.rp.mamba_pool.mamba_cache.temporal[local, SLOT])
    assert th.headers_for(RID) == []  # read by every rank: the park parts are gone
    assert caplog.text.count(f"PDFLIP-TAIL-SKIP-EXTEND rid={RID} prefix={N} first_token={FIRST}") == 3
    assert f"PDFLIP-TAIL-ADOPT rid={RID} page_prefix={WIN} tail_rows={N - WIN} state_at={N} extend=0 " \
           f"fa_rows_written={N - WIN} fa_layers=3 gdn_layers=9 digest=match" in caplog.text


def test_resume_below_the_window_is_todays_extend(arena, caplog):
    _park(_group(seed=4))
    dst = _group()
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt"):
        votes, plans = _resume(dst, prefix_len=0)
    assert plans == [None] * 3
    prompt, out = _ids()
    assert all(r.req.output_ids == out and len(r.req.prefix_indices) == 0 for r in dst)  # untouched
    assert f"skipped:prefix:0!in[{WIN},{C})" in caplog.text


def test_a_changed_token_is_refused(arena, caplog):
    _park(_group(seed=4))
    dst = _group()
    prompt, out = _ids()
    votes = []
    for r in dst:
        with r.active():
            ta.stage(RID, r.tree)
            _join("pdflip-tail-stage")
            votes.append(ta.local_vote(RID))
    for r in dst:
        r.req = _running_req(rpi=D_RPI, prefix_indices=torch.arange(0, 192, dtype=torch.int64),
                             output_ids=out[:-1] + [FIRST + 1])
        with r.active():
            ta.agree(RID, min(votes))
            with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt"):
                assert ta.plan_adopt(r.req, 192, batch_empty=True) is None
    assert "end_only:park_token_mismatch" in caplog.text


# ------------------------------------------------------ P's store bookkeeping
def _touch(d, name):
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"x")


def test_p_prune_never_takes_a_park_part(arena):
    d = th._dir()
    os.makedirs(d, exist_ok=True)
    for part in ("pp0-1", "dpark0-77", "dpark1-78"):
        _touch(d, f"{RID}.tail.{part}.json")
    ages, sizes = th.census(d)
    assert set(sizes) <= {RID} and sizes.get(RID, 0) == 1  # only P's part is counted
    th.remove(RID)
    assert sorted(os.listdir(d)) == [f"{RID}.tail.dpark0-77.json", f"{RID}.tail.dpark1-78.json"]
    # a park clears P's leftovers and its OWN earlier part, never a peer's
    _touch(d, f"{RID}.tail.pp1-2.json")
    th._clear_for_park(RID, "dpark0")
    assert sorted(os.listdir(d)) == [f"{RID}.tail.dpark1-78.json"]
    th.remove(RID, parks=True)
    assert os.listdir(d) == []


# ---------------------------------------------------------- park_running wiring
def test_park_running_writes_every_running_request_and_measures_the_wait(arena, monkeypatch, caplog):
    from flliper.srt.pdflip import d_park_runtime as dpr

    calls = []

    def fake(req, rtp, alloc, page, part, n_parts, window):
        calls.append((str(req.rid), part, n_parts, window))
        return ("", None) if req.rid != "pdflip-9-9" else ("no_tail", None)

    monkeypatch.setattr(th, "publish_park_end", fake)
    sched = SimpleNamespace(ps=SimpleNamespace(tp_rank=1, tp_size=3), page_size=PAGE,
                            server_args=SimpleNamespace(mamba_track_interval=INTERVAL),
                            req_to_token_pool=None, token_to_kv_pool_allocator=None)
    running = [SimpleNamespace(rid=RID), SimpleNamespace(rid="pdflip-9-9")]
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.d_park_runtime"):
        assert dpr._park_end(sched, running) == 1
    pid = os.getpid()
    assert calls == [(RID, f"dpark1-{pid}", 3, 2 * INTERVAL), ("pdflip-9-9", f"dpark1-{pid}", 3, 2 * INTERVAL)]
    assert "F4 PARK-END park: parts=1 of 2 running sync_ms=" in caplog.text
    assert "'pdflip-9-9': 'no_tail'" in caplog.text
    calls.clear()
    with _park_switch(False):
        assert dpr._park_end(sched, running) == 0
    assert calls == []
