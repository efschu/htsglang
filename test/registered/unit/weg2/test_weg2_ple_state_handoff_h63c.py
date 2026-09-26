"""fnFL2 H63c: the PLE side states (n-gram history, short-conv window) across every P -> D path.

Hermetic (no CUDA). The model's OWN PLE state code runs on CPU pools:
``qwen4_exp._prepare_ple_batch`` (n-gram history read, trigram windows),
``Qwen4ExpPLELayer._short_conv`` (the dilated conv over [window | inputs] and
its window update) and ``_commit_ple_batch`` (history commit), on real
``ShortConvPool`` / ``NGramPool`` slots. A PURE P RUN processes positions
[0, M) on one slot; a RESUMED run processes [0, p) on P, hands over through
the real path, and D processes [p, p+9) on a CLEARED slot of its own (EOS
history, zero window -- what a slot starts from). Per position of those 9 it
compares the n-gram lookup ids (the model's hash, host mirror
``ple_ngram_lookup_ids``) and the short-conv output of every channel (a
depthwise conv with random taps: equal outputs = equal inputs) with the pure
run:

* arena anchor (page resume): ``ple_state.side_write`` / ``side_read`` through a
  side table file beside a stub arena, tag = the slot's stem;
* END after N (E2 skip): P's fold (``arm_fold``, the whole prompt in one chunk,
  its extra_buffer track pending at the finish) published by the REAL
  ``UnifiedRadixCache.cache_finished_req`` (H63d: the publish reads all N rows
  before the retention truncation frees the tail), D's vote/agree/admission/
  skip (``run_skip`` -> ``_install_end``);
* E1 at c: P's stash capture, the tail forward, the REAL finish (E1 + END),
  D's E1 admission (``commit_adopt`` -> queued rows) applied by
  ``_prepare_ple_batch`` of the extend that carries the rid.

Armed: identical ids and outputs for all 9 positions (red on the pre-H63c tree:
no ``weg2.ple_state``). Unarmed: D's first 2 positions hash against EOS and
all 9 convolve against zeros -- the unchanged default, named.
"""

import contextlib
import logging
import os
import threading
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs

EOS = 7
C, L, K, DIL = 8, 9, 4, 3  # channels, window = (K-1)*DIL, kernel, dilation
PLE_LAYER = 1
SLOTS, SLOT = 5, 2
REQ = 3  # the request's req_pool_idx (maps to SLOT)
PAGE, RATIO = 64, 4
N = 241  # prompt: c = 240, page prefix 192
PREFIX, C_CUT = 192, 240
FIRST = 151645
M = N + 16  # positions the pure run covers


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


def _armed(on: bool):
    if not on and not hasattr(envs, "SGLANG_WEG2_PLE_STATE_HANDOFF"):
        return contextlib.nullcontext()
    return envs.SGLANG_WEG2_PLE_STATE_HANDOFF.override(on)


# ------------------------------------------------------------------ the model's PLE state code
def _seq(seed=5):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(10, 5000, (M,), generator=g)
    xs = torch.randn(M, C, generator=g)  # the short conv's per-position inputs (gated PLE values)
    weight = torch.randn(C, 1, K, generator=g)
    return tokens, xs, weight


def _hash_params():
    from sglang.srt.models.qwen4_exp_ple_prefetch import PleHashParams

    sizes = torch.tensor([97, 89, 83, 79])
    return PleHashParams(
        layer_multipliers=torch.tensor([1000003, 999983, 998857]),
        head_vocab_sizes=sizes,
        head_offsets=torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), sizes[:-1]]), 0),
        heads_per_ngram=2, ngram_size=3, eos_token_id=EOS,
    )


def _ple_pools(rp):
    from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

    rp.short_conv_pool = ShortConvPool(size=SLOTS, state_shape=(C, L), layer_ids=[PLE_LAYER],
                                       dtype=torch.float32, device="cpu")
    rp.ngram_pool = NGramPool(size=SLOTS, context_len=2, eos_token_id=EOS, device="cpu")
    rp.layer_transfer_counter = None
    rp._mamba_transfer_frame = None
    rp.ple_window_cache = None
    rp.req_index_to_mamba_index_mapping = torch.full((8,), SLOT, dtype=torch.int32)
    return rp


def _bare_pool():
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    return _ple_pools(object.__new__(HybridReqToTokenPool))


class _Model:
    """The model's own functions on a given pool (qwen4_exp's pool accessor
    points at it while a run lasts)."""

    def __init__(self, monkeypatch):
        import sglang.srt.models.qwen4_exp as q

        self.q = q
        self.pool = None
        monkeypatch.setattr(q, "get_req_to_token_pool", lambda: self.pool)
        tokens, xs, weight = _seq()
        self.tokens, self.xs = tokens, xs
        self.layer = SimpleNamespace(layer_id=PLE_LAYER, conv1d=SimpleNamespace(weight=weight),
                                     short_conv_dilation=DIL, short_conv_state_len=L, conv_channels=C)
        self.params = _hash_params()

    def run(self, pool, start: int, stop: int, rids=None):
        """One extend forward over [start, stop) on the slot of REQ: (ids, outputs)."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.qwen4_exp_ple_prefetch import ple_ngram_lookup_ids

        self.pool = pool
        k = stop - start
        fb = SimpleNamespace(
            tbo_parent_token_range=None, spec_algorithm=None, spec_info=None, _original_forward_mode=None,
            forward_mode=ForwardMode.EXTEND, global_num_token_non_padded_cpu=None,
            num_token_non_padded_cpu=None, extend_seq_lens=torch.tensor([k]), extend_seq_lens_cpu=[k],
            req_pool_indices=torch.tensor([REQ]), out_cache_loc=None, mamba_track_indices=None,
            mamba_track_mask=None, rids=rids,
        )
        batch = self.q._prepare_ple_batch(self.tokens[start:stop], fb, ngram_size=3, ngram_eos_token_id=EOS)
        windows = batch.ngram_context[0].unfold(0, 3, 1)[:k]  # (t-2, t-1, t) per position
        ids = ple_ngram_lookup_ids(windows, self.params)
        out = self.q.Qwen4ExpPLELayer._short_conv(self.layer, self.xs[start:stop], fb, batch)
        self.q._commit_ple_batch(batch, fb)
        return ids, out

    def pure(self, p: int):
        """The pure P run's ids and outputs for positions [p, p+9)."""
        pool = _bare_pool()
        self.run(pool, 0, p)
        return self.run(pool, p, p + 9)


def _assert_same(got, want):
    ids, out = got
    ids_w, out_w = want
    assert torch.equal(ids, ids_w), "n-gram lookup ids differ after the resume"
    assert torch.allclose(out, out_w, atol=1e-6), "short-conv outputs differ after the resume"


def _assert_reset_differs(got, want):
    """The unarmed resume: EOS history (the first 2 positions' ids) and a zero
    window (the first 9 positions' conv outputs) -- not the pure run's."""
    ids, out = got
    ids_w, out_w = want
    assert not torch.equal(ids[0], ids_w[0]) and not torch.equal(ids[1], ids_w[1])
    assert torch.equal(ids[2:], ids_w[2:])  # the history is the chunk's own from position 2 on
    diff = [(out[j] - out_w[j]).abs().max().item() > 1e-6 for j in range(9)]
    assert all(diff), diff


# ------------------------------------------------------------------ arena anchor (page resume)
class _Arena:
    def __init__(self, path):
        self.path = path
        self.slots = 4
        self.stems = {1: "hashA.mamba", 3: "hashB.mamba"}

    def slot_stem(self, slot):
        return self.stems.get(int(slot), "")


def test_arena_resume_sees_p_ple_state(monkeypatch, tmp_path):
    from sglang.srt.weg2 import ple_state

    model = _Model(monkeypatch)
    arena = _Arena(str(tmp_path / "mamba.arena"))
    p = PREFIX
    want = model.pure(p)
    with _armed(True):
        p_pool = _bare_pool()
        model.run(p_pool, 0, p)  # the anchor's state: after p tokens
        assert ple_state.side_write(arena, p_pool, [1], torch.tensor([SLOT])) == 1
        d_pool = _bare_pool()  # D: a cleared slot (EOS history, zero window)
        assert ple_state.side_read(arena, d_pool, [1], [SLOT]) == (1, 0)
        got = model.run(d_pool, p, p + 9)
    _assert_same(got, want)
    ple_state._TABLES.clear()


def test_arena_row_of_another_stem_is_never_installed(monkeypatch, tmp_path):
    from sglang.srt.weg2 import ple_state

    model = _Model(monkeypatch)
    arena = _Arena(str(tmp_path / "mamba.arena"))
    with _armed(True):
        p_pool = _bare_pool()
        model.run(p_pool, 0, PREFIX)
        ple_state.side_write(arena, p_pool, [1], torch.tensor([SLOT]))
        arena.stems[1] = "hashC.mamba"  # the slot was recycled by another node
        d_pool = _bare_pool()
        assert ple_state.side_read(arena, d_pool, [1], [SLOT]) == (0, 1)
        assert torch.equal(d_pool.ngram_pool.context[SLOT], torch.full((2,), EOS))
        assert not d_pool.short_conv_pool.conv_state.any()
    ple_state._TABLES.clear()


def test_arena_unarmed_is_todays_reset(monkeypatch, tmp_path):
    model = _Model(monkeypatch)
    want = model.pure(PREFIX)
    with _armed(False):
        d_pool = _bare_pool()
        got = model.run(d_pool, PREFIX, PREFIX + 9)
    _assert_reset_differs(got, want)


# ------------------------------------------------------------------ the tail parts (E1, END)
RID = "weg2-0-4"
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
REQ_SLOTS, P_RPI, D_RPI = 6, 4, 1
KV_ROWS = 6 * PAGE
D_PAGE = 3
FP8 = torch.float8_e4m3fn


def _p_pools(seed=24):
    """P's PP0: QSA KV rows + GDN slots + the PLE pools (the PLE layer is here)."""
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
    rp = _bare_pool()
    rp.mamba_map = {0: 0, 1: 1, 2: 2}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.randn(3, SLOTS, 2, 4, 4, generator=g),
        conv=[torch.randn(3, SLOTS, 6, 3, generator=g)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, N + PAGE, dtype=torch.int32)
    rp.req_to_token[P_RPI, :N] = torch.arange(N, dtype=torch.int32) + PAGE  # row of token i = i + 64
    alloc = SimpleNamespace(get_kvcache=lambda: kv, free=lambda idx: None)
    return kv, rp, alloc


def _p_req(ids, start, end, output_ids=(FIRST,), rid=RID):
    ids = array("q", ids)
    return SimpleNamespace(rid=rid, origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
                           extend_range=SimpleNamespace(start=start, end=end), mamba_pool_idx=torch.tensor(SLOT),
                           req_pool_idx=P_RPI, output_ids=array("q", output_ids), return_logprob=False,
                           return_hidden_states=False, prefix_indices=range(start), cache_protected_len=0,
                           mamba_last_track_seqlen=None, priority=None, swa_uuid_for_lock=None, last_node=None,
                           pop_committed_kv_cache=lambda: len(ids))


class _TrackComp:
    """The mamba component's answer at the finish: the pending extra_buffer
    track is the retention (mamba_component.py ``cache_len =
    req.mamba_last_track_seqlen``)."""

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        return req.mamba_last_track_seqlen if is_finished else None

    def cleanup_after_caching_req(self, req, is_finished, insert_result=None, insert_params=None):
        pass


def _finish(rp, alloc, req):
    """P's REAL finish (``UnifiedRadixCache.cache_finished_req``) on the
    collaborators it touches; the tail publish runs inside it."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

    tree = SimpleNamespace(
        session=SimpleNamespace(try_cache_finished_req=lambda r, **kw: False), disable=False, is_eagle=True,
        bigram_anchor_exact=True, page_size=PAGE, pp_rank=0, pp_size=1, req_to_token_pool=rp,
        token_to_kv_pool_allocator=alloc, _components_tuple=(_TrackComp(),),
        insert=lambda params: SimpleNamespace(prefix_len=0, mamba_exist=False),
        _weg2_note_end_anchor=lambda r, ids: None, _weg2_handoff_write=lambda r, key: None,
        _weg2_publish_at_retain=lambda r, key: None, _anchor_dec_skip=lambda r, prm: None,
        dec_lock_ref=lambda node, prm, skip_swa=False: None,
    )
    tree._note_protected_beyond_retention = lambda r, ecl: U._note_protected_beyond_retention(tree, r, ecl)
    # 27B 34965fc3fa (per-path mamba cap, unified S7c 92666055b7): the finish
    # runs the cap after the #1481 mark. Bound off the real class; this tree's
    # insert anchors nothing (`_weg2_cap_tail` unset), so it caps nothing.
    tree._weg2_cap_after_insert = lambda: U._weg2_cap_after_insert(tree)
    U.cache_finished_req(tree, req)
    _join("weg2-tail-publish")


def _d_pools(worker: bool):
    """D's ranks for P's one part (pp0-1, n_parts=1): TP0 holds exactly the
    layers and row formats P's PP0 publishes (FA 3/7, GDN 0/1/2, fp32 rows),
    in its own local order, plus the PLE layer; a Form-A worker holds none
    (0 heads, 0-byte GDN rows) and no PLE layer."""
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {7: 0, 3: 1}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.zeros(KV_ROWS, heads, 8) for _ in range(2)],
        v_buffer=[torch.zeros(KV_ROWS, heads, 8) for _ in range(2)],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.zeros(KV_ROWS // RATIO + 1, 1, 4) for _ in range(2)]
    kv.qsa_key_state_buffer_pool = [torch.zeros(REQ_SLOTS * RATIO, 1, 4) for _ in range(2)]
    kv.qsa_rope_position_buffer = torch.zeros(REQ_SLOTS * RATIO, 3, dtype=torch.int64)
    rp = object.__new__(HybridReqToTokenPool) if worker else _bare_pool()
    rp.mamba_map = {0: 2, 1: 1, 2: 0}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(3, SLOTS, 0 if worker else 2, 4, 4),
        conv=[torch.zeros(3, SLOTS, 0 if worker else 6, 3)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, 512, dtype=torch.int32)
    return kv, rp


class Rank:
    def __init__(self, worker: bool):
        from sglang.srt.weg2 import tail_adopt as ta

        self.ta = ta
        self.kv, self.rp = _d_pools(worker)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: self.kv),
                                    req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending, self.skips = {}, {}, [], {}
        self.req = None

    @contextlib.contextmanager
    def active(self):
        ta = self.ta
        saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
        ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.jobs, self.agreed, self.pending, self.skips
        try:
            yield
        finally:
            ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = saved


def _req(ids, **kw):
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    base = dict(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                prefix_indices=torch.arange(PAGE, PAGE + PREFIX, dtype=torch.int64),
                mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=D_RPI, return_logprob=False,
                return_hidden_states=False, grammar=None, sampling_params=sp)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def arena_dir(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp
    from sglang.srt.weg2 import tail_handoff as th

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # P publishes
    th._CAPTURES.clear()
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True), envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield tmp_path
    th._CAPTURES.clear()


def _as_d(monkeypatch):
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.common as common
    from sglang.srt.weg2 import tail_adopt as ta

    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)  # D adopts
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    return ta


def _run_group(ranks, ids, skip_server: bool, monkeypatch):
    ta = _as_d(monkeypatch)
    monkeypatch.setattr(ta, "_SKIP_SERVER", [skip_server])
    votes = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            _join("weg2-tail-stage")
            votes.append(ta.local_vote(RID))
    plans = []
    for r in ranks:
        r.req = _req(ids)
        with r.active():
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
    rank.rp.req_to_token[D_RPI, n_prefix:N] = torch.arange(
        int(req.prefix_indices[-1]) + 1, int(req.prefix_indices[-1]) + 1 + N - n_prefix, dtype=torch.int32)


def _skip_on(rank):
    batch = SimpleNamespace(reqs=[rank.req], hicache_consumer_index=-1)
    with rank.active():
        rank.ta.skip_tokens(batch)
        rank.ta.run_skip(batch, counter=None)
    _join("weg2-tail-verify")


def _pending() -> dict:
    """ple_state.PENDING where H63c exists ({} on the pre-H63c tree)."""
    try:
        from sglang.srt.weg2 import ple_state
    except ImportError:
        return {}
    return ple_state.PENDING


def _clear_d_slot(rp):
    rp.ngram_pool.reset_slots(torch.tensor([SLOT]))
    rp.short_conv_pool.reset_slots(torch.tensor([SLOT]))


@pytest.mark.parametrize("armed", [True, False])
def test_end_skip_resume_sees_p_ple_state(arena_dir, monkeypatch, caplog, armed):
    from sglang.srt.weg2 import tail_handoff as th

    model = _Model(monkeypatch)
    ids = model.tokens[:N].tolist()
    want = model.pure(N)
    kv, rp, alloc = _p_pools()
    with _armed(armed), envs.SGLANG_WEG2_ENABLE_P_TAIL_FOLD.override(True), \
            caplog.at_level(logging.INFO, logger="sglang.srt.weg2.ple_state"):
        # P: the fold runs the whole prompt in its last chunk -- its track
        # (floor_page(N) = 192) is still pending at the finish -- and the real
        # finish publishes the END-only part (H63d: before the truncation)
        req = _p_req(ids, 0, N)
        assert th.arm_fold([req], alloc, PAGE, None) == 1
        model.run(rp, 0, N)
        req.mamba_last_track_seqlen = PREFIX
        _finish(rp, alloc, req)
        assert th.headers_for(RID), "no END part: the finish did not publish (H63d missing?)"
        # D: a cleared slot, the group skips the extend, run_skip installs the END state
        ranks = [Rank(worker=False), Rank(worker=True)]
        _clear_d_slot(ranks[0].rp)
        votes, plans = _run_group(ranks, ids, skip_server=True, monkeypatch=monkeypatch)
        assert votes == [2, 2] and all(p is not None and p.skip for p in plans)
        for r in ranks:
            _prepare_for_extend(r)
            _skip_on(r)
        got = model.run(ranks[0].rp, N, N + 9)
    if armed:
        _assert_same(got, want)
        text = caplog.text
        p_line = [ln for ln in text.splitlines() if "side=P path=end" in ln]
        d_line = [ln for ln in text.splitlines() if "side=D path=end" in ln]
        assert len(p_line) == 1 and len(d_line) == 1
        dig = p_line[0].split("digest=")[1].split()[0]
        assert f"digest={dig}" in d_line[0] and "ctx_ok=yes installed=yes" in d_line[0]
    else:
        _assert_reset_differs(got, want)
    assert not _pending()


@pytest.mark.parametrize("armed", [True, False])
def test_e1_resume_sees_p_ple_state(arena_dir, monkeypatch, caplog, armed):
    from sglang.srt.weg2 import tail_handoff as th

    model = _Model(monkeypatch)
    ids = model.tokens[:N].tolist()
    want = model.pure(C_CUT)
    kv, rp, alloc = _p_pools()
    with _armed(armed), caplog.at_level(logging.INFO, logger="sglang.srt.weg2.ple_state"):
        # P: the body [0, c), the stash captures at c (the unfinished insert
        # there consumes the track), the tail [c, N), the real finish (E1 + END)
        model.run(rp, 0, C_CUT)
        req = _p_req(ids, 0, C_CUT)
        assert th.capture_state(req, rp, alloc, PAGE, None)
        req.cache_protected_len = PREFIX
        model.run(rp, C_CUT, N)
        req.extend_range = SimpleNamespace(start=C_CUT, end=N)
        _finish(rp, alloc, req)
        # D: no skip branch -> E1; the rows at c are queued for the extend
        ranks = [Rank(worker=False), Rank(worker=True)]
        _clear_d_slot(ranks[0].rp)
        votes, plans = _run_group(ranks, ids, skip_server=False, monkeypatch=monkeypatch)
        assert votes == [1, 1] and all(p is not None and not p.skip for p in plans)
        # the extend [c, ...) that carries the rid reads (and first applies) them
        got = model.run(ranks[0].rp, C_CUT, C_CUT + 9, rids=[RID])
    if armed:
        _assert_same(got, want)
        assert "side=D path=e1" in caplog.text and "installed=queued" in caplog.text
    else:
        _assert_reset_differs(got, want)
    assert not _pending()


def test_a_queued_row_waits_for_its_own_rid(monkeypatch):
    """E1 rows are applied only in the forward that carries the rid (host
    lookup, no device sync) -- another request's forward leaves them queued."""
    from sglang.srt.weg2 import ple_state

    model = _Model(monkeypatch)
    with _armed(True):
        p_pool = _bare_pool()
        model.run(p_pool, 0, C_CUT)
        rows = ple_state.snapshot(p_pool, torch.tensor([SLOT]))
        d_pool = _bare_pool()
        ple_state.queue(RID, torch.tensor([SLOT]), rows)
        assert ple_state.apply_pending(d_pool, ["weg2-9-9"]) == 0 and RID in ple_state.PENDING
        assert ple_state.apply_pending(d_pool, [RID]) == 1 and not ple_state.PENDING
        assert ple_state.digest(ple_state.snapshot(d_pool, torch.tensor([SLOT]))) == ple_state.digest(rows)


def test_rows_of_another_geometry_are_refused(monkeypatch):
    from sglang.srt.weg2 import ple_state

    with _armed(True):
        pool = _bare_pool()
        rows = {"conv": torch.zeros(1, 1, C, L - 1), "ngram": torch.zeros(1, 2, dtype=torch.long)}
        assert ple_state.install(pool, torch.tensor([SLOT]), rows).startswith("conv_shape")
        assert ple_state.refusal(pool, None) == "absent"
