"""MIXED-D (S32, 18.10.): mixed chunk for SHORT D prefills -- upstream PR #36933 port + the short gate.

Order 18.10. "mixed chunk fuer kurze D-Prefills wieder an"; planner: gate = u < halbX (SOLL R2) on group D only, never P.
Covers: the gate (boundary), the call edge in the scheduler (collective entry guard, chunk stamp, default ON), the mixed batch
arithmetic of ``ScheduleBatch.mix_with_running`` + ``FutureMap.resolve_mixed_spec_tails`` (spec, overlap lag), and TOKEN
IDENTITY mixed vs unmixed on a deterministic toy model that reads/writes KV through the batch's own tensors.

WHAT THIS DOES NOT PROVE: that the real EAGLE draft re-seed of a tail row, the GDN/mamba state of a 1-token extend, and the
attention backends accept the mixed spec batch. That is the metal acceptance (report section 'Metall-Pruefplan')."""
from __future__ import annotations

import ast
import os
import pathlib
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.weg2 import mixed_d_short as mds  # noqa: E402

X = 6000
SRT = pathlib.Path(__import__("sglang.srt", fromlist=["x"]).__path__[0])


# ------------------------------------------------------------------------------------------------------ the gate
# S33 review A1: the bound is the FRONT's halbX (weg2_front_halfx, min over the batch, SOLL R2a), X only says "group D".
HX = 3000


def test_gate_boundary_is_strictly_below_the_front_halfx():
    assert mds.short_gate(2999, X, HX, False) is True       # u < halbX
    assert mds.short_gate(3000, X, HX, False) is False      # u == halbX is halb (R3), not kurz (R2)
    assert mds.short_gate(3001, X, HX, False) is False
    assert mds.short_gate(0, X, HX, False) is True


def test_gate_never_on_p_never_chunked_never_without_a_front_halfx_never_an_e2_handoff():
    assert mds.short_gate(10, 0, HX, False) is False        # P / no riegel: X = 0
    assert mds.short_gate(10, X, HX, True) is False         # chunked request in or behind this batch (R6 rest)
    assert mds.short_gate(10, X, 0, False) is False         # no halbX carried by the front: no invented bound
    assert mds.short_gate(10, X, HX, False, no_forward=True) is False   # E2 END-state hand-off: no target forward


def _r(rid, hx):
    return types.SimpleNamespace(rid=rid, weg2_front_halfx=hx)


def test_batch_half_x_is_the_smallest_front_halfx_and_zero_when_one_is_missing():
    assert mds.batch_half_x([_r("a", 2807), _r("b", 2400)]) == 2400      # R2a: the smallest X of the members
    assert mds.batch_half_x([_r("a", 2807), _r("b", None)]) == 0         # one without a carried halbX: never
    assert mds.batch_half_x([_r("a", 0)]) == 0
    assert mds.batch_half_x([]) == 0


def test_the_rank_riegel_is_not_the_bound():
    """NF: --tp-prefill-max-tokens 12288 -> 0.5 x 12288 = 6144 lies above every curve X (4096..5800) the front decides
    with; a 4000-token D prefill under front X 5614 (halbX 2807) is halb, not kurz -- never mixed."""
    hx = mds.batch_half_x([_r("a", 2807)])
    assert mds.short_gate(4000, 12288, hx, False) is False
    assert mds.short_gate(2806, 12288, hx, False) is True


def test_an_e2_handoff_in_the_batch_is_never_mixed(monkeypatch):
    from sglang.srt.weg2 import tail_adopt

    monkeypatch.setattr(tail_adopt, "SKIP_PLANS", {"pdflip-29-190": object()})
    assert mds.end_state_handoff([_r("pdflip-29-190", 2807)]) is True
    assert mds.end_state_handoff([_r("pdflip-30-192", 2807)]) is False
    monkeypatch.setattr(tail_adopt, "SKIP_PLANS", {})
    assert mds.end_state_handoff([_r("pdflip-29-190", 2807)]) is False


def test_default_on_and_switch(monkeypatch):
    assert envs.SGLANG_WEG2_MIXED_D_SHORT.default is True
    monkeypatch.setenv("SGLANG_WEG2_MIXED_D_SHORT", "0")
    assert mds.enabled() is False
    monkeypatch.delenv("SGLANG_WEG2_MIXED_D_SHORT")
    assert mds.enabled() is True


def test_spec_algorithm_support_is_the_eagle_family_only():
    S = SpeculativeAlgorithm
    assert S.EAGLE.supports_mixed_chunk() and S.EAGLE3.supports_mixed_chunk()
    assert not S.NONE.supports_mixed_chunk() and not S.NGRAM.supports_mixed_chunk()


# --------------------------------------------------------------------------------------------- the scheduler call edge
def _tree(rel):
    return ast.parse((SRT / rel).read_text())


def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_call_edge_collective_entry_is_replicated_and_the_verdict_is_the_group_min():
    t = _tree("managers/scheduler.py")
    raw = ast.unparse(_fn(t, "_get_new_batch_prefill_raw"))
    assert "if self._mixed_d_short:" in raw
    assert "not running_batch.is_empty() and self._mixed_d_agreed(" in raw
    ag = ast.unparse(_fn(t, "_mixed_d_agreed"))
    assert "self._weg2_group_min_flags([mine])" in ag       # R11: MIN over the TP cpu group
    assert "mds.short_gate(u, x, hx, chunked, no_fwd)" in ag
    assert "hx = _mds.batch_half_x(new_batch.reqs)" in ag     # A1: the front's halbX
    assert "no_fwd = _mds.end_state_handoff(new_batch.reqs)" in ag   # A2: never an E2 hand-off
    assert "local_ok and" in ag                              # rank-local blockers ride in the vote, not in the entry
    assert "_mixed_d_was_chunked" in ag                      # lifetime-sticky chunk stamp is read


def test_call_edge_chunk_stamp_and_arming():
    src = (SRT / "managers/scheduler.py").read_text()
    assert "self.chunked_req._mixed_d_was_chunked = True" in src
    init = src[src.index("self._mixed_d_short = bool("):][:900]
    assert "tp_prefill_max_tokens" in init                   # D group only (P has no riegel)
    assert "supports_mixed_chunk()" in init and "enable_kv_session_offload" in init
    assert "self.is_mixed_chunk = True" in src


def test_result_processor_does_not_stamp_a_decode_tail_as_a_prefill_sample():
    src = ast.unparse(_fn(_tree("managers/scheduler_components/batch_result_processor.py"), "process_batch_result_prefill"))
    assert "_is_mixed_tail = bool(batch.decoding_reqs and req in batch.decoding_reqs)" in src
    i = src.index("if not _is_mixed_tail:")
    assert "note_prefill_finished" in src[i:i + 400] and "set_prefill_finished_time" in src[i:i + 400]
    assert "req.kv_committed_len += 1" in src               # PR #36933: the tail committed its bonus token


# ------------------------------------------------------------------------------- mixed batch arithmetic + token identity
def toy_next(ctx):
    """deterministic 'greedy model': next token = f(whole context)."""
    return (sum((i + 1) * t for i, t in enumerate(ctx)) * 31 + len(ctx) * 7) % 997


class FakeReq:
    def __init__(self, rid, prompt, out):
        self.rid, self.origin_input_ids, self.output_ids = rid, list(prompt), list(out)
        self.extend_range = None
        self.full_untruncated_fill_ids = None

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def set_extend_range(self, a, b):
        self.extend_range = (a, b)


def _ns(**kw):
    base = dict(
        sampling_info=types.SimpleNamespace(merge_batch=lambda o: None),
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        multimodal_inputs=None, return_logprob=False, has_grammar=False, return_hidden_states=False,
        is_prefill_only=True, spec_info=None, input_ids=None, mamba_track_indices=None, mamba_track_mask=None,
        mamba_track_seqlens=None, weg2_turn_tracks=None, top_logprobs_nums=[], token_ids_logprobs=[],
        enable_overlap=True, spec_algorithm=SpeculativeAlgorithm.EAGLE, prefill_input_ids_cpu=None,
        mix_running_indices=None, mix_running_indices_cpu=None, device="cpu", forward_mode=None,
    )
    base.update(kw)
    ns = types.SimpleNamespace(**base)
    ns.merge_batch = types.MethodType(ScheduleBatch.merge_batch, ns)
    ns.batch_size = lambda: len(ns.reqs)
    return ns


class Rig:
    """req_to_token pool + KV (slot -> token) + a FutureMap stand-in, all CPU."""

    W = 64

    def __init__(self):
        self.req_to_token = torch.full((8, self.W), -1, dtype=torch.int64)
        self.kv = {}
        self.next_slot = 100

    def alloc(self, row, lo, hi):
        for p in range(lo, hi):
            self.req_to_token[row, p] = self.next_slot
            self.next_slot += 1

    def future_map(self, new_seq_lens, tokens):
        fm = FutureMap.__new__(FutureMap)
        fm.req_to_token = self.req_to_token
        fm.new_seq_lens_buf = torch.full((8,), -1, dtype=torch.int64)
        fm.output_tokens_buf = torch.full((8,), -1, dtype=torch.int64)
        fm.publish_ready = None
        fm.fwd_prepare_d2h_stream = None
        fm.new_seq_lens_cpu_pinned = None
        fm.device = "cpu"
        fm.spec_algo = SpeculativeAlgorithm.EAGLE
        for row, (L, tok) in {r: (new_seq_lens[r], tokens[r]) for r in new_seq_lens}.items():
            fm.new_seq_lens_buf[row] = L
            fm.output_tokens_buf[row] = tok
        return fm

    def forward(self, b):
        """toy model forward over the batch's OWN tensors. Row i: prefix_lens[i] cached positions + extend_lens[i] tokens."""
        ids = b.input_ids.tolist()
        loc = b.out_cache_loc.tolist()
        off, out = 0, []
        for i, row in enumerate(b.req_pool_indices.tolist()):
            p, e = b.prefix_lens[i], b.extend_lens[i]
            assert int(b.seq_lens[i]) == p + e, (i, int(b.seq_lens[i]), p, e)          # attention's row length
            if b.seq_lens_cpu is not None:
                assert int(b.seq_lens_cpu[i]) == p + e
            ctx = [self.kv[int(self.req_to_token[row, q])] for q in range(p)]            # the cached prefix, via the pool
            for k in range(e):
                assert int(self.req_to_token[row, p + k]) == loc[off + k], "out_cache_loc != req_to_token slot of the position"
                self.kv[loc[off + k]] = ids[off + k]
                ctx.append(ids[off + k])
            out.append(toy_next(ctx))
            off += e
        assert off == len(ids) == len(loc)
        return out


def _reference_decode(prompt, n):
    ctx, out = list(prompt), []
    for _ in range(n):
        t = toy_next(ctx)
        out.append(t)
        ctx.append(t)
    return out


@pytest.mark.parametrize("lag", [0, 2])
def test_token_identity_mixed_vs_unmixed_under_spec_v2_overlap(lag):
    """A decodes; a short prefill B arrives; A's row is folded into B's forward. A's and B's streams equal the unmixed reference.

    lag=2: the in-flight verify round accepted 2 tokens the CPU does not know yet -> req.seqlen lags the device truth and
    resolve_mixed_spec_tails must late-bind seq_lens / slot / prefix / input token (SGLang PR #36933)."""
    rig = Rig()
    pa, pb = [3, 5, 8, 13], [21, 34, 55, 89, 144, 233]
    ref_a, ref_b = _reference_decode(pa, 8), _reference_decode(pb, 3)

    # --- A: prefilled and decoded up to a committed base L (device truth); bonus token pending = ref_a[k]
    k = 3 + lag                                     # tokens of A already committed on the device (incl. pending bonus)
    ctxA = pa + ref_a[:k]                           # prompt + k tokens; the LAST one is the pending bonus (KV not written yet)
    L = len(ctxA) - 1                               # committed length: positions [0, L) have KV, bonus sits at position L
    rig.alloc(0, 0, L + 8)
    for q in range(L):
        rig.kv[int(rig.req_to_token[0, q])] = ctxA[q]
    a = FakeReq("A", pa, ref_a[: k - lag])          # CPU view lags `lag` tokens behind the device
    running = _ns(reqs=[a], req_pool_indices=torch.tensor([0]), req_pool_indices_cpu=torch.tensor([0]),
                  seq_lens=torch.tensor([len(a.origin_input_ids) + len(a.output_ids)]), orig_seq_lens=torch.tensor([0]),
                  seq_lens_cpu=None, spec_info=object())
    # --- B: short prefill, whole prompt as ONE extend (u = 6 < halbX)
    rig.alloc(1, 0, len(pb) + 8)
    b_req = FakeReq("B", pb, [])
    b_req.set_extend_range(0, len(pb))
    new = _ns(reqs=[b_req], req_pool_indices=torch.tensor([1]), req_pool_indices_cpu=torch.tensor([1]),
              seq_lens=torch.tensor([len(pb)]), orig_seq_lens=torch.tensor([len(pb)]), seq_lens_cpu=torch.tensor([len(pb)]),
              out_cache_loc=rig.req_to_token[1, :len(pb)].clone(), extend_num_tokens=len(pb), prefix_lens=[0],
              extend_lens=[len(pb)], extend_logprob_start_lens=[0], req_to_token_pool=types.SimpleNamespace(
                  req_to_token=rig.req_to_token),
              prefill_input_ids_cpu=torch.tensor(pb))
    new.spec_algorithm = SpeculativeAlgorithm.EAGLE

    ScheduleBatch.mix_with_running(new, running)
    assert new.forward_mode == ForwardMode.MIXED and running.spec_info is None
    assert new.extend_num_tokens == len(pb) + 1 and new.extend_lens == [len(pb), 1]

    fm = rig.future_map({0: L}, {0: ctxA[-1]})      # the device truth the in-flight step published
    resolve_forward_inputs(new, fm)                 # real code: late-binds the tails, gathers the bonus token
    assert new.input_ids.tolist() == pb + [ctxA[-1]]
    assert new.prefix_lens == [0, L] and int(new.seq_lens[1]) == L + 1

    out = rig.forward(new)                          # B's first token, A's next token
    assert out[0] == ref_b[0], "B's first token differs mixed vs unmixed"
    assert out[1] == ref_a[k], "A's next token differs mixed vs unmixed (the decode tail row is not a plain decode row)"


def test_non_overlap_spec_tail_uses_request_state():
    """non-overlap spec: no late binding, the tail comes straight from request state (delta -1) -- same row."""
    rig = Rig()
    pa = [3, 5, 8, 13]
    ref_a = _reference_decode(pa, 6)
    k = 3
    ctxA = pa + ref_a[:k]
    L = len(ctxA) - 1
    rig.alloc(0, 0, L + 8)
    for q in range(L):
        rig.kv[int(rig.req_to_token[0, q])] = ctxA[q]
    a = FakeReq("A", pa, ref_a[:k])
    running = _ns(reqs=[a], req_pool_indices=torch.tensor([0]), req_pool_indices_cpu=torch.tensor([0]),
                  seq_lens=torch.tensor([L]), orig_seq_lens=torch.tensor([0]), seq_lens_cpu=None, enable_overlap=False)
    rig.alloc(1, 0, 8)
    pb = [21, 34, 55]
    b_req = FakeReq("B", pb, [])
    new = _ns(reqs=[b_req], req_pool_indices=torch.tensor([1]), req_pool_indices_cpu=torch.tensor([1]),
              seq_lens=torch.tensor([3]), orig_seq_lens=torch.tensor([3]), seq_lens_cpu=torch.tensor([3]),
              out_cache_loc=rig.req_to_token[1, :3].clone(), extend_num_tokens=3, prefix_lens=[0], extend_lens=[3],
              extend_logprob_start_lens=[0], req_to_token_pool=types.SimpleNamespace(req_to_token=rig.req_to_token),
              enable_overlap=False)
    new.prefill_input_ids_cpu = torch.tensor(pb)
    ScheduleBatch.mix_with_running(new, running)
    # S33 review A3 (PR #36933 1:1): non-overlap spec never writes the relay -- the scheduler stashes the tails' pending
    # token (output_ids[-1]) before the resolve; the resolve then gathers it like the overlap path.
    fm = rig.future_map({}, {})
    fm.stash_bonus_tokens(running.req_pool_indices, torch.tensor([a.output_ids[-1]]))
    resolve_forward_inputs(new, fm)
    assert new.input_ids.tolist() == pb + [ctxA[-1]]
    assert new.prefix_lens == [0, L] and int(new.seq_lens[1]) == L + 1
    assert rig.forward(new)[1] == ref_a[k]


def test_call_edge_non_overlap_spec_stashes_the_tails_pending_token():
    src = ast.unparse(_fn(_tree("managers/scheduler.py"), "_get_new_batch_prefill_raw"))
    i = src.index("new_batch.mix_with_running(running_batch)")
    tail = src[i:i + 900]
    assert "if not self.enable_overlap and (not self.spec_algorithm.is_none()):" in tail
    assert "self.future_map.stash_bonus_tokens(running_batch.req_pool_indices, last_tokens)" in tail
    assert "[r.output_ids[-1] for r in running_batch.reqs]" in tail


def test_a_mixed_forward_is_reported_untimed_and_never_reaches_the_cost_ring():
    """S33 review A4: a MIXED forward's gpu-ms holds the decode rows; it must not price D's #new-token (X-COST-LINE)."""
    from sglang.srt.managers.scheduler_components.metrics_reporter import RankPrefillLog
    from sglang.srt.weg2 import prefill_clock as pc

    class _Timer:
        def __init__(self, log):
            self.log, self.completed = log, []

        def _report(self):
            while self.completed:
                self.log._on_duration(self.completed.pop(0))

    pc._reset_for_tests()
    log = RankPrefillLog()
    log.timer = _Timer(log)
    log.record(new_tokens=168, cached_tokens=90000, timed=True, mixed=True)      # MIXED: prefill 168 + decode rows
    log.record(new_tokens=900, cached_tokens=80000, timed=True)                  # a plain prefill
    log.timer.completed += [0.200, 0.150]
    log.flush()
    recent = pc.cost_snapshot()["recent"]
    assert [r["n"] for r in recent] == [900], recent                            # only the plain one is a cost sample
    assert abs(recent[0]["ms"] - 150.0) < 1e-6                                   # paired with ITS own duration
    assert log.cum["new_tokens"] == 168 + 900 and log.cum["chunks"] == 2
    assert not log._pending and not log._durations
    pc._reset_for_tests()


def test_mamba_track_of_the_prefill_rows_survives_the_merge_and_tails_get_no_tracking():
    rig = Rig()
    rig.alloc(0, 0, 16)
    rig.alloc(1, 0, 16)
    a = FakeReq("A", [1, 2, 3, 4], [5, 6])
    running = _ns(reqs=[a], req_pool_indices=torch.tensor([0]), req_pool_indices_cpu=torch.tensor([0]),
                  seq_lens=torch.tensor([5]), orig_seq_lens=torch.tensor([0]), seq_lens_cpu=None)
    new = _ns(reqs=[FakeReq("B", [7, 8], [])], req_pool_indices=torch.tensor([1]), req_pool_indices_cpu=torch.tensor([1]),
              seq_lens=torch.tensor([2]), orig_seq_lens=torch.tensor([2]), seq_lens_cpu=torch.tensor([2]),
              out_cache_loc=rig.req_to_token[1, :2].clone(), extend_num_tokens=2, prefix_lens=[0], extend_lens=[2],
              extend_logprob_start_lens=[0], req_to_token_pool=types.SimpleNamespace(req_to_token=rig.req_to_token),
              mamba_track_indices=torch.tensor([9]), mamba_track_mask=torch.tensor([True]),
              mamba_track_seqlens=torch.tensor([2]))
    ScheduleBatch.mix_with_running(new, running)
    assert new.mamba_track_mask.tolist() == [True, False]
    assert new.mamba_track_indices.tolist() == [9, 0] and new.mamba_track_seqlens.tolist() == [2, 0]
