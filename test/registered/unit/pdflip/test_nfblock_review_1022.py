"""NFBLOCK 1022 review: the danger directions of the D-EIGENTUM fixes as BEHAVIOUR, not as source strings.

``test_nfblock_1022`` pins most of the new decisions by ``inspect.getsource`` (a string in the source passes whatever the code does).
These tests run ``plan_d_residency`` on the toy NF checkpoint of ``test_pdflip_owned_planner_base_1001`` (48 FA layers, no layer_types:
no cut form carries there) and ``checkpoint_weight_terms`` on a real header file:

* Form A kept is not a refusal -- but only for ``owned``; a forced cut (``owned:<vector>``, the measurement arm M1b) still refuses.
* Form A kept does not swallow Form A's OWN refusals: a budget Form A does not fit is still refused (W122), only the ownership refusal goes.
* Form A kept publishes nothing a later reader would take as a cut: no ownership vector, no token cut, no solved FR_D.
* The checkpoint memo reads again when a shard changes WITHOUT a size change (mtime), and when the path is another checkpoint.

CPU only, no GPU, no launcher run.
"""

from __future__ import annotations

import json
import os
import struct
import time
import types

import pytest

from flliper.srt.planner import expert_residency as er
from flliper.srt.planner import pp_cut as P
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
POOLED = (0.201074, 0.314024)


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320, "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(P, "checkpoint_weight_terms",
                        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0, num_experts=512, n_layers=48))
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_BASE", "stated")
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_BS_WEIGHTS", "bs1")
    return str(tmp_path / MODEL)


def _plan(ckpt, cut, budgets=(26824, 17640, 17520)):
    return er.plan_d_residency(
        model_path=ckpt, budgets_mib=list(budgets), ratios=[183.0, 137.0, 168.0],
        fractions=[0.06, 0.51, 0.48], scratch_rows=[104, 48, 48], rank_tp_ratio="1,0,0",
        env_d={"FLLIPER_UNEVEN_MOE_EXPERT_SHARD": "1", "FLLIPER_PDFLIP_DENSE_REPACK_OUTSIDE_POOL": "1"},
        reference_logs="", kv_tokens=262144, label="T", marker="T",
        kv_token_shares=cut, kv_dcp_cell_bytes=12288, owned_miss_ms=POOLED, owned_miss_source="test")


def test_owned_without_a_carrying_cut_keeps_form_a_and_publishes_no_cut(ckpt):
    plan = _plan(ckpt, er.KV_TOKEN_CUT_OWNED)
    rec = dict(plan.owner_record)
    assert rec["feasible"] == 0                                    # the toy carries no cut form
    assert rec.get("form_a_kept") is True
    assert "D-EIGENTUM (#239 S3f)" not in (plan.refusal or "")
    assert any("FORM A BLEIBT (nfblock 1022)" in ln for ln in plan.lines)
    # nothing a later reader takes for a solved cut form
    assert not plan.solved_owner_ratio and not plan.kv_token_cut and not plan.solved_fractions
    # Form A is the uncut D: every rank priced with the full context on the host
    assert all(float(f.kv_token_share) == -1.0 for f in plan.fits)            # -1 = no token cut on this rank


def test_a_forced_cut_still_refuses(ckpt):
    # the forced cut onto workers too small to hold it: no form carries (the kept refusal is the point)
    plan = _plan(ckpt, "owned:0,48,16", budgets=(26824, 8000, 8000))
    rec = dict(plan.owner_record)
    assert rec["feasible"] == 0
    assert "form_a_kept" not in rec
    assert plan.refusal is not None and "D-EIGENTUM (#239 S3f)" in plan.refusal
    assert not any("FORM A BLEIBT" in ln for ln in plan.lines)


def test_form_a_kept_does_not_swallow_form_a_own_refusal(ckpt):
    # a budget Form A cannot carry on the host: the ownership refusal goes, the budget refusal of Form A stays
    plan = _plan(ckpt, er.KV_TOKEN_CUT_OWNED, budgets=(9000, 17640, 17520))
    assert dict(plan.owner_record).get("form_a_kept") is True
    assert plan.refusal is not None
    assert "D-EIGENTUM (#239 S3f)" not in plan.refusal


def _write_shard(path, tensors):
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in tensors.items()}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)


def test_memo_reads_again_after_a_same_size_rewrite(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    shard = str(d / "model-00001.safetensors")
    # same byte length, different meaning: layer 1 is attention in the first, linear in the second
    a_t = {"model.layers.0.self_attn.q_proj.weight": [8, 8], "model.layers.1.self_attn.q_proj.weight": [8, 8]}
    b_t = {"model.layers.0.self_attn.q_proj.weight": [8, 8], "model.layers.1.xelf_attn.q_proj.weight": [8, 8]}
    _write_shard(shard, a_t)
    size_a = os.path.getsize(shard)
    first = P.checkpoint_weight_terms(str(d))
    assert first.attention_layer_indices == (0, 1)
    st = os.stat(shard)
    _write_shard(shard, b_t)
    assert os.path.getsize(shard) == size_a
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a real rewrite moves mtime; force it past the fs granularity
    second = P.checkpoint_weight_terms(str(d))
    assert second.attention_layer_indices == (0,)


def test_memo_is_per_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_shard(str(a / "m.safetensors"), {"model.layers.0.self_attn.q.weight": [8, 8]})
    _write_shard(str(b / "m.safetensors"), {"model.layers.0.self_attn.q.weight": [8, 8], "model.layers.1.self_attn.q.weight": [8, 8]})
    t = time.time_ns()
    for p in (a / "m.safetensors", b / "m.safetensors"):
        os.utime(str(p), ns=(t, t))
    assert P.checkpoint_weight_terms(str(a)).n_layers == 1
    assert P.checkpoint_weight_terms(str(b)).n_layers == 2


# ---- the rig (3 ranks) keeps the exact ownership search: OWNED_CLASS_MIN_RANKS 4 -> 3 must not reach it ----------------------------
# a 3-rank board whose two workers are IDENTICAL (one class): the class search would be a different (smaller) lattice, so the candidate
# count of the default equals the exact search's only while 3 ranks stay below the class threshold

_CAP3 = (200, 150, 150)
_SCR3 = (51, 40, 40)


def _fit3(rank, ratio, cut):
    e = int(ratio) + 10
    kv = (53 if rank == 0 else 0) if cut is None else 53 * cut[rank] / float(sum(cut))
    ceiling = int(_CAP3[rank] - kv)
    rows = min(ceiling - _SCR3[rank], e - 2)
    return types.SimpleNamespace(rank=rank, local_experts=e, ceiling_max_rows=ceiling, scratch_rows=_SCR3[rank],
                                 ceiling_fraction=(rows / float(e)) if rows >= 1 else None,
                                 budget_mib=17000.0 if rank else 26000.0, fixed_mib=100.0, slot_mib=27.0, draft_vocab_delta_mib=0.0)


def _solve3(rat, sh):
    return tuple(_fit3(r, rat[r], sh) for r in range(3))


def test_three_ranks_run_the_exact_owned_search(monkeypatch):
    kw = dict(num_experts=512, n_layers=48, ids_per_step=40, x1_scope="round")
    default = er.solve_owned_cut(_solve3, (180, 166, 166), 0, **kw)
    monkeypatch.setattr(er, "OWNED_CLASS_MIN_RANKS", 99)
    exact = er.solve_owned_cut(_solve3, (180, 166, 166), 0, **kw)
    assert default.candidates == exact.candidates
    assert (default.ratios, default.cut) == (exact.ratios, exact.cut)


def test_three_ranks_derive_the_base_exactly(monkeypatch):
    kw = dict(num_experts=512, n_layers=48, ids_per_step=40)
    default = er.derive_owned_base(_solve3, (180, 166, 166), 0, **kw)
    monkeypatch.setattr(er, "OWNED_CLASS_MIN_RANKS", 99)
    exact = er.derive_owned_base(_solve3, (180, 166, 166), 0, **kw)
    assert (default.ratios, default.candidates) == (exact.ratios, exact.candidates)


def test_memo_key_of_a_file_follows_its_mtime_and_its_sibling_config(tmp_path):
    # a GGUF launch names the FILE; the census reads the sibling config.json beside it -- both are in the key
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 60)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"num_hidden_layers": 4}))
    k0 = P._weight_terms_key(str(f))
    st = os.stat(str(f))
    os.utime(str(f), ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    k1 = P._weight_terms_key(str(f))
    assert k1 != k0                                                    # same size, new mtime: another key
    cs = os.stat(str(cfg))
    cfg.write_text(json.dumps({"num_hidden_layers": 5}))               # same length, other depth
    os.utime(str(cfg), ns=(cs.st_atime_ns, cs.st_mtime_ns + 1_000_000))
    assert P._weight_terms_key(str(f)) != k1
