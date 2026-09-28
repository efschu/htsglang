"""#239: under the token cut the H33 card credits the attention host with the
full-attention KV it gave away (Nutzer 28.09.: "gibt die 5090 ihre KV ab,
bekommt sie Platz fuer Experten").

The card reference (fnFL2x151 + x158, Form A) measured its headroom WITH the
host's 262144 x 14143 B KV in it. Before this fix the card never moved under
the cut: the budget path booked the host's KV at 464 MiB, the card still at
3536, so the 5090 got no row (M1a dry run 18:25Z: 136 rows at n=1 in both
forms, FR_D 0.186 -> 0.088, ZIELFORM +8.12 ms -- a booking error, not
physics). Now every card moves by exactly (reference KV - this boot's KV):
the host gains what it gave away, the workers carry what they took, and the
non-FA part of the host cell (QSA keys, 14143 - 12288 B) stays where it is.
Form A is byte-identical.
"""

from __future__ import annotations

import json
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
BUDGETS = (29368, 18184, 17784)  # M1a dry run, D(d-only, expectation)
KV = 262144
DCP_CELL = 12288  # 12 FA layers x 2 x 512 B (fp8), the part the cut moves
MIB = float(1 << 20)


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    from sglang.srt.planner import pp_cut

    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    return str(tmp_path / MODEL)


def _plan(ckpt, shares=None, kv=KV):
    kw = {} if shares is None else dict(kv_token_shares=shares, kv_dcp_cell_bytes=DCP_CELL)
    return er.plan_d_residency(
        model_path=ckpt, budgets_mib=list(BUDGETS), ratios=[215.0, 113.0, 160.0],
        fractions=[0.06, 0.51, 0.48], scratch_rows=[100, 48, 48], rank_tp_ratio="1,0,0",
        env_d={"SGLANG_UNEVEN_MOE_EXPERT_SHARD": "1",
               "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1"},
        reference_logs="", kv_tokens=kv, label="T", marker="T", **kw)


def test_the_5090_gains_rows_when_it_gives_its_kv_away(ckpt):
    """RED before the fix: host share 0, yet the 5090's card ceiling stayed
    at Form A's (the card never saw the KV leave)."""
    form_a = _plan(ckpt)
    cut = _plan(ckpt, (0, 48, 16))
    assert form_a.card_fits and cut.card_fits
    a0, c0 = form_a.card_fits[0], cut.card_fits[0]
    assert c0.ceiling_max_rows > a0.ceiling_max_rows
    # exactly the FA KV the host no longer holds: 262144 x 12288 B = 3072 MiB
    gained = (c0.ceiling_max_rows - a0.ceiling_max_rows) * c0.layer_row_mib
    assert gained == pytest.approx(KV * DCP_CELL / MIB, abs=c0.layer_row_mib)


def test_the_shift_is_reference_kv_minus_this_boots_kv_per_rank():
    ref = er.D_CARD_REFERENCE_FNFL2_H39
    cells = er.kv_token_cut_cells(er.D_RESIDENCY_REFERENCE_FNFL2_H39, (0, 48, 16), DCP_CELL)
    fits = [types.SimpleNamespace(rank=r, kv_tokens=KV, kv_cell_bytes=int(round(c)))
            for r, c in enumerate(cells)]
    shift, line = er.card_kv_cut_shift(ref, fits)
    assert shift == pytest.approx((3072.0, -2304.0, -768.0), abs=0.1)
    assert "rang0 262144 x 14143 B = 3536 -> 262144 x 1855 B = 464 MiB (+3072)" in line
    # the non-FA part of the host cell (QSA keys, draft) stays on the host
    assert int(round(cells[0])) == 14143 - DCP_CELL


def test_the_card_line_names_the_credit(ckpt):
    lines = _plan(ckpt, (0, 48, 16)).lines
    kv = [ln for ln in lines if "KV-SCHNITT (#239)" in ln]
    assert kv and "(+3072)" in kv[0] and "GERECHNET" in kv[0]


def test_form_a_is_byte_identical(ckpt):
    """No cut: no shift, no line, the same card as before the fix; a vector
    that leaves the whole FA KV on the host moves nothing either."""
    plan = _plan(ckpt)
    assert not any("KV-SCHNITT" in ln for ln in plan.lines)
    ref = er.D_CARD_REFERENCE_FNFL2_H39
    fits = [types.SimpleNamespace(rank=r, kv_tokens=KV, kv_cell_bytes=c)
            for r, c in enumerate(ref.kv_cell_bytes)]
    shift, _ = er.card_kv_cut_shift(ref, fits)
    assert shift == (0.0, 0.0, 0.0)
    host_only = _plan(ckpt, (64, 0, 0))
    assert [c.ceiling_max_rows for c in host_only.card_fits] == [
        c.ceiling_max_rows for c in plan.card_fits]


def test_a_reference_without_kv_geometry_names_itself():
    ref = er.DCardReference(
        source="x", model=MODEL, rank_tp_ratio="1,0,0", headroom0_mib=(1.0,),
        free_decode0_mib=(None,), buffer_rows=(1,), cap_mib=(1.0,), peak_mib=(1.0,),
        private_free_mib=(0.0,), phase=("decode",), precision_mib=(0.0,),
        draft_host_rank=0, draft_vocab_held=False)
    shift, line = er.card_kv_cut_shift(
        ref, [types.SimpleNamespace(rank=0, kv_tokens=KV, kv_cell_bytes=1855)])
    assert shift is None and "ENTFAELLT" in line and "KEINE Gutschrift" in line


def test_the_shipped_references_carry_the_kv_they_measured():
    for ref in er.D_CARD_REFERENCES:
        assert ref.kv_tokens == 262144
        assert ref.kv_cell_bytes == (14143, 768, 768)


def test_the_reference_from_logs_reads_its_kv_geometry():
    boots = []
    for name in ("a", "b"):
        boots.append((name, "\n".join([
            "[2026-09-24 14:07:01 TP%d] KV pool sizing: available_bytes=1 (0 GiB), "
            "cell_size=%d, page_size=64 -> max_total_num_tokens=1" % (r, c)
            for r, c in enumerate((14143, 768, 768))] + [
            "[2026-09-24 14:07:01 TP%d] KV Cache is allocated. dtype: torch.float8_e4m3fn, "
            "#tokens: 262144, K size: 0.00 GB, V size: 0.00 GB" % r for r in range(3)])))
    geoms = set()
    for _n, text in boots:
        obs = er._observe_boot(text, n_layers=48)
        assert obs["kv_tokens"] == {0: 262144.0, 1: 262144.0, 2: 262144.0}
        geoms.add((262144, tuple(int(obs["cell"][r]) for r in range(3))))
    assert er._card_kv_geometry(geoms) == {"kv_tokens": 262144,
                                           "kv_cell_bytes": (14143, 768, 768)}
    assert er._card_kv_geometry({(262144, (14143, 768, 768)), (131072, (14143, 768, 768))}) == {}
