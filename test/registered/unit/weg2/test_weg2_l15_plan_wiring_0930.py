# SPDX-License-Identifier: Apache-2.0
"""L15-01b tests: wire the accepted pure L1.5 planner (weg2/l15_plan.py)
into the launcher.

Covers:
  (a) budgets_from_dc(..., l15_mib=None) is byte-identical to the legacy call
      (same budgets, same log lines, no ``l15`` term) -- backward compat.
  (b) budgets_from_dc(..., l15_mib=[...]) deducts the L1.5 post from each
      card's budget BEFORE the // 8 * 8 rounding, in BOTH budget formulas
      (the non-booked and the booked-rest branches), tags the terms dict with
      an ``l15`` key, and appends the ``- l15 {mib} (L1.5 post)`` suffix to
      the budget log line.
  (c) resolve_dual_layout refuses a --dual-layout launch (W-L15-DUAL) when
      the L1.5 master switch is on, and does NOT refuse when it is off.

Plain pytest functions on purpose: CustomTestCase wraps tests in retry() and
would surface a deterministic assertion failure only as
``retry() exceed maximum number of retries`` (the real cause hides in
"Captured stderr call"). Plain functions surface the real assertion directly.

Hermetic: no GPU, no model, no record I/O. PYTHONPATH must include
/spinning/wt-l15-0930/python.
"""

import argparse

import pytest

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import l15_plan

# Three-card fixture copied from test_weg2_budget_rest_no_reserve_0929.py
# (U5090 / U0 / U2, L.Card, L.order_cards). order_cards puts the 5090 first,
# so the card order after ordering is [5090(nvml1), 3080(nvml0), 3080(nvml2)].
U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U0, name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid=U5090, name="NVIDIA GeForce RTX 5090",
               total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid=U2, name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
    ])


def _dc():
    # D dormant residue per card UUID (the P pass's "other group" residue).
    return {U0: 588, U5090: 1104, U2: 814}


def _reserve_zero(cards):
    return {c.uuid: 0 for c in cards}


def _budget_line(lines, ordinal):
    hits = [ln for ln in lines if f"budget P group=P ordinal={ordinal} " in ln]
    assert len(hits) == 1, (
        f"expected one budget line for ordinal {ordinal}, got {len(hits)}")
    return hits[0]


def _unrounded(terms):
    """The pre-//8*8 budget reconstructed from the terms the launcher
    recorded itself.  Both budget formulas subtract exactly these five
    terms (plus l15 once the post is wired):
      non-booked:  total - (floor + awake_builtin) - dc - grow - over - carve
      booked-rest: total - carve - dc - grow - rest
    and the term dict stores floor, dormant(=dc+grow) and awake(= the rest,
    or the builtin+measured overshoot), so one reconstruction covers both.
    """
    return (terms["total"] - terms["carve"] - terms["floor"]
            - terms["dormant"] - terms["awake"] - terms.get("l15", 0))


def test_l15_mib_none_is_byte_identical():
    """(a) l15_mib=None (the default) must produce byte-identical budgets
    and log lines to the legacy call without the keyword at all."""
    cards = _cards()
    dc = _dc()
    reserve = _reserve_zero(cards)
    base_lines = []
    base_terms = []
    base = L.budgets_from_dc(cards, dc, base_lines.append, "P",
                             overshoot_mib=[400, 100, 100],
                             overshoot_provenance="t",
                             user_reserve_by_card=reserve,
                             terms_out=base_terms)
    none_lines = []
    none_terms = []
    none = L.budgets_from_dc(cards, dc, none_lines.append, "P",
                             overshoot_mib=[400, 100, 100],
                             overshoot_provenance="t",
                             user_reserve_by_card=reserve,
                             l15_mib=None, terms_out=none_terms)
    assert none == base
    assert none_lines == base_lines, "l15_mib=None must not change any log line"
    assert none_terms == base_terms
    for t in none_terms:
        assert "l15" not in t, f"l15 term leaked with l15_mib=None: {t}"


def _assert_deduction(cards, dc, l15_vec, **extra):
    """(b) per card: new budget == ((old unrounded - l15) // 8) * 8, terms
    carry l15 and no other term drifts, and the budget line carries the
    suffix."""
    extra = dict(extra)
    extra.setdefault("user_reserve_by_card", _reserve_zero(cards))
    discard = []
    old_terms = []
    old_lines = []
    old = L.budgets_from_dc(cards, dc, old_lines.append, "P",
                            terms_out=old_terms, **extra)
    new_lines = []
    new_terms = []
    new = L.budgets_from_dc(cards, dc, new_lines.append, "P",
                           terms_out=new_terms, l15_mib=l15_vec, **extra)
    assert len(new) == len(cards)
    assert len(new_terms) == len(cards)
    for i in range(len(cards)):
        l15_i = int(l15_vec[i])
        expected = ((_unrounded(old_terms[i]) - l15_i) // 8) * 8
        assert new[i] == expected, (
            f"ordinal {i}: budget {new[i]} != "
            f"floor(({_unrounded(old_terms[i])} - {l15_i}) / 8) * 8 "
            f"= {expected} (legacy no-l15 was {old[i]})")
        assert new_terms[i].get("l15") == l15_i, (
            f"ordinal {i}: terms_out l15 key {new_terms[i].get('l15')!r} "
            f"!= {l15_i}")
        assert ({k: v for k, v in new_terms[i].items() if k != "l15"}
                == old_terms[i]), (
            f"ordinal {i}: a term other than l15 drifted: {new_terms[i]}")
        line = _budget_line(new_lines, i)
        assert f"- l15 {l15_i} (L1.5 post)" in line, line
    for i, ln in enumerate(old_lines):
        assert " - l15 " not in ln, f"legacy line {i} carries an l15 suffix"


def test_l15_deducts_before_rounding_nonbooked_formula():
    """(b, formula 2) the non-booked path: total - corridor - dc - grow
    - over - carve - awake, floored to 8."""
    _assert_deduction(_cards(), _dc(),
                      [0, 1000, 500],
                      overshoot_mib=[400, 200, 300],
                      overshoot_provenance="t")


def test_l15_deducts_before_rounding_booked_rest_formula():
    """(b, formula 1) the booked-rest path: total - carve - dc - grow
    - rest, floored to 8."""
    _assert_deduction(_cards(), _dc(),
                      [0, 1000, 500],
                      overshoot_mib=[400, 200, 300],
                      overshoot_provenance="t",
                      booked_rest_mib=[800, 1200, 900],
                      booked_rest_provenance="t")


def test_l15_partial_vector_raises():
    """Fail fast on a length mismatch: a partial vector never prices a card
    it never saw."""
    cards = _cards()
    discard = []
    with pytest.raises(L.Weg2LaunchRefused, match="partial vector"):
        L.budgets_from_dc(cards, _dc(), discard.append, "P",
                          l15_mib=[128])


def _dual_ns():
    return argparse.Namespace(dual_layout=True, dual_share=False,
                              dual_mps="off", dual_unified_kv="off",
                              flip_weights="family")


def test_resolve_dual_layout_refuses_l15_master(monkeypatch):
    """(c, on) SGLANG_WEG2_L15=1 + --dual-layout refuses by name."""
    monkeypatch.delenv(l15_plan.HOT_HANDOVER_ENV, raising=False)
    monkeypatch.setenv(l15_plan.L15_MASTER_ENV, "1")
    with pytest.raises(L.Weg2LaunchRefused) as exc:
        L.resolve_dual_layout(_dual_ns())
    msg = str(exc.value)
    assert msg.startswith(l15_plan.DUAL_REFUSAL_CODE), msg
    assert l15_plan.L15_MASTER_ENV in msg


def test_resolve_dual_layout_refuses_hot_handover_alone(monkeypatch):
    """(c, on) the two switches are coupled at this gate: hot handover
    alone also refuses --dual-layout."""
    monkeypatch.delenv(l15_plan.L15_MASTER_ENV, raising=False)
    monkeypatch.setenv(l15_plan.HOT_HANDOVER_ENV, "1")
    with pytest.raises(L.Weg2LaunchRefused) as exc:
        L.resolve_dual_layout(_dual_ns())
    msg = str(exc.value)
    assert msg.startswith(l15_plan.DUAL_REFUSAL_CODE), msg
    assert l15_plan.HOT_HANDOVER_ENV in msg


def test_resolve_dual_layout_allows_when_both_off(monkeypatch):
    """(c, off) no refusal when neither switch is on; the resolver still
    applies its normal --dual-layout side effects."""
    monkeypatch.delenv(l15_plan.L15_MASTER_ENV, raising=False)
    monkeypatch.delenv(l15_plan.HOT_HANDOVER_ENV, raising=False)
    ns = _dual_ns()
    L.resolve_dual_layout(ns)  # must not raise
    assert ns.flip_weights == "resident"
