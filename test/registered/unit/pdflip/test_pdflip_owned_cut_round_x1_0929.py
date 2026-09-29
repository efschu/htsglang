"""#239 S3f round guard (29.09., z30n/z30r3 gegen z30k): the x1 rule keeps
every WORKER at or below its Form-A miss time and lets the HOST take the rest.
On metal the host then became the rank the synchronized round waits for:
TP0 misses per seat +18..23 %, decode round bs1 +2.8..5.6 ms against Form A;
the planner's own line said so beforehand (ZIELFORM max T_r 38.45 > Form A
36.11 on the M1s dry run).

``x1_scope="round"`` (``FLLIPER_PDFLIP_OWNED_CUT_X1=round``) reports x1 but does
not enforce it -- the synchronized round (max T_r over ALL ranks) decides.
Default stays ``workers`` (byte-identical form).
"""

from __future__ import annotations

import os
import types

import pytest

from flliper.srt.planner import expert_residency as er
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

BASE = (183, 137, 168)
NUM_EXPERTS = 512
N_LAYERS = 48
IDS = 40
FA_LAYERS = 12


def _model(scratch, rows_form_a, rows_cut_at0, rows_per_share):
    def solve_at(rat, sh):
        out = []
        for r in range(3):
            E = int(round(NUM_EXPERTS * rat[r] / float(sum(BASE)))) + 1
            rows = rows_form_a[r] if sh is None else int(rows_cut_at0[r] - rows_per_share[r] * sh[r])
            S = scratch[r]
            held = min(rows - S, E - 2)
            out.append(types.SimpleNamespace(
                rank=r, local_experts=E, scratch_rows=S, ceiling_max_rows=rows,
                ceiling_fraction=(held / float(E)) if held >= 1 else None))
        return tuple(out)

    return solve_at


# the serving profile nf-h91-dpr-sa-vis-noadopt-st-cut (same model as
# test_pdflip_d_owner_free_search_239): the x1 solve ends above Form A
SERVING = _model(scratch=(100, 48, 48), rows_form_a=(95, 127, 127),
                 rows_cut_at0=(122, 127.9, 127.6), rows_per_share=(0.0, 0.414, 0.414))


def _solve(scope):
    return er.solve_owned_cut(SERVING, BASE, 0, num_experts=NUM_EXPERTS, n_layers=N_LAYERS,
                              ids_per_step=IDS, fa_layers=FA_LAYERS, x1_scope=scope)


def test_workers_scope_is_the_old_x1_rule():
    sol = _solve("workers")
    assert sol.feasible > 0 and sol.x1_ok
    workers = (1, 2)
    assert all(sol.round_ms[w] <= sol.base_round_ms[w] + 1e-9 for w in workers)


def test_round_scope_never_ends_above_the_x1_form():
    x1 = _solve("workers")
    rnd = _solve("round")
    assert rnd.feasible >= x1.feasible
    assert max(rnd.round_ms) <= max(x1.round_ms) + 1e-9


def test_round_scope_ranks_by_the_synchronized_round():
    """the metal case: the x1 form is above Form A on the host; the round
    form must not be, when the grid holds one that is not"""
    x1 = _solve("workers")
    rnd = _solve("round")
    if max(x1.round_ms) > max(x1.base_round_ms):
        assert max(rnd.round_ms) < max(x1.round_ms)


def test_env_default_and_refusal(monkeypatch):
    monkeypatch.delenv(er.OWNED_X1_ENV, raising=False)
    assert er.owned_x1_scope() == "workers"
    monkeypatch.setenv(er.OWNED_X1_ENV, "round")
    assert er.owned_x1_scope() == "round"
    monkeypatch.setenv(er.OWNED_X1_ENV, "bogus")
    with pytest.raises(ValueError):
        er.owned_x1_scope()
