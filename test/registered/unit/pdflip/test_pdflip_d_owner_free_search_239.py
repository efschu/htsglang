"""#239 S3f (main 28.09. ~19:00Z): the owned solve searches the WHOLE
ownership space -- the host may give ownership, the workers may trade among
themselves, the cut stays free on its 64ths grid.

The host-only space (every worker gives to the host) was a search bound, not
physics: with a worker that has room (serving profile -st-cut: TP1 8.82 ms
against 12.28 in Form A) and one at its x1 edge (TP2 35.47 against 36.11),
the best form may move experts TP2 -> TP1 or keep them off the host. The x1
rule and the objective (min max T_r) stay. These tests hold the solver to the
brute-force optimum of the free grid, so a smaller search can never pass.
"""

from __future__ import annotations

import inspect
import itertools
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
    """A per-rank edge model: Form A rows, and under the cut rows(share) =
    rows_cut_at0 - rows_per_share x share (the FA KV the rank holds)."""

    def solve_at(rat, sh):
        out = []
        for r in range(3):
            E = int(round(NUM_EXPERTS * rat[r] / float(sum(BASE)))) + 1
            if sh is None:
                rows = rows_form_a[r]
            else:
                rows = int(rows_cut_at0[r] - rows_per_share[r] * sh[r])
            S = scratch[r]
            held = min(rows - S, E - 2)
            out.append(types.SimpleNamespace(
                rank=r, local_experts=E, scratch_rows=S, ceiling_max_rows=rows,
                ceiling_fraction=(held / float(E)) if held >= 1 else None))
        return tuple(out)

    return solve_at


# the serving profile nf-h91-dpr-sa-vis-noadopt-st-cut on 85386b1df1
# (dry_stcut_z28.log, rc12c card ledger): TP0 122 rows with the host's KV at
# 1855 B/token (95 in Form A, 3536 MiB more KV); TP1 108 rows at 48/64, TP2
# 121 at 16/64, 0.414 rows per 64th (262144 x 12288 B / 64 / 116 MiB a row)
SERVING = _model(scratch=(100, 48, 48), rows_form_a=(95, 127, 127),
                 rows_cut_at0=(122, 127.9, 127.6), rows_per_share=(0.0, 0.414, 0.414))


def _free_vectors(base, step, cap):
    """The whole ownership grid (independent of the solver's generator)."""
    total = sum(base)
    rng = [range(b - (cap // step) * step, b + cap + 1, step) for b in base]
    for a, b in itertools.product(rng[0], rng[1]):
        c = total - a - b
        if min(a, b, c) >= 1 and (c - base[2]) % step == 0 and abs(c - base[2]) <= cap:
            yield (a, b, c)


def _brute(solve_at, host=0):
    kw = dict(host=host, num_experts=NUM_EXPERTS, ids_per_step=IDS, n_layers=N_LAYERS,
              fa_layers=FA_LAYERS)
    base_ms = er.owned_round_ms(solve_at(BASE, None), shares=(1, 0, 0), merged=False, **kw)
    cap = (sum(BASE) - max(BASE)) // 2
    best = None
    for rat in _free_vectors(BASE, er.OWNED_RATIO_STEP, cap):
        for a in range(0, 65, er.OWNED_SHARE_STEP):
            sh = (0, a, 64 - a)
            fits = solve_at(rat, sh)
            if any(f.ceiling_fraction is None or f.ceiling_max_rows < f.scratch_rows + 2
                   for f in fits):
                continue
            ms = er.owned_round_ms(fits, shares=sh, merged=True, **kw)
            if any(ms[w] > base_ms[w] + 1e-9 for w in (1, 2)):
                continue
            if best is None or max(ms) < best[0] - 1e-9:
                best = (max(ms), rat, sh)
    return best, base_ms


def _solve(solve_at):
    return er.solve_owned_cut(solve_at, BASE, 0, num_experts=NUM_EXPERTS, n_layers=N_LAYERS,
                              ids_per_step=IDS, fa_layers=FA_LAYERS)


def test_the_solver_reaches_the_optimum_of_the_free_grid_on_the_serving_form():
    best, base_ms = _brute(SERVING)
    sol = _solve(SERVING)
    assert best is not None and sol.feasible > 0
    assert max(sol.round_ms) == pytest.approx(best[0], abs=1e-6)
    # min max T_r reaches Form A where the grid has such a form, else the
    # plan names the best form and why (ZIELFORM line, below)
    if best[0] <= max(base_ms) + 1e-9:
        assert max(sol.round_ms) <= max(sol.base_round_ms) + 1e-9


def test_the_host_gives_and_the_workers_trade_when_that_is_the_optimum():
    """TP1 has room under the cut (its Form A edge was bound elsewhere), the
    host little: the best form takes ownership OFF the host and moves TP2 ->
    TP1 -- outside the host-only space, whose best form is ~9 ms worse
    (RED before: the solver never looked there)."""
    solve_at = _model(scratch=(100, 48, 48), rows_form_a=(96, 100, 128),
                      rows_cut_at0=(110, 160, 128.0), rows_per_share=(0.0, 0.2, 0.414))
    best, base_ms = _brute(solve_at)
    sol = _solve(solve_at)
    assert best is not None
    assert max(sol.round_ms) == pytest.approx(best[0], abs=1e-6)
    assert sol.ratios[0] < BASE[0] and sol.ratios[1] > BASE[1] and sol.ratios[2] < BASE[2]
    assert max(sol.round_ms) < max(base_ms)


def test_the_host_may_give_ownership():
    vecs = er.owned_ratio_vectors_free(BASE, step=8, max_shift=16)
    assert vecs[0] == BASE
    assert all(sum(v) == sum(BASE) and min(v) >= 1 for v in vecs)
    assert (175, 145, 168) in vecs          # host gives to TP1
    assert (183, 145, 160) in vecs          # TP2 -> TP1, host untouched
    assert (199, 129, 160) in vecs          # the old host-only moves stay
    assert set(er.owned_ratio_vectors(BASE, 0, step=8, max_shift=16)) <= set(vecs)


def test_the_line_names_candidates_solves_and_runtime():
    src = inspect.getsource(er.plan_d_residency)
    assert "Budget-Loesungen in %.1f s" in src
    assert "keine haelt x1 und Form A zugleich" in src
    assert '("solves", sol.solves)' in src and '("elapsed_s", sol.elapsed_s)' in src
    sol = _solve(SERVING)
    assert sol.solves > 0 and sol.elapsed_s >= 0.0
    assert sol.candidates == len(er.owned_ratio_vectors_free(
        BASE, step=er.OWNED_RATIO_STEP)) * (64 // er.OWNED_SHARE_STEP + 1)
