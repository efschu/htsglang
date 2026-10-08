"""#287 (30.09.): the owned solve's tie among the WORKERS is decided by the
next-worst rank, not by the sum.

Metal (NF, same replayed load, median prompt 17249 tokens, 48 twin pairs):
y3j 09291933 (cut [0,36,28], TP2 FR_D 0.393, 62 D-resident experts) decoded
bs3 at ~42 ms a round; 09300002 (cut [0,28,36], TP2 FR_D 0.256, 40 residents)
at ~55-61 ms. TP0 missed the same (5.4-7.3 per layer and forward at bs2/3 in
both), TP2 2.5x more (1.1 -> 2.2-3.8): from bs2 on the 3080 worker with the
fewest residents binds the synchronized round. bs1 did not move (the host
binds there, 35.92 ms in the bs1 model of every boot of the series).

Why the solver flipped: the host binds max T_r, so every worker split ties
on the primary key; the old tie-break was the SUM of T_r, and the sum does
not depend on the split between two workers with the same row cost -- the KV
rows one worker gives up are the rows the other takes (y3j [7.35, 26.11] and
korr 09292034 [5.17, 28.29] both sum to 33.46; 09300002 [5.17, 30.46] against
[7.35, 28.28] for [0,36,28], both 35.63). The flip was rounding noise and the
lexical order of the share vector, which puts (0, 28, 36) before (0, 36, 28)
-- i.e. the KV onto TP2 up to its x1 edge. Leximax (T_r sorted descending,
compared lexicographically) keeps the bs1 optimum and picks, among the tied
forms, the one whose worst worker carries the least miss time -- the most
residents on the rank that binds from bs2 on.
"""

from __future__ import annotations

import types

import pytest

from flliper.srt.planner import expert_residency as er
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

BASE = (215, 121, 152)
NUM_EXPERTS = 512
N_LAYERS = 48
IDS = 40
FA_LAYERS = 12


def _model(rows_form_a, rows_cut_at0, rows_per_share, scratch=(100, 48, 48)):
    """Only the stated ownership is buildable (every other vector has no edge),
    so the search is the cut alone. Worker rows under the cut fall by
    ``rows_per_share`` per 64th of FA-KV they hold -- the same on both 3080s,
    as on the rig (one cell size, one card type)."""

    def solve_at(rat, sh):
        out = []
        for r in range(3):
            E = int(round(NUM_EXPERTS * rat[r] / float(sum(BASE)))) + 1
            rows = rows_form_a[r] if sh is None else rows_cut_at0[r] - rows_per_share[r] * sh[r]
            rows = int(rows) if tuple(rat) == BASE else 0  # no other ownership has an edge
            S = scratch[r]
            held = min(rows - S, E - 2)
            frac = (held / float(E)) if held >= 1 else None
            out.append(types.SimpleNamespace(
                rank=r, local_experts=E, scratch_rows=S, ceiling_max_rows=rows,
                ceiling_fraction=frac))
        return tuple(out)

    return solve_at


# host binds (few rows); both workers lose exactly one row per 64th step of 4
# (rows_per_share 0.25) -> the worker sum of T_r is EXACTLY the same for every
# split with KV on both workers, and x1 (Form A) leaves TP1 room for 36/64 at
# most and TP2 for 44/64: feasible cuts (0,20,44) .. (0,36,28).
METAL_SHAPE = _model(rows_form_a=(110, 107, 123), rows_cut_at0=(110, 118, 136),
                     rows_per_share=(0.0, 0.25, 0.25))


def _solve(solve_at):
    return er.solve_owned_cut(solve_at, BASE, 0, num_experts=NUM_EXPERTS, n_layers=N_LAYERS,
                              ids_per_step=IDS, fa_layers=FA_LAYERS)


def _all_feasible(solve_at):
    kw = dict(host=0, num_experts=NUM_EXPERTS, ids_per_step=IDS, n_layers=N_LAYERS,
              fa_layers=FA_LAYERS)
    base_ms = er.owned_round_ms(solve_at(BASE, None), shares=(1, 0, 0), merged=False, **kw)
    out = []
    for a in range(0, 65, er.OWNED_SHARE_STEP):
        sh = (0, a, 64 - a)
        fits = solve_at(BASE, sh)
        if any(f.ceiling_fraction is None or f.ceiling_max_rows < f.scratch_rows + 2
               for f in fits):
            continue
        ms = er.owned_round_ms(fits, shares=sh, merged=True, **kw)
        if any(ms[w] > base_ms[w] + 1e-9 for w in (1, 2)):
            continue
        out.append((sh, ms))
    return out, base_ms


def test_the_host_binds_and_the_worker_sum_is_the_same_for_every_split():
    """The premise, stated: the primary key ties and the sum cannot break it."""
    forms, _ = _all_feasible(METAL_SHAPE)
    assert len(forms) >= 3
    maxes = {round(max(ms), 9) for _, ms in forms}
    assert len(maxes) == 1  # the host binds every form
    assert all(max(ms) == ms[0] for _, ms in forms)
    sums = {round(ms[1] + ms[2], 6) for _, ms in forms}
    assert len(sums) == 1  # the worker split moves no miss time in the sum


def test_the_tie_goes_to_the_form_whose_worst_worker_misses_least():
    """RED before #287: the sum tie fell to the lexical order of the share
    vector -- (0, 20, 44), TP2 at its x1 edge (the 09300002 shape). GREEN: the
    worst worker is minimized, TP2 keeps the most residents x1 allows."""
    forms, base_ms = _all_feasible(METAL_SHAPE)
    want_sh, want_ms = min(forms, key=lambda f: max(f[1][1], f[1][2]))
    assert want_sh == (0, 36, 28)
    sol = _solve(METAL_SHAPE)
    assert sol.feasible > 0
    assert tuple(sol.ratios) == BASE
    assert max(sol.round_ms) == pytest.approx(max(want_ms), abs=1e-9)  # bs1 optimum kept
    assert tuple(sol.cut) == tuple(want_sh)
    assert max(sol.round_ms[1], sol.round_ms[2]) == pytest.approx(
        max(want_ms[1], want_ms[2]), abs=1e-9)
    # every worker stays within its Form A (x1) and TP2 is not left at its edge
    assert all(sol.round_ms[w] <= base_ms[w] + 1e-9 for w in (1, 2))
    lopsided = min(forms, key=lambda f: f[0])  # the old lexical pick
    assert sol.round_ms[2] < lopsided[1][2]


def test_the_rank_key_is_leximax_then_sum():
    k = er.owned_rank_key
    # same max, same sum: the smaller next-worst wins
    assert k((35.92, 7.35, 28.28)) < k((35.92, 5.17, 30.46))
    # the max still decides first (bs1 objective unchanged)
    assert k((35.0, 5.0, 34.9)) < k((35.92, 7.35, 28.28))
    # equal sorted vectors: the key is the same whatever the rank order
    assert k((35.92, 7.35, 28.28)) == k((35.92, 28.28, 7.35))
