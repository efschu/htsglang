"""Hard limits of the owned solve (Koordinator 01.10. ~20:35Z, y6n).

y6n (desk/nf-y6n-1001 @ b0bf738b39) carried ede8266c5d with
SGLANG_WEG2_OWNED_BASE=derive as the default. Its dry run published
190,127,171 / cut 0,52,12: TP1 (E 134, R 64, scratch 44, trim cell
9984 B/token) keeps 21 rows at the top KV stage (524288 tokens), the captured
step at six seats routes 70 ids -- 2 x 21 < 70, so the launcher raised the
overflow waves 2 -> 4 and the group's top stage fell to 7 (262144 tokens):
the 3 L1.5 workers with 120-161k prompts each no longer fit D's KV.

The solve now holds every candidate to the launcher's own floor
(``rank_wave_floor`` over ``kv_stage_table``) at the line's wave cap with the
top stage mapped; a form that needs it gets its rows moved from resident to
scratch (the #251c LRU-floor path, FR_D solved with it), one that cannot is
not tragbar. And the vector moves away from the base only for more than 3 %
of the weighted round.

Edge model: the live launcher's (18:23:32Z), as in
test_weg2_owned_planner_base_1001.py, with the KV cells the budget solve
carries (host 1855 B/token incl. its QSA keys, a worker share x 12288 B).
"""

from __future__ import annotations

import types

from sglang.srt.planner import expert_residency as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

NUM_E = 512
N_LAYERS = 48
FA = 12
VERIFY = 4
IDS = 40
ROW_MIB = 48 * 2.417
A = (15430 + 541, 13574 + 1967, 14386 + 1157)  # Puffer + Rest vor KV, MiB
HOST_REST_MIB = 464
FA_MIB = 3072
FA_CELL = 12288
HOST_CELL = 1855
S = (104, 48, 48)
HAND = (183, 137, 168)
POOLED = (0.201074, 0.314024)
PER_RANK = (0.2027, 0.4734, 0.2171)
KW = dict(num_experts=NUM_E, n_layers=N_LAYERS, ids_per_step=IDS, fa_layers=FA,
          rows_per_round=VERIFY)
#: the y6n line: 6 seats, the derived H95 cap 2, stages up to S0 x 2, the
#: stage row = expert row minus scales (112.5 MiB, launcher RESIDENZ line)
STAGE_ROW = int(112.5 * (1 << 20))
LIMITS = dict(seats_cap=6, waves_cap=2, stage_tokens=262144, stage_row_bytes=STAGE_ROW)


def solve_at(rat, sh):
    spans = er.expert_span_by_rank(num_experts=NUM_E, ratios=list(rat))
    out = []
    for r in range(3):
        if sh is None:
            kv = (HOST_REST_MIB + FA_MIB) if r == 0 else 0
        else:
            kv = (HOST_REST_MIB if r == 0 else 0) + FA_MIB * sh[r] / 64.0
        rows = int((A[r] - kv) // ROW_MIB)
        E = int(spans[r]) + 1
        frac = er.largest_fraction_for_rows(local_experts=E, scratch_rows=S[r], max_rows=rows)
        stage_cell = 0 if (r == 0 or sh is None) else FA_CELL * sh[r] // 64
        out.append(types.SimpleNamespace(
            rank=r, local_experts=E, scratch_rows=S[r], ceiling_max_rows=rows,
            ceiling_fraction=frac, kv_cell_bytes=HOST_CELL if r == 0 else 0,
            kv_stage_cell_bytes=stage_cell))
    return tuple(out)


def _edge(rank, E, C, rows, cell):
    return er._EdgeFit(rank=rank, local_experts=E, scratch_rows=C, ceiling_max_rows=rows,
                       ceiling_fraction=0.5, trim_cell=cell)


FLOOR = dict(ids_cap=6 * IDS, waves=2, stage_tokens=262144, stage_row_bytes=STAGE_ROW)


# ---- the floor, one formula with the launcher --------------------------------

def test_the_y6n_form_breaks_the_floor_at_two_waves():
    """dry run 20:18Z (y6m tree, -pl): TP1 E 134, R 64, C 44 -> D 70 over
    44 - 23 = 21 rows: 70 > 2 x 21 (the launcher raised the waves to 4)."""
    tp1 = _edge(1, 134, 44, 108, 9984)
    assert er.owned_wave_floor([tp1], **FLOOR) == ("rang1 D 70 > 2 x (44 - 23 Stufenzeilen) = 42",)


def test_the_live_slot1_form_holds_it():
    """slot 1 (y6m -dres, launcher.log): WELLENBODEN rang0 D 201 <= 2 x (98+4),
    rang1 D 76 <= 2 x (32+15), rang2 D 96 <= 2 x (40+9) -- the same rows the
    floor counts here (scratch minus the top stage's rows)."""
    fits = [_edge(0, 227, 107, 133, HOST_CELL), _edge(1, 128, 65, 117, 7680),
            _edge(2, 160, 60, 124, 4608)]
    assert er.owned_wave_floor(fits, **FLOOR) == ()


def test_the_raise_is_the_launchers_lru_raise():
    """TP2 of the live form at the solve's scratch 48: C >= 58 (live: the
    #251c LRU floor raised it to 60); TP1 of the y6n form needs 72 of 108."""
    assert er.owned_scratch_raise([_edge(2, 160, 48, 124, 4608)], **FLOOR) == (10,)
    assert er.owned_scratch_raise([_edge(1, 134, 44, 108, 9984)], **FLOOR) == (28,)
    # no split of the rows carries it: not tragbar
    assert er.owned_scratch_raise([_edge(1, 200, 44, 60, 9984)], **FLOOR) is None


# ---- the solve ---------------------------------------------------------------

def _solve(base, **extra):
    return er.solve_owned_cut(solve_at, base, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW, **extra)


def _held(sol):
    """The floor of the published form with its scratch raise applied."""
    edges = []
    for f, a in zip(solve_at(sol.ratios, sol.cut), sol.scratch_raise or (0, 0, 0)):
        cell = er._edge_trim_cell(f, 0)
        edges.append(_edge(f.rank, f.local_experts, f.scratch_rows + a, f.ceiling_max_rows,
                           cell))
    return er.owned_wave_floor(edges, **FLOOR)


def test_without_limits_the_derived_form_breaks_the_floor():
    """The red case: the derived base + per-card cost + bs mix, unlimited,
    ends on a form whose floor breaks at the given scratch (the y6n class)."""
    ob = er.derive_owned_base(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    free = _solve(ob.ratios)
    assert free.feasible
    edges = [_edge(f.rank, f.local_experts, f.scratch_rows, f.ceiling_max_rows,
                   er._edge_trim_cell(f, 0)) for f in solve_at(free.ratios, free.cut)]
    assert er.owned_wave_floor(edges, **FLOOR)


def test_with_limits_the_chosen_form_holds_them():
    ob = er.derive_owned_base(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    sol = _solve(ob.ratios, **LIMITS)
    assert sol.feasible and sol.guard_refused > 0
    assert dict(sol.guard)["waves"] == 2
    assert _held(sol) == ()
    # FR_D is solved with the raised scratch: R + S stays the edge
    for f, a, fr in zip(solve_at(sol.ratios, sol.cut), sol.scratch_raise or (0, 0, 0),
                        sol.fractions):
        assert fr == er.largest_fraction_for_rows(
            local_experts=f.local_experts, scratch_rows=f.scratch_rows + a,
            max_rows=f.ceiling_max_rows)


def test_the_stated_base_keeps_the_live_form_with_the_lru_raise():
    """Stated 183,137,168 under the limits: the live slot-1 form 215,121,152 /
    0,40,24, TP2 scratch +10 (live: the LRU floor's 48 -> 60)."""
    sol = _solve(HAND, **LIMITS)
    assert sol.ratios == (215, 121, 152) and sol.cut == (0, 40, 24)
    assert sol.scratch_raise == (0, 0, 10)
    assert _held(sol) == ()


def test_no_room_no_form():
    """A wave cap of 1 at six seats cannot carry the top stage on the KV
    workers: every cut is refused by name, none is published."""
    sol = _solve(HAND, seats_cap=6, waves_cap=1, stage_tokens=262144,
                 stage_row_bytes=STAGE_ROW)
    assert not sol.feasible and sol.guard_refused > 0


# ---- the switch gain ---------------------------------------------------------

def test_a_small_gain_keeps_the_base_ownership():
    """x1 reported only (round scope), the limits on: the best other vector
    183,145,160 / 0,16,48 is 0.1 % faster than the best form on the base
    ownership 183,137,168 -- inside the 3 % the vector stays."""
    sol = _solve(HAND, switch_gain=er.OWNED_SWITCH_GAIN, x1_scope="round", **LIMITS)
    assert sol.ratios == HAND and sol.stayed
    assert sol.best_other_objective_ms < sol.objective_ms < sol.best_other_objective_ms * 1.03


def test_without_the_gate_it_moves():
    sol = _solve(HAND, switch_gain=0.0, x1_scope="round", **LIMITS)
    assert sol.ratios != HAND and not sol.stayed


# ---- wiring --------------------------------------------------------------------

def test_the_planner_pass_holds_the_limits_and_the_launcher_raises_the_scratch():
    """plan_d_residency hands the line's limits and the 3 % gate to every
    owned solve (also the stated-base one it compares the derived base
    against); the launcher raises SGLANG_MOE_SCRATCH_SLOTS by the plan's
    owned_scratch_raise and solves again (the #251c path)."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(er.plan_d_residency)
    assert "owned_form_limits(" in src and "switch_gain=OWNED_SWITCH_GAIN" in src
    assert "_owned_from(_stated_base)" in src
    assert "owned_scratch_raise=solved_raise" in src
    lsrc = inspect.getsource(launcher.log_d_rank_vram_solve)
    assert 'getattr(plan, "owned_scratch_raise", ())' in lsrc


def test_limits_from_the_line_env():
    terms = types.SimpleNamespace(expert_layer_weight_bytes=512 * 2.417 * (1 << 20),
                                  num_experts=512, n_layers=48)
    cfg = {"moe_intermediate_size": 512, "hidden_size": 2048,
           "linear_num_value_heads": 32, "linear_value_head_dim": 128,
           "linear_key_head_dim": 128, "layer_types": ["linear_attention"] * 3}
    kw = dict(text_cfg=cfg, terms=terms, seats=6, kv_tokens=262144, rank_tp_ratio="1,0,0",
              n_ranks=3)
    off = er.owned_form_limits({"SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "2"}, **kw)
    assert off == {"seats_cap": 6, "waves_cap": 2}
    on = er.owned_form_limits({"SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "2",
                               "SGLANG_OPT_WEG2_D_SEAT_VRAM": "1"}, **kw)
    assert on["stage_tokens"] == 262144
    assert 0 < on["stage_row_bytes"] <= 48 * 2.417 * (1 << 20)
