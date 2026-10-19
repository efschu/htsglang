"""S54 (19.10.): the D expert split priced at each rank's LINK, x1 held on the
synchronized round.

rc14 (boot ...dauer10190107_92dd662831_1019_010742, front.log l.169-176)
published --rank-moe-ratio 215,121,152 / cut 0,36,28: T_r bs1 68.42 / 21.26 /
29.92 ms against Form A 63.13 / 23.81 / 33.68 -- the x1 rule held every WORKER
to its Form-A miss time, so the KV cut's worker rows were bought with host
residency, and TP0 (5090, PCIe Gen4 x8, 14.42 GB/s) bound every D extend
(H107 misses per layer 91.5 / 32.1 / 48.1 -> 16.1 / 12.6 / 9.1 ms copy).

The edge model below is that launcher's (budget minus fixed, mamba, spec and
activation posts; host KV rest 1855 B/token, FA cell 12288 B/token x share);
the old solve on it reproduces the published form to the hundredth.
"""

from __future__ import annotations

import types

import pytest

from flliper.srt.planner import expert_residency as er
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

NUM_E = 512
N_LAYERS = 48
FA = 12
VERIFY = 4
IDS = 40
ROW_MIB = 48 * 2.417
#: budget [26800, 17320, 17520] minus fest/mamba/spec/Aktivierung (front.log l.172-175)
A = (26800 - 7481 - 2134 - 134 - 1104, 17320 - 1075 - 1024, 17520 - 953 - 1024)
HOST_REST_MIB = 464
FA_MIB = 3072
FA_CELL = 12288
HOST_CELL = 1855
S = (104, 48, 48)
HAND = (183, 137, 168)
POOLED = (0.199992, 0.31532)
PER_RANK = (0.2000, 0.4711, 0.2120)
STAGE_ROW = int(112.5 * (1 << 20))
LIMITS = dict(seats_cap=6, waves_cap=2, stage_tokens=262144, stage_row_bytes=STAGE_ROW)
KW = dict(num_experts=NUM_E, n_layers=N_LAYERS, ids_per_step=IDS, fa_layers=FA,
          rows_per_round=VERIFY)
#: card probe card_probe-20ae5edfc9b2.json, pinned H2D GB/s, D rank order (5090, 3080 x4, 3080 x8)
H2D = (14.42, 6.47, 13.40)
ROW_BYTES = 2534400  # one expert row of one MoE layer (RW WAKE-FIRST-TOKEN row_bytes)
EXT_IDS = 256 * 10


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


def _solve(scope, rate=0.0, **extra):
    row_ms = er.owned_extend_row_ms(H2D, ROW_BYTES)
    return er.solve_owned_cut(
        solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
        bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, switch_gain=er.OWNED_SWITCH_GAIN,
        x1_scope=scope, extend_row_ms=row_ms, extend_ids=EXT_IDS, extend_per_round=rate,
        **KW, **{**LIMITS, **extra})


def _ext_at(sol):
    return er.owned_extend_ms(solve_at(sol.ratios, sol.cut), num_experts=NUM_E,
                              n_layers=N_LAYERS, extend_ids=EXT_IDS,
                              row_ms=er.owned_extend_row_ms(H2D, ROW_BYTES))


# ---- the link prices -----------------------------------------------------------

def test_link_cost_per_row_from_the_card_probe():
    got = er.owned_extend_row_ms(H2D, ROW_BYTES)
    assert got == pytest.approx((0.17576, 0.39172, 0.18913), abs=1e-4)
    # an unmeasured rank switches the term off, never a nameplate
    assert er.owned_extend_row_ms((14.42, None, 13.4), ROW_BYTES) is None
    assert er.owned_extend_row_ms((14.42, 0.0, 13.4), ROW_BYTES) is None


def test_extend_copy_is_missed_rows_times_layers_times_link():
    fits = solve_at((215, 121, 152), (0, 36, 28))
    row_ms = er.owned_extend_row_ms(H2D, ROW_BYTES)
    got = er.owned_extend_ms(fits, num_experts=NUM_E, n_layers=N_LAYERS, extend_ids=EXT_IDS,
                             row_ms=row_ms)
    for f, x, c in zip(fits, got, row_ms):
        miss = er.owned_miss_rows(f, num_experts=NUM_E, ids_per_step=EXT_IDS)
        assert x == pytest.approx(miss * N_LAYERS * c)
    # 2560 ids touch ~99 % of a rank's experts: TP0's 94 non-held experts set the clock
    assert max(got) == got[0] and got[0] == pytest.approx(94 * 0.9933 * 48 * 0.17576, rel=0.01)


# ---- the old solve stays the old solve ---------------------------------------------

def test_workers_scope_reproduces_the_published_rc14_form():
    sol = _solve("workers")
    assert sol.ratios == (215, 121, 152) and sol.cut == (0, 36, 28)
    assert [round(x, 2) for x in sol.round_ms] == [68.42, 21.26, 29.92]
    assert [round(x, 2) for x in sol.base_round_ms] == [63.13, 23.81, 33.68]
    assert sol.extend_ms == () and sol.extend_per_round == 0.0


# ---- sync --------------------------------------------------------------------------

def test_sync_holds_the_synchronized_round_against_form_a():
    sol = _solve("sync")
    by, base = dict(sol.round_ms_by_bs), dict(sol.base_round_ms_by_bs)
    assert sol.x1_ok
    for b in (1, 2):
        assert max(by[b]) <= max(base[b]) + 1e-9
    # the rc14 form breaks exactly this: workers-x1 yet above Form A on TP0
    live = _solve("workers")
    assert max(live.round_ms) > max(live.base_round_ms)


def test_sync_beats_the_live_form_on_round_and_extend():
    live = _solve("workers")
    new = _solve("sync", rate=er.OWNED_EXTEND_PER_ROUND_SEED)
    for (b, ms), (_, lms) in zip(new.round_ms_by_bs, live.round_ms_by_bs):
        assert max(ms) < max(lms), b
    assert max(new.extend_ms) < max(_ext_at(live))
    # the extend is balanced over the links: no rank more than 10 % above the mean
    mean = sum(new.extend_ms) / 3.0
    assert max(new.extend_ms) <= 1.10 * mean
    # and TP0 no longer carries the most non-held experts
    fits = solve_at(new.ratios, new.cut)
    cold = [f.local_experts - min(f.ceiling_max_rows, f.local_experts) for f in fits]
    assert cold[0] < 94


def test_extend_term_enters_the_objective():
    off = _solve("sync", rate=0.0)
    on = _solve("sync", rate=0.0256)
    assert off.extend_ms == () and on.extend_ms
    assert on.objective_ms == pytest.approx(
        sum(w * max(ms) for (b, ms), (_, w) in zip(on.round_ms_by_bs, on.bs_weights))
        + 0.0256 * max(on.extend_ms), rel=1e-9)
    assert on.base_objective_ms == pytest.approx(
        off.base_objective_ms + 0.0256 * max(on.base_extend_ms), rel=1e-9)


def test_extend_term_alone_moves_the_split_off_the_slow_link():
    """The danger direction: priced at its own link the x4 card must not be
    handed the extend -- with the decode term made negligible the solve moves
    the non-held experts to the x8 cards, not to TP1."""
    row_ms = er.owned_extend_row_ms(H2D, ROW_BYTES)
    sol = er.solve_owned_cut(
        solve_at, HAND, 0, miss_ms=(1e-6, 1e-6), miss_ms_rank=(1e-6, 1e-6, 1e-6),
        bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, x1_scope="sync", extend_row_ms=row_ms,
        extend_ids=EXT_IDS, extend_per_round=1.0, **KW, **LIMITS)
    x = sol.extend_ms
    assert max(x) <= max(sol.base_extend_ms) + 1e-9
    fits = solve_at(sol.ratios, sol.cut)
    cold = [f.local_experts - min(f.ceiling_max_rows, f.local_experts) for f in fits]
    # copy time, not cold count, is balanced: the x4 TP1 carries the fewest
    assert cold[1] < cold[0] and cold[1] < cold[2]


def test_sync_ranks_a_form_behind_instead_of_refusing():
    """No ownership move allowed and the workers priced 100x the host: every
    cut form puts KV on a worker and raises its round above Form A. workers
    has nothing tragbar; sync still publishes the best round, x1 reported
    missed (a planner gap never stops a boot)."""
    kw = dict(miss_ms=POOLED, miss_ms_rank=(0.002, 1.0, 1.0), bs_weights=None,
              max_shift=0, **KW)
    w = er.solve_owned_cut(solve_at, HAND, 0, x1_scope="workers", **kw)
    s = er.solve_owned_cut(solve_at, HAND, 0, x1_scope="sync", **kw)
    assert w.feasible == 0 and not w.ratios
    assert s.ratios == HAND and s.feasible > 0 and not s.x1_ok


# ---- env -----------------------------------------------------------------------------

def test_x1_default_is_sync(monkeypatch):
    monkeypatch.delenv(er.OWNED_X1_ENV, raising=False)
    assert er.owned_x1_scope() == "sync"
    monkeypatch.setenv(er.OWNED_X1_ENV, "workers")
    assert er.owned_x1_scope() == "workers"


def test_extend_form_seed_override_refusal(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_OWNED_EXTEND_TOKENS", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_OWNED_EXTEND_PER_ROUND", raising=False)
    tok, rate, src = er.owned_extend_form()
    assert (tok, rate) == (er.OWNED_EXTEND_TOKENS_SEED, er.OWNED_EXTEND_PER_ROUND_SEED)
    assert "1019_010742" in src
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_EXTEND_TOKENS", "512")
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_EXTEND_PER_ROUND", "0")
    assert er.owned_extend_form()[:2] == (512, 0.0)
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_EXTEND_PER_ROUND", "-1")
    with pytest.raises(ValueError):
        er.owned_extend_form()
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_EXTEND_PER_ROUND", "x")
    with pytest.raises(ValueError):
        er.owned_extend_form()


# ---- launcher: the link per D rank, by UUID ------------------------------------------

def test_launcher_reads_each_d_ranks_link_by_uuid(monkeypatch):
    from flliper.srt.layers.moe import expert_offload as eo
    from flliper.srt.pdflip import launcher as L

    table = {"GPU-5090": 14.42, "GPU-x4": 6.47, "GPU-x8": 13.4}
    monkeypatch.setattr(eo, "_card_probe_h2d_table", lambda: dict(table))
    cards = [types.SimpleNamespace(uuid="GPU-5090", nvml_index=1),
             types.SimpleNamespace(uuid="GPU-x4", nvml_index=0),
             types.SimpleNamespace(uuid="GPU-x8", nvml_index=2)]
    got, src = L.d_owned_extend_h2d(cards)
    assert got == [14.42, 6.47, 13.4] and "nvml1 14.42" in src
    # rank order follows the cards, not the table
    got2, _ = L.d_owned_extend_h2d(list(reversed(cards)))
    assert got2 == [13.4, 6.47, 14.42]
    del table["GPU-x4"]
    none, why = L.d_owned_extend_h2d(cards)
    assert none is None and "nvml0" in why
