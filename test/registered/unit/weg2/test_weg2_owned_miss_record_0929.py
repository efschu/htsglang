"""#239 S3f Miss-Record (29.09., z30w): the owned solve priced a missed expert
row with the seed 0.1 ms (5090) / 0.2 ms (3080), UNMEASURED. On metal the
round rule moved misses from the 5090 to TP1 and the decode round got slower
at bs1/bs2 -- the solve steered on the seed.

* the cost per missed row is read from a D log (pool.fetch device ms per rank
  over missed rows) and written as a sidecar RECORD;
* RECORD > BUILTIN > UNMEASURED picks the cost the solve runs with;
* with a measured cost the solve chooses another form than with the seed;
* a log without the graph reader's split is refused, never guessed;
* 'Decode rank batch' can carry the requests' depth (off by default).
"""

from __future__ import annotations

import json
import types

import pytest

from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog, RoundAcc
from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2.tools import owned_miss_record as omr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

BASE = (183, 137, 168)
NUM_EXPERTS = 512
N_LAYERS = 48
IDS = 40
FA_LAYERS = 12
MODEL = "/models/Qwen3.8-Flash-Next-INT4"


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


SERVING = _model(scratch=(100, 48, 48), rows_form_a=(95, 127, 127),
                 rows_cut_at0=(122, 127.9, 127.6), rows_per_share=(0.0, 0.414, 0.414))


def _solve(miss_ms):
    return er.solve_owned_cut(SERVING, BASE, 0, num_experts=NUM_EXPERTS, n_layers=N_LAYERS,
                              ids_per_step=IDS, fa_layers=FA_LAYERS, x1_scope="round",
                              miss_ms=miss_ms)


# -- the solve steers on the measured cost -----------------------------------

def test_measured_cost_changes_the_chosen_form():
    """z30w: a 3080 miss costs far more than the seed says, the 5090's less --
    with that the solve must not move ownership onto the workers as the seed
    form does."""
    seed = _solve(er.OWNED_MISS_MS_PER_ROW_SEED)
    measured = _solve((0.05, 0.6))
    assert seed.feasible > 0 and measured.feasible > 0
    assert (measured.ratios, measured.cut) != (seed.ratios, seed.cut)
    # the worker-heavy cost pushes ownership back to the host
    assert measured.ratios[0] >= seed.ratios[0]


# -- RECORD > BUILTIN > UNMEASURED ------------------------------------------

def _rec(pair, at, model=MODEL, source="z30w"):
    return {"kind": er.OWNED_MISS_KIND, "miss_ms_per_row": list(pair), "at": at,
            "model": model, "source": source, "rounds": 900}


def test_resolve_without_anything_is_the_named_seed():
    ms, tier, src = er.resolve_owned_miss_ms((), model=MODEL)
    assert ms == er.OWNED_MISS_MS_PER_ROW_SEED
    assert tier == er.OWNED_MISS_UNMEASURED and "UNMEASURED" in src


def test_resolve_builtin_beats_seed_and_record_beats_builtin():
    ms, tier, _ = er.resolve_owned_miss_ms((), model=MODEL, builtin=(0.08, 0.3),
                                           builtin_source="boot z30x")
    assert tier == er.OWNED_MISS_BUILTIN and ms == (0.08, 0.3)
    ms, tier, src = er.resolve_owned_miss_ms([_rec((0.05, 0.6), "2026-09-29 09:00:00")],
                                             model=MODEL, builtin=(0.08, 0.3))
    assert tier == er.OWNED_MISS_LOG_BOOTSTRAP and ms == (0.05, 0.6) and "z30w" in src
    assert src.startswith(er.OWNED_MISS_LOG_PROVENANCE)


def test_resolve_takes_the_youngest_record_of_this_model():
    recs = [_rec((0.05, 0.6), "2026-09-29 09:00:00", source="old"),
            _rec((0.07, 0.5), "2026-09-29 10:00:00", source="new"),
            _rec((0.01, 0.01), "2026-09-29 11:00:00", model="/models/other", source="foreign")]
    ms, tier, src = er.resolve_owned_miss_ms(recs, model=MODEL)
    assert tier == er.OWNED_MISS_LOG_BOOTSTRAP and ms == (0.07, 0.5) and "new" in src


def test_a_broken_entry_is_skipped_never_read_as_zero():
    recs = [_rec((0.0, 0.6), "2026-09-29 12:00:00"), {"kind": er.OWNED_MISS_KIND,
                                                      "miss_ms_per_row": "x", "model": MODEL}]
    ms, tier, _ = er.resolve_owned_miss_ms(recs, model=MODEL)
    assert tier == er.OWNED_MISS_UNMEASURED and ms == er.OWNED_MISS_MS_PER_ROW_SEED


def test_sidecar_reader_filters_the_kind(tmp_path):
    p = tmp_path / "weg2_measured_record.json"
    p.write_text(json.dumps({"samples": [_rec((0.05, 0.6), "t"), {"kind": "pd_free0"}]}))
    assert [e["kind"] for e in er.read_owned_miss_records(str(p))] == [er.OWNED_MISS_KIND]
    assert er.read_owned_miss_records(str(tmp_path / "missing.json")) == []


# -- the tool reads the two numbers a D log carries --------------------------

def _round(tp, n, fetch_ms, fwd=2, split=True):
    tail = (" (compute 20.0, wait %.1f) (wait by family: tp.all_reduce 4.0/96x, "
            "spec_verify:pool.fetch %.1f/12x min0.100)" % (fetch_ms + 4.0, fetch_ms)
            if split else " (split unavailable: graph-replay-reader-off, graphed-fwd 2/2)")
    return ("[2026-09-29 09:%02d:%02d TP%d] Decode rank batch, rank: %d, #round: %d, "
            "t: 1790670681.230, bs: 1, #rows: 4, #fwd: %d, gpu-ms: 30.0%s"
            % (n // 60, n % 60, tp, tp, n, fwd, tail))


def _miss(tp, fwd, miss):
    return ("[2026-09-29 09:10:00 TP%d] MoE expert pool layer 0: %d decode forwards since "
            "last sync, %d misses (%.2f per forward, top-k 10); prefetch predicted 0, "
            "fetched 0, hits 0, wasted 0, skipped 0" % (tp, fwd, miss, miss / fwd))


def _log(split=True):
    lines = []
    for n in range(100):
        lines += [_round(0, n, 2.4, split=split), _round(1, n, 9.6, split=split),
                  _round(2, n, 4.8, split=split)]
    # per forward and layer: TP0 0.5, TP1 1.0, TP2 0.5 missed rows
    lines += [_miss(0, 200, 100), _miss(1, 200, 200), _miss(2, 200, 100)]
    return "\n".join(lines)


def test_tool_divides_fetch_ms_by_missed_rows():
    rec = omr.owned_miss_from_log(_log(), n_moe_layers=48, host_rank=0, source="t")
    # TP0: 240 ms / (0.5 x 48 x 200 fwd) = 0.05; TP1 960 / (1.0 x 48 x 200) = 0.1;
    # TP2 480 / (0.5 x 48 x 200) = 0.1 -> workers 0.1
    assert rec["miss_ms_per_row"] == [0.05, 0.1]
    assert rec["kind"] == er.OWNED_MISS_KIND and rec["rounds"] == 100


def test_tool_refuses_a_log_without_the_split():
    with pytest.raises(ValueError, match="graph reader was off"):
        omr.owned_miss_from_log(_log(split=False), n_moe_layers=48, source="t")


def test_tool_refuses_a_rank_without_miss_lines():
    text = "\n".join(l for l in _log().splitlines() if "TP2] MoE expert pool" not in l)
    with pytest.raises(ValueError, match="rank 2"):
        omr.owned_miss_from_log(text, n_moe_layers=48, source="t")


# -- depth per round ---------------------------------------------------------

def test_round_depth_is_min_median_max_of_the_requests():
    assert DecodeRoundLog.round_depth([90000, 12000, 40000]) == (12000, 40000, 90000)
    assert DecodeRoundLog.round_depth([]) is None


def test_depth_off_by_default_leaves_the_round_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_DEBUG_DECODE_ROUND_DEPTH", raising=False)
    drl = DecodeRoundLog(clock=None, rank=0)
    assert drl.depth_on is False
    assert RoundAcc(1, 2, 8).depth is None


def test_depth_on_is_read_once_at_construction(monkeypatch):
    monkeypatch.setenv("SGLANG_DEBUG_DECODE_ROUND_DEPTH", "1")
    assert DecodeRoundLog(clock=None, rank=0).depth_on is True


# -- the launcher hands the record to the planner ----------------------------

def test_launcher_reads_the_log_bootstrap(tmp_path, monkeypatch):
    from sglang.srt.weg2 import launcher

    p = tmp_path / "weg2_measured_record.json"
    p.write_text(json.dumps({"samples": [_rec((0.05, 0.6), "2026-09-29 09:00:00")]}))
    monkeypatch.setattr(launcher, "measured_record_path", lambda: str(p))
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    ms, src = launcher.d_owned_miss_ms(ns, env_d={}, host=0)
    assert ms == (0.05, 0.6) and src.startswith(er.OWNED_MISS_LOG_PROVENANCE)


def test_launcher_without_a_record_keeps_the_seed_byte_identical(tmp_path, monkeypatch):
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(launcher, "measured_record_path", lambda: str(tmp_path / "none.json"))
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    assert launcher.d_owned_miss_ms(ns, env_d={}, host=0) == (None, "")


# -- the ranks write the record themselves (IPC, not the log) ----------------

from sglang.srt.layers.moe import pool_miss_cost as pmc


@pytest.fixture
def miss_dir(tmp_path, monkeypatch):
    d = tmp_path / "owned_miss"
    monkeypatch.setenv("SGLANG_WEG2_OWNED_MISS_RECORD", str(d))
    pmc._reset_for_test()
    yield d
    pmc._reset_for_test()


def _phase(rank, fetch_per_round, miss_per_sync, rounds=100):
    for _ in range(rounds):
        pmc.note_round({"spec_verify:pool.fetch": [fetch_per_round, 12, 0.1],
                        "tp.all_reduce": [4.0, 96, 0.01]})
    for _ in range(48):  # every pool layer syncs
        pmc.note_sync(200, miss_per_sync)
    return pmc.flush(rank=rank, group="D", reason="sleep", model=MODEL, phase_index=3)


def test_rank_record_kind_is_the_one_the_planner_reads():
    assert pmc.RECORD_KIND == er.OWNED_MISS_RANK_KIND


def test_rank_writes_its_record_at_the_sleep(miss_dir):
    path = _phase(0, 2.4, 10)
    rec = json.loads(open(path).read())
    # 240 ms over 48 layers x 10 missed rows = 0.5 ms per row
    assert rec["kind"] == er.OWNED_MISS_RANK_KIND and rec["rank"] == 0
    assert rec["miss_rows"] == 480 and rec["rounds"] == 100
    assert abs(rec["ms_per_row"] - 240.0 / 480) < 1e-9
    # counters zeroed: the next phase starts from nothing
    assert pmc.rank_record(rank=0, group="D", reason="x", model=MODEL) is None


def test_off_counts_nothing_and_writes_nothing(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_OWNED_MISS_RECORD", raising=False)
    pmc._reset_for_test()
    assert _phase(0, 2.4, 10) is None
    assert pmc._ACC["rounds"] == 0 and pmc._ACC["miss_rows"] == 0


def test_a_phase_without_split_writes_no_record(miss_dir):
    for _ in range(48):
        pmc.note_sync(200, 10)
    assert pmc.flush(rank=1, group="D", reason="sleep", model=MODEL) is None
    assert not miss_dir.exists() or not any(miss_dir.iterdir())


def test_planner_reads_the_rank_records(miss_dir):
    _phase(0, 2.4, 10)   # host 0.5 ms/row
    _phase(1, 9.6, 10)   # worker 2.0 ms/row
    _phase(2, 4.8, 20)   # worker 0.5 ms/row, twice the rows
    recs = er.read_owned_miss_rank_records(str(miss_dir))
    ms, tier, src = er.resolve_owned_miss_ms((), rank_records=recs, host=0, model=MODEL)
    assert tier == er.OWNED_MISS_RECORD and src.startswith("RECORD")
    # worker = (960 + 480) ms / (480 + 960) rows = 1.0
    assert ms == pytest.approx((0.5, 1.0))


def test_log_bootstrap_ranks_under_every_rank_record(miss_dir):
    _phase(0, 2.4, 10)
    _phase(1, 9.6, 10)
    ranks = er.read_owned_miss_rank_records(str(miss_dir))
    # a YOUNGER log entry still loses to the ranks' own record
    log = [_rec((0.05, 0.6), "2099-01-01 00:00:00")]
    ms, tier, _ = er.resolve_owned_miss_ms(log, rank_records=ranks, host=0, model=MODEL)
    assert tier == er.OWNED_MISS_RECORD and ms == pytest.approx((0.5, 2.0))
    # without the ranks the bootstrap stands, named as the transition
    ms, tier, src = er.resolve_owned_miss_ms(log, rank_records=(), host=0, model=MODEL)
    assert tier == er.OWNED_MISS_LOG_BOOTSTRAP and src.startswith(er.OWNED_MISS_LOG_PROVENANCE)


def test_rank_records_without_a_worker_do_not_count(miss_dir):
    _phase(0, 2.4, 10)
    ranks = er.read_owned_miss_rank_records(str(miss_dir))
    _ms, tier, _ = er.resolve_owned_miss_ms((), rank_records=ranks, host=0, model=MODEL)
    assert tier == er.OWNED_MISS_UNMEASURED


def test_launcher_takes_the_rank_records_from_group_d_env(miss_dir, tmp_path, monkeypatch):
    from sglang.srt.weg2 import launcher

    _phase(0, 2.4, 10)
    _phase(1, 9.6, 10)
    side = tmp_path / "weg2_measured_record.json"
    side.write_text(json.dumps({"samples": [_rec((0.05, 0.6), "2099-01-01 00:00:00")]}))
    monkeypatch.setattr(launcher, "measured_record_path", lambda: str(side))
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    ms, src = launcher.d_owned_miss_ms(
        ns, env_d={"SGLANG_WEG2_OWNED_MISS_RECORD": str(miss_dir)}, host=0)
    assert ms == pytest.approx((0.5, 2.0)) and src.startswith("RECORD")


def test_the_round_log_feeds_the_rank_record():
    """DecodeRoundLog hands every split round's families to the record."""
    import inspect

    from sglang.srt.managers.scheduler_components import decode_round_log

    assert "_miss_cost.note_round(family_acc)" in inspect.getsource(
        decode_round_log.DecodeRoundLog._emit)
