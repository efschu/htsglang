# SPDX-License-Identifier: Apache-2.0
"""PP-COST (27B, 01.10.): the P cut solver prices the TIME axis with the
per-layer-type, per-card stage model over the predecessor boot's chunk mix.

Metal (rank lines, PP0..PP2 'Prefill rank batch' + '#1469 RETAIN'): 43,11,10
(36 boots 28./29.09.) is +4.5 % faster than 44,10,10 (19 boots 30.09.-01.10.)
with the 44,10,10 boots' own chunk mix and holds +15 % P pool; the solver of
N3b (dkr27browauthoritybar1fs10011917) shipped 44,10,10 because its family /
card-rate model priced it 425.8 against 447.1 ms ('PP-CUT solver' lines).

DANGER DIRECTIONS guarded here:
* the rank-line calibration reads the chunk POSITION (token_ids_len - #new-token),
  pairs a rank line only with the RETAIN of the same rank, and drops an unpaired one;
* the multi-cut fit recovers per-layer, attention and fixed terms, and FOLDS a fixed
  term it cannot separate (or that comes out negative) into the rate at every width
  of that mode;
* the solver, on the shipped calibration (27b_int8_rc12z30) and N3b's chunk mix,
  ranks 43,11,10 ahead of 44,10,10 and does not ship 44,10,10.
MUTANT (in-suite): the same solve with the old family / card-rate model (the very
numbers N3b logged) -> the ranking test is red.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.planner import pp_cut as P
from flliper.srt.planner.pp_cut_launch import solve_launch_cut
from flliper.srt.pdflip import p_stage_model as M
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FAMILIES = tuple(P.LAYER_FAMILY_ATTENTION if i % 4 == 3 else P.LAYER_FAMILY_LINEAR for i in range(64))
LT = M.layer_types_every(64, 4, 3)
CARDS3 = ("RTX5090", "RTX3080", "RTX3080")

#: N3b's PP0 chunk mix (dkr27browauthoritybar1fs10011917, p_stage_model.chunk_mix_from_samples):
#: (width, prefix bin centre, g=graph/e=eager, count)
N3B_MIX = ((24, 94208, 'g', 1), (27, 77824, 'g', 1), (46, 86016, 'g', 1), (55, 118784, 'g', 1),
           (88, 36864, 'g', 1), (89, 53248, 'g', 1), (89, 86016, 'g', 1), (91, 94208, 'g', 1),
           (119, 53248, 'g', 1), (148, 45056, 'g', 1), (155, 151552, 'g', 1), (179, 126976, 'g', 1),
           (193, 126976, 'g', 1), (200, 86016, 'g', 1), (214, 94208, 'g', 1), (236, 45056, 'g', 1),
           (243, 94208, 'g', 1), (246, 61440, 'g', 1), (277, 4096, 'g', 1), (280, 45056, 'g', 1),
           (302, 143360, 'g', 1), (351, 53248, 'g', 1), (395, 69632, 'g', 1), (434, 36864, 'g', 1),
           (439, 4096, 'g', 1), (441, 20480, 'g', 1), (447, 53248, 'g', 1), (476, 53248, 'g', 1),
           (495, 126976, 'g', 1), (501, 61440, 'g', 1), (507, 12288, 'g', 1), (512, 4096, 'g', 6),
           (512, 20480, 'g', 1), (512, 28672, 'g', 2), (512, 36864, 'g', 1), (512, 45056, 'g', 4),
           (512, 53248, 'g', 10), (512, 61440, 'g', 7), (512, 69632, 'g', 17), (512, 77824, 'g', 22),
           (512, 86016, 'g', 27), (512, 94208, 'g', 15), (512, 102400, 'g', 16), (512, 110592, 'g', 16),
           (512, 118784, 'g', 13), (512, 126976, 'g', 3), (512, 135168, 'g', 8), (512, 143360, 'g', 2),
           (512, 151552, 'g', 12), (703, 135168, 'e', 1), (722, 45056, 'e', 1), (1024, 4096, 'e', 6),
           (1024, 12288, 'e', 10), (1024, 20480, 'e', 26), (1024, 28672, 'e', 34), (1024, 36864, 'e', 21),
           (1024, 45056, 'e', 27), (1024, 53248, 'e', 8), (1024, 61440, 'e', 8), (1024, 69632, 'e', 8),
           (1024, 77824, 'e', 8), (1024, 86016, 'e', 8), (1024, 94208, 'e', 2), (1024, 118784, 'e', 1),
           (2048, 12288, 'e', 1), (2048, 53248, 'e', 2), (2048, 61440, 'e', 4), (2048, 69632, 'e', 4),
           (2048, 77824, 'e', 4), (2048, 86016, 'e', 4), (2048, 94208, 'e', 4), (2048, 102400, 'e', 4),
           (2048, 110592, 'e', 4), (2048, 118784, 'e', 3))
MIX = tuple((w, p, M.MODE_GRAPH if m == 'g' else M.MODE_EAGER, float(n)) for w, p, m, n in N3B_MIX)

#: the old pricing exactly as N3b logged it ('PP-CUT depth axis ... family split: ... linear
#: 10.19,42.98,43.85 ms/layer, attn 1.84,7.78,7.94 ms/layer at the reference depth', chunk 2048,
#: reference prefix 4096, design prefix 21615)
OLD_FAMILY = P.FamilyDepthCost(linear_ms_per_layer=(10.19, 42.98, 43.85),
                               attn_ms_per_layer_at_ref=(1.84, 7.78, 7.94),
                               chunk_tokens=2048, ref_prefix_tokens=4096.0)
DESIGN_PREFIX = 21615


# -- the rank-line samples ---------------------------------------------------------


def test_rank_lines_pair_with_the_same_ranks_retain_and_read_the_position():
    lines = [
        "[t PP0] PREFILL-GRAPH captured backend=full pp_rank=0 buckets=[512] slots=1",
        "[t PP0] Prefill rank batch, #new-token: 512, #cached-token: 0, #chunks: 1, gpu-ms: 100.0 (compute 100.0, wait 0.0)",
        "[t PP1] Prefill rank batch, #new-token: 512, #cached-token: 0, #chunks: 1, gpu-ms: 70.0",
        "[t PP1] #1469 RETAIN rid=r is_finished=False token_ids_len=66048 cache_len=66048 value=True",
        "[t PP0] #1469 RETAIN rid=r is_finished=False token_ids_len=66048 cache_len=66048 value=True",
        "[t PP2] Prefill rank batch, #new-token: 1024, #cached-token: 4096, #chunks: 1, gpu-ms: 90.0",
        "[t PP2] Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 1, gpu-ms: 300.0",
        "[t PP2] #1469 RETAIN rid=r is_finished=True token_ids_len=10240 cache_len=10240 value=True",
    ]
    s = M.samples_from_rank_lines(lines)
    got = sorted((x.rank, x.width, x.prefix, x.gpu_ms, x.mode) for x in s)
    assert got == [(0, 512, 65536, 100.0, M.MODE_GRAPH), (1, 512, 65536, 70.0, M.MODE_GRAPH),
                   (2, 2048, 8192, 300.0, M.MODE_EAGER)], got   # the unpaired PP2 1024 line is dropped
    assert M.cut_of_lines(["x server_args=ServerArgs(pp_layer_ratio=[43, 11, 10], y=1)"]) == (43, 11, 10)


# -- the multi-cut fit ---------------------------------------------------------------


def _truth():
    g, k = {"RTX5090": 1.0, "RTX3080": 3.0}, {"RTX5090": 0.28, "RTX3080": 0.68}
    last_fixed = 4.0

    def stage(cut, r, w, p):
        attn = M.LayerCostModel({c: M.CardCost(c, {M.MODE_GRAPH: [(512, 1.0)]}, {M.MODE_GRAPH: [(512, 0.0)]})
                                 for c in set(CARDS3)}, CARDS3, LT).attn_counts(cut)
        c = CARDS3[r]
        return cut[r] * g[c] * w / 512 + attn[r] * k[c] * M.attn_work(w, p) + (last_fixed if r == 2 else 0.0)

    return stage, g, k, last_fixed


def _synthetic_samples(stage, cut):
    out = []
    for p in range(8192, 131072, 4096):
        for r in range(3):
            out.append(M.WidthSample(r, 512, p, stage(cut, r, 512, p), 0.0, 1, M.MODE_GRAPH))
    return out


def test_two_cuts_recover_rate_attention_and_the_last_stage_fixed_term():
    stage, g, k, last_fixed = _truth()
    probe = M.LayerCostModel({c: M.CardCost(c, {M.MODE_GRAPH: [(512, 1.0)]}, {M.MODE_GRAPH: [(512, 0.0)]})
                              for c in set(CARDS3)}, CARDS3, LT)
    fits = []
    for cut in ((43, 11, 10), (44, 10, 10)):
        lines = M.fit_width_lines(_synthetic_samples(stage, cut), widths=(512,))
        fits.append((cut, probe.attn_counts(cut), lines))
    cards, notes = M.card_costs_from_cuts(fits, CARDS3)
    assert cards["RTX5090"].gemm(512, M.MODE_GRAPH) == pytest.approx(g["RTX5090"], abs=1e-3)
    assert cards["RTX3080"].gemm(512, M.MODE_GRAPH) == pytest.approx(g["RTX3080"], abs=1e-3)
    assert cards["RTX5090"].attn_coeff(512, M.MODE_GRAPH) == pytest.approx(k["RTX5090"], abs=1e-4)
    assert cards["RTX3080"].attn_coeff(512, M.MODE_GRAPH) == pytest.approx(k["RTX3080"], abs=1e-4)
    assert cards["RTX3080"].last_fixed(512, M.MODE_GRAPH) == pytest.approx(last_fixed, abs=1e-2)
    # the 5090's first-stage fixed term is 0 in truth and separable (n0 43 vs 44): kept or folded, the
    # per-stage total is reproduced either way
    model = M.LayerCostModel(cards, CARDS3, LT)
    for cut in ((43, 11, 10), (44, 10, 10)):
        for r in range(3):
            assert model.stage_ms(cut, r, 512, 65536, M.MODE_GRAPH) == pytest.approx(
                stage(cut, r, 512, 65536), rel=1e-3)


def test_an_inseparable_or_negative_fixed_term_is_folded_at_every_width():
    stage, *_ = _truth()
    probe = M.LayerCostModel({c: M.CardCost(c, {M.MODE_GRAPH: [(512, 1.0)]}, {M.MODE_GRAPH: [(512, 0.0)]})
                              for c in set(CARDS3)}, CARDS3, LT)
    cut = (44, 10, 10)
    lines = M.fit_width_lines(_synthetic_samples(stage, cut), widths=(512,))
    cards, notes = M.card_costs_from_cuts([(cut, probe.attn_counts(cut), lines)], CARDS3)
    assert not cards["RTX5090"].first_fixed_ms, "one cut cannot separate the 5090's fixed term"
    assert any("RTX5090" in n and "folded" in n for n in notes), notes
    model = M.LayerCostModel(cards, CARDS3, LT)
    assert model.stage_ms(cut, 0, 512, 65536, M.MODE_GRAPH) == pytest.approx(stage(cut, 0, 512, 65536), rel=1e-3)


def test_the_chunk_mix_bins_one_ranks_chunks_and_the_makespan_is_the_mean_bottleneck():
    s = [M.WidthSample(0, 512, 70000, 1.0, 0.0, 1, M.MODE_GRAPH), M.WidthSample(0, 512, 71000, 1.0, 0.0, 1, M.MODE_GRAPH),
         M.WidthSample(1, 512, 70000, 1.0, 0.0, 1, M.MODE_GRAPH), M.WidthSample(0, 1024, 100, 1.0, 0.0, 1, M.MODE_EAGER)]
    mix = M.chunk_mix_from_samples(s)
    assert mix == ((512, 69632, M.MODE_GRAPH, 2.0), (1024, 4096, M.MODE_EAGER, 1.0))
    model = M.load_model(os.path.join(os.path.dirname(M.__file__), "p_stage_model_data", "27b_int8_rc12z30.json"))
    cut = (43, 11, 10)
    want = (2 * max(model.stage_ms(cut, r, 512, 69632, M.MODE_GRAPH) for r in range(3))
            + max(model.stage_ms(cut, r, 1024, 4096, M.MODE_EAGER) for r in range(3))) / 3
    assert M.weighted_makespan(model, cut, mix) == pytest.approx(want)


# -- the solver on the shipped calibration -------------------------------------------


def _pool_model():
    """N3b's P budgets (PP-CUT inputs: free=[26344, 15928, 15656] MiB) with the 27B posts."""
    from flliper.srt.pdflip import form as F

    fixed = tuple(float(x) for x in str(F.profile_constant("P_PP_STAGE_FIXED_MIB", "qwen27b")).split(","))
    return P.PhasePoolModel(
        free_mib=(26344.0, 15928.0, 15656.0), weight_mib_per_layer=363.4, kv_mib_per_token_per_attn_layer=2048 / P.MIB,
        arming_floor_mib=(1229.0,) * 3, stage_fixed_mib=fixed, activation_reserve_mib=0.0, corridor_holdback_mib=0.0,
        mamba_mib_per_linear_layer_per_slot=float(F.profile_constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", "qwen27b")),
        mamba_slots=24, prefill_graph_pool_mib=(0.156 * 1024,) * 3, extra_cell_bytes_by_stage=(0, 0, 10240),
        zero_posts_acknowledged=("mamba pre-capture reserve", "speculative intermediate state", "GGUF dequant scratch",
                                 "prefill activation reserve"))


def _solve(stage_model=None, family_cost=None):
    return solve_launch_cut(
        layer_families=FAMILIES, incumbent_layers=(32, 18, 14), measured_ms_per_layer=(8.1, 35.16, 33.59),
        measured_provenance="test", card_names=("SYNTHETIC A", "SYNTHETIC B", "SYNTHETIC B"),
        pool_model=_pool_model(), cap_tokens=262144, family_cost=family_cost, design_prefix_tokens=DESIGN_PREFIX,
        per_pair_crossing_ms={(0, 1): 4.11, (1, 2): 4.11, (0, 2): 2.31, (1, 0): 4.11, (2, 1): 4.11, (2, 0): 2.31},
        enumerate_gapped=False, objective="makespan", pool_floor=264192,
        stage_model=stage_model, chunk_mix=MIX if stage_model is not None else None,
        stage_model_provenance="27b_int8_rc12z30")


def _shipped_model():
    return M.load_model(os.path.join(os.path.dirname(M.__file__), "p_stage_model_data", "27b_int8_rc12z30.json"))


def _rank_43_before_44(decision):
    by = {tuple(c.layers): c for c in (decision.servable or decision.ranked)}
    a, b = by[(43, 11, 10)], by[(44, 10, 10)]
    ta, tb = a.makespan_ms + a.crossing_ms, b.makespan_ms + b.crossing_ms
    assert ta * 1.03 <= tb, "43,11,10 must price at least 3 %% under 44,10,10: %.1f vs %.1f" % (ta, tb)
    assert tuple(decision.makespan.layers) != (44, 10, 10), decision.makespan.layers


def test_the_stage_model_ranks_43_11_10_ahead_of_44_10_10_and_does_not_ship_44():
    d = _solve(stage_model=_shipped_model())
    _rank_43_before_44(d)
    assert "STAGE MODEL" in d.cost_provenance


def test_the_old_family_card_rate_mutant_turns_the_ranking_test_red():
    d = _solve(family_cost=OLD_FAMILY)
    by = {tuple(c.layers): c for c in (d.servable or d.ranked)}
    assert by[(44, 10, 10)].makespan_ms == pytest.approx(425.8, abs=0.2), "the N3b number is reproduced"
    with pytest.raises(AssertionError):
        _rank_43_before_44(d)


# -- the launcher's resolution --------------------------------------------------------


def test_auto_resolves_only_for_the_27b_int8_checkpoint():
    from flliper.srt.pdflip import form as F
    from flliper.srt.pdflip import launcher as L

    from flliper.srt.pdflip import card_identity as CI

    ref = CI.REFERENCE_INVENTORY  # HW-GENERIC 1002: the cards are part of the key
    ckpt = F.profile_row("qwen27b").formats["int8"].checkpoint
    model, prov = L.resolve_pp_cut_stage_model("auto", "qwen27b", ckpt, inventory=ref)
    assert model is not None and "27b_int8_rc12z30" in prov
    assert L.resolve_pp_cut_stage_model("off", "qwen27b", ckpt, inventory=ref)[0] is None
    nvfp4 = F.profile_row("qwen27b").formats["nvfp4"].checkpoint
    assert L.resolve_pp_cut_stage_model("auto", "qwen27b", nvfp4, inventory=ref)[0] is None
    assert L.resolve_pp_cut_stage_model("auto", "nextflash", ckpt, inventory=ref)[0] is None
    # a foreign inventory gets no model measured on the reference cards: named
    foreign = ("RTX3090/24576MiB/sm86",) * 3
    model, prov = L.resolve_pp_cut_stage_model("auto", "qwen27b", ckpt, inventory=foreign)
    assert model is None and "RTX3090/24576MiB/sm86" in prov and "previous pricing" in prov
    assert L.build_parser().parse_args(["--tree", "/t", "--tag", "t"]).pp_cut_stage_model == "auto"


def test_without_a_predecessor_log_the_mix_is_one_named_cell():
    from flliper.srt.pdflip import launcher as L

    mix, prov = L.pp_cut_chunk_mix(None, 2048, 21615, 512)
    assert mix == ((2048, 21615, M.MODE_EAGER, 1.0),) and prov.startswith("FALLBACK")
