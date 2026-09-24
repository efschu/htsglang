"""fnFL2 H41 (Task #114): the P card per stage as a function of the chunk.

fnFL2x121/x122 booted the 5090 stage (PP0) at residency 0.45/0.40 with chunk
16384; the dry run said PASST and both died in the FIRST chunk's forward
(CUDA OOM, 320/384 MiB). The #140 ceiling and the pool model decide the
BUDGET; what fills the CARD in a chunk forward was in no post. The P-KARTE
prices it from measurement: headroom = cap - peak - private_free at the
binding point of a reference boot (x145/x146), shifted by the expert buffer,
the KV pool, the chunk transient T_s(chunk) (measured support points, linear
between, W131 outside) and the draft.

The fixtures under fixtures/p_card_h41/ are the P logs' own lines (verbatim,
filtered: chunk, [vram-peak], WEG2-GRAPH-POOL, KV pool sizing, the first MoE
buffer line per stage, the draft-group line, the OOM text; for x145/x146/
x149/x150/x160 also ``#969 EXTENT``, ``Scheduler hit an exception``,
``WEG2-ARENA-WRITE`` and ``WEG2-SLEEP-LMEM``).

H41c: fnFL2x149 (FR_P[0] 0.351) died in PP0's SECOND chunk of the
arena run-write's local memory (H47: cap -1457 MiB after 'WEG2-ARENA-WRITE
... mode=run'), not of a row price: at chunk 0 it sat on the row image.

H41d: fnFL2x160 (tree 76ce5580d4, FR_P 0.332,0.605) ran the 97k AND a
259441-token prompt (16 chunks): the peak grew 3 x ~330 MiB and then never
again (PP0 4252 -> 3903 -> 3582 -> 3262, no new high-water to chunk 15). The
chunk term SATURATES; the 262k edge is the 97k edge. x146/x150 stay as the
historical reference on which x149/x121/x122 are judged.

H59: fnFL2x164 (2e2aac1e4e, 4 seats, FR_P 0.410,0.712 = the H41d edge) died in
PP0's FOURTH chunk (320 MiB asked, 287 free, 627 MiB split fragmentation). Not
the transient (in the planner's definition 3663/3265/3263 MiB, under the
support) and not the seats (x162 -> x163, 1 -> 4 seats: +3 MiB allocated, -3
private_free): the CO-TENANT. The sleeping D rank on the same card held 1782
MiB in x164's first flip against 1698 in x160's, and creeps to 2004 over later
flips (x165). The x160 reference carried 1698 in its cap. The shipped reference
is now the 4-seat form (x163 + x164 + x165, D logs beside the P logs), the card
books the measured co-tenant maximum, and the edge is 238/395 rows (0.402/
0.708). x160 stays as P_CARD_REFERENCE_FNFL2_X160 (1 seat, history).

The H55 window's ``transient_mib`` is ``peak - start``; the planner's is ``peak
- allocated AFTER the chunk``. The difference is what the chunk leaves behind
(x164 chunk 0: 411/649/651 MiB, the chunk-growth term), which is exactly the
"377/627/647 MiB larger transient" the H55 reader printed.

Hermetic: no GPU, no torch device.
"""

import os
import types

import msgspec
import pytest

from sglang.srt.planner import p_card_chunk as pc

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "p_card_h41")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
#: 1237.5234375 MiB per layer / 512 experts, from the checkpoint's safetensors
#: headers (pp_cut.checkpoint_weight_terms; dry runs print "2.42 MiB/row").
ROW_MIB = 2.4170379638671875
CUT = (29, 11, 8)
LRU = (32, 32, 32)
#: The H25 draft post on P's last stage, as the dry run prints it
#: (dry_fnFL2x145.log: "freed_mib=3422 (weights_draft 2799 ... producer
#: transient 623)").
DRAFT_MIB = 3422.0

SUPPORT_BOOTS = (
    "fnFL2x130", "fnFL2x134", "fnFL2x12", "fnFL2x14", "fnFL2x101", "fnFL2x107",
    "fnFL2x113", "fnFL2x116", "fnFL2x118", "fnFL2x141", "fnFL2x145", "fnFL2x146",
)


def _boot(name):
    with open(os.path.join(FIX, name + ".P.lines")) as fh:
        return name, fh.read()


def _dlog(name):
    with open(os.path.join(FIX, name + ".D.lines")) as fh:
        return name, fh.read()


X160 = pc.P_CARD_REFERENCE_FNFL2_X160
FOUR_SEAT_BOOTS = ("fnFL2x163", "fnFL2x164", "fnFL2x165")
CO_TENANT_BOOTS = ("fnFL2x160", "fnFL2x163", "fnFL2x164", "fnFL2x165")


def _form_of(name):
    """(fractions, kv_mib, draft_on_p, chunk) as the boot's own log states them."""
    obs = pc.observe_p_log(_boot(name)[1])
    fr = tuple(obs["fraction"][s] for s in range(3))
    kv = tuple(obs["tokens"][s] * obs["cell"][s] / pc.MIB for s in range(3))
    return fr, kv, bool(obs["draft_on_p"][0]), obs["chunk"][0]


def _solve(fr, kv, draft_on_p, chunk, **kw):
    return pc.solve_p_card(
        reference=kw.pop("reference", pc.P_CARD_REFERENCE_FNFL2),
        support=kw.pop("support", pc.P_TRANSIENT_SUPPORT_FNFL2),
        fractions=fr, lru_rows=LRU, stage_layers=CUT, chunk=chunk, kv_mib=kv,
        num_experts=512, row_mib=ROW_MIB, draft_on_p=draft_on_p,
        draft_mib_last_stage=DRAFT_MIB, cards=("nvml1", "nvml0", "nvml2"), **kw,
    )


class TestMeasuredSupport:
    def test_the_shipped_support_is_the_logs_own_measurement(self):
        got = pc.transient_support_from_logs(
            [_boot(b) for b in SUPPORT_BOOTS], n_stages=3, model=MODEL
        )
        assert got == pc.P_TRANSIENT_SUPPORT_FNFL2

    def test_pp2_is_the_target_forward_not_the_draft_forward(self):
        # x118 PP2 prints two extend lines per chunk: 13.09-9.91 (target) and
        # 12.34-9.91 (draft, after a peak reset). #114 read the second.
        obs = pc.observe_p_log(_boot("fnFL2x118")[1])
        # max over the target lines: 13.09-9.91 = 3.18, 13.42-10.23 = 3.19
        assert abs(obs["transient"][2] / 1024.0 - 3.19) < 0.006

    def test_exact_at_the_points_and_linear_between(self):
        sup = pc.P_TRANSIENT_SUPPORT_FNFL2
        for p in sup.points:
            for s in range(3):
                assert pc.transient_mib(sup, s, p.chunk)[0] == p.mib[s]
        v, src = pc.transient_mib(sup, 0, 12288)
        assert v == pytest.approx((1863.7 + 3696.6) / 2.0)
        assert src.startswith("interpoliert 8192")

    @pytest.mark.parametrize("chunk", [256, 511, 16385, 32768])
    def test_outside_the_support_is_refused_by_name(self, chunk):
        with pytest.raises(pc.PChunkUnmeasured, match="W131 Weg2PChunkTransientUnmeasured"):
            pc.transient_mib(pc.P_TRANSIENT_SUPPORT_FNFL2, 0, chunk)

    def test_the_activation_line_names_chunk_stage_value_and_boots(self):
        lines = pc.activation_lines(pc.P_TRANSIENT_SUPPORT_FNFL2, 16384)
        assert lines[0].startswith(
            "PP-CUT ACTIVATION chunk=16384 stage0 transient_mib=3697 "
            "source=gemessen=fnFL2x118+fnFL2x141+fnFL2x145+fnFL2x146"
        )
        assert len(lines) == 3


class TestCardReference:
    def test_the_x160_reference_is_x160s_own_measurement(self):
        got = pc.p_card_reference_from_logs(
            [_boot("fnFL2x160")], stage_layers=CUT, row_mib=ROW_MIB,
            support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
            d_logs={"fnFL2x160": _dlog("fnFL2x160")[1]},
        )
        assert got == X160
        assert got.seats == 1 and got.co_tenant_mib == (1698.0, 768.0, 766.0)

    def test_the_shipped_card_reference_is_the_4_seat_forms_own_measurement(self):
        got = pc.p_card_reference_from_logs(
            [_boot(b) for b in FOUR_SEAT_BOOTS], stage_layers=CUT, row_mib=ROW_MIB,
            support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
            d_logs={b: _dlog(b)[1] for b in FOUR_SEAT_BOOTS},
        )
        assert got == pc.P_CARD_REFERENCE_FNFL2
        assert got.seats == 4
        # growth rate from x164 (335 MiB per 16k), saturation from x163/x165
        assert got.growth_mib_per_token[0] * 16384 == pytest.approx(335.0, abs=0.5)
        assert got.growth_measured_chunks == (3, 3, 1)

    def test_the_minimum_is_taken_over_k0_plus_co_tenant(self):
        # x164's chunk-0 point lies 72 MiB under x163's (23925.1 vs 23997.0), but
        # its co-tenant held 84 MiB more: normalised, x163 is the binding boot.
        boots = [_boot("fnFL2x163"), _boot("fnFL2x164")]
        with_co = pc.p_card_reference_from_logs(
            boots, stage_layers=CUT, row_mib=ROW_MIB, support=pc.P_TRANSIENT_SUPPORT_FNFL2,
            model=MODEL, d_logs={b: _dlog(b)[1] for b, _t in boots},
        )
        blind = pc.p_card_reference_from_logs(
            boots, stage_layers=CUT, row_mib=ROW_MIB, support=pc.P_TRANSIENT_SUPPORT_FNFL2,
            model=MODEL,
        )
        assert with_co.headroom0_mib[0] == pytest.approx(23997.0, abs=0.1)
        assert blind.headroom0_mib[0] == pytest.approx(23925.4, abs=0.2)
        assert blind.co_tenant_mib == ()

    def test_a_reference_does_not_mix_seats(self):
        with pytest.raises(ValueError, match="Sitze"):
            pc.p_card_reference_from_logs(
                [_boot("fnFL2x160"), _boot("fnFL2x163")], stage_layers=CUT,
                row_mib=ROW_MIB, support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
            )

    def test_the_shipped_co_tenant_span_is_the_logs_own_measurement(self):
        got = pc.co_tenant_span_from_logs([_dlog(b) for b in CO_TENANT_BOOTS], n_stages=3)
        assert got.max_mib == pc.P_CARD_CO_TENANT_FNFL2.max_mib == (2004.0, 838.0, 836.0)
        # TP0 jumps in the second flip and then creeps (x165)
        tp0 = [m for _t, m in pc.co_tenant_by_flip(_dlog("fnFL2x165")[1])[0]]
        assert tp0 == [1698.0, 1956.0, 1982.0, 1994.0, 2004.0]
        # x164's first flip already held 84 MiB more than x160's / x163's
        assert pc.co_tenant_by_flip(_dlog("fnFL2x164")[1])[0][0][1] == 1782.0

    def test_co_tenant_at_the_chunk_0_point_is_the_last_release_before_it(self):
        d = _dlog("fnFL2x165")[1]
        assert pc.co_tenant_at(d, 0, "2026-09-24 19:37:34") == 1698.0
        assert pc.co_tenant_at(d, 0, "2026-09-24 19:39:00") == 1994.0
        assert pc.co_tenant_at(d, 0, "2026-09-24 19:30:00") is None

    def test_the_historical_reference_is_x146_x150_with_x149_row_price(self):
        got = pc.p_card_reference_from_logs(
            [_boot("fnFL2x146"), _boot("fnFL2x150")], stage_layers=CUT,
            row_mib=ROW_MIB, support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
            over_boots=[_boot("fnFL2x149")],
        )
        assert got == pc.P_CARD_REFERENCE_FNFL2_X150
        # (4584 - 1300) / 46 = 71.4 on that tree; image 70.1
        assert got.row_card_mib[0] == pytest.approx(71.4, abs=0.05)

    def test_x160_growth_saturates_after_chunk_3_over_16_chunks(self):
        h = pc.headroom_by_chunk_index(_boot("fnFL2x160")[1])
        # PP0 4252 -> 3903 -> 3582 -> 3262, then no new high-water to chunk 15
        assert h[0] == {0: 4252.0, 1: 3903.0, 2: 3582.0, 3: 3262.0}
        assert h[1] == {0: 2859.0, 1: 2524.0, 2: 2204.0, 3: 1883.0}
        assert h[2] == {0: 4057.0, 1: 3726.0}
        assert pc.longest_prompt_tokens(_boot("fnFL2x160")[1]) == 259441
        ref = X160
        assert ref.growth_measured_chunks == (3, 3, 1)
        assert ref.growth_mib_per_token[0] * 16384 * 3 == pytest.approx(990, abs=1)

    def test_the_tree_changed_the_private_pools(self):
        # 2394 -> 225 MiB private_free on PP0 between x150 and x160
        assert pc.P_CARD_REFERENCE_FNFL2_X150.private_free_mib[0] == 2394.0
        assert X160.private_free_mib[0] == 225.0
        assert X160.lmem_mib[0] == 0.0
        assert pc.P_CARD_REFERENCE_FNFL2.lmem_mib[0] == 0.0

    def test_x149_death_is_the_lmem_case_not_a_row_price(self):
        d = pc.death_from_log(*_boot("fnFL2x149"))
        assert (d.stage, d.chunk_index, d.buffer_rows, d.cause) == (0, 1, 212, "lmem")
        assert d.cap_before_mib - d.cap_mib == pytest.approx(1457.4, abs=0.5)
        assert d.used_mib == pytest.approx(24443.7, abs=0.5)
        assert d.headroom_without_lmem_mib == pytest.approx(1656.3, abs=0.5)

    def test_x149_first_chunk_sat_on_the_row_image(self):
        h149 = pc.headroom_by_chunk_index(_boot("fnFL2x149")[1])[0][0]
        h146 = pc.headroom_by_chunk_index(_boot("fnFL2x146")[1])[0][0]
        assert abs(h146 - 46 * 29 * ROW_MIB - 73 - h149) < 5.0

    def test_near_oom_mirror_is_the_corridor_law(self):
        from sglang.srt.managers import corridor_guard

        assert pc.NEAR_OOM_MIB == float(corridor_guard.NEAR_OOM_MIB)

    def test_lmem_law_constants(self):
        assert pc.P_LMEM_RUN_WRITE_MIB == (1457.0, 150.0, 48.0)


H146 = (0.26, 0.45, 0.733887)
H149 = (0.351, 0.642, 0.733887)
H160 = (0.332, 0.605, 0.733887)
KV_H25 = (1904.0, 816.0, 544.0)
OLD = pc.P_CARD_REFERENCE_FNFL2_X150


class TestVerdicts:
    @pytest.mark.parametrize("prompt", [97841, 262144])
    def test_x160_form_passes_with_its_measured_last_chunk_headroom(self, prompt):
        fits = _solve(H160, KV_H25, False, 16384, prompt_tokens=prompt, reference=X160)
        assert [f.verdict for f in fits] == ["PASST"] * 3
        # the reference's own last-chunk high-water: 3262 / 1883 (+2 LMEM) / 3726 (+2)
        assert fits[0].headroom_mib == pytest.approx(3262, abs=1)
        assert fits[1].headroom_mib == pytest.approx(1885, abs=1)
        assert not any(f.growth_extrapolated for f in fits)

    def test_262k_edge_equals_97k_edge_on_x160(self):
        a = _solve(H160, KV_H25, False, 16384, prompt_tokens=97841, reference=X160)
        b = _solve(H160, KV_H25, False, 16384, prompt_tokens=262144, reference=X160)
        assert [f.ceiling_fraction for f in a] == [f.ceiling_fraction for f in b]
        assert [f.ceiling_fraction for f in a] == [0.410, 0.712, 0.996]
        assert [f.ceiling_max_rows for f in a] == [242, 397, 580]

    def test_the_h41d_edge_passes_on_x160_and_one_row_more_dies(self):
        edge = _solve((0.410, 0.712, 0.733887), KV_H25, False, 16384, prompt_tokens=262144,
                      reference=X160)
        assert (edge[0].buffer_rows, edge[1].buffer_rows) == (242, 397)
        assert not any(f.refused for f in edge)
        over = _solve((0.412, 0.714, 0.733887), KV_H25, False, 16384, prompt_tokens=262144,
                      reference=X160)
        assert over[0].buffer_rows == 243 and over[0].refused
        assert over[1].buffer_rows == 398 and over[1].refused

    def test_on_its_own_tree_x149_is_refused(self):
        # historical reference (tree of x149, private_free 2394): saturated growth
        # 1300 - 3 x 320.7 = 338 < 400 even with the LMEM fix; without it -1119
        fits = _solve(H149, KV_H25, False, 16384, reference=OLD)
        assert fits[0].refused and fits[0].headroom_mib == pytest.approx(338, abs=2)
        nofix = _solve(H149, KV_H25, False, 16384, reference=OLD, lmem_fixed=False)
        assert nofix[0].headroom_mib < fits[0].headroom_mib - 1400
        text = pc.p_card_refusal_text(fits, OLD, chunk=16384)
        assert text.startswith("W132 Weg2PCardChunkOom (P, chunk 16384")

    def test_mutant_without_the_chunk_term_lets_x149_through_on_its_tree(self):
        fits = _solve(H149, KV_H25, False, 16384, reference=OLD, use_growth=False)
        assert not any(f.refused for f in fits)
        assert fits[0].headroom_mib == pytest.approx(1300, abs=2)

    def test_mutant_without_the_chunk_term_raises_the_x160_edge(self):
        with_term = _solve(H160, KV_H25, False, 16384, reference=X160)[0].ceiling_max_rows
        without = _solve(H160, KV_H25, False, 16384, use_growth=False,
                         reference=X160)[0].ceiling_max_rows
        assert without - with_term == 14  # 990 MiB / 70.1 per row

    def test_a_longer_prompt_than_measured_is_named_extrapolated(self):
        fits = _solve(H160, KV_H25, False, 16384, prompt_tokens=300000, reference=X160)
        assert all(f.growth_extrapolated for f in fits)
        assert fits[0].headroom_mib == pytest.approx(3262, abs=1)  # still saturated

    @pytest.mark.parametrize("boot,f0", [("fnFL2x121", 0.45), ("fnFL2x122", 0.40)])
    def test_the_oom_boots_are_refused_on_their_tree(self, boot, f0):
        fr, kv, draft, chunk = _form_of(boot)
        assert fr[0] == f0 and draft is True and chunk == 16384
        assert "OutOfMemoryError" in _boot(boot)[1]
        fits = _solve(fr, kv, draft, chunk, reference=OLD)
        assert fits[0].refused
        text = pc.p_card_refusal_text(fits, OLD, chunk=chunk)
        assert "stage0 (nvml1) f %.4f" % f0 in text

    def test_without_the_lmem_fix_the_run_write_is_priced(self):
        fits = _solve(H160, KV_H25, False, 16384, lmem_fixed=False, reference=X160)
        assert fits[0].headroom_mib == pytest.approx(3262 - 1457, abs=1)
        assert [f.ceiling_fraction for f in fits][0] == 0.371

    def test_a_narrower_chunk_buys_rows_by_the_measured_difference(self):
        at16 = _solve(H160, KV_H25, False, 16384, use_growth=False, reference=X160)[0]
        at8 = _solve(H160, KV_H25, False, 8192, use_growth=False, reference=X160)[0]
        assert at8.headroom_mib - at16.headroom_mib == pytest.approx(3696.6 - 1863.7)

    def test_mutant_without_the_chunk_transient_lets_x122_through(self, monkeypatch):
        fr, kv, draft, chunk = _form_of("fnFL2x122")
        image = msgspec.structs.replace(OLD, row_card_mib=())
        kw = dict(reference=image, use_growth=False)
        assert _solve(fr, kv, draft, chunk, **kw)[0].refused
        monkeypatch.setattr(pc, "transient_mib", lambda sup, s, c: (1024.0, "mutant"))
        assert not _solve(fr, kv, draft, chunk, **kw)[0].refused

    def test_another_cut_is_named_not_priced(self):
        with pytest.raises(ValueError, match="Schnitt"):
            pc.solve_p_card(
                reference=pc.P_CARD_REFERENCE_FNFL2, support=pc.P_TRANSIENT_SUPPORT_FNFL2,
                fractions=(0.26, 0.45, 0.73), lru_rows=LRU, stage_layers=(33, 7, 8),
                chunk=16384, kv_mib=KV_H25, num_experts=512, row_mib=ROW_MIB,
            )


H164 = (0.410, 0.712, 0.733887)
H59_EDGE = (0.402, 0.708, 0.733887)
FOUR = pc.P_CARD_REFERENCE_FNFL2
SPAN = pc.P_CARD_CO_TENANT_FNFL2


def _span(*mib):
    return pc.CoTenantSpan(max_mib=tuple(float(x) for x in mib), source="test")


class TestFourSeatCardH59:
    """fnFL2x164 died where the 4-seat card says; the edge is 238/395 rows."""

    def test_x164_death_is_a_card_death_in_its_fourth_chunk(self):
        d = pc.death_from_log(*_boot("fnFL2x164"))
        assert (d.stage, d.chunk_index, d.buffer_rows, d.cause) == (0, 3, 242, "card")
        # 320 MiB asked with 594 MiB of torch headroom left: 627 MiB of it
        # were split blocks the retry could not release (card free 287)
        assert d.request_mib == 320.0
        assert d.headroom_mib == pytest.approx(594.2, abs=0.2)
        assert "627.40 MiB is reserved by PyTorch but unallocated" in _boot("fnFL2x164")[1]

    @pytest.mark.parametrize("last_chunk,measured", [(0, 1362.0), (1, 1013.0), (2, 692.0)])
    def test_the_card_walks_x164s_own_measured_path(self, last_chunk, measured):
        # x164's own co-tenant (1782 in its first flip) against the reference's 1698
        fits = _solve(H164, KV_H25, False, 16384, prompt_tokens=16384 * (last_chunk + 1),
                      co_tenant=_span(1782, 768, 766), seats=4)
        assert fits[0].headroom_mib == pytest.approx(measured, abs=15.0)

    def test_the_card_refuses_x164_at_its_fourth_chunk(self):
        fits = _solve(H164, KV_H25, False, 16384, prompt_tokens=97841,
                      co_tenant=_span(1782, 768, 766), seats=4)
        assert fits[0].refused and fits[0].headroom_mib == pytest.approx(345, abs=2)
        assert fits[0].co_tenant_mib == 84.0

    def test_mutant_without_the_co_tenant_term_lets_x164_through(self):
        # the H41d answer: the reference's own cap, nothing for a fuller D-TP0
        fits = _solve(H164, KV_H25, False, 16384, prompt_tokens=97841, seats=4)
        assert not fits[0].refused
        assert fits[0].headroom_mib == pytest.approx(429, abs=2)

    def test_the_4_seat_edge_with_the_measured_co_tenant_is_238_395(self):
        fits = _solve(H59_EDGE, KV_H25, False, 16384, prompt_tokens=262144,
                      co_tenant=SPAN, seats=4)
        assert [f.buffer_rows for f in fits] == [238, 395, 408]
        assert not any(f.refused for f in fits)
        assert [f.ceiling_max_rows for f in fits][:2] == [238, 395]
        assert [f.ceiling_fraction for f in fits] == [0.402, 0.708, 0.996]
        # the co-tenant term: 2004 - 1698 on PP0, 838 - 768 on PP1
        assert (fits[0].co_tenant_mib, fits[1].co_tenant_mib) == (306.0, 70.0)
        assert fits[0].headroom_mib == pytest.approx(403, abs=2)

    def test_one_row_over_the_4_seat_edge_dies(self):
        over = _solve((0.404, 0.710, 0.733887), KV_H25, False, 16384, prompt_tokens=262144,
                      co_tenant=SPAN, seats=4)
        assert (over[0].buffer_rows, over[1].buffer_rows) == (239, 396)
        assert over[0].refused and over[1].refused

    def test_the_x163_form_passes_with_room(self):
        fits = _solve((0.36, 0.64, 0.733887), KV_H25, False, 16384, prompt_tokens=262144,
                      co_tenant=SPAN, seats=4)
        assert not any(f.refused for f in fits)
        # its own saturated high-water 2196 (x163/x165), 15 MiB conservative from
        # x164's growth rate, minus the co-tenant span 306
        assert fits[0].headroom_mib == pytest.approx(2196 - 15 - 306, abs=3)

    def test_more_seats_than_the_reference_is_named_not_priced(self):
        with pytest.raises(pc.PCardFormUnmeasured, match="8 Sitze"):
            _solve(H59_EDGE, KV_H25, False, 16384, seats=8)
        # fewer seats are covered (x162 -> x163: 1 -> 4 seats cost +3/-3 MiB)
        assert not any(f.refused for f in _solve(H59_EDGE, KV_H25, False, 16384, seats=1,
                                                 co_tenant=SPAN))

    def test_the_card_line_names_seats_and_co_tenant(self):
        fit = _solve(H59_EDGE, KV_H25, False, 16384, co_tenant=SPAN, seats=4)[0]
        line = pc.describe_p_card(fit, FOUR)
        assert "(4 Sitze, hier 4)" in line
        assert "Mitbewohner D-TP0 schlafend 1698 am Referenzpunkt, gemessen bis 2004 -> 306 MiB" in line
        blind = pc.describe_p_card(_solve(H59_EDGE, KV_H25, False, 16384, reference=X160)[0], X160)
        assert "Hochstand ungemessen (Term 0)" in blind


class TestTransientDefinitionH59:
    """The H55 window transient is peak - START; the planner's is peak - END."""

    def test_the_gap_is_what_the_chunk_leaves_behind(self):
        w = pc.chunk_windows(_boot("fnFL2x164")[1])
        first = [w[s][0] for s in range(3)]
        assert [x.transient_h55_mib for x in first] == [4074.0, 3913.0, 3914.0]
        assert [x.persist_mib for x in first] == [411.0, 649.0, 651.0]
        assert [x.transient_planner_mib for x in first] == [3663.0, 3264.0, 3263.0]
        support = pc.transient_vector_mib(pc.P_TRANSIENT_SUPPORT_FNFL2, 16384)
        # the planner-definition transient stays under the support point ...
        assert all(x.transient_planner_mib <= t for x, t in zip(first, support))
        # ... and the "377/627/647 MiB larger" is the H55 number against it
        gap = [round(x.transient_h55_mib - t) for x, t in zip(first, support)]
        assert gap == [377, 626, 647]

    def test_the_vram_peak_line_agrees_with_the_window(self):
        # [vram-peak] prints the planner definition itself: 3.58 / 3.19 / 3.19 GiB
        obs = pc.observe_p_log(_boot("fnFL2x164")[1])
        assert [round(obs["transient"][s] / 1024.0, 2) for s in range(3)] == [3.58, 3.19, 3.19]

    def test_persist_saturates_after_chunk_3_in_the_4_seat_form(self):
        w = pc.chunk_windows(_boot("fnFL2x165")[1])
        full = [x for x in w[0] if x.rows == 16384]
        assert [x.persist_mib for x in full] == [411.0, 349.0, 321.0, 320.0, 1.0]
        # once saturated the H55 number IS the planner's (x165 chunk 4: 3664)
        assert full[4].transient_h55_mib == pytest.approx(full[4].transient_planner_mib + 1)

    def test_card_free_under_400_is_not_death(self):
        # x165 (the x163 form, 217 rows) ran all six 16k chunks without OOM
        # with 7 / 95 MiB card free at chunk ends: the allocator's cache fills
        # the card. NVML rest is not the P card's death line.
        text = _boot("fnFL2x165")[1]
        assert "OutOfMemoryError" not in text
        free = [x.card_free_mib for x in pc.chunk_windows(text)[0] if x.rows == 16384]
        assert min(free) < 100.0


class TestLauncherSeam:
    def _ns(self, draft="off", prompt=0, **kw):
        ns = types.SimpleNamespace(
            p_card_reference_logs="", draft_kv_on_p=draft, p_card_prompt_tokens=prompt,
            extra_p="--speculative-draft-model-path /nonexistent/draft", extra_d="",
        )
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def _cards(self):
        return [types.SimpleNamespace(nvml_index=i) for i in (1, 0, 2)]

    def _run(self, ns, fr, kv, chunk=16384):
        from sglang.srt.weg2 import launcher

        lines = []
        launcher.p_card_verdict(
            ns, self._cards(), lines.append, model="/m/" + MODEL, chunk_tokens=chunk,
            fracs=fr, lru_rows=LRU, stage_layers=CUT, kv_mib=kv, num_experts=512,
            row_mib=ROW_MIB,
        )
        return lines

    def test_x160_form_passes_at_the_full_context(self):
        lines = self._run(self._ns(), H160, KV_H25)
        assert sum(l.startswith("PP-CUT ACTIVATION chunk=16384 stage") for l in lines) == 3
        card = [l for l in lines if l.startswith("PP-CUT P-KARTE stage")]
        assert len(card) == 3
        assert all("prompt=262144" in l and "-> PASST |" in l for l in card)
        assert all("saettigt nach Chunk" in l for l in card)
        assert not any("BEFUND" in l for l in lines)

    def test_one_row_over_the_edge_is_refused_w132(self):
        from sglang.srt.weg2 import launcher

        with pytest.raises(launcher.Weg2LaunchRefused, match="W132 Weg2PCardChunkOom"):
            self._run(self._ns(), (0.412, 0.605, 0.733887), KV_H25)

    def test_x164s_form_is_refused_and_the_edge_named(self):
        from sglang.srt.weg2 import launcher

        with pytest.raises(launcher.Weg2LaunchRefused) as ei:
            self._run(self._ns(p_bs=4, extra_p="--max-running-requests 4"), H164, KV_H25)
        assert "W132 Weg2PCardChunkOom" in str(ei.value)
        assert "Groesste tragbare Fraction je Stufe: 0.402,0.708,0.996" in str(ei.value)

    def test_the_4_seat_edge_passes_and_names_the_co_tenant(self):
        lines = self._run(self._ns(p_bs=4, extra_p="--max-running-requests 4"), H59_EDGE, KV_H25)
        card = [l for l in lines if l.startswith("PP-CUT P-KARTE stage")]
        assert all("-> PASST |" in l for l in card)
        assert "gemessen bis 2004 -> 306 MiB" in card[0]
        assert "(4 Sitze, hier 4)" in card[0]

    def test_more_seats_than_measured_entfaellt_by_name(self):
        lines = self._run(self._ns(p_bs=4, extra_p="--max-running-requests 8"), H59_EDGE, KV_H25)
        assert any("ENTFAELLT" in l and "8 Sitze" in l for l in lines)

    def test_reference_logs_bring_their_d_log_as_co_tenant(self, tmp_path):
        for b in FOUR_SEAT_BOOTS:
            (tmp_path / (b + ".P.log")).write_text(_boot(b)[1])
            (tmp_path / (b + ".D.log")).write_text(_dlog(b)[1])
        refs = ",".join(str(tmp_path / (b + ".P.log")) for b in FOUR_SEAT_BOOTS)
        lines = self._run(self._ns(p_card_reference_logs=refs, p_bs=4), H59_EDGE, KV_H25)
        card = [l for l in lines if l.startswith("PP-CUT P-KARTE stage")]
        assert len(card) == 3 and all("-> PASST |" in l for l in card)
        assert "fnFL2x163.P.log + fnFL2x164.P.log + fnFL2x165.P.log (4 Sitze" in card[0]
        assert "schlafend 1698 am Referenzpunkt, gemessen bis 2004" in card[0]

    def test_unmeasured_chunk_is_refused_w131(self):
        from sglang.srt.weg2 import launcher

        with pytest.raises(launcher.Weg2LaunchRefused, match="W131"):
            self._run(self._ns(), H160, KV_H25, chunk=32768)

    def test_other_model_is_named_not_priced(self):
        from sglang.srt.weg2 import launcher

        lines = []
        launcher.p_card_verdict(
            self._ns(), self._cards(), lines.append, model="/m/Qwen3.8-27B",
            chunk_tokens=16384, fracs=(0.3, 0.3, 0.3), lru_rows=LRU, stage_layers=CUT,
            kv_mib=(1, 1, 1), num_experts=512, row_mib=ROW_MIB,
        )
        assert lines and "ENTFAELLT" in lines[0]
