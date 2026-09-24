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
chunk term SATURATES; the 262k edge is the 97k edge. x160 is the shipped
reference (private_free 225 instead of 2394 MiB on this tree); x146/x150 stay
as the historical reference on which x149/x121/x122 are judged.

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
    def test_the_shipped_card_reference_is_x160s_own_measurement(self):
        got = pc.p_card_reference_from_logs(
            [_boot("fnFL2x160")], stage_layers=CUT, row_mib=ROW_MIB,
            support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
        )
        assert got == pc.P_CARD_REFERENCE_FNFL2

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
        ref = pc.P_CARD_REFERENCE_FNFL2
        assert ref.growth_measured_chunks == (3, 3, 1)
        assert ref.growth_mib_per_token[0] * 16384 * 3 == pytest.approx(990, abs=1)

    def test_the_tree_changed_the_private_pools(self):
        # 2394 -> 225 MiB private_free on PP0 between x150 and x160
        assert pc.P_CARD_REFERENCE_FNFL2_X150.private_free_mib[0] == 2394.0
        assert pc.P_CARD_REFERENCE_FNFL2.private_free_mib[0] == 225.0
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
        fits = _solve(H160, KV_H25, False, 16384, prompt_tokens=prompt)
        assert [f.verdict for f in fits] == ["PASST"] * 3
        # the reference's own last-chunk high-water: 3262 / 1883 (+2 LMEM) / 3726 (+2)
        assert fits[0].headroom_mib == pytest.approx(3262, abs=1)
        assert fits[1].headroom_mib == pytest.approx(1885, abs=1)
        assert not any(f.growth_extrapolated for f in fits)

    def test_262k_edge_equals_97k_edge(self):
        a = _solve(H160, KV_H25, False, 16384, prompt_tokens=97841)
        b = _solve(H160, KV_H25, False, 16384, prompt_tokens=262144)
        assert [f.ceiling_fraction for f in a] == [f.ceiling_fraction for f in b]
        assert [f.ceiling_fraction for f in a] == [0.410, 0.712, 0.996]
        assert [f.ceiling_max_rows for f in a] == [242, 397, 580]

    def test_the_edge_passes_and_one_row_more_dies(self):
        edge = _solve((0.410, 0.712, 0.733887), KV_H25, False, 16384, prompt_tokens=262144)
        assert (edge[0].buffer_rows, edge[1].buffer_rows) == (242, 397)
        assert not any(f.refused for f in edge)
        over = _solve((0.412, 0.714, 0.733887), KV_H25, False, 16384, prompt_tokens=262144)
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
        with_term = _solve(H160, KV_H25, False, 16384)[0].ceiling_max_rows
        without = _solve(H160, KV_H25, False, 16384, use_growth=False)[0].ceiling_max_rows
        assert without - with_term == 14  # 990 MiB / 70.1 per row

    def test_a_longer_prompt_than_measured_is_named_extrapolated(self):
        fits = _solve(H160, KV_H25, False, 16384, prompt_tokens=300000)
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
        fits = _solve(H160, KV_H25, False, 16384, lmem_fixed=False)
        assert fits[0].headroom_mib == pytest.approx(3262 - 1457, abs=1)
        assert [f.ceiling_fraction for f in fits][0] == 0.371

    def test_a_narrower_chunk_buys_rows_by_the_measured_difference(self):
        at16 = _solve(H160, KV_H25, False, 16384, use_growth=False)[0]
        at8 = _solve(H160, KV_H25, False, 8192, use_growth=False)[0]
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


class TestLauncherSeam:
    def _ns(self, draft="off", prompt=0):
        return types.SimpleNamespace(
            p_card_reference_logs="", draft_kv_on_p=draft, p_card_prompt_tokens=prompt,
            extra_p="--speculative-draft-model-path /nonexistent/draft", extra_d="",
        )

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
