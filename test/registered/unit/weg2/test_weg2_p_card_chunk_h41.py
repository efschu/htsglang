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
buffer line per stage, the draft-group line, the OOM text; for x145/x146/x149
also the ``#969 EXTENT`` lines and ``Scheduler hit an exception``).

H41b: fnFL2x149 (FR_P[0] 0.351, the v1 edge) passed v1 and died in PP0's
SECOND chunk. A buffer row above the reference costs the card ~2 row images
(142.2 MiB on PP0, lower bound from the death); the edge at 16384 is 0.304.

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
    def test_the_shipped_card_reference_is_the_logs_own_measurement(self):
        got = pc.p_card_reference_from_logs(
            [_boot("fnFL2x145"), _boot("fnFL2x146")], stage_layers=CUT,
            row_mib=ROW_MIB, support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL,
            death_boots=[_boot("fnFL2x149")],
        )
        assert got == pc.P_CARD_REFERENCE_FNFL2

    def test_x149_death_is_read_off_its_own_oom_line(self):
        d = pc.death_from_log(*_boot("fnFL2x149"))
        # 12:18:30 PP0, second chunk (#969 EXTENT 16384..32768), 212 rows;
        # cap = 292.81 free + 26.28 GiB allocated + 290.04 unallocated,
        # need = allocated + 384, private_free 2467 (WEG2-GRAPH-POOL extend)
        assert (d.stage, d.chunk_index, d.buffer_rows) == (0, 1, 212)
        assert d.headroom_bound_mib == pytest.approx(-2268.1, abs=0.2)
        assert d.private_free_mib == 2467.0

    def test_x149_first_chunk_sat_on_the_row_image(self):
        # chunk 0: 4594 (x146) - 46 rows x 29 x 2.417 - 73 private_free = 1297,
        # measured 1300 -- the extra cost appears from chunk 1 on.
        h149 = pc.headroom_by_chunk_index(_boot("fnFL2x149")[1])[0][0]
        h146 = pc.headroom_by_chunk_index(_boot("fnFL2x146")[1])[0][0]
        assert abs(h146 - 46 * 29 * ROW_MIB - 73 - h149) < 5.0

    def test_row_card_cost_is_about_two_row_images(self):
        ref = pc.P_CARD_REFERENCE_FNFL2
        assert ref.row_card_mib[0] == pytest.approx(142.2, abs=0.1)
        assert 2.0 < ref.row_card_mib[0] / (29 * ROW_MIB) < 2.1
        # stages without their own death inherit the ratio (named)
        assert "uebertragen (Hochrechnung)" in ref.row_card_source

    def test_pp0_binds_at_the_last_high_water_of_x146(self):
        ref = pc.P_CARD_REFERENCE_FNFL2
        # 28953 - 22927 - 2394 (WEG2-GRAPH-POOL rank=0 phase=high-water 11:31:00)
        assert ref.headroom_mib[0] == 3632.0
        assert ref.buffer_rows == (166, 263, 408)
        assert ref.kv_mib == (1904.0, 816.0, 544.0)
        assert ref.draft_on_p is False

    def test_near_oom_mirror_is_the_corridor_law(self):
        from sglang.srt.managers import corridor_guard

        assert pc.NEAR_OOM_MIB == float(corridor_guard.NEAR_OOM_MIB)


class TestVerdicts:
    def test_x145_form_passes(self):
        # x145: FR_P 0.26,0.45,0.39 -> H25 0.26,0.45,0.733887, no draft on P.
        fr, kv, draft, chunk = _form_of("fnFL2x145")
        assert fr == (0.26, 0.45, 0.734) and draft is False and chunk == 16384
        fits = _solve((0.26, 0.45, 0.733887), kv, draft, chunk)
        assert [f.verdict for f in fits] == ["PASST"] * 3
        assert pc.p_card_refusal_text(fits, pc.P_CARD_REFERENCE_FNFL2, chunk=chunk) is None

    @pytest.mark.parametrize("boot,f0", [("fnFL2x121", 0.45), ("fnFL2x122", 0.40)])
    def test_the_oom_boots_are_refused_on_the_5090_stage(self, boot, f0):
        fr, kv, draft, chunk = _form_of(boot)
        assert fr[0] == f0 and draft is True and chunk == 16384
        assert "OutOfMemoryError" in _boot(boot)[1]
        fits = _solve(fr, kv, draft, chunk)
        assert fits[0].refused and not fits[1].refused and not fits[2].refused
        text = pc.p_card_refusal_text(fits, pc.P_CARD_REFERENCE_FNFL2, chunk=chunk)
        assert text.startswith("W132 Weg2PCardChunkOom (P, chunk 16384)")
        assert "stage0 (nvml1) f %.4f" % f0 in text

    def test_x149_form_is_refused(self):
        # x149: FR_P 0.351,0.642,0.39 -> H25 0.733887, died in PP0's 2nd chunk.
        fr, kv, draft, chunk = _form_of("fnFL2x149")
        assert fr[:2] == (0.351, 0.642) and draft is False and chunk == 16384
        assert "OutOfMemoryError" in _boot("fnFL2x149")[1]
        fits = _solve((0.351, 0.642, 0.733887), kv, draft, chunk)
        assert fits[0].refused and fits[0].buffer_rows == 212
        assert fits[1].refused  # 361 rows at the transferred price
        text = pc.p_card_refusal_text(fits, pc.P_CARD_REFERENCE_FNFL2, chunk=chunk)
        assert text.startswith("W132 Weg2PCardChunkOom (P, chunk 16384)")

    def test_mutant_with_the_row_image_lets_x149_through(self):
        # H41 v1: a row above the reference priced at its image L x row.
        mutant = msgspec.structs.replace(pc.P_CARD_REFERENCE_FNFL2, row_card_mib=())
        fr, kv, draft, chunk = _form_of("fnFL2x149")
        fits = _solve((0.351, 0.642, 0.733887), kv, draft, chunk, reference=mutant)
        assert not any(f.refused for f in fits)

    def test_the_card_ceiling_at_16384_is_the_edge(self):
        fr, kv, draft, chunk = _form_of("fnFL2x145")
        fits = _solve((0.26, 0.45, 0.733887), kv, draft, chunk)
        assert [f.ceiling_max_rows for f in fits] == [188, 311, 446]
        assert [f.ceiling_fraction for f in fits] == [0.304, 0.544, 0.808]
        # the edge itself passes, one row above it dies
        edge = _solve((0.304, 0.544, 0.733887), kv, draft, chunk)
        assert edge[0].buffer_rows == 188 and not edge[0].refused
        assert edge[1].buffer_rows == 311 and not edge[1].refused
        over = _solve((0.306, 0.45, 0.733887), kv, draft, chunk)
        assert over[0].buffer_rows == 189 and over[0].refused

    def test_a_narrower_chunk_buys_rows_by_the_measured_difference(self):
        fr, kv, draft, _ = _form_of("fnFL2x145")
        at16 = _solve((0.26, 0.45, 0.733887), kv, draft, 16384)[0]
        at8 = _solve((0.26, 0.45, 0.733887), kv, draft, 8192)[0]
        assert at8.headroom_mib - at16.headroom_mib == pytest.approx(3696.6 - 1863.7)

    def test_mutant_without_the_chunk_term_lets_x122_through(self, monkeypatch):
        # The #1286 world: the activation priced as the reference boots'
        # 1024 MiB on every chunk width. x122 (died) then passes the card.
        fr, kv, draft, chunk = _form_of("fnFL2x122")
        # Isolated from the H41b row price (row image only), so the chunk term
        # alone decides: with it x122 is refused, without it x122 passes.
        image = msgspec.structs.replace(pc.P_CARD_REFERENCE_FNFL2, row_card_mib=())
        assert _solve(fr, kv, draft, chunk, reference=image)[0].refused
        monkeypatch.setattr(pc, "transient_mib", lambda sup, s, c: (1024.0, "mutant"))
        fits = _solve(fr, kv, draft, chunk, reference=image)
        assert not any(f.refused for f in fits)

    def test_another_cut_is_named_not_priced(self):
        with pytest.raises(ValueError, match="Schnitt"):
            pc.solve_p_card(
                reference=pc.P_CARD_REFERENCE_FNFL2, support=pc.P_TRANSIENT_SUPPORT_FNFL2,
                fractions=(0.26, 0.45, 0.73), lru_rows=LRU, stage_layers=(33, 7, 8),
                chunk=16384, kv_mib=(1904, 816, 544), num_experts=512, row_mib=ROW_MIB,
            )


class TestLauncherSeam:
    def _ns(self, draft="off"):
        return types.SimpleNamespace(
            p_card_reference_logs="", draft_kv_on_p=draft,
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

    def test_x145_form_prints_activation_and_card_and_passes(self):
        lines = self._run(self._ns(), (0.26, 0.45, 0.733887), (1904.0, 816.0, 544.0))
        assert sum(l.startswith("PP-CUT ACTIVATION chunk=16384 stage") for l in lines) == 3
        card = [l for l in lines if l.startswith("PP-CUT P-KARTE stage")]
        assert len(card) == 3 and all("-> PASST |" in l for l in card)

    def test_x121_form_is_refused_w132(self, monkeypatch):
        from sglang.srt.weg2 import draft_post, launcher

        monkeypatch.setattr(draft_post, "p_draft_post_mib", lambda path: (2799.0, 623.0))
        with pytest.raises(launcher.Weg2LaunchRefused, match="W132 Weg2PCardChunkOom"):
            self._run(self._ns("on"), (0.45, 0.45, 0.39), (2176.0, 1088.0, 816.0))

    def test_x149_form_is_refused_w132(self):
        from sglang.srt.weg2 import launcher

        with pytest.raises(launcher.Weg2LaunchRefused, match="W132 Weg2PCardChunkOom"):
            self._run(self._ns(), (0.351, 0.642, 0.733887), (1904.0, 816.0, 544.0))

    def test_unmeasured_chunk_is_refused_w131(self):
        from sglang.srt.weg2 import launcher

        with pytest.raises(launcher.Weg2LaunchRefused, match="W131"):
            self._run(self._ns(), (0.26, 0.45, 0.733887), (1904.0, 816.0, 544.0), chunk=32768)

    def test_other_model_is_named_not_priced(self):
        from sglang.srt.weg2 import launcher

        lines = []
        launcher.p_card_verdict(
            self._ns(), self._cards(), lines.append, model="/m/Qwen3.8-27B",
            chunk_tokens=16384, fracs=(0.3, 0.3, 0.3), lru_rows=LRU, stage_layers=CUT,
            kv_mib=(1, 1, 1), num_experts=512, row_mib=ROW_MIB,
        )
        assert lines and "ENTFAELLT" in lines[0]
