# SPDX-License-Identifier: Apache-2.0
"""H92c: P's mamba slots are the argv's ``--max-mamba-cache-size``, and the P card
prices them.

THE BEFUND (fnFL2h91bb2, H92b). The pool model's mamba post charged
``ceil(--p-bs x 2 x 1.25)`` slots -- 20 at P_BS 8 -- while argv_p ALWAYS states
``--max-mamba-cache-size`` (24 by default, the arm's 32 in --extra-p) and the
runtime takes an explicit value as the pool: bb2's P log ``Mamba Cache is
allocated. max_mamba_cache_size: 32`` on every stage. 12 slots x 34.29 / 12.47 /
9.35 MiB = 411 / 150 / 112 MiB were missing from the pool budget; x178 (P_BS 4,
the default 24) was charged 10. And ``solve_p_card`` had no mamba term at all:
its K0 is the 4-seat reference x163+x164+x165 WITH 24 slots, and any other seat
count was ``PCardFormUnmeasured`` (the card check entfiel -- the launch went on
unchecked).

THE FIX. ``launcher.p_mamba_slots`` reads the slots off P's argv as the launcher
builds it (with --extra-p); the pool model and the P card take that one number.
The card subtracts ``(slots - reference slots) x 1.5588 MiB x linear layers`` per
stage, and more seats than the reference become computable ONLY through that
term -- everything else a seat changes stays unmeasured and is named on the line.

Control: the x178 launcher line (4 seats, 24 slots, FR_P 0.332,0.64,0.733887,
prompt 262144, co-tenant 2004/838/836) printed Kopfraum 2926 / 1348 / 3658.

Hermetic: no GPU, no torch device.
"""

import ast
import inspect
import json
import os
import tempfile
import textwrap
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.planner import p_card_chunk as pc
from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as lc

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "p_card_h41")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ROW_MIB = 2.4170379638671875
CUT = (29, 11, 8)
LRU = (32, 32, 32)
KV = (1904.0, 816.0, 544.0)
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 12
#: 1.5588 MiB x 22 / 8 / 6 linear layers of the 29,11,8 cut
SLOT_MIB = (1.5588 * 22, 1.5588 * 8, 1.5588 * 6)
X178 = (0.332, 0.64, 0.733887)
BB2 = (0.324, 0.637, 0.733887)
RELEASE = (0.328, 0.64, 0.733887)
FOUR_SEAT_BOOTS = ("fnFL2x163", "fnFL2x164", "fnFL2x165")


def _boot(name):
    with open(os.path.join(FIX, name + ".P.lines")) as fh:
        return name, fh.read()


def _model_dir(d):
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 48,
            "layer_types": LAYER_TYPES,
            "hidden_size": 2560,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
        },
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {"model.language_model.layers.0.h92c.weight": "x.safetensors"}}, f)
    return d


@pytest.fixture
def model(tmp_path):
    with mock.patch.object(
        hl, "read_pp_calibration",
        lambda digest, calib_dir=hl.CALIB_DIR: (None, "no PP calibration (test)"),
    ):
        yield _model_dir(str(tmp_path))


def _ns(extra_p="", p_bs=8, profile="nextflash"):
    # unified tree: the argv reading is the registry row's
    # p_mamba_slots_from_argv (nextflash on, qwen27b off)
    return types.SimpleNamespace(extra_p=extra_p, p_bs=p_bs, profile=profile,
                                 pp_cut_mamba_slots_per_running_request=2)


# ---------------------------------------------------------------------------
# 1. the pool model's mamba post
# ---------------------------------------------------------------------------


class TestMambaPost:
    def test_the_arms_flag_is_the_post(self, model):
        # bb2: P_BS 8 + --max-mamba-cache-size 32 -> the boot allocated 32
        slots, src = lc.p_mamba_slots(_ns("--max-mamba-cache-size 32"), model, 8)
        assert slots == 32
        assert "--max-mamba-cache-size 32 on P's argv" in src
        # the old post: ceil(8 x 2 x 1.25) = 20
        assert slots != 20

    def test_last_occurrence_and_equals_form_win(self, model):
        assert lc.p_mamba_slots(_ns("--max-mamba-cache-size=28"), model, 6)[0] == 28
        assert lc.p_mamba_slots(
            _ns("--max-mamba-cache-size 28 --max-mamba-cache-size 32"), model, 6)[0] == 32

    def test_without_the_arm_flag_it_is_argv_ps_own_default(self, model):
        # x178: P_BS 4, no flag in --extra-p -> argv_p states 24; the old post said 10
        slots, src = lc.p_mamba_slots(_ns("", p_bs=4), model, 4)
        assert slots == lc.P_MAX_MAMBA_CACHE_SIZE == 24
        assert "on P's argv" in src

    def test_qwen27b_keeps_the_demand_formula(self, model):
        """Operator rule 26.09. (27B byte-identical): on the qwen27b row the post
        stays ceil(p_bs x 2 x 1.25) even with the flag on P's argv -- the 27B
        stage constants were fitted with it (reading the argv moves the solved
        27B cut 42,11,11 -> 39,13,12)."""
        slots, src = lc.p_mamba_slots(_ns("--max-mamba-cache-size 32", profile="qwen27b"), model, 8)
        assert slots == 20 and "UPPER BOUND" in src
        assert lc.p_mamba_slots(_ns("", p_bs=2, profile="qwen27b"), model, 2)[0] == 5

    def test_the_demand_bound_only_for_an_argv_without_the_flag(self, model):
        real = lc.argv_p

        def no_mamba(*a, **kw):
            argv = real(*a, **kw)
            i = argv.index("--max-mamba-cache-size")
            return argv[:i] + argv[i + 2:]

        with mock.patch.object(lc, "argv_p", no_mamba):
            slots, src = lc.p_mamba_slots(_ns("", p_bs=8), model, 8)
        assert slots == 20 and "UPPER BOUND" in src

    def test_solve_p_cut_charges_the_argv_slots(self):
        """The PhasePoolModel's ``mamba_slots`` is p_mamba_slots' number, and the
        P card gets the SAME number (one reader, two consumers)."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(lc.solve_p_cut)))
        pool_kw, card_kw, assigned = None, None, False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                for kw in node.keywords:
                    if kw.arg == "mamba_slots" and name == "PhasePoolModel":
                        pool_kw = ast.unparse(kw.value)
                    if kw.arg == "mamba_slots" and name == "p_card_verdict":
                        card_kw = ast.unparse(kw.value)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if getattr(node.value.func, "id", "") == "p_mamba_slots":
                    assigned = any(
                        isinstance(t, ast.Tuple) and t.elts[0].id == "_p_mamba_slots"
                        for t in node.targets)
        assert pool_kw == "int(_p_mamba_slots)", pool_kw
        assert card_kw == "int(_p_mamba_slots)", card_kw
        assert assigned

    def test_slot_price_per_stage_counts_the_linear_layers(self):
        got = lc.p_mamba_mib_per_slot_by_stage(LAYER_TYPES, CUT, 1.5588)
        assert got == pytest.approx(SLOT_MIB)
        assert [round(x, 2) for x in got] == [34.29, 12.47, 9.35]


class TestMambaFloor:
    """The runtime's hard floor per P seat and the retention budget above it."""

    def test_the_constant_is_the_runtimes_floor_on_ps_posture(self, model):
        from sglang.srt.mem_cache import mamba_pool_floor as mf

        argv = lc.argv_p("py", model, [1, 1, 1], 1, 1, lc.RING_FORM_SENTINEL_STORE_CFG, [],
                         p_bs=8, stage_ratio="", attn_stage_ratio="")
        # P's posture as its own argv states it
        assert "--disable-overlap-schedule" in argv
        policy = argv[argv.index("--hicache-write-policy") + 1]
        assert policy == "write_back"
        sa = types.SimpleNamespace(
            disable_radix_cache="--disable-radix-cache" in argv,
            disable_overlap_schedule=True,
            enable_hierarchical_cache="--enable-hierarchical-cache" in argv,
            hicache_write_policy=policy,
            mamba_anchor_ack_release=False,
            enable_mamba_extra_buffer=lambda: True,
        )
        assert mf.mamba_slots_per_running_req(sa) == lc.P_MAMBA_FLOOR_SLOTS_PER_SEAT == 4

    def test_bb2_form_is_warned_retention_zero(self):
        # bb2 P log: 'MAMBA-FLOOR pool=32 floor=32 retention_budget=0 (8 running requests x 4 = 32)'
        text = lc.p_mamba_floor_text(32, 8)
        assert "mamba floor 8 seats x 4 = 32, retention budget 0" in text
        assert "WARNUNG MAMBA-RETENTION" in text
        # never a W-code: the arm counts 'W[0-9]+ ' lines as refusals
        import re

        assert not re.search(r"W[0-9]+ ", text)

    def test_release_form_keeps_eight_retention_slots(self):
        text = lc.p_mamba_floor_text(32, 6)
        assert text == "mamba floor 6 seats x 4 = 24, retention budget 8"


# ---------------------------------------------------------------------------
# 2. the P card's mamba term
# ---------------------------------------------------------------------------


def _card(fr, seats, **kw):
    return pc.solve_p_card(
        reference=kw.pop("reference", pc.P_CARD_REFERENCE_FNFL2),
        support=pc.P_TRANSIENT_SUPPORT_FNFL2, fractions=fr, lru_rows=LRU,
        stage_layers=CUT, chunk=16384, kv_mib=KV, num_experts=512, row_mib=ROW_MIB,
        prompt_tokens=262144, seats=seats, co_tenant=pc.P_CARD_CO_TENANT_FNFL2,
        cards=("nvml1", "nvml0", "nvml2"), **kw)


class TestPCardMambaTerm:
    def test_the_reference_holds_24_slots_from_its_own_lines(self):
        got = pc.p_card_reference_from_logs(
            [_boot(b) for b in FOUR_SEAT_BOOTS], stage_layers=CUT, row_mib=ROW_MIB,
            support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL)
        assert got.mamba_slots == 24 == pc.P_CARD_REFERENCE_FNFL2.mamba_slots
        assert pc.P_CARD_REFERENCE_FNFL2_X160.mamba_slots == 24

    def test_a_reference_mixing_slot_counts_is_refused(self):
        name, text = _boot("fnFL2x164")
        mixed = text.replace("max_mamba_cache_size: 24", "max_mamba_cache_size: 32")
        with pytest.raises(ValueError, match="mischt Mamba-Slots"):
            pc.p_card_reference_from_logs(
                [_boot("fnFL2x163"), (name, mixed)], stage_layers=CUT, row_mib=ROW_MIB,
                support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL)

    def test_x178_control_4_seats_24_slots(self):
        # the x178 launcher line: 2926 / 1348 / 3658
        for kw in ({}, dict(mamba_slots=24, mamba_mib_per_slot=SLOT_MIB)):
            fits = _card(X178, 4, **kw)
            assert [round(f.headroom_mib) for f in fits] == [2926, 1348, 3658]
            assert all(f.mamba_mib == 0.0 for f in fits)

    def test_more_slots_at_4_seats_cost_the_pool(self):
        fits = _card(X178, 4, mamba_slots=32, mamba_mib_per_slot=SLOT_MIB)
        assert [round(f.mamba_mib) for f in fits] == [274, 100, 75]
        assert [round(f.headroom_mib) for f in fits] == [2926 - 274, 1348 - 100, 3658 - 75]

    @pytest.mark.parametrize("seats,slots,fr,head", [
        (6, 32, RELEASE, [2792, 1248, 3583]),
        (6, 28, RELEASE, [2929, 1298, 3621]),
        (8, 32, BB2, [2932, 1274, 3583]),
    ])
    def test_six_and_eight_seats_are_computable(self, seats, slots, fr, head):
        fits = _card(fr, seats, mamba_slots=slots, mamba_mib_per_slot=SLOT_MIB)
        assert [round(f.headroom_mib) for f in fits] == head
        assert not any(f.refused for f in fits)
        assert all(f.seats_over_reference == seats - 4 for f in fits)

    def test_more_seats_without_the_mamba_term_stay_unmeasured(self):
        with pytest.raises(pc.PCardFormUnmeasured, match="8 Sitze"):
            _card(BB2, 8)
        # a reference that does not know its slots cannot price the difference
        blind = pc.P_CARD_REFERENCE_FNFL2.__class__(
            **{f: getattr(pc.P_CARD_REFERENCE_FNFL2, f)
               for f in pc.P_CARD_REFERENCE_FNFL2.__struct_fields__ if f != "mamba_slots"})
        with pytest.raises(pc.PCardFormUnmeasured, match="Mamba-Term ist nicht rechenbar"):
            _card(BB2, 8, reference=blind, mamba_slots=32, mamba_mib_per_slot=SLOT_MIB)

    def test_the_line_names_the_term_and_the_unmeasured_rest(self):
        fit = _card(RELEASE, 6, mamba_slots=32, mamba_mib_per_slot=SLOT_MIB)[0]
        line = pc.describe_p_card(fit, pc.P_CARD_REFERENCE_FNFL2)
        assert "(4 Sitze, hier 6)" in line
        assert "Mamba 32 Slots gegen 24 der Referenz -> +274 MiB -> Kopfraum 2792 MiB" in line
        assert "SITZE UEBER DER REFERENZ (6 statt 4): gerechnet ist nur der Mamba-Term" in line
        four = pc.describe_p_card(_card(X178, 4)[0], pc.P_CARD_REFERENCE_FNFL2)
        assert "Mamba-Slots ungemessen (hier ?, Referenz 24: Term 0)" in four
        assert "SITZE UEBER" not in four


class TestLauncherSeamH92c:
    def _run(self, extra_p, p_bs, fr, **kw):
        lines = []
        ns = types.SimpleNamespace(
            p_card_reference_logs="", draft_kv_on_p="off", p_card_prompt_tokens=0,
            extra_p=extra_p, extra_d="", p_bs=p_bs)
        lc.p_card_verdict(
            ns, [types.SimpleNamespace(nvml_index=i) for i in (1, 0, 2)], lines.append,
            model="/m/" + MODEL, chunk_tokens=16384, fracs=fr, lru_rows=LRU,
            stage_layers=CUT, kv_mib=KV, num_experts=512, row_mib=ROW_MIB, **kw)
        return [ln for ln in lines if ln.startswith("PP-CUT P-KARTE")]

    def test_eight_seats_with_the_mamba_term_are_checked_not_skipped(self):
        card = self._run("--max-running-requests 8", 8, BB2,
                         mamba_slots=32, mamba_mib_per_slot=SLOT_MIB)
        assert len(card) == 3 and all("-> PASST |" in c for c in card)
        assert "(4 Sitze, hier 8)" in card[0] and "+274 MiB" in card[0]

    def test_a_computed_deficit_at_eight_seats_refuses_w132(self):
        with pytest.raises(lc.Weg2LaunchRefused, match="W132 Weg2PCardChunkOom"):
            self._run("--max-running-requests 8", 8, (0.40, 0.70, 0.733887),
                      mamba_slots=32, mamba_mib_per_slot=SLOT_MIB)

    def test_without_the_term_it_still_entfaellt(self):
        card = self._run("--max-running-requests 8", 8, BB2)
        assert len(card) == 1 and "ENTFAELLT" in card[0] and "8 Sitze" in card[0]
