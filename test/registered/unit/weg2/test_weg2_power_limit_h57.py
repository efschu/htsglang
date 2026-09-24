"""fnFL2 H57: das Power-Limit je Karte -- Startzeile, Datierung der Zeit-/Raten-
Referenzen, Schnitt aus gemessenen Stufenraten.

Nutzer 24.09. 18:53Z: "die geschwindigkeiten der karten ist noch im powerlimit
bei 400 und 230 deswegen muss der schnitt aufjedenfall anpassbar sein, da ich
spaeter das powerlimit ggf. erhoehen werde".

Hermetisch: ein Fake-NVML (dieselben Abfragen wie ``nvidia-smi
--query-gpu=power.limit,power.max_limit,clocks.max.sm``), eine
Fake-Kartenbibliothek, die woertlichen ``Prefill rank batch``-Zeilen der
Referenz-Boots als Fixture (``fixtures/power_limit_h57``). Keine GPU, kein
NVML, kein CUDA.
"""

import ast
import inspect
import json
import os
import re
import tempfile
import textwrap
from unittest import mock

import pytest

try:
    from sglang.srt.distributed import pp_crossing_transport
    from sglang.srt.planner import expert_residency, p_card_chunk
    from sglang.srt.planner import power_limit as P
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import wake_credit, wake_credit_pd, wake_credit_pd_refs
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "power_limit_h57")

#: Next Flash, config.json text_config.layer_types: full_attention an 3,7,...,47
#: (full_attention_interval 4) -- 12 von 48.
NF_FULL_ATTENTION = [i % 4 == 3 for i in range(48)]
PIN = [29, 11, 8]
FR_P_X163 = "0.36,0.64,0.39"


def _fixture(tag: str) -> str:
    with open(os.path.join(FIX, tag + ".P.lines")) as fh:
        return fh.read()


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeNvml:
    """pynvml-artig; Watt/MHz je Karte in NVML-Reihenfolge."""

    NVML_CLOCK_SM = 1

    def __init__(self, cards):
        self.cards = cards

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return self.cards[h]["uuid"].encode()

    def nvmlDeviceGetName(self, h):
        return self.cards[h]["name"]

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return int(self.cards[h]["limit"] * 1000)

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
        return (100000, int(self.cards[h]["max"] * 1000))

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h):
        return int(self.cards[h]["default"] * 1000)

    def nvmlDeviceGetMaxClockInfo(self, h, clock):
        assert clock == self.NVML_CLOCK_SM
        return self.cards[h]["clock"]


def rig_nvml(w5090=400.0, w3080_nvml0=230.0, w3080_nvml2=230.0) -> FakeNvml:
    """Das Rig in NVML-Reihenfolge: nvml0 3080, nvml1 5090, nvml2 3080."""
    return FakeNvml(
        [
            dict(
                uuid=P.RIG_3080_NVML0[0],
                name=P.RIG_3080_NVML0[2],
                limit=w3080_nvml0,
                max=320.0,
                default=320.0,
                clock=2100,
            ),
            dict(
                uuid=P.RIG_5090[0],
                name=P.RIG_5090[2],
                limit=w5090,
                max=600.0,
                default=575.0,
                clock=3090,
            ),
            dict(
                uuid=P.RIG_3080_NVML2[0],
                name=P.RIG_3080_NVML2[2],
                limit=w3080_nvml2,
                max=320.0,
                default=320.0,
                clock=2100,
            ),
        ]
    )


def reading(*a, **k) -> "P.PowerReading":
    return P.read_card_power(nvml=rig_nvml(*a, **k))


def stage_cards():
    """Die Karten, wie der Launcher sie nach order_cards haelt (Ordinal = P-Stufe)."""
    return L.order_cards(
        [
            L.Card(0, P.RIG_3080_NVML0[0], P.RIG_3080_NVML0[2], 20480),
            L.Card(1, P.RIG_5090[0], P.RIG_5090[2], 32607),
            L.Card(2, P.RIG_3080_NVML2[0], P.RIG_3080_NVML2[2], 20480),
        ]
    )


def cut_line(rd, **over):
    kw = dict(
        stage_cards=stage_cards(),
        model_name=P.NF_MODEL,
        is_full_attention=NF_FULL_ATTENTION,
        pinned=PIN,
        chunk_tokens=16384,
        fr_p=FR_P_X163,
    )
    kw.update(over)
    return P.rate_cut_line(rd, **kw)


class FakeVariant:
    def __init__(self, gemm, rate_env):
        self.gemm_tflops = gemm
        self.rate_env = rate_env


class FakeLibrary:
    """card_library.json wie am 14.08. gemessen: 3080 bei 200 W (#584 rate_env)."""

    def __init__(self, plimit_5090_mw=400000, plimit_3080_mw=200000):
        self.by = {
            "5090": [
                FakeVariant(203.42, "v1;drv=595.58.03;plimit_mw=%d" % plimit_5090_mw)
            ],
            # zuerst der Seed ohne Rate, wie variants() ihn liefert (kuerzester Schluessel)
            "3080": [
                FakeVariant(None, None),
                FakeVariant(50.97, "v1;drv=595.58.03;plimit_mw=%d" % plimit_3080_mw),
            ],
        }

    def variants(self, name):
        return self.by["5090" if "5090" in name else "3080"]


# --------------------------------------------------------------------------
# 1. die Zeile im Log und im Zustands-JSON
# --------------------------------------------------------------------------


def test_the_launch_line_names_limit_max_and_sm_clock_per_nvml_card():
    assert P.launch_line(reading()) == (
        "POWER-LIMIT nvml0=230/320W nvml1=400/600W nvml2=230/320W "
        "sm_clock_max=nvml0:2100,nvml1:3090,nvml2:2100MHz source=nvml"
    )


def test_the_launcher_logs_the_line_and_writes_it_into_the_boot_state_json():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "front.log")
        log = L.Log(log_path)
        state = L.BootState(tag="h57t", tip="0" * 10, tree=tmp, stamp="0924_190000")
        with mock.patch(
            "sglang.srt.planner.card_rate_pass.load_measured_library",
            return_value=FakeLibrary(),
        ):
            got = L.log_power_limits(state, stage_cards(), log, reading=reading())
        log.fh.close()
        with open(log_path) as fh:
            text = fh.read()
        assert re.search(
            r"^\[\S+\] WEG2-LAUNCH POWER-LIMIT nvml0=230/320W nvml1=400/600W nvml2=230/320W "
            r"sm_clock_max=nvml0:2100,nvml1:3090,nvml2:2100MHz source=nvml$",
            text,
            re.M,
        ), text
        assert got.known
        with mock.patch.object(L, "GPU_ARB", tmp):
            L._write_state(state)
            with open(L.state_path(state)) as fh:
                saved = json.load(fh)
    pl = saved["power_limits"]
    assert (
        pl["cards"]["nvml1"]["limit_w"] == 400.0
        and pl["cards"]["nvml1"]["max_w"] == 600.0
    )
    assert (
        pl["cards"]["nvml0"]["limit_w"] == 230.0
        and pl["cards"]["nvml2"]["sm_clock_max_mhz"] == 2100
    )
    assert pl["cards"]["nvml1"]["uuid"] == P.RIG_5090[0]
    assert pl["line"] == P.launch_line(reading())


def test_the_line_the_launcher_writes_is_the_line_the_rate_reader_parses():
    """Schreiber und Leser einer Naht fragen dasselbe: was der Launcher
    druckt, stempelt spaeter die Raten dieses Boots (stage_rates_from_boot)."""
    line = "[2026-09-24T19:00:00Z] WEG2-LAUNCH " + P.launch_line(reading(525, 320, 320))
    assert P.limits_from_launch_log(line) == {0: 320.0, 1: 525.0, 2: 320.0}


def test_an_unreadable_nvml_prints_unknown_and_never_raises():
    from sglang.srt.registry import nvml as reg

    def boom():
        raise reg.NvmlUnavailableError("nvmlInit() failed (driver not loaded)")

    with mock.patch.dict(os.environ, {reg.ENV_NVML_REPLAY: ""}), mock.patch.object(
        reg, "nvml_session", boom
    ):
        rd = P.read_card_power()
    assert rd.cards == () and not rd.known
    assert P.launch_line(rd).startswith(
        "POWER-LIMIT unbekannt (NvmlUnavailableError: nvmlInit()"
    )
    lines = P.reference_lines(rd)
    assert all(P.STALE_MARKER not in ln for ln in lines)
    assert sum(ln.startswith(P.UNDATED_MARKER) for ln in lines) == len(
        P.TIMED_REFERENCES
    )
    assert (
        "keine Raten fuer nvml1=?W nvml0=?W nvml2=?W (Power-Limit unbekannt"
        in cut_line(rd)
    )


def test_a_replay_reads_the_recorded_rows_and_never_the_running_rig():
    """#1377: ein Replay mit Live-Limits waere ein Boot auf zwei Maschinen."""
    from sglang.srt.registry import nvml as reg

    rows = [
        {
            "index": 0,
            "uuid": "GPU-aaa",
            "name": "NVIDIA GeForce RTX 3080",
            "total_bytes": 21474836480,
        }
    ]

    def live_rig():
        raise AssertionError("replay read the running rig")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cards.json")
        with open(path, "w") as fh:
            json.dump(rows, fh)
        with mock.patch.dict(
            os.environ, {reg.ENV_NVML_REPLAY: path}
        ), mock.patch.object(reg, "nvml_session", live_rig):
            bare = P.read_card_power()
            rows[0].update(
                power_limit_w=230.0, power_max_w=320.0, sm_clock_max_mhz=2100
            )
            with open(path, "w") as fh:
                json.dump(rows, fh)
            stamped = P.read_card_power()
    assert bare.source == "replay" and not bare.known
    assert "Replay ohne Power-Felder" in P.launch_line(bare)
    assert (
        P.launch_line(stamped)
        == "POWER-LIMIT nvml0=230/320W sm_clock_max=nvml0:2100MHz source=replay"
    )


# --------------------------------------------------------------------------
# 2. die Referenzen: veraltet bei 450 W, VRAM-Riegel unberuehrt
# --------------------------------------------------------------------------


def test_at_400_230_every_stamped_timed_reference_is_current():
    lines = P.reference_lines(reading())
    assert lines == [lines[-1]] and lines[-1].startswith(
        "POWER-LIMIT REFERENZEN: 3 von 3 Zeit-/Raten-Referenzen passen zum Limit "
        "(launcher.MEASURED_MS_PER_LAYER, launcher.ATTN_ANCHOR_MS, wake_credit_pd_refs.REFERENCES)"
    )


def test_at_450_w_on_the_5090_every_reference_stamped_400_is_named_stale():
    lines = P.reference_lines(reading(w5090=450.0))
    stale = [ln for ln in lines if ln.startswith(P.STALE_MARKER)]
    assert [ln.split("): ", 1)[1].split(" [", 1)[0] for ln in stale] == [
        "launcher.MEASURED_MS_PER_LAYER",
        "wake_credit_pd_refs.REFERENCES",
    ]
    for ln in stale:
        assert ln.startswith("REFERENZ VERALTET (Power-Limit nvml1 400->450 W): ")
        assert "gemessen bei nvml1=400W nvml0=230W nvml2=230W, jetzt nvml1=450W" in ln
    # 3080-only: ATTN_ANCHOR_MS haengt nicht an der 5090 und bleibt aktuell.
    assert "(launcher.ATTN_ANCHOR_MS)" in lines[-1]


def test_the_card_library_measured_at_200_w_is_stale_at_todays_230_w():
    """Der reale Befund: card_library.json (14.08.) traegt fuer die 3080
    plimit_mw=200000, das Rig laeuft seit spaetestens 06.09. mit 230 W -- und
    ``pp_cut_launch.ms_per_layer_from_card_library`` liest sie ohne Datierung."""
    rd = reading()
    ref = P.card_library_reference(
        rd, stage_cards(), library=FakeLibrary(), path="lib.json"
    )
    assert [(st.label, st.limit_w) for st in ref.stamps] == [
        ("nvml1", 400.0),
        ("nvml0", 200.0),
        ("nvml2", 200.0),
    ]
    lines = P.reference_lines(rd, P.TIMED_REFERENCES + (ref,))
    assert lines[0].startswith(
        "REFERENZ VERALTET (Power-Limit nvml0 200->230 W, nvml2 200->230 W): "
        "card_library.json (gemm_tflops, card_rate_pass) [lib.json;"
    )
    assert "neu messen: `python -m sglang.srt.planner.card_rate_pass --run`" in lines[0]
    fresh = P.card_library_reference(
        rd, stage_cards(), library=FakeLibrary(plimit_3080_mw=230000)
    )
    assert P.stamp_verdict(fresh.stamps, rd)[0] == "aktuell"
    with mock.patch(
        "sglang.srt.planner.card_rate_pass.load_measured_library", return_value=None
    ):
        assert (
            P.card_library_reference(rd, stage_cards()) is None
        )  # keine Bibliothek, kein Leser


def test_the_vram_riegel_stay_untouched():
    """W122/W130/W132/W126 lesen nur Bytes: keine VRAM-Referenz wird datiert,
    keine Zeile verweigert, der Block wirft nie, und keiner der Riegel liest
    die Lesung."""
    rd = reading(w5090=450.0, w3080_nvml0=320.0, w3080_nvml2=320.0)
    lines = P.reference_lines(rd)
    dated = [ln for ln in lines if ln.startswith((P.STALE_MARKER, P.UNDATED_MARKER))]
    assert len(dated) == len(P.TIMED_REFERENCES)
    for ln in dated:
        assert not any(n.name in ln for n in P.LIMIT_NEUTRAL_REFERENCES)
        assert not re.search(r"\bW\d+[a-z]? Weg2", ln)  # kein Verweigerungs-Code
        assert "Keine Verweigerung, VRAM-Riegel unberuehrt." in ln
    with tempfile.TemporaryDirectory() as tmp:
        log = L.Log(os.path.join(tmp, "l.log"))
        state = L.BootState(tag="h57v", tip="0", tree=tmp, stamp="s")
        with mock.patch(
            "sglang.srt.planner.card_rate_pass.load_measured_library",
            side_effect=RuntimeError("kaputte Bibliothek"),
        ):
            L.log_power_limits(
                state, stage_cards(), log, reading=rd
            )  # darf nicht werfen
        log.fh.close()
    gates = [
        L.p_card_verdict,
        p_card_chunk.solve_p_card,
        p_card_chunk.p_card_refusal_text,
        expert_residency.solve_d_rank_residency,
        expert_residency.solve_d_card,
        wake_credit.plan_wake_credit,
        wake_credit_pd.plan_wake_credit_pd,
    ]
    for fn in gates:
        assert not re.search(
            r"_power\b|power_limit|PowerReading|power_reading", inspect.getsource(fn)
        ), fn.__name__


def test_no_h57_line_is_counted_as_a_w_line_by_the_arm():
    """Der Arm zaehlt nach dem Dry-Run ``grep -acE 'W[0-9]+ '`` als W-Zeilen und
    druckt ``W[0-9]+ |Refus|Error`` (arm_fnFL2_long.sh): eine Info-Zeile, die
    dort mitzaehlt, liest sich wie eine Verweigerung."""
    lib = FakeLibrary()
    lines = []
    for rd in (reading(), reading(w5090=450.0), reading(525, 320, 320)):
        ref = P.card_library_reference(rd, stage_cards(), library=lib, path="lib.json")
        lines += [P.launch_line(rd)] + P.reference_lines(
            rd, P.TIMED_REFERENCES + (ref,)
        )
        lines.append(cut_line(rd))
    lines.append(cut_line(reading(), model_name="Qwen3.8-27B-INT4"))
    for ln in lines:
        assert not re.search(r"W[0-9]+ |Refus", ln), ln


def test_every_stamped_name_is_a_live_constant_with_the_source_it_claims():
    modules = {
        "launcher": L,
        "wake_credit_pd_refs": wake_credit_pd_refs,
        "p_card_chunk": p_card_chunk,
        "expert_residency": expert_residency,
        "wake_credit": wake_credit,
        "pp_crossing_transport": pp_crossing_transport,
    }
    for name in [r.name for r in P.TIMED_REFERENCES] + [
        n.name for n in P.LIMIT_NEUTRAL_REFERENCES
    ]:
        mod, attr = name.split(" (", 1)[0].split(".", 1)
        assert hasattr(modules[mod], attr), name
    # Wer eine Referenz auffrischt, muss ihren Stempel mitpruefen: die Quellen
    # sind an die Konstanten gebunden.
    boots = sorted({k.split("/")[0] for k in wake_credit_pd_refs.REFERENCES})
    assert next(
        r for r in P.TIMED_REFERENCES if r.name == "wake_credit_pd_refs.REFERENCES"
    ).source.startswith(" + ".join(boots))
    for n in P.LIMIT_NEUTRAL_REFERENCES:
        mod, attr = n.name.split(" (", 1)[0].split(".", 1)
        src = getattr(getattr(modules[mod], attr), "source", None)
        if isinstance(src, str):
            assert src == n.source, (n.name, src)


# --------------------------------------------------------------------------
# 3. Stufenraten und der empfohlene Schnitt
# --------------------------------------------------------------------------


def test_the_shipped_stage_rates_are_the_fixture_logs_own_measurement():
    for shipped, tags in (
        (P.STAGE_RATES_FNFL2, ("fnFL2x160", "fnFL2x162", "fnFL2x163")),
        (P.STAGE_RATES_FNFL2_X148, ("fnFL2x148",)),
    ):
        got = P.stage_rates_from_logs(
            [_fixture(t) for t in tags],
            name=shipped.name,
            source=shipped.source,
            model=shipped.model,
            trees=shipped.trees,
            fr_p=shipped.fr_p,
            stage_layers=shipped.stage_layers,
            chunk_tokens=shipped.chunk_tokens,
            stamps=shipped.stamps,
            stamp_source=shipped.stamp_source,
        )
        assert got == shipped
        assert shipped.source == " + ".join(tags)


def test_at_400_230_the_measured_rates_recommend_26_10_12_and_apply_nothing():
    line = cut_line(reading())
    assert line.startswith(
        "PP-CUT RATEN-SCHNITT (H57) Power-Limit nvml1=400W nvml0=230W nvml2=230W "
        "= Raten P_STAGE_RATES_FNFL2 (fnFL2x160 + fnFL2x162 + fnFL2x163,"
    )
    assert (
        "voller Chunk compute 3868.1/3810.3/2303.6 ms (Median ueber 30/30/30)" in line
    )
    assert (
        "= 133.4/346.4/287.9 ms je Schicht = 8.14/21.14/17.58 us je Schicht und Token"
        in line
    )
    assert "gepinnt 29,11,8 attn 7,3,2 -> 3868/3810/2304 ms, Takt 3868 ms" in line
    assert (
        "empfohlen 26,10,12 attn 6,3,3 -> 3468/3464/3455 ms, Takt 3468 ms (-10.3 %)"
        in line
    )
    assert (
        "NICHT angewendet" in line and "PP_RATIO=26,10,12 PP_ATTN_RATIO=6,3,3" in line
    )
    assert "dieser Boot FR_P 0.36,0.64,0.39" in line


def test_the_printed_pair_is_what_the_runtime_derives():
    """Die Zeile nennt ein realisierbares Paar: derive_pp_layer_split (die
    Autoritaet, die group P selbst fragt) liefert genau den Schnitt, und das
    Paar ist die Attention-Zahl des Schnitts -- kein W40."""
    from sglang.srt.distributed.utils import derive_pp_layer_split

    for rd, want in ((reading(), [26, 10, 12]), (reading(525, 320, 320), [27, 9, 12])):
        m = re.search(r"PP_RATIO=([0-9,]+) PP_ATTN_RATIO=([0-9,]+)", cut_line(rd))
        cut = [int(x) for x in m.group(1).split(",")]
        attn = [int(x) for x in m.group(2).split(",")]
        assert cut == want
        assert (
            derive_pp_layer_split(
                cut, is_full_attention=NF_FULL_ATTENTION, attn_scores=attn
            )
            == cut
        )
        assert list(P.attention_per_stage(NF_FULL_ATTENTION, cut)) == attn


def test_at_525_320_the_x148_rates_recommend_their_own_cut():
    line = cut_line(reading(525, 320, 320))
    assert "= Raten P_STAGE_RATES_FNFL2_X148 (fnFL2x148," in line
    assert (
        "empfohlen 27,9,12 attn 6,3,3 -> 3237/3103/3109 ms, Takt 3237 ms (-14.7 %)"
        in line
    )


def test_at_450_w_there_are_no_rates_and_the_cut_stays_pinned():
    line = cut_line(reading(w5090=450.0))
    assert line.startswith(
        "PP-CUT RATEN-SCHNITT (H57): keine Raten fuer nvml1=450W nvml0=230W "
        "nvml2=230W, Schnitt bleibt gepinnt (29,11,8). Gemessen sind: "
        "P_STAGE_RATES_FNFL2 bei nvml1=400W nvml0=230W nvml2=230W"
    )
    assert "empfohlen" not in line and "PP_RATIO=" not in line


def test_a_balanced_pin_is_confirmed_not_moved():
    rates = P.STAGE_RATES_FNFL2
    balanced = [26, 10, 12]
    line = cut_line(reading(), pinned=balanced)
    assert "empfohlen = gepinnt: der Schnitt ist fuer dieses Limit balanciert" in line
    assert rates.stage_layers == (29, 11, 8)  # die Raten bleiben die des Messschnitts


def test_a_caller_without_a_reading_and_a_layout_without_a_cut_only_get_a_line():
    """solve_p_cut's power_reading defaults to None; and a layout whose stages
    cannot each hold an attention layer has no cut to recommend -- both are a
    line, never an exception (the line computes, it refuses nothing)."""
    none = cut_line(None)
    assert none.startswith(
        "PP-CUT RATEN-SCHNITT (H57): keine Raten fuer nvml1=?W nvml0=?W nvml2=?W "
        "(Power-Limit unbekannt: keine Lesung), Schnitt bleibt gepinnt (29,11,8)."
    )
    two_attn = [i in (0, 47) for i in range(48)]  # 2 Attention-Schichten, 3 Stufen
    assert cut_line(reading(), is_full_attention=two_attn).startswith(
        "PP-CUT RATEN-SCHNITT (H57) ENTFAELLT: kein Schnitt von 48 Schichten auf 3 Stufen"
    )


def test_the_27b_never_gets_next_flash_rates():
    """27B strikt getrennt (Memory 27b-strikt-getrennt-von-nf)."""
    line = cut_line(reading(), model_name="Qwen3.8-27B-INT4", pinned=[42, 11, 11])
    assert line.startswith(
        "PP-CUT RATEN-SCHNITT (H57) ENTFAELLT: die Stufenraten sind auf "
        "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist gemessen"
    )
    assert line.endswith(
        "dieser Boot faehrt Qwen3.8-27B-INT4; Schnitt bleibt gepinnt (42,11,11)."
    )


def test_rates_from_a_new_boot_are_stamped_by_its_own_power_limit_line():
    launch = "\n".join(
        [
            "[t] WEG2-LAUNCH NVML -> CUDA ordinal map: ordinal 0 = nvml 1 NVIDIA GeForce RTX 5090 "
            "%s total 32607 MiB, ordinal 1 = nvml 0 NVIDIA GeForce RTX 3080 %s total 20480 MiB, "
            "ordinal 2 = nvml 2 NVIDIA GeForce RTX 3080 %s total 20480 MiB"
            % (P.RIG_5090[0], P.RIG_3080_NVML0[0], P.RIG_3080_NVML2[0]),
            "[t] WEG2-LAUNCH " + P.launch_line(reading(600, 320, 320)),
            "[t] WEG2-LAUNCH WEG2-PP-SPLIT group=P ... REALIZED layer split [29, 11, 8] (source: ...)",
        ]
    )
    got = P.stage_rates_from_boot(_fixture("fnFL2x163"), launch, name="NEU", source="x")
    assert got.stamps == P.rig_stamps(600, 320) and got.stage_layers == (29, 11, 8)
    assert (
        got.ms_per_chunk
        == P.stage_rates_from_logs(
            [_fixture("fnFL2x163")],
            name="n",
            source="s",
            model=P.NF_MODEL,
            trees="t",
            fr_p="f",
            stage_layers=(29, 11, 8),
            chunk_tokens=16384,
            stamps=P.rig_stamps(1, 1),
            stamp_source="s",
        ).ms_per_chunk
    )
    with pytest.raises(ValueError, match="aelter als H57"):
        P.stage_rates_from_boot(
            _fixture("fnFL2x163"), launch.split("\n", 1)[0], name="ALT", source="x"
        )


# --------------------------------------------------------------------------
# 4. Verdrahtung: eine Arm-Variable, keine Default-Aenderung, nichts angewendet
# --------------------------------------------------------------------------


def test_the_cut_stays_one_arm_variable_with_no_default():
    ap = L.build_parser()
    act = next(a for a in ap._actions if "--pp-stage-ratio" in a.option_strings)
    assert act.default is None
    assert "PP_RATIO in arm_fnFL2_long.sh" in act.help and "never applied" in act.help


def test_main_logs_the_power_limit_right_after_the_card_map_and_hands_it_to_the_cut():
    src = inspect.getsource(L.main)
    i_map = src.index('log("NVML -> CUDA ordinal map: "')
    i_pow = src.index("power_reading = log_power_limits(state, cards, log)")
    i_cut = src.index("cut = solve_p_cut(")
    assert i_map < i_pow < i_cut
    assert "power_reading=power_reading" in src[i_cut : i_cut + 300]


def test_solve_p_cut_prints_the_rate_cut_first_and_only_logs_it():
    fn = ast.parse(textwrap.dedent(inspect.getsource(L.solve_p_cut))).body[0]
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "rate_cut_line"
    ]
    assert len(calls) == 1
    outer = parents[calls[0]]
    assert isinstance(outer, ast.Call) and getattr(outer.func, "id", None) == "log"
    src = inspect.getsource(L.solve_p_cut)
    assert src.index("_power.rate_cut_line(") < src.index("p_card_verdict(")
    assert src.index("_power.rate_cut_line(") < src.index("_cut.solve_launch_cut(")


# --------------------------------------------------------------------------
# 5. Mutanten: die Tests oben unterscheiden das Richtige vom Falschen
# --------------------------------------------------------------------------


#: Die Originale, bevor ein Mutant sie im Modul ersetzt (sonst ruft er sich selbst).
_REAL_VERDICT = P.stamp_verdict
_REAL_BALANCE = P.balanced_cut


def _mutant_verdict_ignores_the_5090(stamps, rd):
    return _REAL_VERDICT([s for s in stamps if "5090" not in s.name], rd)


def _mutant_verdict_reads_the_max_limit(stamps, rd):
    fake = P.PowerReading(
        tuple(
            P.CardPower(c.nvml_index, c.uuid, c.name, c.max_w, c.max_w)
            for c in rd.cards
        ),
        rd.source,
    )
    return _REAL_VERDICT(stamps, fake)


@pytest.mark.parametrize(
    "mutant", [_mutant_verdict_ignores_the_5090, _mutant_verdict_reads_the_max_limit]
)
def test_a_verdict_mutant_misses_the_450_w_limit(mutant):
    rd = reading(w5090=450.0)
    ref = next(
        r for r in P.TIMED_REFERENCES if r.name == "launcher.MEASURED_MS_PER_LAYER"
    )
    assert P.stamp_verdict(ref.stamps, rd)[0] == "veraltet"
    with mock.patch.object(P, "stamp_verdict", mutant):
        mutated = P.reference_lines(rd)
    assert not any(
        ln.startswith(P.STALE_MARKER + " (Power-Limit nvml1 400->450 W)")
        for ln in mutated
    )


def _mutant_rates_ignore_the_limit(rd, stage_cards, model_name, rates=P.STAGE_RATES):
    return next((r for r in rates if r.model == model_name), None)


def _mutant_balance_by_layer_count(ms_per_layer, is_full_attention, pinned=None):
    return _REAL_BALANCE([1.0] * len(ms_per_layer), is_full_attention, pinned)


def _mutant_gpu_ms_with_partial_chunks(texts, chunk_tokens):
    rx = re.compile(
        r"PP(\d+)\] Prefill rank batch, #new-token: (\d+),.*?gpu-ms: ([0-9.]+)"
    )
    out = {}
    for t in texts:
        for m in rx.finditer(t):
            if int(m.group(2)) >= 8192:
                out.setdefault(int(m.group(1)), []).append(float(m.group(3)))
    return out


@pytest.mark.parametrize(
    "target,mutant",
    [
        ("rates_for", _mutant_rates_ignore_the_limit),
        ("balanced_cut", _mutant_balance_by_layer_count),
        ("full_chunk_compute_ms", _mutant_gpu_ms_with_partial_chunks),
    ],
)
def test_a_cut_mutant_turns_the_rate_tests_red(target, mutant):
    def verdicts():
        at_400 = cut_line(reading())
        at_450 = cut_line(reading(w5090=450.0))
        rates = P.stage_rates_from_logs(
            [_fixture(t) for t in ("fnFL2x160", "fnFL2x162", "fnFL2x163")],
            name=P.STAGE_RATES_FNFL2.name,
            source=P.STAGE_RATES_FNFL2.source,
            model=P.NF_MODEL,
            trees=P.STAGE_RATES_FNFL2.trees,
            fr_p=P.STAGE_RATES_FNFL2.fr_p,
            stage_layers=(29, 11, 8),
            chunk_tokens=16384,
            stamps=P.STAGE_RATES_FNFL2.stamps,
            stamp_source=P.STAGE_RATES_FNFL2.stamp_source,
        )
        return (
            "PP_RATIO=26,10,12 PP_ATTN_RATIO=6,3,3" in at_400,
            "keine Raten fuer nvml1=450W" in at_450,
            rates == P.STAGE_RATES_FNFL2,
        )

    assert verdicts() == (True, True, True)
    with mock.patch.object(P, target, mutant):
        assert verdicts() != (True, True, True)
