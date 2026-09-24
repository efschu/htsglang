# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H33 -- der Posten ausserhalb des D-Budgets steht GEMESSEN im Ledger.

DER BUG, schwarz-weiss (H30 R1): der Dry-Run liess auf der 5090 (D-TP0)
SCRATCH_D[0] 86 und FR_D[0] 0.10/0.15 durch -- W122 PASST, DECKE 0.150 --,
am Metall starb fnFL2x128 mit SCRATCH 86 an OOM ("74.81 MiB is free ... 4.75
GiB allocated in private pools (e.g., CUDA Graphs)"). Der W122-Ledger prueft
das BUDGET; das KV ist auf 262144 Token gedeckelt, der Budget-Rest bleibt
liegen, und was die KARTE fuellt -- freie Bloecke privater Pools (fuer
empty_cache unerreichbar) und die Transiente des schwersten Forwards --
stand in keinem Posten.

Alle Zahlen hier sind Logzeilen: ``fixtures/d_card_h33/`` sind die woertlichen
``[vram-peak]``-, ``#1027``-, Puffer-, Draft-Zensus- und OOM-Zeilen aus
/spinning/evidence-665-f1/boot_weg2_fnFL2x{141,144,128}_*.D.log.
"""

import json
import logging
import os
import types

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.planner import graph_pool_ledger as gpl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_card_h33")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
SLOT_BYTES = 1297637376 / 512
N_LAYERS = 48
LAYER_MIB = N_LAYERS * SLOT_BYTES / (1 << 20)
RATIOS = (183, 137, 168)
LIVE_BUDGETS = (29624, 18664, 18656)  # fnFL2x141 "budget D group=D"
DRY_BUDGETS = (29368, 18184, 17784)  # H30-Leiter "budget D(dry, expectation)"
VOCAB_MIB = er.draft_vocab_mib(vocab_size=248320, hidden_size=2560)
NEAR_OOM = 400.0
BAND_FLOOR = 819.0


def _text(name):
    with open(os.path.join(FIX, name + ".D.lines")) as fh:
        return fh.read()


def _ref(*names):
    return er.d_card_reference_from_logs(
        [(n, _text(n)) for n in names],
        n_ranks=3,
        n_layers=N_LAYERS,
        slot_bytes=SLOT_BYTES,
        model=MODEL,
        rank_tp_ratio="1,0,0",
    )


def _cards(fractions, scratch, *, budgets=LIVE_BUDGETS, reference=None, share=True):
    fits = er.solve_d_rank_residency(
        budgets_mib=budgets,
        fractions=fractions,
        ratios=RATIOS,
        scratch_rows=scratch,
        staging_rows=8,
        num_experts=512,
        pad_rows=1,
        n_layers=N_LAYERS,
        slot_bytes=SLOT_BYTES,
        reference=er.D_RESIDENCY_REFERENCE_FNFL2,
        vocab_mib=VOCAB_MIB,
        share_embed=share,
        kv_tokens=262144,
    )
    ref = reference or er.D_CARD_REFERENCE_FNFL2
    cards = er.solve_d_card(
        fits=fits,
        reference=ref,
        vocab_mib=VOCAB_MIB,
        share_embed=share,
        near_oom_mib=NEAR_OOM,
        band_floor_mib=BAND_FLOOR,
    )
    return fits, cards, ref


X141 = ((0.06, 0.44, 0.365), (82, 48, 48))
X128 = ((0.06, 0.44, 0.365), (86, 48, 48))


# ---------------------------------------------------------------------------
# 1. das Instrument misst, was am Metall entschied -- in BEIDE Richtungen
# ---------------------------------------------------------------------------


def test_the_headroom_separates_the_boot_that_died_from_the_two_that_lived():
    """Indikator-Gesetz: der Kopfraum ``cap - peak - privat_frei`` liegt bei
    beiden Ueberlebenden (x141, x144) ueber der near-OOM-Grenze und bei x128
    (OOM) darunter. Die naheliegende Alternative, ``card free`` im Extend,
    trennt NICHT: x144 lief mit 102 MiB frei im Extend (TP0) und x141/x144 TP2
    mit 41/61 MiB -- der allgemeine Cache ist Luft, die der Allokator abgibt
    (alloc retries), der private nicht."""
    got = {}
    for boot in ("fnFL2x141", "fnFL2x144", "fnFL2x128"):
        s = gpl.samples_from_log(_text(boot))
        got[boot] = {r: gpl.binding_sample(ss) for r, ss in s.items()}
    for boot in ("fnFL2x141", "fnFL2x144"):
        for r in range(3):
            assert got[boot][r].headroom_mib >= NEAR_OOM, (boot, r)
    assert got["fnFL2x128"][0].headroom_mib < NEAR_OOM
    # die Terme selbst, woertlich aus den Zeilen (TP0 decode, x141):
    b = got["fnFL2x141"][0]
    assert b.phase == "decode" and b.source == gpl.SOURCE_LEGACY
    assert b.private_free_mib == pytest.approx(5235679744 / (1 << 20))
    assert b.card_free_mib == pytest.approx(1.05 * 1024)
    assert b.headroom_mib == pytest.approx(618.4, abs=0.2)
    # x128: dieselbe Zahl, die torch im OOM-Text "allocated in private pools" nennt
    assert got["fnFL2x128"][0].private_free_mib / 1024 == pytest.approx(4.75, abs=0.005)
    assert "4.75 GiB allocated in private pools" in _text("fnFL2x128")
    # die Gegenprobe, dass card free im Extend kein Survival-Mass ist:
    ext = [
        s
        for s in gpl.samples_from_log(_text("fnFL2x144"))[0]
        if s.phase == "extend"
    ]
    assert min(s.card_free_mib for s in ext) < NEAR_OOM


def test_the_shipped_card_reference_is_the_logs_own_measurement():
    assert _ref("fnFL2x141", "fnFL2x144") == er.D_CARD_REFERENCE_FNFL2


def test_a_log_without_the_private_term_is_refused_not_zeroed():
    """Ohne ``#1027``/``WEG2-GRAPH-POOL`` gibt es keinen privaten Term; die
    Referenz verweigert, statt ihn als 0 (= 4993 MiB zu viel Kopfraum) zu
    buchen."""
    text = "\n".join(
        ln for ln in _text("fnFL2x141").splitlines() if "#1027" not in ln
    )
    with pytest.raises(ValueError, match="ohne gemessenen privaten Term"):
        er.d_card_reference_from_logs(
            [("x141-ohne-1027", text)],
            n_ranks=3,
            n_layers=N_LAYERS,
            slot_bytes=SLOT_BYTES,
            model=MODEL,
            rank_tp_ratio="1,0,0",
        )


# ---------------------------------------------------------------------------
# 2. die Ledger-Rechnung reproduziert die Metall-Bilanz
# ---------------------------------------------------------------------------


def test_the_ledger_reproduces_the_x141_decode_free_on_the_5090():
    """x141-Form mit dem x141/x144-Term: Decode frei 1,05 GiB (Metall) +-
    Zensus-Genauigkeit (0.005 GiB je gedrucktem Term), Kopfraum 608-618 MiB."""
    fits, cards, ref = _cards(*X141)
    assert all(not f.refused for f in fits)
    c0 = cards[0]
    assert c0.buffer_rows == 94 and c0.expert_delta_mib == 0.0
    assert c0.free_decode_mib == pytest.approx(1.05 * 1024, abs=ref.precision_mib[0])
    assert 608.0 <= c0.headroom_mib <= 618.5
    assert c0.verdict == "PASST" and not any(c.refused for c in cards)
    # die Worker: dieselbe Rechnung, Metall x141 decode card free 4.32 / 4.13 GiB
    assert cards[1].free_decode_mib == pytest.approx(4.32 * 1024, abs=ref.precision_mib[1])
    assert cards[2].free_decode_mib == pytest.approx(4.13 * 1024, abs=ref.precision_mib[2])


def test_scratch_86_on_the_5090_is_refused_like_the_metal_killed_x128():
    """x128s argv (SCRATCH_D 86,48,48): W122 PASST (Budget), die Karte nicht."""
    fits, cards, ref = _cards(*X128)
    assert all(not f.refused for f in fits)  # das Budget-Ledger laesst es durch
    c0 = cards[0]
    assert c0.buffer_rows == 98
    assert c0.expert_delta_mib == pytest.approx(4 * LAYER_MIB)
    assert c0.refused and c0.verdict == "STIRBT AN DER KARTE"
    text = er.card_refusal_text(cards, ref, label="D(dry, expectation)")
    assert text.startswith("W130 Weg2DCardNearOom") and "rang0" in text
    # konservativ gegen das Metall: vorhergesagt ~144 MiB, x128 mass 289 MiB,
    # beide unter der Grenze -- die Differenz ist x128s kleinerer privater Pool
    # (4861 statt 4993 MiB, anderer Baum a1dab24409).
    measured = gpl.binding_sample(gpl.samples_from_log(_text("fnFL2x128"))[0])
    assert c0.headroom_mib <= measured.headroom_mib < NEAR_OOM


@pytest.mark.parametrize("f0", [0.10, 0.15])
def test_the_h30_ladder_tp0_rungs_are_refused(f0):
    """H30 L1/L3: FR_D[0] 0.10/0.15 liefen im Dry-Run mit rc=0 durch."""
    fits, cards, _ = _cards((f0, 0.44, 0.365), (82, 48, 48), budgets=DRY_BUDGETS)
    assert all(not f.refused for f in fits)
    assert cards[0].refused and not cards[1].refused and not cards[2].refused


def test_the_mutant_without_the_private_term_passes_falsely():
    """Der Posten ist tragend: ohne privat_frei (die Ledger-Form vor H33 --
    ``cap - peak`` allein) passt SCRATCH 86 faelschlich, und sogar 0.15."""
    ref = er.D_CARD_REFERENCE_FNFL2
    mutant = msgspec.structs.replace(
        ref,
        headroom0_mib=tuple(h + p for h, p in zip(ref.headroom0_mib, ref.private_free_mib)),
    )
    for args in (X128, ((0.15, 0.44, 0.365), (82, 48, 48))):
        _, cards, _ = _cards(*args, reference=mutant)
        assert not any(c.refused for c in cards)
        _, cards, _ = _cards(*args)
        assert cards[0].refused


def test_the_card_ceiling_and_the_h30_k_stage():
    """KARTEN-DECKE auf der 5090: +1 Zeile ueber x141 (f 0.067 bei S 82); die
    H30-Stufe K (0.06,0.51,0.48) passt die Karte auf allen Raengen."""
    _, cards, _ = _cards(*X141)
    assert [c.ceiling_fraction for c in cards] == [0.067, 0.634, 0.519]
    assert cards[0].ceiling_max_rows == 95
    _, k, _ = _cards((0.06, 0.51, 0.48), (82, 48, 48), budgets=DRY_BUDGETS)
    assert not any(c.refused for c in k)
    _, over, _ = _cards((0.068, 0.44, 0.365), (82, 48, 48))
    assert over[0].buffer_rows == 96 and over[0].refused


def test_the_unshared_draft_vocab_is_charged_on_the_draft_host_only():
    _, shared, _ = _cards(*X141)
    _, own, _ = _cards(*X141, share=False)
    assert own[0].vocab_delta_mib == pytest.approx(VOCAB_MIB)
    assert own[0].headroom_mib == pytest.approx(shared[0].headroom_mib - VOCAB_MIB)
    assert own[1].headroom_mib == shared[1].headroom_mib


# ---------------------------------------------------------------------------
# 3. die Instrumentzeile: Format, Leser, Spiegel der Runtime-Quelle
# ---------------------------------------------------------------------------


def _segments():
    return [
        {"segment_pool_id": (0, 0), "total_size": 3 << 30, "allocated_size": 1 << 30},
        {"segment_pool_id": (1, 7), "total_size": 5 << 30, "allocated_size": 1 << 29},
        {"segment_pool_id": (1, 7), "total_size": 1 << 30, "allocated_size": 1 << 30},
        {"owner_private_pool_id": (2, 1), "total_size": 1 << 28, "allocated_size": 0},
        "not-a-segment",
    ]


def test_segment_reading_mirrors_the_runtime_source():
    from sglang.srt.managers import phase_flip_runtime as pfr

    segs = _segments()
    for s in segs:
        if isinstance(s, dict):
            assert gpl.segment_pool_id(s) == pfr._segment_pool_id(s)
    pools = gpl.private_pools_from_segments(segs)
    free = sum(t - u for t, u in pools.values())
    assert free == pfr.graph_pool_free_bytes_from_segments(segs)


def test_the_instrument_line_round_trips_and_supersedes_the_legacy_reading():
    s = gpl.sample_from_stats(
        rank=0,
        phase="decode",
        free_bytes=1100 << 20,
        total_bytes=32088 << 20,
        reserved_bytes=28000 << 20,
        allocated_bytes=22000 << 20,
        peak_bytes=23000 << 20,
        segments=_segments(),
    )
    assert s.private_total_mib == pytest.approx(6400.0)
    assert s.private_free_mib == pytest.approx(4864.0)
    assert s.headroom_mib == pytest.approx(1100 + 28000 - 23000 - 4864)
    line = "[2026-09-24 12:00:00 TP0] " + gpl.format_line(s)
    for field in ("captured_mib=6400", "reserved_after_mib=28000", "allocator_cache_mib=6000"):
        assert field in line
    legacy = _text("fnFL2x141")
    got = gpl.samples_from_log(legacy + "\n" + line)
    assert [x.source for x in got[0]] == [gpl.SOURCE_INSTRUMENT]
    assert got[0][0].headroom_mib == pytest.approx(s.headroom_mib, abs=1.0)
    assert got[1][0].source == gpl.SOURCE_LEGACY  # andere Raenge unberuehrt
    blind = gpl.sample_from_stats(
        rank=0, phase="decode", free_bytes=1, total_bytes=2, reserved_bytes=1,
        allocated_bytes=1, peak_bytes=1, segments=None)
    assert not gpl.usable(blind)


class _FakeCuda:
    def __init__(self):
        self.snapshots = 0

    def mem_get_info(self):
        return (1100 << 20, 32088 << 20)

    def memory_reserved(self):
        return 28000 << 20

    def memory_allocated(self):
        return 22000 << 20

    def max_memory_allocated(self):
        return 23000 << 20

    def memory_stats(self):
        return {"inactive_split_bytes.all.current": 0, "num_alloc_retries": 0}

    def memory_snapshot(self):
        self.snapshots += 1
        return _segments()

    def is_current_stream_capturing(self):
        return False


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_extend(self):
        return self.kind == "extend"

    def is_decode(self):
        return self.kind == "decode"


def _batch(kind, n):
    return types.SimpleNamespace(
        forward_mode=_Mode(kind), input_ids=types.SimpleNamespace(shape=(n,))
    )


def test_the_runtime_hook_records_after_capture_and_beside_vram_peak(caplog, monkeypatch):
    from sglang.srt.model_executor import vram_family_census as vfc
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    monkeypatch.setitem(vfc._GRAPH_POOL_STATE, "post_capture", False)
    cuda = _FakeCuda()
    runner = types.SimpleNamespace(tp_rank=0, _rank_vector_index=lambda: 0)
    caplog.set_level(logging.INFO, logger=vfc.logger.name)
    with model_capture_mode():
        vfc.maybe_log_vram_peak(runner, _batch("extend", 4096), cuda=cuda)
    assert cuda.snapshots == 0  # nie im Capture
    runner = types.SimpleNamespace(tp_rank=0, _rank_vector_index=lambda: 0)
    vfc.maybe_log_vram_peak(runner, _batch("decode", 1), cuda=cuda)  # post-capture + high-water
    vfc.maybe_log_vram_peak(runner, _batch("decode", 1), cuda=cuda)  # nichts
    vfc.maybe_log_vram_peak(runner, _batch("extend", 4096), cuda=cuda)  # extend
    recs = [r.getMessage() for r in caplog.records if gpl.MARKER in r.getMessage()]
    phases = [m.split("phase=")[1].split()[0] for m in recs]
    assert phases == ["post-capture", "high-water", "extend"], recs
    got = gpl.samples_from_log("\n".join("[x TP0] " + m for m in recs))
    assert [s.phase for s in got[0]] == phases
    assert all(s.headroom_mib == pytest.approx(1100 + 28000 - 23000 - 4864, abs=1)
               for s in got[0])


# ---------------------------------------------------------------------------
# 4. die Launcher-Naht: der Dry-Run verweigert x128 vor dem Laden
# ---------------------------------------------------------------------------


def _launcher_ns(tmp_path, fractions, scratch, card_logs=""):
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction " + fractions),
        env_d=("SGLANG_MOE_POOL_STAGING=12;SGLANG_MOE_SCRATCH_SLOTS=" + scratch
               + ";SGLANG_UNEVEN_MOE_EXPERT_SHARD=1"),
        d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
        d_residency_reference_logs="", d_card_reference_logs=card_logs,
        wake_credit_reference_logs="",
    )


def test_the_launcher_refuses_x128_and_the_h30_rungs_before_a_rank_loads(
    tmp_path, monkeypatch
):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    for i, (fr, sc, budgets) in enumerate((
        ("0.06,0.44,0.365", "86,48,48", LIVE_BUDGETS),   # x128
        ("0.10,0.44,0.365", "82,48,48", DRY_BUDGETS),    # H30 L1
        ("0.15,0.44,0.365", "82,48,48", DRY_BUDGETS),    # H30 L3
    )):
        lines = []
        with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 Weg2DCardNearOom"):
            launcher.log_d_rank_vram_solve(
                _launcher_ns(tmp_path / str(i), fr, sc), cards, budgets,
                lines.append, "D(dry, expectation)")
        assert any("FRACTION-SOLVE D(dry, expectation) rang0" in ln and "PASST" in ln
                   for ln in lines), lines
        assert any("KARTE D(dry, expectation) rang0" in ln
                   and "STIRBT AN DER KARTE" in ln for ln in lines), lines
    lines = []
    launcher.log_d_rank_vram_solve(
        _launcher_ns(tmp_path / "x141", "0.06,0.44,0.365", "82,48,48"), cards,
        LIVE_BUDGETS, lines.append, "D")
    head = [ln for ln in lines if "KARTE D (H33" in ln]
    assert head and "KARTEN-DECKE je Rang ['0.067', '0.634', '0.519']" in head[0], lines
    # dieselbe Antwort aus den Logs selbst (--d-card-reference-logs)
    logs = ",".join(os.path.join(FIX, b + ".D.lines") for b in ("fnFL2x141", "fnFL2x144"))
    lines = []
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 "):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "logs", "0.06,0.44,0.365", "86,48,48", logs),
            cards, LIVE_BUDGETS, lines.append, "D")
    assert any("fnFL2x141.D.lines + fnFL2x144.D.lines" in ln for ln in lines)
