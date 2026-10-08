# SPDX-License-Identifier: Apache-2.0
"""D->P-Flip je PP-Stufe (30.09., NF y4h 351aa9c20f): wann ist welches Band da?

Der Auftrag "P beginnt den ersten Prefill-Chunk, sobald PP0 sein Band hat"
stand auf der Annahme, der kritische Pfad sei das P-Wake-Leg von PP0
(FLIPZEIT-VERLAUF-0929). Auf y4h bindet in 37 von 39 D->P-Flips der
SCHLAFENDE D-Rang TP1 (3080 nvml0); PP0 hat seine Bytes erst, wenn TP1 seine
letzten Deposits geschrieben hat, und der Basis-Tag (PP0s Einbettung) schliesst
per Vertrag jedes Leg. Der Leser ``pdflip.tools.dp_stage_legs`` macht das je Boot
sichtbar: je Stufe Band-Bereitschaft, je D-Rang Leg-Ende und Pausenanteil, und
das Was-waere-wenn der Umordnung (V1 Vertrag gehalten, V2 Basis-Tag frueh).

Die synthetischen Logs unten sind so gebaut, dass jede Zahl von Hand
nachrechenbar ist (Kosten je D-Rang und Tag im Kopf der Datei); der letzte
Test haelt die Regexe an woertlichen y4h-Zeilen fest.
"""

from __future__ import annotations

import datetime as dt

import pytest

from flliper.srt.pdflip.tools import dp_stage_legs as dsl
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

T0 = 1790755950.0          # P leg start = D leg start (epoch s)
BEGIN = T0 - 0.300          # PDFLIP-FLIP begin
DONE = T0 + 0.260           # PDFLIP-FLIP done
#: measured order and per-tag total_ms per D rank; TP1/TP2 hold no piece of
#: the base tag (their step is a pieces=0 no-op, as on y4h)
ORDER = ("weights_2", "weights_1", "weights_0", "weights")
COST = {"TP0": (10, 10, 10, 150), "TP1": (100, 10, 100, 5), "TP2": (10, 50, 50, 5)}
PAUSE = {"TP0": 2, "TP1": 5, "TP2": 4}
NOOP = {("TP1", "weights"), ("TP2", "weights")}
#: P side: the band of each stage (h2d MiB), base on PP0 (embedding) and PP2 (head)
BYTES = {"PP0": {"weights_0": 1700, "weights": 630},
         "PP1": {"weights_1": 3100},
         "PP2": {"weights_2": 4000, "weights": 650}}
CARD = {"PP0": "GPU-31d7ef41-f574", "PP1": "GPU-5c648f96-be1d", "PP2": "GPU-62dbbae1-e859"}
CHUNK_MS = {"PP0": 1001.5, "PP1": 250.0, "PP2": 40.0}


def _front_ts(t):
    d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
    return "[%s,%03d]" % (d.strftime("%Y-%m-%d %H:%M:%S"), int(round((t % 1) * 1000)) % 1000)


def _rank_ts(t, rank):
    d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
    return "[%s %s]" % (d.strftime("%Y-%m-%d %H:%M:%S"), rank)


def _front():
    return [
        "%s INFO pdflip.front: PDFLIP-FLIP begin epoch=2 sleep=D wake=P outstanding=0 queue=1\n" % _front_ts(BEGIN),
        "%s INFO pdflip.front: PDFLIP-FLIP done epoch=3 slept=D woke=P drain+quiesce=10 ms\n" % _front_ts(DONE),
        # a P->D flip afterwards: never read as D->P
        "%s INFO pdflip.front: PDFLIP-FLIP begin epoch=3 sleep=P wake=D outstanding=0 queue=0\n" % _front_ts(T0 + 20),
        "%s INFO pdflip.front: PDFLIP-FLIP done epoch=4 slept=P woke=D drain+quiesce=10 ms\n" % _front_ts(T0 + 22),
    ]


def _d_log(with_noop=True):
    out = []
    for r, costs in COST.items():
        t = T0
        for tag, c in zip(ORDER, costs):
            if with_noop and (r, tag) in NOOP:
                out.append("%s PDFLIP-XCHG DEPOSIT tag=%s group=D rank=%s pieces=0 resident_bytes=0 "
                           "-- this rank's plan carries no desc for this tag\n" % (_rank_ts(t, r), tag, r[-1]))
            pause = 0 if (r, tag) in NOOP else PAUSE[r]
            out.append(
                "%s PDFLIP-SLEEP-TAG-TIME tag=%s deposit_ms=%d sync_ms=0 pause_ms=%d credit_ms=0 gap_ms=0 "
                "total_ms=%d t0=%.3f t=%.3f\n" % (_rank_ts(t, r), tag, c - pause, pause, c, t, t + c / 1000.0))
            t += c / 1000.0
    # a P->D sleep of D later on: outside every D->P window
    out.append("%s PDFLIP-SLEEP-TAG-TIME tag=weights_0 deposit_ms=5 sync_ms=0 pause_ms=1 credit_ms=0 gap_ms=0 "
               "total_ms=6 t0=%.3f t=%.3f\n" % (_rank_ts(T0 + 21, "TP0"), T0 + 21, T0 + 21.006))
    return out


def _p_log():
    out = []
    ends = {"PP0": 0.150, "PP1": 0.090, "PP2": 0.120}
    for s in ("PP0", "PP1", "PP2"):
        t = T0
        for i, tag in enumerate(ORDER):
            te = T0 + ends[s] * (i + 1) / len(ORDER)
            out.append("%s PDFLIP-WAKE-TAG-TIME tag=%s resume_ms=5 t0=%.3f t=%.3f\n" % (_rank_ts(t, s), tag, t, te))
            t = te
    for s in ("PP0", "PP1", "PP2"):
        for tag in ORDER:
            out.append("%s PDFLIP-FLIP-TAG group=P rank=%s card=%s dir=h2d tag=%s bytes=%d MiB "
                       "population=weights-family ms=3\n" % (_rank_ts(T0 + 0.2, s), s[-1], CARD[s], tag,
                                                             BYTES[s].get(tag, 0)))
    for s, lc in (("PP0", 212), ("PP1", 112), ("PP2", 182)):
        out.append("%s PDFLIP-WAKE-TAIL ms pre_leg=0 read_early=0 leg_collects=%d reload=10 dest_hook_compare=2 "
                   "seam_after=0 nvfp4_marlin=0 fence=2 t=%.3f\n" % (_rank_ts(T0 + 0.23, s), lc, T0 + 0.23))
    for s in ("PP0", "PP1", "PP2"):
        for k in range(2):  # only the first prefill after done counts
            out.append("%s Prefill rank batch, #new-token: 77, #cached-token: 0, #chunks: 1, gpu-ms: %.1f "
                       "(compute %.1f, wait 0.0)\n" % (_rank_ts(DONE + 1 + k, s), CHUNK_MS[s] + 7 * k,
                                                       CHUNK_MS[s]))
    return out


def _flips(with_noop=True):
    flips = dsl.flips_from_front(_front())
    dsl.attach_rank_logs(flips, _p_log(), _d_log(with_noop))
    return flips


def test_only_d_to_p_flips_are_read():
    flips = dsl.flips_from_front(_front())
    assert len(flips) == 1
    assert flips[0]["begin"] == pytest.approx(BEGIN, abs=1e-3)
    assert flips[0]["done"] == pytest.approx(DONE, abs=1e-3)


def test_decompose_per_stage_and_d_rank():
    (f,) = _flips()
    dec = dsl.decompose(f)
    assert dec["leg_start_from_begin"] == pytest.approx(300, abs=1)
    # band_ready: the last D deposit of the stage's own tags over all D ranks
    #   PP0 {weights_0, weights}: TP0 180 (base), TP1 210 (weights_0), TP2 110
    #   PP1 {weights_1}: 20 / 110 / 60;  PP2 {weights_2, weights}: 180 / 100 / 10
    st = dec["stages"]
    assert st["PP0"]["band_ready"] == pytest.approx(210, abs=1)
    assert st["PP1"]["band_ready"] == pytest.approx(110, abs=1)
    assert st["PP2"]["band_ready"] == pytest.approx(180, abs=1)
    assert (st["PP0"]["bytes_mib"], st["PP0"]["tags"]) == (2330, 2)
    assert (st["PP1"]["bytes_mib"], st["PP1"]["tags"]) == (3100, 1)
    assert st["PP0"]["card"].startswith("GPU-31d7")
    # resume_end is the mapping, collects_end the bytes (leg start + leg_collects)
    assert st["PP0"]["resume_end"] == pytest.approx(150, abs=1)
    assert st["PP0"]["collects_end"] == pytest.approx(212, abs=1)
    # the D rank with the latest leg end binds
    assert dec["crit"] == "TP1"
    assert dec["legs_end"] == pytest.approx(215, abs=1)
    assert dec["d"]["TP1"]["pause_ms"] == 3 * 5  # the no-op base step pauses nothing
    assert dec["d"]["TP1"]["deposit_ms"] == (100 - 5) + (10 - 5) + (100 - 5) + 5
    assert dec["d"]["TP0"]["leg_end"] == pytest.approx(180, abs=1)


def test_noop_step_gates_no_band():
    """TP1/TP2 walk the base tag's lockstep step without a piece (pieces=0).
    Read as a deposit, their late step would date PP0's and PP2's band to the
    end of TP1's chain (215) -- the band that the collects show complete
    earlier."""
    (with_noop,) = _flips(True)
    (without,) = _flips(False)
    a, b = dsl.decompose(with_noop), dsl.decompose(without)
    assert a["stages"]["PP2"]["band_ready"] == pytest.approx(180, abs=1)
    assert b["stages"]["PP2"]["band_ready"] == pytest.approx(215, abs=1)
    assert b["stages"]["PP0"]["band_ready"] == pytest.approx(215, abs=1)


def test_what_if_orders():
    (f,) = _flips()
    w = dsl.what_if(f)
    # V0 recomputes the measured order and must hit band_ready
    assert w["V0"] == pytest.approx({"PP0": 210, "PP1": 110, "PP2": 180}, abs=1)
    # V1: PP0's bands first, base LAST (the family contract):
    #   TP0 10/20/30/180, TP1 100/110/210/215, TP2 50/100/110/115
    assert w["V1"] == pytest.approx({"PP0": 180, "PP1": 110, "PP2": 210}, abs=1)
    # V2: base right behind PP0's bands:
    #   TP0 10/160/170/180, TP1 100/105/115/215, TP2 50/55/105/115
    assert w["V2"] == pytest.approx({"PP0": 160, "PP1": 170, "PP2": 215}, abs=1)
    orders = dsl._variant_orders(list(ORDER), dsl.own_tags(f))
    assert orders["V1"][-1] == dsl.BASE_TAG
    assert orders["V2"] == ["weights_0", "weights", "weights_1", "weights_2"]
    for o in orders.values():
        assert sorted(o) == sorted(ORDER)


def test_first_chunk_and_report(tmp_path):
    stem = str(tmp_path / "boot_weg2_synth_0930")
    for kind, lines in (("front", _front()), ("P", _p_log()), ("D", _d_log())):
        with open("%s.%s.log" % (stem, kind), "w") as fh:
            fh.writelines(lines)
    rows = dsl.analyze(stem)
    assert len(rows) == 1
    assert rows[0]["first_chunk_ms"] == pytest.approx(CHUNK_MS)
    text = dsl.report(rows)
    assert text.startswith("DP-STAGE-LEGS n=1")
    assert "crit={'TP1': 1}" in text
    assert "WHAT-IF V2 band_ready PP0     160/    160" in text
    assert "PP0-Gewinn      55/     55" in text  # 215 - 160
    assert dsl.main([stem + ".front.log"]) == 0


# woertliche Zeilen aus boot_..._351aa9c20f_0930_080853 (y4h, 30.09.)
Y4H = {
    "front_begin": "[2026-09-30 08:12:30,692] INFO pdflip.front: PDFLIP-FLIP begin epoch=2 sleep=D wake=P outstanding=0 queue=1",
    "front_done": "[2026-09-30 08:12:32,596] INFO pdflip.front: PDFLIP-FLIP done epoch=3 slept=D woke=P drain+quiesce=136 ms sleep=1758 ms (kv RPC + the D leg of the gathered pair) wake=1611 ms",
    "sleep": "[2026-09-30 08:12:31 TP1] PDFLIP-SLEEP-TAG-TIME tag=weights_0 deposit_ms=79 sync_ms=0 pause_ms=25 credit_ms=0 gap_ms=0 total_ms=104 t0=1790755950.990 t=1790755951.094",
    "noop": "[2026-09-30 08:12:32 TP2] PDFLIP-XCHG DEPOSIT tag=weights group=D rank=2 pieces=0 resident_bytes=2097152 -- this rank's plan carries no desc for this tag, so the lockstep step is a no-op",
    "wake": "[2026-09-30 08:12:30 PP0] PDFLIP-WAKE-TAG-TIME tag=weights_0 resume_ms=10 t0=1790755950.967 t=1790755950.977",
    "flip_tag": "[2026-09-30 08:12:32 PP0] PDFLIP-FLIP-TAG group=P rank=0 card=GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d dir=h2d tag=weights_0 bytes=1742 MiB population=weights-family (source: tms_tag_bytes, NOT RssShmem) ms=10 GB/s=177.96 granules=871 allocations=22 map_ms=1.4 copy_ms=0.0",
    "tail": "[2026-09-30 08:12:32 PP0] PDFLIP-WAKE-TAIL ms pre_leg=0 read_early=0 leg_collects=1556 reload=32 dest_hook_compare=9 seam_after=0 nvfp4_marlin=2 fence=2 t=1790755952.567",
    "prefill": "[2026-09-30 08:12:35 PP0] Prefill rank batch, #new-token: 203, #cached-token: 93632, #chunks: 1, gpu-ms: 1246.7 (compute 1246.5, wait 0.2) (wait by family: tp.all_reduce 0.2/29x) bubble_ms=26453.5 (between forwards, mb=0)",
}


def test_regexes_hold_on_y4h_lines():
    flips = dsl.flips_from_front([Y4H["front_begin"], Y4H["front_done"]])
    assert len(flips) == 1 and flips[0]["done"] - flips[0]["begin"] == pytest.approx(1.904, abs=1e-3)
    m = dsl._RX_SLEEP_TAG.search(Y4H["sleep"])
    assert m.group(1, 2, 3, 4, 5) == ("TP1", "weights_0", "79", "25", "104")
    assert dsl._RX_NOOP_DEPOSIT.search(Y4H["noop"]).group(1, 2) == ("weights", "2")
    assert dsl._RX_WAKE_TAG.search(Y4H["wake"]).group(1, 2) == ("PP0", "weights_0")
    m = dsl._RX_FLIP_TAG.search(Y4H["flip_tag"])
    assert m.group(1, 3, 4) == ("PP0", "weights_0", "1742")
    assert dsl._RX_WAKE_TAIL.search(Y4H["tail"]).group(1, 2, 3) == ("PP0", "1556", "1790755952.567")
    m = dsl._RX_PREFILL.search(Y4H["prefill"])
    assert m.group(2, 3, 5) == ("PP0", "203", "1246.7")
    assert dsl._rank_ts(Y4H["prefill"]) == pytest.approx(1790755955.0)
