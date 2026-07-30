# SPDX-License-Identifier: Apache-2.0
"""Dry tests for the s12 compute/wait analysis (#293, Schritt 1).

Hermetic and CPU-only: no card, no host, no ssh. The inputs are REAL server
lines of the 2026-07-30 s12 run, not reconstructions -- see PROVENIENZ.md next
to the fixtures. That matters more here than in most parsers: the whole verdict
of #293 step 1 rests on four numbers pulled out of these lines with a regex,
and a regex that silently stops matching would turn "no evidence of contention"
into "no evidence at all" without changing a single test that only checked
shapes.

So the assertions are the numbers themselves. The ones in
docs/dev/INTEGRATION_R3_VALIDATION.md, section "#293 Schritt 1", are the same
ones; changing one without the other breaks this file on purpose.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "gpu_battery",
    "s12_wait_analyse",
)

sys.path.insert(0, BATTERY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bar1_marker_source as _src  # noqa: E402
from s12_log_analyse import (  # noqa: E402
    auswerten,
    belegung,
    decode_tick_aggregat,
    im_fenster,
    kollektiv_bytes,
    lade_punkte,
    parse_bar1_geometrie,
    parse_decode,
    parse_prefill_rang,
    punkt_fenster,
    runden,
    tabellen,
    wait_aggregat,
)

# The lines the scheduler and htccl_bar1 really write. Copied verbatim out of
# /root/battery-bar1/s12.bar1.1.log and s12.grundlinie.8.log.
ZEILE_PREFILL_BAR1 = (
    "[2026-07-30 13:34:52 TP0] Prefill rank batch, #new-token: 2048, "
    "#cached-token: 0, #chunks: 1, gpu-ms: 1194.7 (compute 143.6, wait 1051.1)"
)
ZEILE_PREFILL_GRUNDLINIE = (
    "[2026-07-30 13:47:31 TP1] Prefill rank batch, #new-token: 80, "
    "#cached-token: 0, #chunks: 1, gpu-ms: 668.4 (compute 463.5, wait 205.0)"
)
ZEILE_DECODE = (
    "[2026-07-30 13:35:13 TP0] Decode batch, #running-req: 1, #full token: 257, "
    "full token usage: 0.00, mamba num: 3, mamba usage: 0.03, accept len: 2.83, "
    "accept rate: 0.61, cuda graph: True, gen throughput (token/s): 93.65, "
    "#queue-req: 0"
)
ZEILE_DECODE_WARMUP = (
    "[2026-07-30 13:35:12 TP0] Decode batch, #running-req: 1, #full token: 143, "
    "full token usage: 0.00, mamba num: 3, mamba usage: 0.03, accept len: 3.35, "
    "accept rate: 0.78, cuda graph: True, gen throughput (token/s): 3.37, "
    "#queue-req: 0"
)
# #315: built from the ACTUAL htccl_bar1.py format string via
# _bar1_marker_source.py, not retyped -- this used to be the German wording
# #295 moved the emitter away from, matching only itself and RE_BAR1_SETUP's
# equally-stale pattern.
ZEILE_AUFBAU = "[2026-07-30 13:44:02 TP2] " + _src.render_setup_line(
    dauer_ms=324, peer_targets=2, region_mib=96.0,
    slots_desc="12 slots (of which 2(R-1) for all_to_all)",
    slot_kib=8188, payload_kib=24564,
)
# A line that carries no split, from before #252. It must not parse as a batch
# with wait 0 -- that would read as "no collective time at all".
ZEILE_OHNE_SPLIT = (
    "[2026-07-30 13:34:52 TP0] Prefill rank batch, #new-token: 2048, "
    "#cached-token: 0, #chunks: 1, gpu-ms: 1194.7"
)


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


def _quellen() -> list:
    return [
        ("bar1", 1, _fixture("bar1_1.log")),
        ("bar1", 8, _fixture("bar1_8.log")),
        ("grundlinie", 1, _fixture("grundlinie_1.log")),
        ("grundlinie", 8, _fixture("grundlinie_8.log")),
    ]


@pytest.fixture(scope="module")
def payload():
    return auswerten(
        _quellen(), lade_punkte(_fixture("punkte.jsonl")), hidden=5120, welt=3
    )


def _w(payload, arm, sessions, rang):
    for w in payload["wait"]:
        if (w["arm"], w["sessions"], w["rang"]) == (arm, sessions, rang):
            return w
    raise AssertionError(f"kein Aggregat fuer {arm}/{sessions}/TP{rang}")


class TestZeilen:
    def test_prefill_split_is_read_field_by_field(self):
        (row,) = parse_prefill_rang([ZEILE_PREFILL_BAR1])
        assert row["rang"] == 0
        assert row["new_token"] == 2048
        assert row["chunks"] == 1
        assert row["gpu_ms"] == 1194.7
        assert row["compute_ms"] == 143.6
        assert row["wait_ms"] == 1051.1
        # The split has to add up, or the two halves are not the same batch.
        assert row["compute_ms"] + row["wait_ms"] == pytest.approx(
            row["gpu_ms"], abs=0.2
        )

    def test_a_line_without_the_split_is_not_a_batch(self):
        """Silence, not a zero. A pre-#252 log has no wait to report."""
        assert parse_prefill_rang([ZEILE_OHNE_SPLIT]) == []

    def test_every_rank_prefix_is_kept_apart(self):
        rows = parse_prefill_rang([ZEILE_PREFILL_BAR1, ZEILE_PREFILL_GRUNDLINIE])
        assert [r["rang"] for r in rows] == [0, 1]

    def test_decode_line_carries_accept_and_rate(self):
        (row,) = parse_decode([ZEILE_DECODE])
        assert row["running_req"] == 1
        assert row["accept_len"] == 2.83
        assert row["gen_tok_s"] == 93.65
        assert row["cuda_graph"] is True

    def test_bar1_geometry_is_read_from_the_aufbau_line(self):
        geo = parse_bar1_geometrie(["irgendwas", ZEILE_AUFBAU])
        assert geo["schlitz_kib"] == 8188
        assert geo["max_nutzlast_kib"] == 24564
        assert geo["schlitze"] == 12
        assert geo["region_mib"] == 96.0

    def test_a_baseline_log_has_no_geometry(self):
        assert parse_bar1_geometrie([ZEILE_PREFILL_GRUNDLINIE]) is None


class TestArithmetik:
    def test_collective_size_is_tokens_times_hidden_times_two(self):
        assert kollektiv_bytes(2048, 5120) == 20971520

    def test_a_2048_chunk_needs_exactly_one_round(self):
        """6,67 MiB shard against a 7,996 MiB slot -- one round, with room."""
        assert runden(kollektiv_bytes(2048, 5120), 8188 * 1024, 3) == 1

    def test_the_tipping_point_is_2457_tokens(self):
        """At welt*slot the payload stops fitting -- and BAR1 stops carrying it
        at all, because handles() says False above max_nutzlast."""
        assert runden(kollektiv_bytes(2456, 5120), 8188 * 1024, 3) == 1
        assert runden(kollektiv_bytes(2457, 5120), 8188 * 1024, 3) == 2

    def test_the_wait_share_is_built_from_sums_not_from_median_ratios(self):
        """A median of per-batch ratios does not add up to a point. Here the
        big batch dominates the seconds and has to dominate the share."""
        batches = [
            {"gpu_ms": 100.0, "compute_ms": 90.0, "wait_ms": 10.0, "new_token": 8},
            {
                "gpu_ms": 1000.0,
                "compute_ms": 100.0,
                "wait_ms": 900.0,
                "new_token": 2048,
            },
        ]
        agg = wait_aggregat(batches)
        assert agg["wait_anteil"] == pytest.approx(910.0 / 1100.0)
        assert agg["new_token_max"] == 2048
        assert wait_aggregat([]) == {"n": 0}

    def test_occupancy_over_100_percent_means_batches_overlapped(self):
        b = [
            {"zeit": "2026-07-30 13:34:50", "gpu_ms": 1200.0},
            {"zeit": "2026-07-30 13:34:52", "gpu_ms": 1200.0},
        ]
        assert belegung(b)["belegung"] == pytest.approx(1.2)
        assert belegung(b[:1]) == {"batches": 1}


class TestFenster:
    def test_the_point_is_the_last_n_large_batches(self):
        """The warmup logs exactly like the point. Counting from the back with
        the point's own request count is the only boundary the log admits."""
        rows = parse_prefill_rang(open(_fixture("bar1_1.log"), errors="replace"))
        alle_gross = [r for r in rows if r["rang"] == 0 and r["new_token"] >= 1000]
        assert len(alle_gross) == 16  # warmup and point together
        fenster = punkt_fenster(rows, requests=11)
        assert len(fenster[0]) == 11
        assert fenster[0] == alle_gross[-11:]

    def test_small_chunk_remainders_stay_out(self):
        """A 2055-token prompt at chunked_prefill_size=2048 leaves a 7-token
        batch. It is a real collective, but not the point's work."""
        rows = parse_prefill_rang(open(_fixture("bar1_1.log"), errors="replace"))
        fenster = punkt_fenster(rows, requests=11)
        assert min(r["new_token"] for r in fenster[0]) >= 1000

    def test_window_by_wall_clock_widens_by_one_second(self):
        """The scheduler stamps whole seconds; a half-second window would drop
        the tick it was opened for."""
        import datetime

        ticks = parse_decode([ZEILE_DECODE])
        t = datetime.datetime.strptime("2026-07-30 13:35:13", "%Y-%m-%d %H:%M:%S")
        mitte = t.timestamp() + 0.4
        assert im_fenster(ticks, mitte, mitte) == ticks
        assert im_fenster(ticks, mitte + 600, mitte + 900) == []


class TestDecodeTicks:
    def test_the_first_tick_of_a_stream_is_dropped(self):
        """3,37 tok/s is the prefill of that request showing up in a decode
        rate. Kept, it halves the median."""
        ticks = parse_decode([ZEILE_DECODE_WARMUP, ZEILE_DECODE])
        agg = decode_tick_aggregat(ticks, running_req=1)
        assert agg["ticks"] == 2
        assert agg["ticks_warmup_verworfen"] == 1
        assert agg["gen_tok_s_median"] == 93.65
        assert agg["accept_len_median"] == 2.83

    def test_accept_comes_out_of_the_log_not_out_of_meta_info(self, payload):
        """The 2026-07-30 run recorded accept: None in all eight points, because
        meta_info is opt-in on the chat endpoint. The scheduler logged it
        anyway."""
        gemessen = {
            (d["arm"], d["sessions"], d["running_req"]): d["accept_len_median"]
            for d in payload["decode"]
            if d.get("ticks_gewertet")
        }
        assert gemessen[("bar1", 1, 16)] == 2.90
        assert gemessen[("grundlinie", 8, 16)] == 2.80
        assert all(2.0 < v < 4.0 for v in gemessen.values())

    def test_the_tick_rate_is_not_the_request_rate(self, payload):
        """bs=1 was reported as ~32 tok/s at the request level. The decode loop
        was turning over 77-94."""
        bs1 = [
            d
            for d in payload["decode"]
            if d["running_req"] == 1 and d["ticks_gewertet"]
        ]
        assert bs1
        assert all(70.0 < d["gen_tok_s_median"] < 100.0 for d in bs1)

    def test_a_batch_size_that_never_ran_yields_no_aggregate(self):
        agg = decode_tick_aggregat(parse_decode([ZEILE_DECODE]), running_req=16)
        assert agg == {"ticks": 0, "ticks_gewertet": 0}


class TestBefund:
    """The four numbers #293 step 1 stands on."""

    def test_compute_is_flat_from_one_to_eight_sessions(self, payload):
        """If the arithmetic had grown, the compression would be a load effect
        and nothing to do with the transport."""
        for arm in ("bar1", "grundlinie"):
            for rang in (0, 1, 2):
                eins = _w(payload, arm, 1, rang)["compute_ms_median"]
                acht = _w(payload, arm, 8, rang)["compute_ms_median"]
                assert acht / eins < 1.10

    def test_bar1_wait_grows_about_twice_as_fast_as_the_baseline(self, payload):
        """The core question, on the compute-critical rank: 652,6 -> 1220,1 ms
        against 867,25 -> 1214,5 ms. The table rounds the 867,25 to 867,2; the
        assertion takes the unrounded value, because that is what the median of
        an even number of batches is."""
        b1, b8 = (_w(payload, "bar1", s, 1)["wait_ms_median"] for s in (1, 8))
        g1, g8 = (_w(payload, "grundlinie", s, 1)["wait_ms_median"] for s in (1, 8))
        assert (b1, b8) == (652.6, 1220.1)
        assert (g1, g8) == (867.25, 1214.5)
        assert (b8 / b1) / (g8 / g1) == pytest.approx(1.34, abs=0.02)

    def test_both_arms_land_on_the_same_level_at_eight_sessions(self, payload):
        """Transport-specific contention would end ABOVE the baseline. It ends
        on it, which is what makes the verdict "shared ceiling" and not "bar1
        got slower"."""
        for rang in (0, 1, 2):
            b = _w(payload, "bar1", 8, rang)["gpu_ms_median"]
            g = _w(payload, "grundlinie", 8, rang)["gpu_ms_median"]
            assert abs(b - g) / g < 0.005

    def test_the_collective_size_is_the_same_at_one_and_at_eight_sessions(
        self, payload
    ):
        """chunked_prefill_size=2048 pins it. A size effect is therefore
        excluded before any timing is even looked at."""
        assert {g["nutzlast_bytes"] for g in payload["groessen"]} == {20971520}
        assert {g["new_token_max"] for g in payload["groessen"]} == {2048}
        assert {g["chunks_max"] for g in payload["groessen"]} == {1}

    def test_no_collective_ever_needed_a_second_round(self, payload):
        assert {g["runden"] for g in payload["groessen"]} == {1}
        assert all(g["traegt_bar1"] for g in payload["groessen"])
        assert payload["kipp_token"] == 2456

    def test_batches_overlap_from_four_sessions_on(self, payload):
        """Over 100 % occupancy is why a per-batch gpu-ms at load is a period
        and not a latency -- the caveat the verdict has to carry."""
        eins = [g for g in payload["groessen"] if g["sessions"] == 1]
        acht = [g for g in payload["groessen"] if g["sessions"] == 8]
        assert all(g["belegung"] > 1.05 for g in acht)
        assert any(g["belegung"] < 1.05 for g in eins)


class TestBericht:
    def test_the_tables_render_and_do_not_judge(self, payload):
        text = tabellen(payload)
        assert "| bar1 | 1 | TP1 | 11 | 547.2 | 652.6 | 1200.7 | 53.8 % |" in text
        assert "20971520" in text
        for urteil in ("gut", "schlecht", "besser", "Gewinn", "Kontention"):
            assert urteil not in text

    def test_the_cli_runs_hermetically(self, tmp_path):
        ziel = tmp_path / "analyse.json"
        argv = [sys.executable, os.path.join(BATTERY, "s12_log_analyse.py")]
        for arm, sessions, pfad in _quellen():
            argv += ["--log", f"{arm}:{sessions}:{pfad}"]
        argv += ["--punkte", _fixture("punkte.jsonl"), "--json", str(ziel)]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr
        assert "compute/wait je Rang" in p.stdout
        payload = json.loads(ziel.read_text())
        assert payload["bar1_geometrie"]["schlitz_kib"] == 8188

    def test_a_malformed_log_spec_fails_loudly(self):
        p = subprocess.run(
            [
                sys.executable,
                os.path.join(BATTERY, "s12_log_analyse.py"),
                "--log",
                "bar1-1-pfad",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 2
        assert "ARM:SESSIONS:PFAD" in p.stderr


class TestHarnessLuecke:
    """The four points of the 2026-07-30 run that the summary got wrong."""

    def test_the_run_recorded_no_accept_at_all(self):
        """The gap this fix closes, asserted against the run's own artefacts so
        it cannot be argued away later."""
        punkte = lade_punkte(_fixture("punkte.jsonl"))
        assert punkte
        for p in punkte.values():
            for d in p["decode"]:
                assert d["spec_accept_length"] is None

    def test_the_measuring_side_harvests_the_ticks_of_its_own_window(self):
        """s12_prefill_kurve calls this right after a decode point, with the
        wall clock it just ran in. Same local clock on both sides, so the
        window is comparable without a timezone argument."""
        import datetime

        from s12_prefill_kurve import ernte_ticks

        def stempel(s):
            return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()

        agg = ernte_ticks(
            _fixture("bar1_1.log"),
            stempel("2026-07-30 13:35:12"),
            stempel("2026-07-30 13:35:14"),
            1,
        )
        assert agg["tick_ticks_gewertet"] == 1
        assert agg["tick_accept_len_median"] == 2.83
        assert agg["tick_gen_tok_s_median"] == 93.65
        # Prefixed, or the request-level ms_pro_token would be overwritten.
        assert "ms_pro_token" not in agg

    def test_a_missing_server_log_says_so_instead_of_inventing_a_number(self):
        from s12_prefill_kurve import ernte_ticks

        agg = ernte_ticks("", 0.0, 1.0, 1)
        assert agg["tick_fehler"] == "kein Serverlog"
        assert "tick_accept_len_median" not in agg

    def test_the_table_names_both_levels(self, tmp_path):
        from s12_prefill_kurve import tabelle

        payload = {
            "kurven": {"bar1": {"1": 1469.0}, "grundlinie": {}},
            "verhaeltnis_bar1_zu_grundlinie": {},
            "decode": [
                {
                    "arm": "bar1",
                    "batch": 1,
                    "decode_tok_s": 33.19,
                    "tick_gen_tok_s_median": 93.65,
                    "tick_ms_pro_token": 10.68,
                    "tick_accept_len_median": 2.83,
                    "tick_ticks_gewertet": 1,
                }
            ],
        }
        text = tabelle(payload)
        assert "tok/s (Tick)" in text and "tok/s (Anfrage)" in text
        assert "| bar1 | 1 | 93.7 | 10.68 | 2.83 | 1 | 33.2 |" in text

    def test_the_table_survives_a_point_without_a_server_log(self):
        from s12_prefill_kurve import tabelle

        payload = {
            "kurven": {"bar1": {"1": 1469.0}, "grundlinie": {}},
            "verhaeltnis_bar1_zu_grundlinie": {},
            "decode": [{"arm": "bar1", "batch": 1, "decode_tok_s": 33.19}],
        }
        assert "| bar1 | 1 | None | None | None | None | 33.2 |" in tabelle(payload)

    def test_the_recorded_request_rate_is_half_the_tick_rate(self, payload):
        punkte = lade_punkte(_fixture("punkte.jsonl"))
        anfrage = {
            (arm, sessions, d["batch"]): d["decode_tok_s"]
            for (arm, sessions), p in punkte.items()
            for d in p["decode"]
        }
        tick = {
            (d["arm"], d["sessions"], d["running_req"]): d["gen_tok_s_median"]
            for d in payload["decode"]
            if d.get("ticks_gewertet")
        }
        for schluessel, wert in anfrage.items():
            assert tick[schluessel] > 2.0 * wert
