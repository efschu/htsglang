# SPDX-License-Identifier: Apache-2.0
"""#315: couples every BAR1 consumer regex/marker in scripts/gpu_battery/
against the ACTUAL emitter source, so the next rename breaks HERE instead of
silently on the next real run.

Background: barlink.py / barlink_bar1.py / benchmark/bar1_graph_check.py were
translated from German to English in #295. Several scripts/gpu_battery/
consumers (s11_bar1_e2e.py's RE_LEDGER/RE_SETUP/RE_CAPTURE_BOLT,
s12_log_analyse.py's RE_BAR1_SETUP, the "Bar1Unverfuegbar" literal in the
s13/s14/s15 booterror harvest lists, the "Zusammenfassung"/"Aufbau"
substrings) kept matching the OLD German wording -- dead on every real run
since, hidden because the accompanying test fixtures were themselves still
German and matched their own stale regexes.

This module does not re-type either side by hand. `_bar1_marker_source.py`
extracts the format-string literal straight out of the emitter source via
`ast`; this file renders a synthetic sample from it and feeds that sample
into the REAL consumer regex objects (imported, not copied) and into the
REAL literal markers the shell scripts grep for (read from the .sh source,
not retyped). Both sides can only agree if the emitter and the consumer
actually agree.

Hermetic and CPU-only: no card, no host, no ssh, no server.
"""

from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")

sys.path.insert(0, BATTERY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bar1_marker_source as src  # noqa: E402
import s11_bar1_e2e as s11  # noqa: E402
import s12_log_analyse as s12  # noqa: E402


# ---------------------------------------------------------------------------
# s11_bar1_e2e.py regexes against the real emitter format strings
# ---------------------------------------------------------------------------


class TestS11RegexesAgainstTheRealEmitters:
    def test_re_setup_matches_the_real_setup_line(self):
        line = src.render_setup_line()
        m = s11.RE_SETUP.search(line)
        assert m, f"RE_SETUP does not match the real setup line: {line!r}"
        assert m.group("ms") == "46"

    def test_re_ledger_matches_the_real_ledger_line(self):
        line = src.render_ledger_line(group="dcp:0")
        m = s11.RE_LEDGER.search(line)
        assert m, f"RE_LEDGER does not match the real ledger line: {line!r}"
        assert m.group("group") == "dcp:0"

    def test_re_capture_bolt_matches_the_real_runtimeerror(self):
        line = src.render_capture_bolt_message(op="broadcast", nbytes=128)
        m = s11.RE_CAPTURE_BOLT.search(line)
        assert m, f"RE_CAPTURE_BOLT does not match the real RuntimeError: {line!r}"
        assert m.group("op") == "broadcast"
        assert m.group("bytes") == "128"

    def test_re_group_matches_the_real_success_line(self):
        line = src.render_group_ok_line(group="tp:0")
        m = s11.RE_GROUP.search(line)
        assert m, f"RE_GROUP does not match the real success line: {line!r}"
        assert m.group("group") == "tp:0"
        assert m.group("requested") == "bar1"
        assert m.group("achieved") == "bar1"

    def test_re_group_matches_the_real_fallback_line(self):
        line = src.render_group_fallback_line(group="dcp:0", achieved="gloo")
        m = s11.RE_GROUP.search(line)
        assert m, f"RE_GROUP does not match the real fallback line: {line!r}"
        assert m.group("group") == "dcp:0"
        assert m.group("achieved") == "gloo"

    def test_re_gate_case_matches_the_real_transcript(self):
        transcript = src.render_graph_check_transcript()
        matches = [
            line for line in transcript.splitlines()
            if s11.RE_GATE_CASE.match(line)
        ]
        assert len(matches) == 5, transcript

    def test_summary_heading_is_recognised(self):
        """parse_graph_check()'s `zusammenfassung_vorhanden` checks for the
        literal word the real script prints -- "Summary", not "Zusammenfassung".
        """
        heading = src.render_graph_check_summary_heading()
        assert heading == "Summary"
        transcript = src.render_graph_check_transcript()
        assert any(
            "Summary" in line for line in transcript.splitlines()
        ), transcript


# ---------------------------------------------------------------------------
# s12_log_analyse.py's RE_BAR1_SETUP against the real setup line
# ---------------------------------------------------------------------------


class TestS12RegexAgainstTheRealEmitter:
    def test_re_bar1_setup_matches_and_extracts_the_geometry(self):
        line = src.render_setup_line(
            dauer_ms=324, peer_targets=2, region_mib=96.0,
            slots_desc="12 slots (of which 2(R-1) for all_to_all)",
            slot_kib=8188, payload_kib=24564,
        )
        m = s12.RE_BAR1_SETUP.search(line)
        assert m, f"RE_BAR1_SETUP does not match the real setup line: {line!r}"
        d = m.groupdict()
        assert d["setup_ms"] == "324"
        assert d["peers"] == "2"
        assert d["region_mib"] == "96.0"
        assert d["schlitze"] == "12"
        assert d["schlitz_kib"] == "8188"
        assert d["max_nutzlast_kib"] == "24564"

    def test_parse_bar1_geometrie_end_to_end(self):
        line = src.render_setup_line()
        out = s12.parse_bar1_geometrie([line])
        assert out is not None
        assert out["peers"] == 2
        assert out["schlitze"] == 12


# ---------------------------------------------------------------------------
# the shell harvest lists (host_grep_into -F patterns) against the real
# markers -- these cannot be imported, so the .sh source is read and the
# literal patterns are checked, both for presence in the script and for
# actually occurring in a source-derived sample.
# ---------------------------------------------------------------------------

#: file -> literal `grep -F` patterns that are meant to catch BAR1/barlink
#: marker lines. Every one of these must (a) still be present verbatim in
#: the shell script, and (b) actually occur in a synthetic emission built
#: from the real format string.
SHELL_HARVEST_MARKERS = {
    "s11_bar1_e2e.sh": [
        "barlink-BAR1: setup in",
        "BAR1 ledger of this card after group",
        "during a CUDA graph capture",
    ],
    "s12_prefill_kurve.sh": [
        "barlink-BAR1: setup in",
        "during a CUDA graph capture",
    ],
    "s13_hebel_messung.sh": [
        "barlink-BAR1: setup in",
        "during a CUDA graph capture",
        "the graph pool of the result ring is exhausted",
    ],
    "s14_decode_verif.sh": [
        "barlink-BAR1: setup in",
        "during a CUDA graph capture",
    ],
}

#: The booterror harvest lists grep for the exception CLASS NAME as it
#: appears in a traceback -- "Bar1Unavailable", not the German
#: "Bar1Unverfuegbar" #295 renamed the class away from.
SHELL_BOOTERROR_CLASS_MARKER_FILES = (
    "s13_hebel_messung.sh",
    "s14_decode_verif.sh",
    "s15_phasen_optima.sh",
)

_SETUP_LINE_SAMPLE = src.render_setup_line()
_LEDGER_LINE_SAMPLE = src.render_ledger_line()
_CAPTURE_BOLT_SAMPLE = src.render_capture_bolt_message()
_POOL_EXHAUSTED_SAMPLE = src.render_pipe_pool_exhausted_line()

_MARKER_SAMPLES = {
    "barlink-BAR1: setup in": _SETUP_LINE_SAMPLE,
    "BAR1 ledger of this card after group": _LEDGER_LINE_SAMPLE,
    "during a CUDA graph capture": _CAPTURE_BOLT_SAMPLE,
    "the graph pool of the result ring is exhausted": _POOL_EXHAUSTED_SAMPLE,
}


def _read(rel_path: str) -> str:
    with open(os.path.join(BATTERY, rel_path), encoding="utf-8") as f:
        return f.read()


def _booterror_harvest_block(text: str) -> str:
    """The host_grep_into(...) call that harvests boot failures, and ONLY
    that call -- not the whole file.

    s13_hebel_messung.sh carries a historical incident comment (2026-07-30)
    that quotes the German exception name VERBATIM as it actually appeared
    in that run's log, before the class was renamed. That is a record of
    what happened, not a live pattern, and must stay exactly as it reads;
    scoping the check to the actual shell command (a backslash-continued
    line group starting at "host_grep_into") keeps the two apart.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "host_grep_into" in line and (
            "bootfehler" in line or "booterror" in line
        ):
            block = [line]
            j = i
            while lines[j].rstrip().endswith("\\"):
                j += 1
                block.append(lines[j])
            return "\n".join(block)
    raise AssertionError("no host_grep_into(...bootfehler/booterror...) call found")


class TestShellHarvestMarkersAgainstTheRealEmitters:
    def test_every_listed_marker_is_still_literally_in_its_script(self):
        for rel_path, markers in SHELL_HARVEST_MARKERS.items():
            text = _read(rel_path)
            for marker in markers:
                assert marker in text, (
                    f"{rel_path}: harvest marker {marker!r} not found -- "
                    "either it was retyped wrong here or the emitter moved "
                    "again"
                )

    def test_every_listed_marker_actually_occurs_in_a_real_emission(self):
        for rel_path, markers in SHELL_HARVEST_MARKERS.items():
            for marker in markers:
                sample = _MARKER_SAMPLES[marker]
                assert marker in sample, (
                    f"{rel_path}: harvest marker {marker!r} does not occur "
                    f"in the real emitter's sample output {sample!r} -- the "
                    "grep -F pattern would not match a real run"
                )

    def test_booterror_lists_grep_the_real_exception_class_name(self):
        class_name = src.bar1_unavailable_class_name()
        assert class_name == "Bar1Unavailable"
        for rel_path in SHELL_BOOTERROR_CLASS_MARKER_FILES:
            block = _booterror_harvest_block(_read(rel_path))
            assert class_name in block, (
                f"{rel_path}: the boot-failure harvest does not grep for "
                f"{class_name!r} -- it would miss a real BAR1 setup failure:"
                f"\n{block}"
            )
            assert "Bar1Unverfuegbar" not in block, (
                f"{rel_path}: the boot-failure harvest still greps the dead "
                f"German exception name:\n{block}"
            )


# ---------------------------------------------------------------------------
# s13_auswertung.py's plain string/count checks against the real emitters
# ---------------------------------------------------------------------------


class TestS13StringChecksAgainstTheRealEmitters:
    def test_bar1_gruppen_marker_matches_a_real_setup_line(self):
        text = _read("s13_auswertung.py")
        m = re.search(r'text\.count\("([^"]+)"\)', text)
        assert m, "s13_auswertung.py: bar1_gruppen marker literal not found"
        marker = m.group(1)
        assert marker in _SETUP_LINE_SAMPLE, (
            f"s13_auswertung.py counts {marker!r}, which does not occur in "
            f"a real setup line: {_SETUP_LINE_SAMPLE!r}"
        )

    def test_vorrat_leer_marker_matches_the_real_pool_exhausted_line(self):
        text = _read("s13_auswertung.py")
        m = re.search(r'"vorrat_leer": \(\s*"([^"]+)" in text', text)
        assert m, "s13_auswertung.py: vorrat_leer marker literal not found"
        marker = m.group(1)
        assert marker in _POOL_EXHAUSTED_SAMPLE, (
            f"s13_auswertung.py checks for {marker!r}, which does not occur "
            f"in a real graph-pool-exhausted line: {_POOL_EXHAUSTED_SAMPLE!r}"
        )
