# SPDX-License-Identifier: Apache-2.0
"""#485 -- the certification judge may only say CERTIFIED for criteria it checked.

WHY THIS FILE EXISTS. `scripts/cert_485/certify_485.py` was written against the
FIRST revision of RUNSHEET_485_CERTIFICATION.md, where the across-window
criteria were exactly C1 + C2 + C3. Revision 2 of that runsheet replaced C2
with C2' and ADDED C4 and C5 (runsheet section 6, "THE GOVERNING CRITERIA"),
and section 7 gate item 1 states the governing set verbatim:

    the criteria are now C1 + C2' + C3 + C4 + C5

The script was never updated, because it lived on a branch that the line never
merged. Ported onto the line it would print a bare

    CERTIFIED: margin N MiB exceeds the observed spread M MiB

on C1+C2+C3 alone -- a verdict for three of the five governing criteria, using
the one word the whole runsheet is built to protect. Runsheet section 7 exists
to prevent exactly this class of error ("certifying cut X and then advertising
cut Y's throughput"); a judge that certifies against a superseded criteria set
is the same failure one level down.

These tests pin, in order:

  T1  the tool refuses CERTIFIED when C2'/C4/C5 carry no evidence, even on
      input that satisfies C1+C2+C3. The refusal names which criteria are
      unattested.
  T2  with every governing criterion attested and passing, it certifies -- so
      T1 is a real gate and not an unconditional refusal (the can-fail proof
      in the other direction).
  T3  C4: a cut certified that is NOT the cut the gate admitted is refused.
      This is runsheet section 5.1b/6a's own rule.
  T4  C5: a window whose census recorded no SEAM_* samples is refused -- a
      census that could not see the cutover cannot certify one.
  T5  C2': a negative C2' margin is refused even when C1+C2+C3 pass.
  T6  the per-window CLEAN inputs the runsheet lists -- flips, soak errors,
      tracebacks -- must be reachable from the command line. They were
      hardcoded to None in cmd_judge, so three of the seven W-criteria could
      never fire from the CLI: unenforceable, not merely unenforced.
  T7  the recorded R12 verdict still reproduces verbatim. One window must
      still fail at C2 with the same sentence, so this change cannot be read
      as having moved the goalposts after a result.
  T8  the usage line names every subcommand it dispatches.

None of these relax a threshold. Every one of them can only turn a CERTIFIED
into a NOT CERTIFIED, which is the only direction a criteria change may go
after windows have already run.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "cert_485" / "certify_485.py"


def _load():
    assert SCRIPT.is_file(), f"certify_485.py is not on the line at {SCRIPT}"
    spec = importlib.util.spec_from_file_location("certify_485_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cert():
    return _load()


def _corridor(path: Path, g0: int, g1: int, g2: int) -> Path:
    with path.open("w") as f:
        f.write("ts_ms,gpu0_free,gpu1_free,gpu2_free\n")
        for i in range(50):
            f.write(f"{1000 + i},{g0 + i},{g1 + i},{g2 + i}\n")
    return path


def _run(cert, argv):
    """Run the judge subcommand, returning (exit_code, stdout)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cert.main(["judge"] + argv)
    return rc, buf.getvalue()


# Two windows that pass C1 + C2 + C3: same binding card, margin 376 > spread 50.
def _two_clean_windows(tmp_path):
    a = _corridor(tmp_path / "w1.csv", 5591, 1400, 6125)
    b = _corridor(tmp_path / "w2.csv", 5591, 1450, 6125)
    return [
        "--window", f"w1={a}",
        "--window", f"w2={b}",
        "--seam-breaches", "w1=0", "--seam-breaches", "w2=0",
        "--abandoned", "w1=0", "--abandoned", "w2=0",
        "--ranks-alive", "w1=3", "--ranks-alive", "w2=3",
    ]


ATTEST = [
    "--cut", "37,14,13",
    "--c4-admitted-cut", "37,14,13",
    "--c2prime-margin", "2589",
    "--c5-seam-samples", "w1=99", "--c5-seam-samples", "w2=99",
]


def test_t1_refuses_certified_when_governing_criteria_unattested(cert, tmp_path):
    """C1+C2+C3 alone must NOT produce the word CERTIFIED."""
    rc, out = _run(cert, _two_clean_windows(tmp_path))
    assert rc == 1, f"C1+C2+C3 alone certified; output was:\n{out}"
    assert "NOT CERTIFIED" in out
    # The refusal must say WHICH criteria have no evidence, or the operator
    # cannot act on it.
    for criterion in ("C2'", "C4", "C5"):
        assert criterion in out, f"refusal does not name {criterion}:\n{out}"


def test_t2_certifies_when_every_governing_criterion_is_attested(cert, tmp_path):
    """The can-fail proof for T1: with evidence attached, it does certify."""
    rc, out = _run(cert, _two_clean_windows(tmp_path) + ATTEST)
    assert rc == 0, f"fully attested input did not certify; output was:\n{out}"
    assert "CERTIFIED" in out
    assert "NOT CERTIFIED" not in out
    # It must state the criteria it actually checked, not just "CERTIFIED".
    assert "C2'" in out and "C4" in out and "C5" in out


def test_t3_c4_refuses_a_cut_the_gate_did_not_admit(cert, tmp_path):
    """Certifying 40,12,12 while the gate admitted 37,14,13 is runsheet 6a's error."""
    argv = _two_clean_windows(tmp_path) + [
        "--cut", "40,12,12",
        "--c4-admitted-cut", "37,14,13",
        "--c2prime-margin", "2589",
        "--c5-seam-samples", "w1=99", "--c5-seam-samples", "w2=99",
    ]
    rc, out = _run(cert, argv)
    assert rc == 1, f"certified a cut the gate refused:\n{out}"
    assert "C4" in out
    assert "40,12,12" in out and "37,14,13" in out


def test_t4_c5_refuses_a_window_whose_census_saw_no_seam(cert, tmp_path):
    argv = _two_clean_windows(tmp_path) + [
        "--cut", "37,14,13",
        "--c4-admitted-cut", "37,14,13",
        "--c2prime-margin", "2589",
        "--c5-seam-samples", "w1=99", "--c5-seam-samples", "w2=0",
    ]
    rc, out = _run(cert, argv)
    assert rc == 1, f"certified a window that never saw a cutover:\n{out}"
    assert "C5" in out and "w2" in out


def test_t5_c2prime_negative_margin_refuses(cert, tmp_path):
    """s50/s51 returned -354 / -765 MiB under C2'. Those must never certify."""
    argv = _two_clean_windows(tmp_path) + [
        "--cut", "37,14,13",
        "--c4-admitted-cut", "37,14,13",
        "--c2prime-margin", "-354",
        "--c5-seam-samples", "w1=99", "--c5-seam-samples", "w2=99",
    ]
    rc, out = _run(cert, argv)
    assert rc == 1, f"certified a negative C2' margin:\n{out}"
    assert "C2'" in out and "-354" in out


def test_t6_per_window_clean_inputs_are_reachable_from_the_cli(cert, tmp_path):
    """W3 flips, W4 soak err and tracebacks were hardcoded None in cmd_judge.

    The runsheet lists them under "Per-window CLEAN (W1-W7, all required)", so a
    CLI that cannot express them makes three of the seven criteria
    unenforceable from the documented command.
    """
    base = _two_clean_windows(tmp_path)

    rc, out = _run(cert, base + ATTEST + ["--soak-err", "w1=3"])
    assert rc == 1, f"a window with 3 soak errors certified:\n{out}"
    assert "W4" in out

    rc, out = _run(cert, base + ATTEST + ["--tracebacks", "w2=1"])
    assert rc == 1, f"a window with a traceback certified:\n{out}"
    assert "W4" in out

    rc, out = _run(cert, base + ATTEST + ["--flips", "w1=0"])
    assert rc == 1, f"a window that never flipped certified:\n{out}"
    assert "W3" in out


def test_t7_the_recorded_r12_verdict_still_reproduces(cert, tmp_path):
    """One window must still fail at C2 with the pre-registered sentence.

    WINDOW_VERDICT_485_R12.md quotes this tool's output verbatim. If the
    reconciliation changed what one clean window returns, the recorded verdict
    would stop being reproducible from the tree -- and a criteria change that
    edits history is the thing pre-registration exists to prevent.
    """
    a = _corridor(tmp_path / "w1.csv", 6537, 3584, 5671)
    rc, out = _run(cert, [
        "--window", f"w1={a}",
        "--seam-breaches", "w1=0", "--abandoned", "w1=0", "--ranks-alive", "w1=3",
    ])
    assert rc == 1
    assert "NOT CERTIFIED (C2)" in out
    assert "one window cannot bound a variance" in out


def test_t8_usage_names_every_dispatched_subcommand(cert, capsys):
    rc = cert.main(["no-such-command"])
    assert rc == 2
    err = capsys.readouterr().err
    for sub in ("flags", "judge", "smoke", "ordering"):
        assert sub in err, f"usage line omits the {sub!r} subcommand: {err!r}"


def test_t9_builtin_smoke_still_passes(cert):
    """The 7 pre-registered red-on-demand cases (runsheet section 9) must survive the port."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cert.smoke()
    assert rc == 0, buf.getvalue()
    assert "7/7 cases behaved as required" in buf.getvalue()


def _long_corridor(path: Path, rows) -> Path:
    """The shape `nvidia-smi --query-gpu=index,memory.free -lms 100` actually emits."""
    with path.open("w") as f:
        for tick in rows:
            for idx, free in enumerate(tick):
                f.write(f"{idx}, {free}\n")
    return path


def test_t11_reads_the_long_form_the_documented_sampler_emits(cert, tmp_path):
    """The runsheet's own sampling command emits one line PER CARD per tick.

    Runsheet section 5.2 specifies:

        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -lms 100

    which is long-form (`0, 20053` / `1, 32080` / `2, 20050`), while this tool
    was written for wide-form `ts_ms,gpu0_free,gpu1_free,gpu2_free`. The R12
    shift hit this and wrote a converter into the evidence directory that never
    landed on the line, so the documented chain does not run as written.

    Reading long-form natively is strictly better than shipping that converter:
    the converter SYNTHESISED timestamps from the file mtime, and the corridor
    verdict is a per-column minimum that needs no timestamps at all.
    """
    p = _long_corridor(tmp_path / "long.csv", [
        (5591, 1400, 6125),
        (5600, 1402, 6130),
        (5595, 1401, 6127),
    ])
    mins = cert.read_corridor(p)
    assert mins == {"gpu0_free": 5591, "gpu1_free": 1400, "gpu2_free": 6125}


def test_t12_long_and_wide_forms_agree_on_the_minimum(cert, tmp_path):
    ticks = [(5591 + i, 1400 + (i % 7), 6125 + i) for i in range(40)]
    lp = _long_corridor(tmp_path / "long.csv", ticks)
    wp = tmp_path / "wide.csv"
    with wp.open("w") as f:
        f.write("ts_ms,gpu0_free,gpu1_free,gpu2_free\n")
        for i, t in enumerate(ticks):
            f.write(f"{1000 + 100 * i},{t[0]},{t[1]},{t[2]}\n")
    assert cert.read_corridor(lp) == cert.read_corridor(wp)


def test_t13_a_partial_trailing_tick_is_dropped_not_half_counted(cert, tmp_path):
    """A truncated final tick must not score one card against another's absence."""
    p = tmp_path / "long.csv"
    with p.open("w") as f:
        for idx, free in enumerate((5591, 1400, 6125)):
            f.write(f"{idx}, {free}\n")
        f.write("0, 99\n")          # sampler killed mid-tick: gpu0 only
    mins = cert.read_corridor(p)
    assert mins["gpu0_free"] == 5591, "a partial trailing tick was counted"


def test_t14_ordering_refuses_a_long_form_series(cert, tmp_path, capsys):
    """`ordering` reasons about wall-clock instants; long-form carries none.

    The converter's own docstring warns it "must not be pointed at a converted
    file". Refusing loudly is the version of that warning the tool can enforce.
    """
    p = _long_corridor(tmp_path / "long.csv", [(5591, 900, 6125)])
    rc = cert.main(["ordering", "--corridor", str(p), "--card", "gpu1_free"])
    out = capsys.readouterr().out
    assert rc == 1, f"ordering accepted a series with no timestamps:\n{out}"
    assert "REFUSED" in out and "ts_ms" in out


def test_t10_arm_json_cache_hit_rule_unchanged(cert, tmp_path):
    """W6 must still reject a scored sample over 5 % cache hit."""
    a = _corridor(tmp_path / "w1.csv", 5591, 1400, 6125)
    arm = tmp_path / "arm.json"
    arm.write_text(json.dumps({
        "n_scored": 6,
        "n_rejected_cache_hit": 0,
        "samples": [{"idx": 1, "warmup": False, "cache_hit_frac": 0.4}],
    }))
    rc, out = _run(cert, [
        "--window", f"w1={a}", "--arm", f"w1={arm}",
        "--seam-breaches", "w1=0", "--abandoned", "w1=0", "--ranks-alive", "w1=3",
    ])
    assert rc == 1
    assert "W6" in out
