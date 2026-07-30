# SPDX-License-Identifier: Apache-2.0
"""Dry tests for the s16 structured-output arm (DFLASH vs NEXTN, task #285).

Hermetic and CPU-only: no card, no host, no ssh, no lock, no server. What is
verified is everything that can be wrong BEFORE a card is booked, because a
GPU window spent on a harness bug is a window nobody gets back:

  * THE PROMPT SET IS INDEPENDENT AND COMPLETE. Three classes, a validator of
    a known kind on every prompt, unique ids, and no prompt quoting another
    prompt's id. The last one is the #156 self-conditioning trap written as an
    assertion rather than as a comment.
  * THE VALIDATORS REJECT WHAT THEY MUST REJECT. A validator that only ever
    says yes turns the output gate into decoration, and the gate is the only
    thing standing between a fast garbage run and a throughput table.
  * THE TICK AGGREGATION IS ARITHMETIC, not a guess: edge ticks dropped,
    foreign batch sizes counted rather than hidden, and ms/Verify exactly
    ``accept * 1000 / tok_s``.
  * THE ANALYSIS IS DRIVEN AS A SUBPROCESS against synthetic points, because
    what the operator depends on is the table and the exit code, not the
    functions behind them. Its floor gate must call a sub-floor difference
    `~` and a supra-floor difference by name, and it must never average two
    content classes together.
  * THE STEP SCRIPT PARSES. ``bash -n`` on the orchestrator.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
PROMPT_FILE = os.path.join(BATTERY, "prompts", "structured_v1.json")
STEP_SCRIPT = os.path.join(BATTERY, "s16_dflash_structured.sh")
ANALYSIS = os.path.join(BATTERY, "s16_analysis.py")

sys.path.insert(0, BATTERY)

from s16_structured_point import (  # noqa: E402
    count_rows,
    extract_code,
    extract_json,
    tick_aggregate,
    validate_output,
)

VALIDATOR_KINDS = {"python_syntax", "bash_syntax", "json_object", "rows"}
EXPECTED_CLASSES = {"code_completion", "json_schema", "list_table"}


# ---------------------------------------------------------------------------
# prompt set
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompt_set() -> dict:
    with open(PROMPT_FILE) as f:
        return json.load(f)


def test_prompt_set_shape(prompt_set):
    prompts = prompt_set["prompts"]
    assert 12 <= len(prompts) <= 20, "the set is specified as 12-20 prompts"
    assert {p["class"] for p in prompts} == EXPECTED_CLASSES
    for cls in EXPECTED_CLASSES:
        n = sum(1 for p in prompts if p["class"] == cls)
        assert n >= 4, f"class {cls} has only {n} prompts -- too thin for a median"
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate prompt id"
    assert {p["lang"] for p in prompts} == {"de", "en"}, "the set must be mixed de/en"


def test_every_prompt_has_a_usable_validator(prompt_set):
    for p in prompt_set["prompts"]:
        v = p.get("validator") or {}
        assert v.get("kind") in VALIDATOR_KINDS, f"{p['id']}: {v}"
        assert p["prompt"].strip(), f"{p['id']}: empty prompt"
        assert int(p["max_new_tokens"]) > 0
        if v["kind"] == "rows":
            assert v["row_kind"] in ("bullet", "numbered", "table")
            assert int(v["min_rows"]) >= 1
        if v["kind"] == "json_object":
            assert v["top_level"] in ("object", "array")


def test_prompts_are_independent(prompt_set):
    """No prompt may reference another prompt or a previous answer.

    The rule this encodes is the one that cost #156 a whole measurement: a
    ladder evaluated on text it produced itself measures the ladder.
    """
    prompts = prompt_set["prompts"]
    ids = {p["id"] for p in prompts}
    for p in prompts:
        text = p["prompt"]
        for other in ids - {p["id"]}:
            assert other not in text, f"{p['id']} references {other}"
        lowered = text.lower()
        for phrase in (
            "previous answer",
            "vorherige antwort",
            "the output above",
            "die obige ausgabe",
            "continue the",
        ):
            assert phrase not in lowered, f"{p['id']} chains onto earlier output"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_extract_code_prefers_the_fence():
    answer = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps."
    assert extract_code(answer).strip() == "def f():\n    return 1"


def test_extract_code_falls_back_to_the_whole_answer():
    assert extract_code("def f():\n    return 1").strip() == "def f():\n    return 1"


def test_extract_json_cuts_the_object_out_of_prose():
    answer = 'Sure! {"a": 1, "b": {"c": [1, 2]}} -- let me know.'
    assert json.loads(extract_json(answer)) == {"a": 1, "b": {"c": [1, 2]}}


def test_extract_json_survives_braces_inside_strings():
    answer = '{"msg": "a } b", "n": 2}'
    assert json.loads(extract_json(answer)) == {"msg": "a } b", "n": 2}


def test_extract_json_handles_arrays_and_fences():
    answer = '```json\n[{"id": 1}, {"id": 2}]\n```'
    assert json.loads(extract_json(answer)) == [{"id": 1}, {"id": 2}]


# ---------------------------------------------------------------------------
# validators -- both directions, always
# ---------------------------------------------------------------------------


def test_python_validator_accepts_and_rejects():
    good = "```python\ndef chunk(items, size):\n    return [items]\n```"
    ok, reason = validate_output(good, {"kind": "python_syntax", "min_chars": 10})
    assert ok and reason == ""

    bad = "```python\ndef chunk(items, size:\n    return\n```"
    ok, reason = validate_output(bad, {"kind": "python_syntax", "min_chars": 10})
    assert not ok and reason.startswith("python_syntax")

    short = "```python\nx=1\n```"
    ok, reason = validate_output(short, {"kind": "python_syntax", "min_chars": 40})
    assert not ok and reason.startswith("too_short")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash missing")
def test_bash_validator_accepts_and_rejects():
    good = '```bash\nf() {\n  local a="$1"\n  echo "$a"\n}\n```'
    ok, reason = validate_output(good, {"kind": "bash_syntax", "min_chars": 10})
    assert ok and reason == ""

    bad = '```bash\nf() {\n  if [ -z "$1" ]; then\n  echo hi\n}\n```'
    ok, reason = validate_output(bad, {"kind": "bash_syntax", "min_chars": 10})
    assert not ok and reason.startswith("bash_syntax")


def test_json_validator_checks_keys_and_shape():
    spec = {
        "kind": "json_object",
        "top_level": "object",
        "required_keys": ["gpu_index", "ranks"],
    }
    ok, _ = validate_output('{"gpu_index": 0, "ranks": [0, 1]}', spec)
    assert ok

    ok, reason = validate_output('{"gpu_index": 0}', spec)
    assert not ok and "json_missing_keys" in reason and "ranks" in reason

    ok, reason = validate_output('{"gpu_index": 0,}', spec)
    assert not ok and reason.startswith("json_parse")

    ok, reason = validate_output("[1, 2]", spec)
    assert not ok and reason.startswith("json_toplevel")


def test_json_validator_checks_array_items():
    spec = {
        "kind": "json_object",
        "top_level": "array",
        "min_items": 2,
        "item_required_keys": ["id", "severity"],
    }
    ok, _ = validate_output(
        '[{"id": 1, "severity": "info"}, ' '{"id": 2, "severity": "error"}]', spec
    )
    assert ok

    ok, reason = validate_output('[{"id": 1, "severity": "info"}]', spec)
    assert not ok and reason.startswith("json_items")

    ok, reason = validate_output('[{"id": 1}, {"id": 2, "severity": "error"}]', spec)
    assert not ok and "json_item0_missing" in reason


def test_row_counting_per_kind():
    bullets = "- one\n- two\n* three\nprose line\n"
    assert count_rows(bullets, "bullet") == 3
    numbered = "1. one\n2) two\n3. three\n"
    assert count_rows(numbered, "numbered") == 3
    table = "| a | b |\n" "|---|---|\n" "| 1 | 2 |\n" "| 3 | 4 |\n"
    assert count_rows(table, "table") == 2

    ok, reason = validate_output(
        table, {"kind": "rows", "row_kind": "table", "min_rows": 3}
    )
    assert not ok and reason.startswith("rows_table")
    ok, _ = validate_output(table, {"kind": "rows", "row_kind": "table", "min_rows": 2})
    assert ok


def test_unknown_validator_is_a_failure_not_a_pass():
    ok, reason = validate_output("anything", {"kind": "vibes"})
    assert not ok and reason.startswith("unknown_validator")
    ok, reason = validate_output("anything", {})
    assert not ok and reason == "no_validator"


# ---------------------------------------------------------------------------
# tick aggregation, against synthetic ticks
# ---------------------------------------------------------------------------


def _tick(running_req: int, tok_s: float, accept: float = 2.0) -> dict:
    return {
        "running_req": running_req,
        "gen_tok_s": tok_s,
        "accept_len": accept,
        "cuda_graph": True,
    }


def test_tick_aggregate_drops_edges_and_counts_foreign_batch_sizes():
    ticks = [
        _tick(8, 50.0),  # partial first interval
        _tick(8, 100.0),
        _tick(8, 100.0),
        _tick(7, 90.0),  # a request finished here -- wrong working point
        _tick(8, 100.0),
        _tick(8, 100.0),
        _tick(8, 50.0),  # partial last interval
    ]
    agg = tick_aggregate(ticks, bs=8)
    assert agg["ticks_window"] == 7
    assert agg["ticks_bs"] == 6
    assert agg["ticks_other_bs"] == 1
    assert agg["ticks_counted"] == 4
    assert agg["gen_tok_s_median"] == 100.0
    assert agg["accept_len_median"] == 2.0
    assert agg["ms_per_token"] == pytest.approx(10.0)
    assert agg["ms_per_verify"] == pytest.approx(20.0)
    assert agg["ms_per_step"] == pytest.approx(160.0)


def test_tick_aggregate_keeps_the_edges_when_that_is_all_there_is():
    agg = tick_aggregate([_tick(1, 200.0), _tick(1, 210.0)], bs=1)
    assert agg["ticks_counted"] == 2
    assert agg["gen_tok_s_median"] == pytest.approx(205.0)


def test_tick_aggregate_reports_an_empty_window_instead_of_inventing_one():
    agg = tick_aggregate([_tick(4, 100.0)], bs=8)
    assert agg["ticks_bs"] == 0
    assert agg["ticks_counted"] == 0
    assert "gen_tok_s_median" not in agg


# ---------------------------------------------------------------------------
# the analysis, driven as a subprocess against synthetic points
# ---------------------------------------------------------------------------


def _point(arm: str, bs: int, cls: str, ms_verify: float, **over) -> dict:
    tok_s = 1000.0 / ms_verify * 2.0
    point = {
        "kind": "s16_structured",
        "schema": 1,
        "arm": arm,
        "algo": "DFLASH" if arm.startswith("dflash") else "NEXTN",
        "bs": bs,
        "content_class": cls,
        "counted": True,
        "tick_ms_per_verify": ms_verify,
        "tick_gen_tok_s_median": tok_s,
        "tick_accept_len_median": 2.0,
        "client_accept_len_pooled": 2.0,
        "valid_ratio": 1.0,
        "requests_in_window": 8,
        "requests_valid": 8,
        "invalid_reasons": {},
        "not_counted_because": [],
    }
    point.update(over)
    return point


@pytest.fixture()
def synthetic_run(tmp_path):
    """A run directory with a floor round, two comparison rounds, one reject.

    The numbers are chosen so the verdict is arithmetically forced:
      * code_completion floor spread 2.0 %, DFLASH 5 % better  -> a finding
      * json_schema     floor spread 3.0 %, DFLASH 0.5 % worse -> inside floor
    """
    step = tmp_path / "s16_dflash_structured"
    (step / "proofs").mkdir(parents=True)
    rows = [
        # A-vs-A floor round
        _point("floor_a_r0", 1, "code_completion", 20.00),
        _point("floor_b_r0", 1, "code_completion", 20.40),  # 2.0 % spread
        _point("floor_a_r0", 1, "json_schema", 30.00),
        _point("floor_b_r0", 1, "json_schema", 30.91),  # 3.0 % spread
        # comparison rounds, interleaved
        _point("nextn_r1", 1, "code_completion", 20.00),
        _point("dflash_r1", 1, "code_completion", 19.00),
        _point("nextn_r2", 1, "code_completion", 20.00),
        _point("dflash_r2", 1, "code_completion", 19.00),
        _point("nextn_r1", 1, "json_schema", 30.00),
        _point("dflash_r1", 1, "json_schema", 30.15),
        _point("nextn_r2", 1, "json_schema", 30.00),
        _point("dflash_r2", 1, "json_schema", 30.15),
        # a point that failed its output gate
        _point(
            "dflash_r1",
            8,
            "list_table",
            21.0,
            counted=False,
            valid_ratio=0.25,
            requests_valid=2,
            invalid_reasons={"rows_table": 6},
            not_counted_because=["valid share 0.25 < 0.75"],
        ),
    ]
    with open(step / "structured_points.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for arm in ("nextn_r1", "dflash_r1"):
        with open(step / "proofs" / f"{arm}.txt", "w") as f:
            f.write("12:[2026-07-30 10:00:00] max_total_num_tokens=131072\n")
            f.write(f"13:[2026-07-30 10:00:00] speculative_algorithm='{arm[:6]}'\n")
    return step


def _section(text: str, header: str) -> str:
    """The block under one `### <header>` heading.

    The floor table and every metric table carry rows with the same leading
    cells, so a test that greps the whole report reads whichever table comes
    first. Sectioning is what makes the assertion say what it means.
    """
    start = text.index(f"### {header}")
    rest = text[start + 4 :]
    end = rest.find("\n###")
    return rest if end < 0 else rest[:end]


def _run_analysis(step, extra=()):
    out_json = os.path.join(str(step), "summary.json")
    proc = subprocess.run(
        [sys.executable, ANALYSIS, "--step-dir", str(step), "--json", out_json, *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, out_json


def test_analysis_floor_gate_separates_finding_from_noise(synthetic_run):
    proc, out_json = _run_analysis(synthetic_run)
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout

    # The derived gate is the MAX over the floor cells, so 3.0 %.
    with open(out_json) as f:
        summary = json.load(f)
    assert summary["floor_derived"]["ms_per_verify"] == pytest.approx(3.0, abs=0.05)

    ms_table = _section(text, "ms/Verify")
    lines = [
        ln for ln in ms_table.splitlines() if ln.startswith("| code_completion | 1 |")
    ]
    assert len(lines) == 1, ms_table
    assert "DFLASH" in lines[0] and "~" not in lines[0], lines[0]

    lines = [ln for ln in ms_table.splitlines() if ln.startswith("| json_schema | 1 |")]
    assert len(lines) == 1, ms_table
    assert "~ inside floor" in lines[0], lines[0]


def test_analysis_never_averages_across_classes(synthetic_run):
    proc, _ = _run_analysis(synthetic_run)
    text = proc.stdout
    assert "Nothing in these tables is averaged across content classes" in text
    # Both classes keep their own row in the ms/Verify table, with their own
    # numbers: 19.00 for code, 30.15 for json. A mean would print neither.
    assert "| 19.00 |" in text
    assert "| 30.15 |" in text
    assert "| 24.5" not in text


def test_analysis_reports_the_uncounted_point_by_name(synthetic_run):
    proc, _ = _run_analysis(synthetic_run)
    text = proc.stdout
    assert "Points that do NOT count (1)" in text
    assert "dflash_r1 bs=8 list_table" in text
    assert "valid share 0.25 < 0.75" in text
    # and it must NOT have entered a comparison cell
    with open(os.path.join(str(synthetic_run), "summary.json")) as f:
        summary = json.load(f)
    assert "dflash:8:list_table" not in summary["cells"]


def test_analysis_marks_a_run_without_a_floor_as_unverdicted(synthetic_run, tmp_path):
    src = os.path.join(str(synthetic_run), "structured_points.jsonl")
    with open(src) as f:
        kept = [ln for ln in f if '"floor_' not in ln]
    step = tmp_path / "nofloor"
    step.mkdir()
    with open(step / "structured_points.jsonl", "w") as f:
        f.writelines(kept)
    proc, _ = _run_analysis(step)
    assert proc.returncode == 0, proc.stderr
    assert "NO FLOOR ROUND IN THIS RUN" in proc.stdout
    assert "unverdicted (no floor)" in proc.stdout


def test_analysis_stops_on_an_empty_points_file(tmp_path):
    step = tmp_path / "empty"
    step.mkdir()
    (step / "structured_points.jsonl").write_text("")
    proc, _ = _run_analysis(step)
    assert proc.returncode == 2
    assert "no points" in proc.stderr


# ---------------------------------------------------------------------------
# the step script
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash missing")
def test_step_script_parses():
    proc = subprocess.run(
        ["bash", "-n", STEP_SCRIPT], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


def test_step_script_uses_the_fp8_vehicle_and_not_gguf():
    """#290 contaminates the GGUF path; the vehicle of this step is FP8."""
    with open(STEP_SCRIPT) as f:
        text = f.read()
    assert "Qwen3.6-27B-FP8" in text
    code = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )
    assert "gguf" not in code.lower(), "a GGUF path leaked into the recipe"
    assert "--speculative-algorithm NEXTN" in text
    assert "--speculative-algorithm DFLASH" in text
