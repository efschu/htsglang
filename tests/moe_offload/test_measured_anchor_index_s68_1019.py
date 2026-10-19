"""S68 (19.10.): the W65 anchor scan answers from a per-log index.

Measured on the NF line 19.10.: the scan read all 490 ``boot_*.D.log`` (15.8
GiB) whole on every boot, rejected all of them, and pushed D's early start
behind P READY (+16 s).  The index keeps each log's FORM facts (model echoes,
budget echoes) and the posts of a form-passing log, keyed by size+mtime.

What is pinned here:

* EQUIVALENCE -- the indexed scan returns the same winner / the same refusal
  enumeration as the pre-S68 cold scan (``use_index=False``), cold and warm;
* a warm scan reads NO log whose size+mtime did not change, a changed log is
  re-read, a new log is read once;
* an unwritable directory, a damaged index and a parser change degrade to the
  cold scan, never to a wrong answer or a crash;
* REPRO -- many large dummy logs: cold reads them all, warm reads none and is
  far faster.
"""

import json
import os
import re
import time

import pytest

from flliper.srt.planner import measured_anchor as ma
from flliper.srt.planner.measured_anchor import (
    INDEX_BASENAME,
    MeasuredAnchorRefused,
    read_measured_anchor,
)

from test_planner_vram_anchor_0921 import EXCERPT  # verbatim rank lines (V72)

BUDGETS = [29560, 18512, 18488]
ARGV = "argv: --rank-gpu-memory-mib %s model_path='/ckpt/form-a'\n"


def _kw(tmp_path, **over):
    kw = dict(
        group="D", tp_size=3, rank_axis="tp", ranks_on_gpu=[1, 1, 1],
        required_free_bytes=[0, 0, 0], mlp_vector=[1, 1, 1],
        budgets_mib=BUDGETS, model_path="/ckpt/form-a",
        evidence_dirs=(str(tmp_path),), now=3_000_000.0,
    )
    kw.update(over)
    return kw


def _log(tmp_path, name, body, mtime):
    p = tmp_path / name
    p.write_text(body)
    os.utime(p, (mtime, mtime))
    return p


def _scan_stats(text):
    m = re.search(r"\[W65-SCAN logs=(\d+) index_hits=(\d+) read=(\d+) ", text)
    assert m, text[-400:]
    return tuple(int(x) for x in m.groups())


def _strip_note(text):
    return re.sub(r" \[W65-SCAN [^\]]*\]$", "", text)


def _populate(tmp_path):
    """newest: other model; then other budgets; then dead of the form; then good."""
    _log(tmp_path, "boot_weg2_vM_x_1019_120000.D.log",
         "argv: --rank-gpu-memory-mib 29560,18512,18488 model_path='/ckpt/other'\n" + EXCERPT, 4_000)
    _log(tmp_path, "boot_weg2_vB_x_1019_110000.D.log",
         ARGV % "1,2,3" + EXCERPT, 3_000)
    _log(tmp_path, "boot_weg2_vDEAD_x_1019_100000.D.log",
         ARGV % ",".join(map(str, BUDGETS)) + "crashed before load\n", 2_000)
    _log(tmp_path, "boot_weg2_vOLD_x_1019_090000.D.log",
         ARGV % ",".join(map(str, BUDGETS)) + EXCERPT, 1_000)


def test_winner_is_identical_cold_warm_and_without_index(tmp_path):
    _populate(tmp_path)
    ref = read_measured_anchor(**_kw(tmp_path, use_index=False))
    assert not (tmp_path / INDEX_BASENAME).exists()  # the old path writes nothing
    cold = read_measured_anchor(**_kw(tmp_path))
    warm = read_measured_anchor(**_kw(tmp_path))
    for a in (cold, warm):
        assert a.components == ref.components
        assert a.provenance == ref.provenance  # provenance text unchanged by S68
        assert a.boot_tag == "vOLD" and a.source_path == ref.source_path
    assert _scan_stats(cold.scan_note) == (4, 0, 4)
    assert _scan_stats(warm.scan_note) == (4, 4, 0)  # nothing read again


def test_refusal_enumeration_is_identical_cold_and_warm(tmp_path):
    _populate(tmp_path)
    os.unlink(tmp_path / "boot_weg2_vOLD_x_1019_090000.D.log")
    msgs = []
    for use_index in (False, True, True):
        with pytest.raises(MeasuredAnchorRefused) as exc:
            read_measured_anchor(**_kw(tmp_path, use_index=use_index))
        msgs.append(str(exc.value))
    assert _strip_note(msgs[0]) == _strip_note(msgs[1]) == _strip_note(msgs[2])
    assert "different model (/ckpt/other)" in msgs[0]
    assert "measured under budgets 1,2,3, planning 29560,18512,18488" in msgs[0]
    assert "vDEAD" in msgs[0] and "missing" in msgs[0]
    assert _scan_stats(msgs[2]) == (3, 3, 0)
    # the note sits behind the enumeration, i.e. inside the oracle's masked tail
    assert msgs[2].index("no boot_*.D.log of this form: ") < msgs[2].index("[W65-SCAN")


def test_changed_and_new_logs_are_read_unchanged_ones_are_not(tmp_path):
    _populate(tmp_path)
    read_measured_anchor(**_kw(tmp_path))
    # the dead log of the form grows into a complete one (same mtime order)
    _log(tmp_path, "boot_weg2_vDEAD_x_1019_100000.D.log",
         ARGV % ",".join(map(str, BUDGETS)) + EXCERPT, 2_000)
    a = read_measured_anchor(**_kw(tmp_path))
    assert a.boot_tag == "vDEAD"
    assert _scan_stats(a.scan_note) == (4, 2, 1)  # vDEAD re-read, vOLD never reached
    _log(tmp_path, "boot_weg2_vNEW_x_1019_130000.D.log", ARGV % "9,9,9", 5_000)
    a = read_measured_anchor(**_kw(tmp_path))
    assert a.boot_tag == "vDEAD"
    assert _scan_stats(a.scan_note) == (5, 3, 1)  # only the new log read


def test_index_drops_entries_of_vanished_logs(tmp_path):
    _populate(tmp_path)
    read_measured_anchor(**_kw(tmp_path))
    os.unlink(tmp_path / "boot_weg2_vM_x_1019_120000.D.log")
    _log(tmp_path, "boot_weg2_vNEW_x_1019_130000.D.log", ARGV % "9,9,9", 5_000)
    read_measured_anchor(**_kw(tmp_path))
    logs = json.loads((tmp_path / INDEX_BASENAME).read_text())["logs"]
    assert "boot_weg2_vM_x_1019_120000.D.log" not in logs
    assert "boot_weg2_vNEW_x_1019_130000.D.log" in logs


def test_damaged_index_is_a_cold_scan_not_a_failure(tmp_path):
    _populate(tmp_path)
    ref = read_measured_anchor(**_kw(tmp_path, use_index=False))
    (tmp_path / INDEX_BASENAME).write_text("{not json")
    a = read_measured_anchor(**_kw(tmp_path))
    assert a.components == ref.components
    assert "unreadable" in a.scan_note
    # a structurally damaged entry is a miss, not a TypeError
    doc = json.loads((tmp_path / INDEX_BASENAME).read_text())
    for e in doc["logs"].values():
        e["models"] = None
        e["posts"] = {"tp": {"x": "y"}}
    (tmp_path / INDEX_BASENAME).write_text(json.dumps(doc))
    a = read_measured_anchor(**_kw(tmp_path))
    assert a.components == ref.components
    assert _scan_stats(a.scan_note) == (4, 0, 4)


def test_parser_change_invalidates_the_index(tmp_path, monkeypatch):
    _populate(tmp_path)
    read_measured_anchor(**_kw(tmp_path))
    monkeypatch.setattr(ma, "INDEX_SCHEMA", ma.INDEX_SCHEMA + 1)
    a = read_measured_anchor(**_kw(tmp_path))
    assert "another parser" in a.scan_note
    assert _scan_stats(a.scan_note) == (4, 0, 4)


def test_unwritable_dir_still_answers(tmp_path, monkeypatch):
    _populate(tmp_path)
    ref = read_measured_anchor(**_kw(tmp_path, use_index=False))

    def _boom(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ma.os, "replace", _boom)
    a = read_measured_anchor(**_kw(tmp_path))
    assert a.components == ref.components
    assert "not written" in a.scan_note
    assert not (tmp_path / INDEX_BASENAME).exists()
    assert not [p for p in os.listdir(tmp_path) if ".tmp." in p]


def test_logs_are_never_written(tmp_path):
    _populate(tmp_path)
    before = {p: (os.stat(tmp_path / p).st_mtime_ns, (tmp_path / p).read_bytes())
              for p in os.listdir(tmp_path)}
    read_measured_anchor(**_kw(tmp_path))
    read_measured_anchor(**_kw(tmp_path))
    for p, (mt, data) in before.items():
        assert os.stat(tmp_path / p).st_mtime_ns == mt
        assert (tmp_path / p).read_bytes() == data


def test_launcher_note_carries_the_scan_stats(tmp_path):
    from flliper.srt.pdflip import launcher

    _populate(tmp_path)
    _anchor, note = launcher._d_measured_anchor(
        cards=[], budgets=[1, 1, 1], model="/ckpt/form-a", maxkv_weights=[1, 1, 1],
        user_reserve_by_card={}, injected=None, evidence_dirs=(str(tmp_path),),
    )
    assert note.startswith(launcher.ANCHOR_NOTE_PREFIX + "no anchor, heuristic path stands")
    assert "[W65-SCAN logs=4 " in note


def test_warm_index_fills_without_planning(tmp_path):
    _populate(tmp_path)
    out = ma.warm_index([str(tmp_path)], "D")
    assert "read=4" in out
    a = read_measured_anchor(**_kw(tmp_path))
    # the form facts are warm; only the two logs that PASS the form (vDEAD,
    # vOLD) are read once more for their posts, which warm_index cannot know
    # the rank axis of
    assert _scan_stats(a.scan_note) == (4, 4, 2)
    a = read_measured_anchor(**_kw(tmp_path))
    assert _scan_stats(a.scan_note) == (4, 4, 0)


def test_repro_many_large_logs_warm_scan_reads_nothing(tmp_path):
    """REPRO of the 19.10. shape at reduced scale: N logs of S MiB, all of
    another budget form (the 240-of-490 case), none an anchor."""
    n, mib = int(os.environ.get("S68_REPRO_N", "60")), int(os.environ.get("S68_REPRO_MIB", "8"))
    filler = ("[2026-10-19 12:00:00 TP0] decode batch #running-req: 1 gen throughput 70.0\n" * 64)
    reps = (mib << 20) // len(filler)
    for i in range(n):
        p = tmp_path / ("boot_weg2_r%03d_x_1019_%06d.D.log" % (i, i))
        with open(p, "w") as f:
            f.write(ARGV % ("%d,1,1" % (20000 + i)))
            for _ in range(reps):
                f.write(filler)
        os.utime(p, (10_000 + i, 10_000 + i))
    msgs, walls = [], []
    for use_index in (False, True, True):
        t0 = time.monotonic()
        with pytest.raises(MeasuredAnchorRefused) as exc:
            read_measured_anchor(**_kw(tmp_path, use_index=use_index))
        walls.append(time.monotonic() - t0)
        msgs.append(str(exc.value))
    assert _strip_note(msgs[0]) == _strip_note(msgs[2])
    assert _scan_stats(msgs[1]) == (n, 0, n)
    assert _scan_stats(msgs[2]) == (n, n, 0)
    print("S68-REPRO n=%d x %d MiB: cold(no index) %.2f s, cold(index build) %.2f s, warm %.3f s"
          % (n, mib, walls[0], walls[1], walls[2]))
    assert walls[2] < walls[0] / 5
