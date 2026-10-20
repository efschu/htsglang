"""P-Durchlauf im Dashboard (Nutzer 20.10.: 06:26Z "Flip auf P fuer 2,4k Token", der 7903er-Pass fehlte).

Fixture prun_nf_1020 = NF boot ...1019_230333, P run epoch 181: D->P 06:26:10.094Z (trigger pdflip-180-3125, u 7903 > X 7441),
P->D 06:26:23.464Z.  Real: the events (flip_*, request_done) and the P log "Prefill rank batch" lines
  06:26:17 PP0/PP1/PP2  7903 new / 99328 cached   gpu-ms 2642.1 / 1660.1 / 668.4  (pdflip-180-3125)
  06:26:21              2294 new / 78848 cached   gpu-ms 2021.4 /  935.5 /  222.1  (pdflip-180-3126)
  06:26:23               185 new / 49536 cached   gpu-ms 1196.4 /  514.1 /   44.6
The log has whole seconds only and the 1-s rankstats ring of that minute is gone, so the ring is rebuilt by
fixtures/prun_nf_1020/make_fixture.py with the flush instants inferred (PP2 17.43 = pp_last_start_ts 16.759 + 668 ms;
PP0/PP1 flush with it -- prefill.last.t = time.time() at the flush, flliper metrics_reporter.py).

Defect (root): the rank stamps prefill.last.t when it FLUSHES the chunk's line.  PP2 flushes at its own forward end, PP0/PP1
only once the chunk has left the pipeline (7903 tokens: PP0 forward 12.06-14.70 s, flushed 17.45 s).  t - gpu_ms put the chunk
at 14.8-17.45 instead of 12.06-17.43; between chunks 1.8 s > CHUNK_GAP_S (1.5) split ONE run into two bursts with an
"unknown" hole, the second burst merged chunk 2+3 (2,479 tok) and the P->D flip row named only that last burst ("2,4k").
"""

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import activity, ipcboot, server  # noqa: E402
from rigdash.tests.boot_replay import replay_boot  # noqa: E402

T0 = 1792477570.0      # 06:26:10Z
FIX = "prun_nf_1020"


@pytest.fixture(scope="module")
def card():
    return replay_boot(FIX, now=T0 + 30.0)


def test_chunk_start_is_not_the_pp0_flush(card):
    keys = ipcboot._ring_keys(card["ring"])
    cs = activity.chunks(card["ring"], keys, "P")
    assert [int(c["tok"]) for c in cs] == [7903, 2294, 185]
    # chunk 1: last-stage end 17.43 minus 2.642 + 1.660 + 0.668 s = 12.46 s (the front's pp_first_forward stamp: 12.056 s);
    # the old rule (PP0 flush 17.45 - 2.642 s) said 14.81 s
    assert cs[0]["s"] - T0 == pytest.approx(2.459, abs=0.02)
    assert cs[0]["s"] - T0 < 2.6
    # contiguous like the true forwards (PP0 17.78 / 21.54): gaps well under CHUNK_GAP_S
    assert cs[1]["s"] - cs[0]["e"] < activity.CHUNK_GAP_S
    assert cs[2]["s"] - cs[1]["e"] < activity.CHUNK_GAP_S


def test_one_p_run_one_bar_all_tokens(card):
    p = [x for x in card["segs"] if x["k"] == "P" and x["e"] > T0]
    assert len(p) == 1                                  # was two bars with an "unknown" hole between them
    assert p[0]["tok"] == pytest.approx(7903 + 2294 + 185)
    assert p[0]["tps"] == pytest.approx(910.0, abs=5)   # true 10382 tok / ~11.4 s = 914
    assert not [x for x in card["segs"] if x["k"] == "unknown" and T0 + 3 < x["s"] < T0 + 13]


def test_trigger_is_named_on_the_run_and_its_bar(card):
    p = next(x for x in card["segs"] if x["k"] == "P" and x["e"] > T0)
    assert p["trigger"]["rid"] == "pdflip-180-3125"
    assert (p["trigger"]["n"], p["trigger"]["cached"]) == (7903, 99328)
    assert p["trigger"]["matched"] is True
    rq = {r["rid"]: r for r in p["depth"]["reqs"]}
    assert rq["pdflip-180-3125"]["trigger"] is True and rq["pdflip-180-3125"]["n"] == 7903
    assert rq["pdflip-180-3126"]["n"] == 2294 and "trigger" not in rq["pdflip-180-3126"]


def test_flip_rows_carry_the_whole_run_and_the_trigger(card):
    fv = [f for f in card["view"]["flip_views"] if f["begin"] > T0 - 5]
    dp, pd = fv[0], fv[1]
    assert (dp["dir"], pd["dir"]) == ("D>P", "P>D")
    assert dp["trigger"]["rid"] == "pdflip-180-3125" and dp["trigger"]["n"] == 7903
    run = pd["p_run"]
    assert run["n"] == 10382 and run["bursts"] == 1             # was: p_depth n = 2479, the last burst only
    assert run["trigger"]["rid"] == "pdflip-180-3125"
    assert [r["rid"] for r in run["reqs"]] == ["pdflip-180-3125", "pdflip-180-3126"]
    # the Flipzeit itself is untouched: P-Ende is still the last chunk's last-stage stamp
    assert pd["kind"] == "ok" and pd["start_src"].startswith("P-Chunk-Ende")


def test_trigger_without_request_done_is_named_without_tokens():
    # the rid is the front's stamp at the flip; u only exists once request_done does -- never invented
    ipc = {"boot_id": "x", "ipc_events": [
        {"type": "flip_begin", "ts": T0, "data": {"flip_begin_ts": T0, "sleep": "D", "wake": "P"}},
        {"type": "flip_done", "ts": T0 + 1.6, "data": {"sleep": "D", "wake": "P", "flip_ms": 1500, "flip_begin_ts": T0, "t": T0 + 1.6,
                                                   "epoch": 181}}],
        "flip_first_work": [{"dir": "D>P", "flip_begin_ts": T0, "rid": "pdflip-180-9", "what": "p_leg1_dispatch",
                             "first_work_ts": T0 + 2.0, "prefill_start_ts": T0 + 2.0, "prefill_start_source": "pp_first_forward"}],
        "flip_user_time": [], "request_done": []}
    rows = ipcboot.flip_views([{"s": T0 - 5.0, "e": T0, "k": "unknown"}], ipc, T0 + 20.0, [])
    assert rows[0]["trigger"] == {"rid": "pdflip-180-9", "n": None, "cached": None,
                                  "src": "front flip_user_time/flip_first_work rid"}


NODE = next((p for p in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node") if p and os.path.exists(p)), None)


def _js(expr, lang="en"):
    html = open(os.path.join(server.STATIC, "index.html"), encoding="utf-8").read()
    blk = html[html.index("/* DEPTH-BEGIN"):html.index("/* DEPTH-END */")]
    prelude = ('const window = {i18nLang: () => "%s"};\n'
               'const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? "-" : Number(v).toLocaleString("en-US", '
               '{maximumFractionDigits: d, minimumFractionDigits: d});\nconst esc = (s) => String(s);\n' % lang)
    out = subprocess.run([NODE, "-e", prelude + blk + "\nprocess.stdout.write(" + expr + ");"], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(NODE is None, reason="no node")
def test_hover_names_the_trigger_and_gives_no_depth_range_for_several_requests(card):
    import json
    p = next(x for x in card["segs"] if x["k"] == "P" and x["e"] > T0)
    tip = _js("triggerTip(%s)" % json.dumps(p))
    assert "trigger of the flip (woke P)" in tip and "pdflip-180-3125" in tip and "7,903 new, 99,328 cached" in tip
    d = _js("depthTip(%s)" % json.dumps(p["depth"]))
    assert "Tokens 10,382 new" in d and "78,848&ndash;107,231" not in d       # no range over unrelated requests
    assert "pdflip-180-3125</span> Tokens 99,328&ndash;107,231 (7,903 new) <b>&#9664; trigger</b>" in d
    assert "pdflip-180-3126</span> Tokens 78,848&ndash;81,142 (2,294 new)" in d
    de = _js("triggerTip(%s)" % json.dumps({"trigger": {"rid": "r1", "n": None}}), "de")
    assert "Auslöser" in de and "erst mit request_done" in de
