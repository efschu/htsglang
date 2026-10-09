"""F0-M (08.10.2026), user decision "pdflip as the name, the metrics are must-keep": the exported METRIC NAMES keep the
spelling they had before the rename, so the series in VictoriaMetrics and the Grafana panels (rig-verlauf) do not break.

  * old names:  the engine's series ``sglang:*`` (and ``sglang_*`` on /v1/loads?format=prometheus) and the P/D-flip subsystem's
                series, labels and Influx measurements ``weg2_*`` (+ the scrape job name ``weg2-front``);
  * new names:  none -- ``pdflip_*`` / ``flliper:*`` / ``flliper_*`` for the same metric must NOT appear next to them (no double export).

The fixture ``fixtures/metric_names_1008/inv_old.json`` is the inventory of the pre-rename tree (tools/release/metric_inventory.py
scan on the base ref).  Four layers: the static scan of this tree against it, the exporters run on synthetic input, the rename
kit's engine on the old spellings (a re-run of the kit leaves them standing), the must-keep list of the translation gate.

F0-F reads both generations of a series: a panel / PromQL may say ``{__name__=~"(<old>|pdflip)_x"}`` (``vmpush.dual_promql``). That
is a read of the OLD name (the old stem is an alternative of the regex) and is accepted wherever a panel is checked; what is never
accepted is a bare ``pdflip_x`` (a series the exporters would have to write) or an exporter that writes a name of the kit's spelling.

The words of the old names are built from pieces so that this file itself passes the mechanical rename unchanged."""
import json
import os
import re
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KIT = os.path.join(REPO, "tools", "release")
FIXTURE = os.path.join(HERE, "fixtures", "metric_names_1008", "inv_old.json")

OLD_SUB = "we" + "g2"                      # the P/D-flip subsystem's old word
OLD_ENG = "sg" + "lang"                    # the engine's old word
NEW_SUB = "pd" + "flip"
NEW_ENG = "fl" + "liper"

pytestmark = pytest.mark.skipif(not (os.path.isfile(os.path.join(KIT, "metric_inventory.py")) and os.path.isfile(FIXTURE)),
                                reason="rename kit / metric fixture not in this tree")


def _kit(name):
    sys.path.insert(0, KIT)
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        sys.path.remove(KIT)


def _old():
    with open(FIXTURE) as f:
        return json.load(f)


def _kit_spelled(name):
    """What the rename kit makes of an old metric name (the spelling that must not leave the process)."""
    for old, new in ((OLD_SUB + "_", NEW_SUB + "_"), (OLD_ENG + ":", NEW_ENG + ":"), (OLD_ENG + "_", NEW_ENG + "_"),
                     (OLD_SUB + "-front", NEW_SUB + "-front")):
        if name.startswith(old):
            return new + name[len(old):]
    return name


def _has_front_exporter():
    return os.path.isfile(os.path.join(REPO, "python", NEW_ENG, "srt", NEW_SUB, "rank_metrics.py"))


def _in_git(fn):
    import subprocess
    try:
        return fn()
    except (subprocess.CalledProcessError, OSError) as e:      # a tree without .git (export): the scan lists tracked files
        pytest.skip(f"not a git checkout: {type(e).__name__}")


def _names(text):
    """Metric names in a Prometheus exposition text (families from # TYPE and every sample name)."""
    out = set()
    for ln in text.splitlines():
        if not ln or ln.startswith("# HELP"):
            continue
        if ln.startswith("# TYPE"):
            out.add(ln.split()[2])
        elif not ln.startswith("#"):
            out.add(re.split(r"[{ ]", ln, 1)[0])
    return out


def _panel_reads(text):
    """The OLD metric names a panel / PromQL text reads: the literal ``<old>_x`` and the dual-read form ``(<old>|pdflip)_x``."""
    pat = (r"(?<![A-Za-z0-9_])%s_([a-z0-9_]+)", r"\(%s\|%s\)_([a-z0-9_]+)")
    out = {OLD_SUB + "_" + n for n in re.findall(pat[0] % OLD_SUB, text)}
    out |= {OLD_SUB + "_" + n for n in re.findall(pat[1] % (OLD_SUB, NEW_SUB), text)}
    return out


def _string_tokens(text, skip_docstring_of=("dual_promql",)):
    """(line, string) of every string literal of a Python source, except the docstring of the functions in skip_docstring_of
    (F0-F's reader construct explains itself with an example of the interim spelling; comments are no exporter output)."""
    import ast
    import io
    import tokenize
    skip = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name in skip_docstring_of and ast.get_docstring(node, clean=False) is not None:
            d = node.body[0]
            skip.update(range(d.lineno, d.end_lineno + 1))
    return [(t.start[0], t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type == tokenize.STRING and t.start[0] not in skip]


# ---- 1. static: the tree against the inventory of the old tree --------------------------------------------------------------

def test_every_old_metric_name_is_still_in_the_tree_and_the_kit_spelling_is_not():
    mi = _kit("metric_inventory")
    new = _in_git(lambda: mi.scan(mi._iter_root(REPO)))
    fails = mi.compare(_old(), new)
    assert fails == [], "\n".join(fails[:20])


def test_old_inventory_is_what_the_exporters_define():
    """Sanity of the fixture itself: the engine and the P/D-flip front are in it, with the dimensions the panels query."""
    names = {h["name"] for h in _old()["hits"]}
    for must in (OLD_ENG + ":num_running_reqs", OLD_ENG + ":time_to_first_token_seconds", OLD_ENG + ":e2e_request_latency_seconds",
                 OLD_SUB + "_req", OLD_SUB + "_front_ttft_ms_sum", OLD_SUB + "-front"):
        assert must in names, must
    if _has_front_exporter():                   # the 27B line's front has its own Prometheus exposition; the NF line's only pushes points
        for must in (OLD_SUB + "_ttft_seconds", OLD_SUB + "_group", OLD_SUB + "_rank_vram_used_bytes", OLD_SUB + "_decode_round_seconds"):
            assert must in names, must


def test_subject_files_carry_no_kit_spelled_word_at_all():
    """The writer / panel files are pure metric files: not one token of the kit's spelling in them."""
    subject = ["python/flliper/srt/pdflip/front_metrics.py", "python/flliper/srt/pdflip/rank_metrics.py",
               "python/flliper/srt/entrypoints/v1_loads.py", "tools/rig_dashboard/rigdash/vmpush.py",
               "tools/rig_dashboard/rigdash/deploy/grafana/make_dashboard.py",
               "tools/rig_dashboard/rigdash/deploy/grafana/dashboards/rig-verlauf.json",
               "tools/rig_dashboard/rigdash/deploy/vm/scrape.yml",
               "examples/monitoring/grafana/dashboards/json/" + NEW_ENG + "-dashboard.json"]
    rx = re.compile(r"(?<![A-Za-z0-9_])(?:%s_|%s[:_]|%s-front)" % (NEW_SUB, NEW_ENG, NEW_SUB))
    bad = []
    for rel in subject:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if rel.endswith(".py"):              # F0-F's dual_promql(): its prose / comments / example are not exporter output
            lines = _string_tokens(text)
            for i, tok in lines:
                bad += [f"{rel}:{i}: {m.group(0)}" for m in rx.finditer(tok)]
        else:
            for i, ln in enumerate(text.splitlines(), 1):
                bad += [f"{rel}:{i}: {m.group(0)}" for m in rx.finditer(ln)]
    assert bad == [], bad[:10]


def test_grafana_panels_query_the_old_names_and_the_scrape_job_keeps_its_name():
    base = os.path.join(REPO, "tools", "rig_dashboard", "rigdash", "deploy")
    panels = open(os.path.join(base, "grafana", "dashboards", "rig-verlauf.json"), encoding="utf-8").read()
    reads = _panel_reads(panels)                                 # literal old name or the dual-read form (<old>|pdflip)_x
    assert OLD_SUB + "_front_ttft_ms_sum" in reads and OLD_ENG + ":time_to_first_token_seconds" in panels
    assert not re.findall(r"(?<![A-Za-z0-9_|(])%s_[a-z]" % NEW_SUB, panels), "a bare interim-spelled series in a panel"
    scrape = open(os.path.join(base, "vm", "scrape.yml"), encoding="utf-8").read()
    assert "job_name: " + OLD_SUB + "-front" in scrape            # the `job` label value of every scraped front series
    ex = os.path.join(REPO, "examples", "monitoring", "grafana", "dashboards", "json", NEW_ENG + "-dashboard.json")
    if os.path.isfile(ex):
        assert OLD_ENG + "_num_running_reqs" in open(ex, encoding="utf-8").read()


# ---- 2. dynamic: the exporters on synthetic input ----------------------------------------------------------------------------

def test_front_exposition_and_influx_points_keep_the_old_names():
    from flliper.srt.pdflip import front_metrics as fm
    if not hasattr(fm, "FrontMetrics"):
        pytest.skip("this line's front pushes Influx points only (no own Prometheus exposition)")
    pushed = []
    m = fm.FrontMetrics(submit=lambda fn, *a: pushed.append((fn, a)), environ={fm.PUSH_URL_ENV: "http://127.0.0.1:1/write"})
    if m.registry is None:
        pytest.skip("prometheus_client not importable")
    m.leg2_first_content("r1", "after_p", 1.0, 100.0, now=102.0)
    m.served_leg("D", "r1", 1.0, 10, 5, 3)
    m.served_leg("P", "r1", 1.0, 10, 5, 0)
    m.on_event("flip_done", {"sleep": "P", "wake": "D", "flip_ms": 2000, "epoch": 1})
    m.on_event("flip_first_work", {"dir": "P>D", "flip_time_ms": 100, "epoch": 1})
    m.on_event("flip_user_time", {"dir": "D>P", "flip_user_ms": 300, "epoch": 2, "parts": {"park_rpc_ms": 5}})
    m.park_rpc_done(0.1)
    m.set_gauges(queue_len=1, outstanding=2, d_seats=3, d_parked=0, awake="D")
    text = m.aggregate([("P", "# TYPE x gauge\nx 1\n"), ("D", None)])
    got = _names(text)
    want = {OLD_SUB + "_ttft_seconds_count", OLD_SUB + "_served_total", OLD_SUB + "_tokens_total", OLD_SUB + "_flips_total",
            OLD_SUB + "_queue_len", OLD_SUB + "_group_scrape_ok", OLD_SUB + "_awake", OLD_SUB + "_park_rpc_seconds_count",
            OLD_SUB + "_flip_seconds_count", OLD_SUB + "_request_seconds_count", OLD_SUB + "_outstanding", OLD_SUB + "_d_seats",
            OLD_SUB + "_d_parked", OLD_SUB + "_leg2_first_content_seconds_count", OLD_SUB + "_metrics_push"}
    assert want <= got, sorted(want - got)
    assert not [n for n in got if n.startswith(NEW_SUB + "_")], sorted(n for n in got if n.startswith(NEW_SUB + "_"))
    assert OLD_SUB + "_group" in text and (NEW_SUB + "_group") not in text            # the label the front puts on P's and D's metrics
    meas = {re.split(r"[, ]", ln, 1)[0] for ln in m.pusher._buf}
    assert meas == {OLD_SUB + "_flip"}, meas                                            # the Influx measurements
    assert fm.GROUP_LABEL == OLD_SUB + "_group"


def test_rank_gauges_keep_the_old_names(tmp_path, monkeypatch):
    prom = pytest.importorskip("prometheus_client")
    if not _has_front_exporter():
        pytest.skip("this line has no rank gauges")
    from flliper.srt.pdflip import rank_metrics as rm
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    rm._reset_for_tests()
    mm = rm._metrics()
    if mm is None:
        pytest.skip("rank metrics not armable here")
    try:
        names = {k: v._name for k, v in mm.items()}
        assert names == {"vram": OLD_SUB + "_rank_vram_used_bytes", "kv": OLD_SUB + "_rank_kv_usage_ratio",
                         "running": OLD_SUB + "_rank_running", "round": OLD_SUB + "_decode_round_seconds",
                         "tokens": OLD_SUB + "_decode_tokens"}
    finally:
        for v in mm.values():
            try:
                prom.REGISTRY.unregister(v)
            except Exception:  # noqa: BLE001
                pass
        rm._reset_for_tests()


def _vmpush():
    sys.path.insert(0, os.path.join(REPO, "tools", "rig_dashboard"))
    try:
        import importlib
        return importlib.import_module("rigdash.vmpush")
    finally:
        sys.path.remove(os.path.join(REPO, "tools", "rig_dashboard"))


def test_sampler_lines_for_victoriametrics_keep_the_old_names():
    vp = _vmpush()
    ipc = {"boot_id": "nfh91xyz-boot-20261001T064831Z-2025", "terminal": False,
           "front": {"served": {"P": 3, "D": 9}, "awake": "D", "queue": [1, 2], "outstanding_n": 1, "d_phase_n": 1,
                     "d_parked_n": 0, "epoch": 4, "oldest_outstanding_age_s": 1.5,
                     "served_tokens": {"D": {"prompt": 100, "cached": 80, "completion": 7, "n": 9}},
                     "arrival_seat": {"ttft_n": 4, "ttft_ms_sum": 8000, "ttft_ms_max": 3100, "verdict_n": 2, "verdict_ms_sum": 5}}}
    rec = {sec: {} for sec in ("prefill", "decode", "sched", "cap", "cache")}
    for sec, field, _ in vp.RANK_FIELDS:
        rec[sec][field] = 1
    rec["cache"].update({"store_incomplete_delivered": 1, "store_incomplete_deliverable": 1, "mamba_tok": 1,
                         "prefetch": {"attempted": 3, "landed": 2}})
    lines = vp.lines_for_boot(ipc, {"D.tp0pp0": rec}, "27B", 1000)
    lines += vp.decode_sum_lines(vp.decode_sums_empty(), "27B", "boot-1", 1000)
    names = {re.split(r"[{ ]", ln, 1)[0] for ln in lines}
    assert names and all(n.startswith(OLD_SUB + "_") for n in names), sorted(n for n in names if not n.startswith(OLD_SUB + "_"))
    for must in (OLD_SUB + "_front_up", OLD_SUB + "_front_ttft_ms_sum", OLD_SUB + "_front_served_tokens_total",
                 OLD_SUB + "_rank_decode_tokens_total", OLD_SUB + "_rank_l3_prefetch_total",
                 OLD_SUB + "_rank_cache_mamba_tok_total", OLD_SUB + "_boot_decode_tokens_total"):
        assert must in names, must
    # the series the panels read are produced under the same names
    panels = open(os.path.join(REPO, "tools", "rig_dashboard", "rigdash", "deploy", "grafana", "dashboards", "rig-verlauf.json"),
                  encoding="utf-8").read()
    read = {n for n in _panel_reads(panels) if re.match(OLD_SUB + r"_(?:front|rank)_", n)}
    assert len(read) >= 10, sorted(read)
    for n in read:
        assert n in names, n


def test_loads_prometheus_exposition_keeps_the_old_prefix():
    try:
        from flliper.srt.entrypoints import v1_loads
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"v1_loads not importable here: {type(e).__name__}")

    class _Load:
        def to_dict(self, include=None):
            return {"dp_rank": 0, "num_running_reqs": 3, "speculative": {"accept_length": 2.5}, "name": "x"}

    resp = v1_loads._format_loads_prometheus([_Load()])
    names = _names(resp.body.decode())
    assert names == {OLD_ENG + "_num_running_reqs", OLD_ENG + "_spec_accept_length"}, names


# ---- 3. the rename kit leaves them standing ----------------------------------------------------------------------------------

def test_kit_engine_keeps_every_inventoried_name_and_still_renames_the_neighbours():
    R = _kit("rename_to_flliper")
    R._MK = None
    assert R.METRIC_KEEP_FILE and os.path.isfile(R.METRIC_KEEP_FILE), "the engine has no must-keep table"
    miss = []
    for h in _old()["hits"]:
        name = h["name"]
        word = len(OLD_SUB) if name.startswith(OLD_SUB) else len(OLD_ENG)
        iv = R.metric_keep_spans(name, h["path"])
        if not any(a == 0 and b >= word for a, b, _ in iv):
            miss.append((h["path"], name))
    assert miss == [], miss[:10]
    # a metric token stays, the identifier / record key next to it is renamed, a docker tag is no metric
    src = ('f"%s_ttft_seconds" _%s_rank() "%s:num_running_reqs" %s:dev %s_served' % (OLD_SUB, OLD_SUB, OLD_ENG, OLD_ENG, OLD_SUB))
    got, _, _ = R.rewrite_all(src, False, True, {}, "python/%s/srt/managers/scheduler.py" % OLD_ENG)
    assert got == ('f"%s_ttft_seconds" _%s_rank() "%s:num_running_reqs" %s:dev %s_served' %
                   (OLD_SUB, NEW_SUB, OLD_ENG, NEW_ENG, OLD_SUB)), got
    rec = R.rewrite_all("%s_prefill_s" % OLD_SUB, False, True, {}, "python/%s/srt/%s/front.py" % (OLD_ENG, OLD_SUB))[0]
    assert rec == NEW_SUB + "_prefill_s"
    # file-scoped words: the label is kept where it is the label, the server_args attribute is renamed where it is the attribute
    assert R.rewrite_all(OLD_SUB + "_group", False, True, {}, "python/%s/srt/%s/front.py" % (OLD_ENG, OLD_SUB))[0] == OLD_SUB + "_group"
    assert R.rewrite_all(OLD_SUB + "_group", False, True, {}, "python/%s/srt/managers/scheduler.py" % OLD_ENG)[0] == NEW_SUB + "_group"
    # idempotency: the renamed path of the file selects the same rule
    assert R.rewrite_all(OLD_SUB + "_group", False, True, {}, "python/%s/srt/%s/front.py" % (NEW_ENG, NEW_SUB))[0] == OLD_SUB + "_group"


def test_translation_gate_must_keep_lists_the_metric_names():
    with open(os.path.join(KIT, "data", "must_keep.txt")) as f:
        entries = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    for must in (OLD_ENG + ":num_running_reqs", OLD_ENG + ":routing_key_all_req_count", OLD_SUB + "_req", OLD_SUB + "_ttft_seconds",
                 OLD_SUB + "_front_", OLD_SUB + "_rank_"):
        assert must in entries, must


#: dashboard files that mention the names but belong to the dashboard work package (F0-F: readers, generated catalog, UI text).
#: Empty this list when they are restored (metric_inventory.py restore --apply on them); until then they are the only exceptions.
#: (fix round 2: server.py is NOT in the list any more -- it is the engine tile's READER and now spells the engine's names; ``weg2line.py``
#: carries the docker-image regexes ``^flliper:(cu\d+)`` which look like a regex head but are image tags, not metrics.)
OWNED_BY_F0F = ["tools/rig_dashboard/README.md", "tools/rig_dashboard/rigdash/README.md",
                "tools/rig_dashboard/rigdash/profil_data/catalog.json", "tools/rig_dashboard/rigdash/static/index.html",
                "tools/rig_dashboard/rigdash/weg2line.py"]


def test_restore_is_a_noop_on_this_tree():
    """The tool that puts the names back finds nothing left to change (apart from the files of OWNED_BY_F0F)."""
    mi = _kit("metric_inventory")
    left = _in_git(lambda: mi.restore(REPO, os.path.join(KIT, "data", "metric_names_1008.json"), apply=False, skip=OWNED_BY_F0F))
    assert left == [], left[:10]


# ---- 4. label VALUES of the VictoriaMetrics push (a translated value cuts a series like a translated name) --------------------

LABELS_FIXTURE = os.path.join(HERE, "fixtures", "metric_names_1008", "label_values_old.json")


def _label_triples(vp):
    """(metric, label, value) of everything the sampler pushes for a synthetic boot with every rank field, both flip directions
    and every part of a flip view -- the same driver is run on the pre-rename tree to make the fixture."""
    ipc = {"boot_id": "nfh91xyz-boot-20261001T064831Z-2025", "terminal": False,
           "front": {"served": {"P": 3, "D": 9}, "awake": "D", "queue": [1, 2], "outstanding_n": 1, "d_phase_n": 1,
                     "d_parked_n": 0, "epoch": 4, "oldest_outstanding_age_s": 1.5,
                     "served_tokens": {"D": {"prompt": 100, "cached": 80, "completion": 7, "n": 9}},
                     "arrival_seat": {"ttft_n": 4, "ttft_ms_sum": 8000, "ttft_ms_max": 3100, "verdict_n": 2, "verdict_ms_sum": 5}}}
    rec = {sec: {} for sec in ("prefill", "decode", "sched", "cap", "cache")}
    for sec, field, _ in vp.RANK_FIELDS:
        rec[sec][field] = 1
    rec["cache"].update({"store_incomplete_delivered": 1, "store_incomplete_deliverable": 1, "mamba_tok": 1,
                         "prefetch": {"attempted": 3, "landed": 2, "refused": 1}})
    lines = vp.lines_for_boot(ipc, {"D.tp0pp0": rec, "P.tp1pp0": rec}, "27B", 1000)
    lines += vp.decode_sum_lines(vp.decode_sums_empty(), "27B", "boot-1", 1000)
    # a flip view row carries the parts under either spelling of the internal key; VIEW_PARTS reads the one it knows
    keys = {k for _, k in vp.VIEW_PARTS} | {"vorlauf_ms", "warmup_ms"}
    done = set()
    for i, d in enumerate(("P>D", "D>P")):
        row = dict({k: 1.0 + i for k in keys}, kind="ok", total_ms=5.0, dir=d, begin=100.0 + i)
        lines += vp.flip_view_points([row], "27B", "boot-1", done)
    out = set()
    for ln in lines:
        name = re.split(r"[{ ]", ln, 1)[0]
        for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', ln):
            out.add((name, k, v))
    return out


@pytest.mark.skipif(not os.path.isfile(LABELS_FIXTURE), reason="label-value fixture not in this tree")
def test_vm_push_label_values_are_the_pre_rename_ones():
    """Fixture = the same driver on the pre-rename dashboard (27B 86ff356d0d / NF a452294dd2; both spell every label value alike).
    An exact comparison: a changed part name (vorlauf -> warmup, F0-M finding 1) or a new label value fails it."""
    with open(LABELS_FIXTURE) as f:
        old = {tuple(x) for x in json.load(f)["triples"]}
    got = _label_triples(_vmpush())
    assert got == old, {"missing": sorted(old - got)[:10], "new": sorted(got - old)[:10]}
    assert ("%s_flip_user_view_ms" % OLD_SUB, "part", "vorlauf") in got
    assert not [t for t in got if t[1] == "part" and t[2] == "warmup"]


# ---- 5. the READERS of the exposition (fix round 2): a regex head is a name too ------------------------------------------------
# F0-M restored the keys of rigmon/sources.py and the dashboard's keep map but not the regex that parses the exposition: the engine
# exported ``sglang:*``, the parser read ``flliper:*``, every live value went empty and nothing raised.  These tests run the PARSERS on
# a REAL exposition (the engine's SchedulerMetricsCollector through prometheus_client.generate_latest), not on strings of the test.

_EXPO_PROG = r'''
import types
from prometheus_client import REGISTRY, generate_latest
from flliper.srt.observability.metrics_collector import QueueCount, SchedulerMetricsCollector, SchedulerStats

sa = types.SimpleNamespace(
    enable_metrics=True, enable_metrics_for_all_schedulers=False, kv_events_config=None,
    prefill_delayer_forward_passes_buckets=None, prefill_delayer_max_delay_passes=0, prefill_delayer_wait_seconds_buckets=None)
labels = {"model_name": "m", "moe_ep_rank": 0, "engine_type": "e", "tp_rank": 0, "pp_rank": 0, "dp_rank": 0}
c = SchedulerMetricsCollector(labels=labels, server_args=sa)
s = SchedulerStats()
s.num_running_reqs = QueueCount(total=3)
s.num_queue_reqs = QueueCount(total=2)
s.gen_throughput = 12.5
s.token_usage = 0.25
s.cache_hit_rate = 0.5
c.log_stats(s)
print("@@EXPO" + generate_latest(REGISTRY).decode().replace("\n", "\x01"))
'''


@pytest.fixture(scope="module")
def real_exposition():
    pytest.importorskip("prometheus_client")
    import subprocess
    out = subprocess.run([sys.executable, "-c", _EXPO_PROG], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "python") + os.pathsep + os.environ.get("PYTHONPATH", "")))
    if out.returncode != 0 and ("ImportError" in out.stderr or "ModuleNotFoundError" in out.stderr):
        pytest.skip("the engine's metrics collector is not importable here: " + out.stderr[-300:])
    assert out.returncode == 0, out.stderr[-1500:]            # a failing driver is a failure, never a skip
    line = [x for x in out.stdout.splitlines() if x.startswith("@@EXPO")]
    assert line, out.stdout[-500:] + out.stderr[-500:]
    return line[0][len("@@EXPO"):].replace("\x01", "\n")


def test_the_real_exposition_is_in_the_old_spelling(real_exposition):
    names = _names(real_exposition)
    assert OLD_ENG + ":num_running_reqs" in names and OLD_ENG + ":gen_throughput" in names, sorted(names)[:10]
    assert not [n for n in names if n.startswith(NEW_ENG + ":")]


def test_rigmon_parser_reads_the_real_exposition(real_exposition):
    from flliper.srt.rigmon import sources as rs
    got = rs.parse_prometheus(real_exposition)
    assert got, "parse_prometheus found nothing in a real exposition: its regex does not read the engine's spelling"
    assert all(k.startswith(OLD_ENG + ":") for k in got), sorted(got)[:5]
    assert got[OLD_ENG + ":num_running_reqs"][0][1] == 3.0 and got[OLD_ENG + ":num_running_reqs"][0][0]["tp_rank"] == "0"
    assert got[OLD_ENG + ":gen_throughput"][0][1] == 12.5
    # every key the collector maps to a short name is the spelling the parser returns (keys and regex agree)
    keys = list(rs._ENGINE_KEYS.values()) + list(rs.PER_RANK_KEYS.values()) + [rs.FORWARD_TIME_METRIC]
    assert all(k.startswith(OLD_ENG + ":") for k in keys), [k for k in keys if not k.startswith(OLD_ENG + ":")]
    present = [k for k in keys if k in got]
    assert OLD_ENG + ":num_running_reqs" in present and OLD_ENG + ":gen_throughput" in present and len(present) >= 4, present


def test_dashboard_server_engine_tile_reads_the_real_exposition(real_exposition, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "tools", "rig_dashboard"))
    try:
        import importlib
        srv = importlib.import_module("server")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"dashboard server not importable here: {type(e).__name__}")
    finally:
        sys.path.remove(os.path.join(REPO, "tools", "rig_dashboard"))
    monkeypatch.setitem(srv._CFG, NEW_ENG, "http://127.0.0.1:1")
    monkeypatch.setattr(srv, "_get", lambda url, timeout=1.5: real_exposition)
    vals = srv.scrape_metrics()
    assert vals, "the dashboard's engine tile reads nothing from a real exposition"
    assert vals["num_running_reqs"] == 3.0 and vals["num_queue_reqs"] == 2.0 and vals["gen_throughput"] == 12.5, vals
    assert vals["token_usage"] == 0.25 and vals["cache_hit_rate"] == 0.5, vals


def test_no_regex_head_or_grep_of_a_reader_spells_the_kit_name():
    """Static: a reader that matches the exposition with a regex / grep (``^sglang:[a-z_]+``, ``^sglang:(a|b)``) keeps the old head.
    Docker-image regexes (``^flliper:(cu\\d+)``, ``(\\d+...)``) are image tags, not metrics."""
    import subprocess
    rx = re.compile(r"(?<![A-Za-z0-9_])" + NEW_ENG + r":[\[(\\]")
    files = _in_git(lambda: subprocess.run(["git", "-C", REPO, "ls-files", "-z"], check=True,
                                           stdout=subprocess.PIPE).stdout.split(b"\0"))
    bad = []
    for raw in files:
        rel = raw.decode("utf-8", "surrogateescape")
        if not rel.endswith((".py", ".sh", ".json", ".yml", ".yaml", ".html", ".js")):
            continue
        if rel.startswith(("tools/release/", "python/flliper/test/")) or "/tests/" in rel or rel.startswith("test/"):
            continue
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for i, ln in enumerate(text.splitlines(), 1):
            for m in rx.finditer(ln):
                if not re.match(r"\((?:cu|\\d)", ln[m.end() - 1:]):
                    bad.append(f"{rel}:{i}: {ln.strip()[:100]}")
    assert bad == [], bad[:10]


def test_kit_keeps_the_regex_head_of_a_reader():
    R = _kit("rename_to_flliper")
    R._MK = None
    head = 're.compile(r"^(%s:[a-z_0-9]+)(?:\\{)") grep -E "^%s:(cache_hit_rate|x)"' % (OLD_ENG, OLD_ENG)
    for path in ("python/%s/srt/rigmon/sources.py" % OLD_ENG, "tools/rig_dashboard/server.py",
                 "scripts/dev/543_yarn/validate_544.sh", "python/%s/srt/any/new_reader.py" % OLD_ENG):
        assert R.rewrite_all(head, False, True, {}, path)[0] == head, path
    # the neighbours are still renamed: a docker-image regex, a process title, a docker tag
    other = "^%s:(cu\\d+)-x %s::scheduler %s:dev" % (OLD_ENG, OLD_ENG, OLD_ENG)
    assert R.rewrite_all(other, False, True, {}, "tools/rig_dashboard/rigdash/weg2line.py")[0] == \
        "^%s:(cu\\d+)-x %s::scheduler %s:dev" % (NEW_ENG, NEW_ENG, NEW_ENG)
    # the table says so (and the translation gate keeps the head in string units)
    tb = _kit("metric_inventory")
    assert tb.REGEX_HEADS == (OLD_ENG + ":",)
    with open(os.path.join(KIT, "data", "metric_names_1008.json")) as f:
        table = json.load(f)
    assert table["regex_heads"] == [OLD_ENG + ":"]
    for rd in ("python/%s/srt/rigmon/sources.py" % OLD_ENG, "tools/rig_dashboard/server.py"):
        assert table["files"][rd] == [OLD_ENG + ":"], rd
    with open(os.path.join(KIT, "data", "must_keep.txt")) as f:
        entries = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    assert {OLD_ENG + ":[", OLD_ENG + ":("} <= entries
