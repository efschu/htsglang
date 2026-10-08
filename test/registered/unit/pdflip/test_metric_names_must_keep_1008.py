"""F0-M (08.10.2026), user decision "pdflip as the name, the metrics are must-keep": the exported METRIC NAMES keep the
spelling they had before the rename, so the series in VictoriaMetrics and the Grafana panels (rig-verlauf) do not break.

  * old names:  the engine's series ``sglang:*`` (and ``sglang_*`` on /v1/loads?format=prometheus) and the P/D-flip subsystem's
                series, labels and Influx measurements ``weg2_*`` (+ the scrape job name ``weg2-front``);
  * new names:  none -- ``pdflip_*`` / ``flliper:*`` / ``flliper_*`` for the same metric must NOT appear next to them (no double export).

The fixture ``fixtures/metric_names_1008/inv_old.json`` is the inventory of the pre-rename tree (tools/release/metric_inventory.py
scan on the base ref).  Four layers: the static scan of this tree against it, the exporters run on synthetic input, the rename
kit's engine on the old spellings (a re-run of the kit leaves them standing), the must-keep list of the translation gate.

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
            for i, ln in enumerate(f, 1):
                bad += [f"{rel}:{i}: {m.group(0)}" for m in rx.finditer(ln)]
    assert bad == [], bad[:10]


def test_grafana_panels_query_the_old_names_and_the_scrape_job_keeps_its_name():
    base = os.path.join(REPO, "tools", "rig_dashboard", "rigdash", "deploy")
    panels = open(os.path.join(base, "grafana", "dashboards", "rig-verlauf.json"), encoding="utf-8").read()
    assert OLD_SUB + "_front_ttft_ms_sum" in panels and OLD_ENG + ":time_to_first_token_seconds" in panels
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
    for n in set(re.findall(r"(?<![A-Za-z0-9_])" + OLD_SUB + r"_(?:front|rank)_[a-z0-9_]+", panels)):
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
OWNED_BY_F0F = ["tools/rig_dashboard/server.py", "tools/rig_dashboard/README.md", "tools/rig_dashboard/rigdash/README.md",
                "tools/rig_dashboard/rigdash/profil_data/catalog.json", "tools/rig_dashboard/rigdash/static/index.html"]


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
