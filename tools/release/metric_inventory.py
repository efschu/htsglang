#!/usr/bin/env python3
"""metric_inventory.py -- which metric names / labels leave the process, old tree vs renamed tree (F0-M, 08.10.2026).

User decision 08.10.2026 ~19:30Z: "pdflip as the name, the metrics are must-keep".  The exported SERIES names and label
names (Prometheus exposition of the servers and of the front, the Influx-line points pushed to VictoriaMetrics, the
rig dashboard's sampler writing ``/api/v1/import/prometheus``, the Grafana panels that query them) keep the spelling
they had before the rename: ``weg2_*`` and ``sglang:*`` / ``sglang_*``.  Reason: the history in VictoriaMetrics and
the Grafana panels (rig-verlauf) must not break.  Module, flag, env, marker and path names stay pdflip / flliper.

What this tool does
-------------------
  scan     read one tree (git ref or directory) and list every METRIC-NAME LITERAL it emits or reads:
             * ``prom-def``      a prometheus_client / wrapper constructor call (Counter, Gauge, Histogram, Summary, Info, Enum,
                                 ``*MetricFamily``, ``Ray*Wrapper``) with a literal name,
             * ``influx``        the measurement of an ``influx_line("...")`` / ``_push("...")`` call,
             * ``emitter``       every string literal that starts with a metric-family prefix inside an EMITTER MODULE (a module
                                 that has a ``prom-def`` or ``influx`` hit, or is listed in EMITTER_MODULES: the sampler's
                                 ``vmpush.py`` builds its names by concatenation, ``v1_loads.py`` by f-string),
             * ``reader``        the same literals in a module that only READS metrics (planner/live_metrics.py, rigmon/sources.py ...),
                                 found by the family prefix + the set of defined names,
             * ``text``          non-Python files (Grafana dashboard JSON, scrape.yml, shell probes): every metric-family token.
           Output: JSON ``{"family": ..., "hits": [{"path", "line", "kind", "name"}]}``.
  keepfile make the engine's must-keep table (data/metric_names_1008.json) from the scans of the OLD trees and prove it covers every
           hit of every scan (the engine reads it: rename_to_flliper.py `METRIC-NAME MUST-KEEP`).
  restore  put the metric names back in a tree the kit renamed BEFORE the must-keep table existed (F0-M): every token the table
           protects, spelled the kit's way (``pdflip_*`` / ``flliper:*`` / ``flliper_*``), goes back to the old spelling.  The same
           table, the same per-file scope as the engine -- a re-run of the kit on the old tree gives the same tokens.
  compare  OLD scan (the fixture) against the NEW tree: every old name must still be in the new tree (same spelling), and the
           new tree must not carry the kit's renamed spelling of a metric (``pdflip_*`` / ``flliper:*`` ...) in an emitter, a reader
           or a panel.  Exit 1 on any difference.  This is the check the F0-M test runs.
  table    old -> kit-renamed -> now, per name (the inventory table of the F0-M report).

Name families (old spelling -> what the kit makes of it)
--------------------------------------------------------
  weg2_*   -> pdflip_*        the P/D-flip subsystem's series (front, ranks, sampler, Influx points)
  sglang:* -> flliper:*       the engine's series (colon form of upstream sglang)
  sglang_* -> flliper_*       the ``/v1/loads?format=prometheus`` exposition (underscore form)
Docker tags and org ids that look alike (``lmsysorg/sglang:v0.5``, ``sglang:dev``) are NOT metrics: a token counts only when
it is DEFINED as a metric somewhere in the same scan, or sits in an emitter / panel position (see ``scan``).
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import subprocess
import sys
import warnings
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

# old spelling prefix -> kit spelling prefix (the three families)
FAMILIES = (("weg2_", "pdflip_"), ("sglang:", "flliper:"), ("sglang_", "flliper_"), ("weg2-front", "pdflip-front"))
OLD_RX = re.compile(r"(?<![A-Za-z0-9_])(?:weg2_|sglang[:_])[A-Za-z0-9_:%{}.]*|(?<![A-Za-z0-9_])weg2-front(?![A-Za-z0-9_-])")
NEW_RX = re.compile(r"(?<![A-Za-z0-9_])(?:pdflip_|flliper[:_])[A-Za-z0-9_:%{}.]*|(?<![A-Za-z0-9_])pdflip-front(?![A-Za-z0-9_-])")
CONSTRUCTORS = {"Counter", "Gauge", "Histogram", "Summary", "Info", "Enum", "CounterMetricFamily", "GaugeMetricFamily",
                "HistogramMetricFamily", "SummaryMetricFamily", "InfoMetricFamily", "RayCounterWrapper", "RayGaugeWrapper",
                "RayHistogramWrapper", "RaySummaryWrapper"}
INFLUX_CALLS = {"influx_line", "_push"}
# modules that EMIT names by concatenation / f-string (no literal constructor call): added by hand, proved by the test
EMITTER_MODULES = (
    "tools/rig_dashboard/rigdash/vmpush.py",          # sampler -> /api/v1/import/prometheus: "weg2_front_" + k, "weg2_rank_cache_%s_total"
    "python/{pkg}/srt/entrypoints/v1_loads.py",       # /v1/loads?format=prometheus: f"sglang_{prefix}_{sub_key}"
    "python/{pkg}/srt/{sub}/front_metrics.py",
    "python/{pkg}/srt/{sub}/rank_metrics.py",
)
# modules that only READ (or lay out panels over) weg2_ series: their weg2_ literals count, other modules' weg2_ literals are record keys
READER_MODULES = (
    "tools/rig_dashboard/rigdash/deploy/grafana/make_dashboard.py",
)
# trees whose content the rename never touched (foreign, docs) -- not scanned
SKIP_DIRS = ("3rdparty/", "sgl-kernel/", "sgl-model-gateway/", "experimental/", "rust/", "proto/", ".github/", "docs/",
             "docs_new/", ".claude/", "tools/release/", "test/", "python/sglang/test/", "python/flliper/test/")
TEXT_EXT = (".json", ".yml", ".yaml", ".sh", ".html", ".js", ".toml")


def _iter_ref(repo: str, ref: str) -> Iterator[Tuple[str, bytes]]:
    rows = []
    for rec in subprocess.run(["git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", ref], check=True,
                              stdout=subprocess.PIPE).stdout.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha = meta.split()
        if typ == b"blob" and mode != b"120000":
            rows.append((path.decode("utf-8", "surrogateescape"), sha.decode()))
    rows.sort()
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    for path, sha in rows:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        size = int(proc.stdout.readline().split()[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        yield path, data
    proc.stdin.close()
    proc.wait()


def _iter_root(root: str) -> Iterator[Tuple[str, bytes]]:
    out = subprocess.run(["git", "-C", root, "ls-files", "-z", "-c"], check=True, stdout=subprocess.PIPE).stdout
    for p in sorted({x.decode("utf-8", "surrogateescape") for x in out.split(b"\0") if x}):
        full = os.path.join(root, p)
        if os.path.isfile(full) and not os.path.islink(full):
            with open(full, "rb") as f:
                yield p, f.read()


# the rig dashboard READS the series (F0-F: it queries both spellings, ``(weg2|pdflip)_x``): only its WRITER (the sampler's
# vmpush.py) and the shipped Grafana / VictoriaMetrics deployment files are part of the metric contract
DASHBOARD_SUBJECT = ("tools/rig_dashboard/rigdash/vmpush.py", "tools/rig_dashboard/rigdash/deploy/")


def _in_scope(path: str) -> bool:
    if path.startswith("tools/rig_dashboard/") and not path.startswith(DASHBOARD_SUBJECT):
        return False
    return not any(path.startswith(d) for d in SKIP_DIRS) and "/tests/" not in path and "/test/" not in path


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _joined_head(node: ast.AST) -> Optional[str]:
    """Text of an f-string / %-format / concatenation with formatted parts as ``{}`` (names built dynamically)."""
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        left = _const_str(node.left) or _joined_head(node.left)
        if left is not None:
            right = _const_str(node.right) if isinstance(node.op, ast.Add) else None
            return left + (right if right is not None else "{}")
    return None


_CTOR_WORD = re.compile(r"(Counter|Gauge|Histogram|Summary|Info|Enum)")


def _is_ctor(name: str) -> bool:
    """prometheus_client classes and our own wrappers/subclasses (``GaugeHistogram``, ``Ray*Wrapper``, ``*MetricFamily``)."""
    return name in CONSTRUCTORS or (bool(name) and name[0].isupper() and _CTOR_WORD.search(name) is not None)


def _call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _name_arg(call: ast.Call) -> Optional[ast.AST]:
    for kw in call.keywords:
        if kw.arg == "name":
            return kw.value
    return call.args[0] if call.args else None


def _norm(s: str) -> str:
    """The metric-name token at the head of a literal (``name%s %s %d`` -> ``name``; ``a_%s_total`` keeps its hole)."""
    m = re.match(r"[A-Za-z0-9_:]+(?:(?:%s|\{\})[A-Za-z0-9_:]*)*", s)
    tok = m.group(0) if m else s
    for hole in ("%s", "{}"):
        if tok.endswith(hole) and len(s) > len(tok) - 0 and s[len(tok):len(tok) + 1] in (" ", "{"):
            tok = tok[: -len(hole)]
    return tok


def _py_hits(path: str, text: str) -> Tuple[List[dict], List[dict]]:
    """(definition hits, every family-prefixed string constant of the module)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except SyntaxError:
        return [], []
    defs: List[dict] = []
    lits: List[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            cn = _call_name(node)
            if _is_ctor(cn) or cn in INFLUX_CALLS:
                arg = _name_arg(node)
                s = (_const_str(arg) or _joined_head(arg)) if arg is not None else None
                if s is not None and (OLD_RX.match(s) or NEW_RX.match(s)):
                    defs.append({"path": path, "line": node.lineno, "kind": "influx" if cn in INFLUX_CALLS else "prom-def",
                                 "name": _norm(s)})
        elif isinstance(node, ast.JoinedStr):
            s = _joined_head(node)
            if s and (OLD_RX.match(s) or NEW_RX.match(s)):
                lits.append({"path": path, "line": node.lineno, "kind": "literal", "name": _norm(s)})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if len(s) < 200 and (OLD_RX.match(s) or NEW_RX.match(s)):
                lits.append({"path": path, "line": node.lineno, "kind": "literal", "name": _norm(s)})
    return defs, lits


def _is_sglang_family(name: str) -> bool:
    return old_spelling(name).startswith(("sglang:", "sglang_"))


def scan(files: Iterable[Tuple[str, bytes]]) -> dict:
    hits: List[dict] = []
    py_defs: Dict[str, List[dict]] = {}
    py_lits: Dict[str, List[dict]] = {}
    texts: Dict[str, str] = {}
    for path, data in files:
        if not _in_scope(path):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if path.endswith(".py"):
            d, l = _py_hits(path, text)
            if d:
                py_defs[path] = d
            if l:
                py_lits[path] = l
        elif path.endswith(TEXT_EXT) and (OLD_RX.search(text) or NEW_RX.search(text)):
            texts[path] = text
    # 1. definitions: constructor / influx calls with a literal name (anywhere), and every literal of an EMITTER MODULE
    #    (names built by concatenation: no constructor call to find)
    for path, ds in py_defs.items():
        hits.extend(ds)
    emitters = {p for p in py_lits if any(_match_module(p, m) for m in EMITTER_MODULES)}
    for path in sorted(emitters):
        have = {h["name"] for h in py_defs.get(path, [])}
        for h in py_lits[path]:
            if h["name"] not in have:
                hits.append(dict(h, kind="emitter"))
                have.add(h["name"])
    defined = {h["name"] for h in hits if h["kind"] in ("prom-def", "influx", "emitter") and not h["name"].endswith(("_", ":"))}
    defined.add("weg2-front")   # the `job` label value of the VictoriaMetrics scrape job (series identity)
    defined_keys = {_key(n) for n in defined}
    # 2. the engine's own emitters outside the listed modules: a literal of the sglang families that is a defined name
    #    (other modules that DEFINE with a literal constructor are already in `hits`); weg2_ literals elsewhere are record keys
    #    and request fields (``weg2_prefill_s``, ``weg2_decode_progress``), never metrics -- they are not matched here.
    for path, ls in py_lits.items():
        if path in emitters:
            continue
        reader_ok = any(_match_module(path, m) for m in READER_MODULES)
        seen_here = {h["name"] for h in py_defs.get(path, [])}
        for h in ls:
            if h["name"] in seen_here:
                continue
            if _key(_strip_hist(h["name"])) in defined_keys and (_is_sglang_family(h["name"]) or reader_ok):
                hits.append(dict(h, kind="reader"))
    # 3. panels / probes / config: every defined name inside non-Python files
    for path, text in texts.items():
        for i, line in enumerate(text.splitlines(), 1):
            for rx in (OLD_RX, NEW_RX):
                for m in rx.finditer(line):
                    tok = m.group(0).rstrip(".:")
                    k = _key(_strip_hist(tok))
                    # a dashboard JSON may spell the colon family with an underscore (examples/monitoring: sglang_num_running_reqs)
                    if k in defined_keys or (k.startswith("sglang_") and "sglang:" + k[len("sglang_"):] in defined_keys):
                        hits.append({"path": path, "line": i, "kind": "text", "name": tok})
    # line-level duplicates (a constructor call and its literal) collapse by (path, name, kind-class)
    uniq: Dict[Tuple[str, str, int], dict] = {}
    for h in hits:
        uniq.setdefault((h["path"], h["name"], h["line"]), h)
    hits = sorted(uniq.values(), key=lambda h: (h["path"], h["line"], h["name"]))
    return {"family": "weg2_/sglang:/sglang_", "defined": sorted(defined), "hits": hits}


def _match_module(path: str, pattern: str) -> bool:
    rx = "^" + re.escape(pattern).replace(r"\{pkg\}", r"(?:sglang|flliper)").replace(r"\{sub\}", r"(?:weg2|pdflip)") + "$"
    return re.match(rx, path) is not None


_HIST_SUFFIX = ("_bucket", "_sum", "_count", "_created", "_total")


def _strip_hist(name: str) -> str:
    for s in _HIST_SUFFIX:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def old_spelling(name: str) -> str:
    """The kit-renamed spelling mapped back (flliper: -> sglang:, pdflip_ -> weg2_ ...); old names map to themselves."""
    for old, new in FAMILIES:
        if name.startswith(new):
            return old + name[len(new):]
    return name


def kit_spelling(name: str) -> str:
    for old, new in FAMILIES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def _key(name: str) -> str:
    return old_spelling(name)


def is_new_spelling(name: str) -> bool:
    return any(name.startswith(new) for _, new in FAMILIES)


def compare(old: dict, new: dict) -> List[str]:
    """Differences: an old name that vanished, a kit-spelled metric that is back or was never restored."""
    fails: List[str] = []
    new_names = collections.Counter(h["name"] for h in new["hits"])
    for name, n in sorted(collections.Counter(h["name"] for h in old["hits"]).items()):
        if new_names.get(name, 0) < n:
            fails.append(f"old name {name!r}: {n} in the old tree, {new_names.get(name, 0)} in the new tree"
                         + (f" (kit spelling {kit_spelling(name)!r}: {new_names.get(kit_spelling(name), 0)})"
                            if kit_spelling(name) != name else ""))
    for h in new["hits"]:
        if is_new_spelling(h["name"]):
            fails.append(f"kit spelling {h['name']!r} at {h['path']}:{h['line']} ({h['kind']}) -- metric names keep the old spelling")
    return fails


def table(old: dict, new: dict) -> List[Tuple[str, str, int, int]]:
    """(old name, kit spelling, count in old tree, count in the new tree under the old spelling)."""
    oc = collections.Counter(h["name"] for h in old["hits"])
    nc = collections.Counter(h["name"] for h in new["hits"])
    return [(n, kit_spelling(n), c, nc.get(n, 0)) for n, c in sorted(oc.items())]


# ---- policy of the must-keep table (data, regenerated by `keepfile`) ---------------------------------------------------------
#: token families that occur in the old trees ONLY as metric names (checked 08.10.2026 on 86ff356d0d and a452294dd2 with
#: `git grep -E "(^|[^A-Za-z0-9_])weg2_(front|rank|boot_decode|gpu_pcie)_"`: writer, panels, tests -- no record key, no attribute)
GLOBAL_WEG2_PREFIXES = ("weg2_front_", "weg2_rank_", "weg2_boot_decode_", "weg2_gpu_pcie_")
#: metric names the sampler / the front mention that no scan finds as a literal definition (prose, per-field series)
EXTRA_WEG2_EXACT = ("weg2_flip_user_view_ms", "weg2_flip_user_ms", "weg2_flip_time_ms", "weg2_decode_tokens", "weg2_metrics_errors")
#: also record keys / server_args attributes elsewhere in the trees: kept only inside the files that mean the metric / label
AMBIGUOUS_WEG2 = ("weg2_group", "weg2_d_parked")
#: colon-family names that are READ (a probe script, a test) but have no literal definition in our trees: legacy upstream names and a family name
#: without its ``_total`` suffix (``sglang:num_prefill_retries`` is the family of ``..._total``)
EXTRA_COLON = ("num_generation_tokens_total", "time_per_output_token_seconds", "num_prefill_retries")
#: docs / prose wildcards of the colon family (`sglang:spill_tier_*_bytes`): kept as written
COLON_WILDCARDS = ("encoder_*", "hicache_host_*_tokens", "spill_tier_*_bytes")
#: REGEX HEADS (fix round 2): the family prefix written as the head of a regex in a READER of the exposition (`^(sglang:[a-z_0-9]+)`,
#: rigmon/sources.py + rig_dashboard/server.py); the engine keeps `sglang:` where a `[`, `(` or a backslash follows
REGEX_HEADS = ("sglang:",)
#: old path -> entries.  A trailing "_" or ":" is a PREFIX (every token starting with it), anything else an exact token
FILE_ENTRIES = {
    "python/sglang/srt/weg2/front_metrics.py": ["weg2_"],
    "python/sglang/srt/weg2/rank_metrics.py": ["weg2_"],
    "python/sglang/srt/weg2/front.py": ["weg2_group"],            # comments: the label the front sets on the group metrics
    "python/sglang/srt/entrypoints/v1_loads.py": ["sglang_"],     # /v1/loads?format=prometheus: f"sglang_{prefix}_{sub_key}"
    "tools/rig_dashboard/rigdash/vmpush.py": ["weg2_"],           # the sampler's writer
    # the two READERS of the engine's exposition: their regex and key map spell the engine's names (fix round 2: the restore had missed the
    # regex head, every live value of rigmon / the dashboard's engine tile went empty)
    "python/sglang/srt/rigmon/sources.py": ["sglang:"],
    "tools/rig_dashboard/server.py": ["sglang:"],
    "tools/rig_dashboard/rigdash/deploy/grafana/make_dashboard.py": ["weg2_"],
    "tools/rig_dashboard/rigdash/deploy/grafana/dashboards/rig-verlauf.json": ["weg2_"],
    # job_name `weg2-front` is the `job` LABEL VALUE of every series scraped from the front: renamed, the history splits in two series
    "tools/rig_dashboard/rigdash/deploy/vm/scrape.yml": ["weg2_", "sglang:", "weg2-front"],
    "examples/monitoring/grafana/dashboards/json/sglang-dashboard.json": ["sglang_"],   # the example panels query sglang_<name>
    # the OpenTelemetry-sanitised form of a colon name is its underscore twin: the test pins one real name (the other names there are test-local)
    "test/registered/unit/observability/test_ray_wrappers.py": ["sglang_num_running_reqs"],
    # tests that pin the exported names (they travel with the writer)
    "test/registered/unit/weg2/test_weg2_tsdb_metrics_1001.py": ["weg2_"],
    "tools/rig_dashboard/rigdash/tests/test_vm_1001.py": ["weg2_"],
    "tools/rig_dashboard/rigdash/tests/test_vm_boot_rates_1002.py": ["weg2_"],
    "tools/rig_dashboard/rigdash/tests/test_flipzeit_punkte_1006.py": ["weg2_"],
    "tools/rig_dashboard/rigdash/tests/test_flipzeit_1002.py": ["weg2_"],
}


MK_BEGIN = "# --- METRIC NAMES (F0-M, generated by metric_inventory.py keepfile --must-keep; do not edit by hand) ---"
MK_END = "# --- end METRIC NAMES ---"


def must_keep_section(table: dict) -> List[str]:
    """english_audit must-keep lines for the metric names: OLD spelling, a translated log/help/string unit keeps every occurrence."""
    names = ["sglang:" + n for n in table["sglang_colon"]] + list(table["sglang_colon_patterns"] and ["sglang:" + n for n in table["sglang_colon_patterns"]])
    names += table["weg2_global"] + table["weg2_global_prefixes"]
    for ents in table["files"].values():
        names += [e for e in ents if e not in ("weg2_", "sglang_", "sglang:")]
    names += ["weg2_group", "weg2_d_parked"]
    names += [h + c for h in table.get("regex_heads", []) for c in "[("]
    return [MK_BEGIN] + sorted(set(names)) + [MK_END]


def write_must_keep(path: str, table: dict) -> None:
    with open(path) as f:
        lines = f.read().split("\n")
    if MK_BEGIN in lines:
        a, b = lines.index(MK_BEGIN), lines.index(MK_END)
        lines = lines[:a] + lines[b + 1:]
    while lines and lines[-1] == "":
        lines.pop()
    lines += must_keep_section(table) + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def keepfile(scans: List[dict]) -> dict:
    colon, wg = set(), set()
    for sc in scans:
        for h in sc["hits"]:
            n = h["name"]
            if n.startswith("sglang:") and h["kind"] in ("prom-def", "emitter"):
                colon.add(n[len("sglang:"):])
            elif n.startswith("weg2_") and not n.endswith("_") and "%" not in n and "{" not in n:
                wg.add(n)
    wg.update(EXTRA_WEG2_EXACT)
    wg -= set(AMBIGUOUS_WEG2)
    wg = {n for n in wg if not n.startswith(GLOBAL_WEG2_PREFIXES)}
    return {
        "schema": "metric-names-keep/1",
        "_comment": ("F0-M 08.10.2026: series and label names that leave the process keep their pre-rename spelling (user decision "
                     "'pdflip as the name, the metrics are must-keep'). Entries are OLD spellings. Read by rename_to_flliper.py "
                     "(METRIC-NAME MUST-KEEP). Regenerate: metric_inventory.py keepfile; prove: metric_inventory.py compare."),
        "sglang_colon": sorted(colon | set(EXTRA_COLON)),
        "sglang_colon_patterns": sorted(COLON_WILDCARDS),
        "regex_heads": list(REGEX_HEADS),
        "weg2_global": sorted(wg),
        "weg2_global_prefixes": list(GLOBAL_WEG2_PREFIXES),
        "files": {k: v for k, v in sorted(FILE_ENTRIES.items())},
    }


def _kit_spelled_table(table: dict) -> dict:
    """The must-keep table with every entry spelled the way the kit spells it (to find the tokens in a renamed tree)."""
    def sp(n: str) -> str:
        return kit_spelling(n)
    out = dict(table)
    out["sglang_colon"] = list(table.get("sglang_colon", []))          # kept after the prefix: handled by the prefix swap below
    out["weg2_global"] = [sp(n) for n in table.get("weg2_global", [])]
    out["weg2_global_prefixes"] = [sp(n) for n in table.get("weg2_global_prefixes", [])]
    out["files"] = {k: [sp(e) for e in v] for k, v in table.get("files", {}).items()}
    return out


def restore(root: str, table_path: str, apply: bool, skip: Iterable[str] = ()) -> List[Tuple[str, int]]:
    """Rewrite kit-spelled metric tokens back; returns [(path, n_tokens)]."""
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rename_to_flliper as R
    with open(table_path) as f:
        table = json.load(f)
    kit = _kit_spelled_table(table)
    # colon family: the engine pattern is `sglang:<name>`; in the renamed tree it is `flliper:<name>`
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(kit, tf)
        tmp = tf.name
    glob, files = R._load_metric_keep(tmp)
    os.unlink(tmp)
    glob = [(n, re.compile(rx.pattern.replace("sglang:", "flliper:", 1) if "sglang:" in rx.pattern else rx.pattern)) for n, rx in glob]
    changed: List[Tuple[str, int]] = []
    for path, data in _iter_root(root):
        if not R.in_scope(path) or os.path.splitext(path)[1] in R.CXX_EXT or path in skip:
            continue
        text = R.as_text(data)
        if text is None:
            continue
        spans = [(m.start(), m.end()) for _, rx in glob for m in rx.finditer(text)]
        for rx in files.get(path, ()):
            spans.extend((m.start(), m.end()) for m in rx.finditer(text))
        if not spans:
            continue
        spans = sorted(set(spans))
        out, last, n = [], 0, 0
        for a, b in spans:
            if a < last:
                continue
            old = old_spelling(text[a:b])
            out.append(text[last:a])
            out.append(old)
            last = b
            n += old != text[a:b]
        out.append(text[last:])
        new = "".join(out)
        if new != text:
            changed.append((path, n))
            if apply:
                with open(os.path.join(root, path), "w", encoding="utf-8", newline="") as f:
                    f.write(new)
    return changed


def _load(args: argparse.Namespace, which: str) -> dict:
    ref, root, js = getattr(args, which + "_ref", None), getattr(args, which + "_root", None), getattr(args, which + "_json", None)
    if js:
        with open(js) as f:
            return json.load(f)
    if ref:
        return scan(_iter_ref(args.repo, ref))
    return scan(_iter_root(root))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--repo", default=".")
    s.add_argument("--ref")
    s.add_argument("--root")
    s.add_argument("--out")
    r = sub.add_parser("restore")
    r.add_argument("--root", required=True)
    r.add_argument("--table", required=True)
    r.add_argument("--apply", action="store_true")
    r.add_argument("--skip", action="append", default=[], help="exact path (renamed spelling) left alone, e.g. files another work package owns")
    k = sub.add_parser("keepfile")
    k.add_argument("--scan-json", action="append", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--must-keep", help="english_audit must_keep.txt to extend with the metric names (section between markers)")
    for name in ("compare", "table"):
        c = sub.add_parser(name)
        c.add_argument("--repo", default=".")
        for w in ("old", "new"):
            c.add_argument(f"--{w}-ref")
            c.add_argument(f"--{w}-root")
            c.add_argument(f"--{w}-json")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        res = scan(_iter_ref(a.repo, a.ref) if a.ref else _iter_root(a.root or a.repo))
        txt = json.dumps(res, indent=1, sort_keys=True)
        if a.out:
            with open(a.out, "w") as f:
                f.write(txt + "\n")
        else:
            print(txt)
        return 0
    if a.cmd == "restore":
        ch = restore(a.root, a.table, a.apply, a.skip)
        for p_, n in ch:
            print(f"{n:5d}  {p_}")
        print(f"{'APPLIED' if a.apply else 'DRY'}: {len(ch)} files, {sum(n for _, n in ch)} tokens")
        return 0
    if a.cmd == "keepfile":
        scans = [json.load(open(x)) for x in a.scan_json]
        table = keepfile(scans)
        with open(a.out, "w") as f:
            f.write(json.dumps(table, indent=1) + "\n")
        if a.must_keep:
            write_must_keep(a.must_keep, table)
        return 0
    old, new = _load(a, "old"), _load(a, "new")
    if a.cmd == "table":
        print("old name\tkit spelling\told count\tnew count (old spelling)")
        for r in table(old, new):
            print("\t".join(str(x) for x in r))
        return 0
    fails = compare(old, new)
    for f in fails:
        print("FAIL", f)
    print("VERDICT:", "PASS" if not fails else "FAIL", f"({len(old['hits'])} old hits, {len(new['hits'])} new hits)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
