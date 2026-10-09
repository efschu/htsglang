#!/usr/bin/env python3
"""rest_inventory_1008.py -- F0-I: classify EVERY remaining old name of a renamed tree (read-only).

The kit's ``rename_to_flliper.py inventory`` counts per category; this tool lists each hit (file:line) and gives it exactly one
verdict, so that "every remaining old name is justified" is a statement about hits, not about categories.

Old names searched (the three families of RENAME_PLAN 8.1):
  sglang   : sglang, SGLANG, SGLang, Sglang ... (not ``htsglang``)   -- the kit's MAIN regex
  weg2     : weg2, WEG2, Weg2, Weg2Flip                              -- the kit's W_MAIN regex
  htsglang : htsglang, HTSGLANG_ (any case)                          -- the product/container layer

Verdicts (``kind``):
  keep     the hit is on the R2 must-keep list (RENAME_PLAN 2, F0-M), with the rule that clears it
  finding  the hit is understood and classified but is NOT on the R2 list: a decision or a later phase (prose pass, product layer)
  residue  the hit matches no rule: an unexplained hit (the target is 0)

Usage:  python3 tools/release/rest_inventory_1008.py --root <worktree> --line 27b|nf [--json out.json] [--md out.md] [--list-residue N]
The rules are the ``RULES`` table below, evaluated in order, first match wins; they use the kit's own deny spans, scope tables and
metric-name table (``rename_to_flliper``), so the verdicts follow the same definitions as the rename pass.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rename_to_flliper as R  # noqa: E402

FAMILIES = (("sglang", R.MAIN), ("weg2", R.W_MAIN), ("htsglang", re.compile(r"(?i)htsglang")))
KERNEL = re.compile(r"\bsgl[-_]kernel[\w-]*|\bsglang-kernel\b")
NEW_WORD = {
    "sglang": re.compile(r"flliper|FLLIPER|Flliper|fLLiper"),
    "weg2": re.compile(r"pdflip|PDFLIP|PdFlip|Pdflip"),
    "htsglang": re.compile(r"flliper|FLLIPER|fLLiper"),
}
# R2 whole trees (RENAME_PLAN 2): the kit's EXCLUDE list
R2_TREES = ("3rdparty/", "sgl-kernel/", "sgl-model-gateway/", "experimental/", "rust/", "proto/", ".github/", "docs/", "docs_new/",
            ".claude/", ".deps/")


def _tok(line: str, col: int, length: int) -> str:
    """the maximal [\\w./:@-]* token around the hit"""
    a = col
    while a > 0 and re.match(r"[\w./:@=-]", line[a - 1]):
        a -= 1
    b = col + length
    while b < len(line) and re.match(r"[\w./:@-]", line[b]):
        b += 1
    return line[a:b]


class Hit:
    __slots__ = ("path", "line_no", "col", "fam", "word", "line", "tok", "after", "before")

    def __init__(self, path, line_no, col, fam, word, line):
        self.path, self.line_no, self.col, self.fam, self.word, self.line = path, line_no, col, fam, word, line
        self.tok = _tok(line, col, len(word))
        self.after = line[col + len(word): col + len(word) + 40]
        self.before = line[max(0, col - 40): col]


def rx(p: str):
    return re.compile(p)


# ---- rules on the token / line (ordered; first match wins) -----------------------------------------------------
# (id, kind, family or None, test, why).  ``test`` is a regex applied to the token (``tok``) or a callable(hit).
# The kit's own spans (deny_intervals for sglang, W_DENY_SPAN for weg2) are applied BEFORE these rules (see _verdict): what the
# engine would not rewrite again is clear by the engine's own definition.  The rules here cover what the engine cannot see.
RULES: List[Tuple[str, str, Optional[str], object, str]] = [
    # --- metric family prefixes spelled with a glob in prose (F0-M): ``weg2_*``, ``sglang:*``, ``sglang_*``, the scrape job -----
    ("metric-family-text", "keep", None, rx(r"^(weg2_\*|sglang[:_]\*|weg2-front)"),
     "F0-M: metric family named in prose / docstrings (weg2_*, sglang:*, sglang_*) and the scrape job weg2-front"),
    # --- persisted identities of the htsglang product layer ------------------------------------------------------------
    ("persisted-id", "keep", "htsglang", rx(r"htsglang-rig-artifact/v\d+"),
     "versioned payload id of the rig artifact (a format id is data, not a name; same rule as weg2-footprint/1)"),
    ("github-repo-name", "keep", "htsglang", rx(r"efschu/htsglang|org-htsglang"),
     "R2: the old GitHub repository name (URL / org id class, RENAME_PLAN 2 'old GitHub repo links')"),
    # --- evidence images / records carried by captured fixtures --------------------------------------------------------------
    ("image-tag-htsglang", "keep", "htsglang", rx(r"htsglang:(cu\d+|rc|x|nope)"),
     "R2: published image tags htsglang:cu130-... (and the placeholder tags tests build from them)"),
]


def _load_collision_ok(root: str, line: str) -> Dict[str, frozenset]:
    out: Dict[str, frozenset] = {}
    d = os.path.join(root, "tools", "release", "data")
    names = [f"collision_ok_1007_{line}.json", f"collision_ok_y8t_{line}.json"]
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                if k.startswith("_"):
                    continue
                for pp in {k, R.rewrite_path(k, True)}:
                    out[pp] = frozenset(out.get(pp, frozenset())) | frozenset(v)
    for k, v in R.COLLISION_OK.items():
        for pp in {k, R.rewrite_path(k, True)}:
            out[pp] = frozenset(out.get(pp, frozenset())) | frozenset(v)
    return out


# files whose old-name hits are the point of the file (decisions of F0-B/F0-C/F0-F/F0-G, documented in RENAME_PLAN 8.14-8.18 and the F0 reports)
DECISION_FILES = {
    "README.md": "F0-I: the release README; its migration table and the product-layer note name the old spellings on purpose",
    "docker/README.htsglang.md": "F0-I: the htsglang container README carries the sleep notice (names the old project) and documents the product layer",
    "docker/flliper/Dockerfile.flliper": "F0-G: the recipe bakes the five variables the DOUBLE-READING entrypoint reads under the legacy spelling (header comment, "
                                         "lines 10-14) and names the old layout in comments; /opt|/var/lib/htsglang paths are the product layer (phase 2b)",
    "docker/flliper/make_flat_ctx.sh": "F0-G: the context builder must also recognise a PRE-rename tree (RENAMED=0 branch, B1/B6 gates name SGLANG_*/--weg2-*)",
    "docker/flliper/host_publish_flliper.sh": "F0-G: the publish gate refuses tag parts that carry an old name (probe|sglang|weg2|htsglang)",
    "docker/flliper/host_publish_flliper_selftest.sh": "F0-G: selftest of the publish gate builds a pre-rename tree on purpose",
    "docker/flliper/drycmp/dry_force.sh": "F0-G: the dry-run comparison runs on a pre-rename AND a renamed tree (maps old census paths)",
    "test/registered/unit/docker/test_flliper_container_f0g_1008.py": "F0-G test: old spelling is the input of the profile-conversion / old-vs-new checks",
    "test/registered/unit/pdflip/test_metric_names_must_keep_1008.py": "F0-M test: the old metric spellings are what the test pins",
    "tools/rig_dashboard/rigdash/tests/test_f0f_dashboard_1008.py": "F0-F test: builds old-spelling inputs from split tokens on purpose",
}
# captured evidence carried in the tree as data (boot records / dry-run protocols / dashboard state fixtures of boots run on the old names)
EVIDENCE_PATH = re.compile(r"^(docker/flliper/drycmp/protocols/|tools/rig_dashboard/rigdash/tests/fixtures/|test/registered/unit/[\w/]*fixtures/|scripts/fixtures/)")
# product layer (phase 2b of RENAME_PLAN 5: image name, units, x-htsglang headers, /opt|/var/lib|/etc/htsglang, volume names): NOT renamed in F0
PRODUCT_BUCKETS = [
    ("htsglang-api-namespace", rx(r"x-htsglang|X-Htsglang"), "HTTP namespace of the product API (x-htsglang headers, /x-htsglang/workbench routes, x-htsglang.* fields): clients depend on it"),
    ("htsglang-units-config", rx(r"htsglang(-serving@?|-planner|-watchdog@?|-preflight|\.target)|/(etc|var/log|run|var/tmp)/htsglang|\.config/htsglang|XDG_CACHE_HOME/htsglang|\.cache/htsglang"),
     "systemd units / stack config / state dirs of the turnkey deployment"),
    ("htsglang-container", rx(r"/(opt|var/lib)/htsglang|htsglang(-pip|-apt|-acc-|-build|-gui|-node\d|-entrypoint|-chat_template|-constraints|-qwen|-rig-|-wheels|\.yml|\.env|\.Dockerfile|\.htsglang)|"
                              r"docker/(README\.)?htsglang|CONTAINER_NAME"),
     "container layer: compose files, Dockerfiles, image-internal paths, volume / container names"),
]


def classify_file(path: str, text: str, ctx: dict) -> List[Tuple[Hit, str, str, str]]:
    """-> [(hit, kind, rule id, why)] for every old-name hit in ``text``"""
    ext = os.path.splitext(path)[1]
    lines = text.split("\n")
    starts_of_lines = [0]
    for ln in lines[:-1]:
        starts_of_lines.append(starts_of_lines[-1] + len(ln) + 1)
    iv = R.deny_intervals(text, path)                                  # sglang family: attribution, URLs, foreign packages, metrics
    istarts = [a for a, _, _ in iv]
    wiv = R._intervals(text, R.W_DENY_SPAN, R.metric_keep_spans(text, path))   # weg2 family: the engine's own keep spans
    wstarts = [a for a, _, _ in wiv]
    kern = [(m.start(), m.end()) for m in KERNEL.finditer(text)]
    tree = next((t for t in R2_TREES if path.startswith(t)), None)
    in_path = R.in_path_scope(path)
    excl_content = R._glob(path, R.EXCLUDE_CONTENT)
    outside = (not in_path) and (not excl_content) and tree is None
    coll_words = ctx["collision_ok"].get(path)
    sp = dict(ext=ext, iv=iv, istarts=istarts, wiv=wiv, wstarts=wstarts, kern=kern, tree=tree, outside=outside, excl_content=excl_content,
              coll_words=coll_words)
    out: List[Tuple[Hit, str, str, str]] = []
    for fam, rxm in FAMILIES:
        for m in rxm.finditer(text):
            s, e = m.span()
            ln = text.count("\n", 0, s) + 1
            line = lines[ln - 1]
            h = Hit(path, ln, s - starts_of_lines[ln - 1], fam, m.group(0), line)
            out.append((h,) + _verdict(h, s, e, sp))
    return out


def _verdict(h: Hit, s: int, e: int, sp: dict) -> Tuple[str, str, str]:
    if sp["tree"]:
        return ("keep", "tree:" + sp["tree"], "R2 whole tree (upstream / foreign / docs, RENAME_PLAN 2)")
    d = None
    if h.fam == "sglang":
        d = R._denied(sp["iv"], sp["istarts"], s, e)
    elif h.fam == "weg2":
        d = R._denied(sp["wiv"], sp["wstarts"], s, e)
    if d:
        if d.startswith("metric"):
            return ("keep", "metric-name", "F0-M: metric names are must-keep (user decision 08.10.2026, RENAME_PLAN 8.17)")
        return ("keep", "kit-span:" + d, "the engine's own keep span (attribution / URL / upstream id / foreign package / wire name; weg2: boot tag, "
                                         "evidence prefix, host path, image tag, doc ref, wire magic, persisted id)")
    if any(a - 3 <= s < b for a, b in sp["kern"]):
        return ("keep", "kernel-wheel", "R2 kernel wheel sgl_kernel / sglang-kernel (Phase 2c)")
    if sp["ext"] in R.CXX_EXT:
        return ("keep", "cxx-namespace", "R2 C/C++/CUDA sources (Phase 2e)")
    if h.fam == "htsglang" and re.search(r"/spinning/htsglang", h.line[max(0, h.col - 30): h.col + len(h.word) + 12]):
        return ("keep", "host-path-htsglang", "R2 host path /spinning/htsglang*")
    if (h.fam == "htsglang" and h.word.startswith("HTSGLANG")) or h.line[h.col:h.col + 9] == "HTSGLANG_":
        return ("keep", "HTSGLANG_env", "R2 product env HTSGLANG_*")
    if h.path.endswith(".alt"):
        return ("keep", "env.alt", "F0-G: *.env.alt are the old-spelling copies of the converted profiles (rollback; plan row F0-G)")
    if sp["excl_content"] and "/_vendor/" in "/" + h.path and re.search(r"sglang\.srt|python/sglang", h.line):
        return ("finding", "vendor-stale-module-path", "a vendored (EXCLUDE_CONTENT) file that names OUR module path in a comment / README: stale after the rename, prose only")
    if sp["excl_content"]:
        return ("keep", "exclude-content", "kit EXCLUDE_CONTENT: bytes kept by rule (evidence fixtures, kit data, kartenplan_data, double-reader scripts)")
    if sp["coll_words"] is not None and h.fam != "htsglang":
        return ("keep", "collision_ok-file", "kit COLLISION_OK: the file reads old and new spelling on purpose (double reader)")
    if h.path in DECISION_FILES:
        return ("decision", "decision-file", DECISION_FILES[h.path])
    if sp["outside"]:
        return ("finding", "outside-kit-scope", "not in the kit's INCLUDE scope (root-level notes, .devcontainer ...): prose / separate pass")
    if EVIDENCE_PATH.match(h.path):
        return ("keep", "evidence-data", "captured evidence / fixtures of boots run on the old names (R2 Evidence/Log-Bestand)")
    for rid, kind, fam, test, why in RULES:
        if fam is not None and fam != h.fam:
            continue
        if (test(h) if callable(test) else bool(test.search(h.tok))):
            return (kind, rid, why)
    if h.path in DECISION_FILES:
        return ("decision", "decision-file", DECISION_FILES[h.path])
    if NEW_WORD[h.fam].search(h.line):
        return ("keep", "dual-line", "old and new spelling on the same line: the double reader of F0-B/F0-C/F0-F/F0-G (name_compat, readers, shims)")
    if h.fam == "htsglang":
        ctxt = h.line
        for rid, rxp, why in PRODUCT_BUCKETS:
            if rxp.search(h.tok) or rxp.search(ctxt[max(0, h.col - 12): h.col + 40]):
                return ("finding", rid, why)
        return ("finding", "htsglang-prose", "the product name 'htsglang' in comments, docstrings, log or help text and identifiers of the product layer: prose pass / phase 2b")
    return ("residue", "unexplained", "no rule")


def scan(root: str, line: str) -> List[Tuple[Hit, str, str, str]]:
    ctx = {"collision_ok": _load_collision_ok(root, line)}
    files = [p for p in subprocess.run(["git", "-C", root, "ls-files", "-z", "-c"], capture_output=True, check=True).stdout.split(b"\0") if p]
    res: List[Tuple[Hit, str, str, str]] = []
    for p in files:
        path = p.decode()
        fp = os.path.join(root, path)
        if os.path.islink(fp) or not os.path.isfile(fp):
            continue
        text = R.as_text(open(fp, "rb").read())
        if text is None:
            continue
        res.extend(classify_file(path, text, ctx))
    return res


def summarize(res) -> dict:
    by_kind = collections.Counter()
    by_rule: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    files_by_rule: Dict[str, set] = collections.defaultdict(set)
    for h, kind, rid, why in res:
        by_kind[(kind, h.fam)] += 1
        by_rule[rid][h.fam] += 1
        files_by_rule[rid].add(h.path)
    rules = {}
    why_of = {}
    for h, kind, rid, why in res:
        why_of[rid] = (kind, why)
    for rid, c in sorted(by_rule.items(), key=lambda kv: -sum(kv[1].values())):
        rules[rid] = {"kind": why_of[rid][0], "why": why_of[rid][1], "hits": dict(c), "total": sum(c.values()), "files": len(files_by_rule[rid])}
    return {"total": len(res), "by_kind": {f"{k}:{f}": n for (k, f), n in sorted(by_kind.items())}, "rules": rules}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--line", required=True, choices=["27b", "nf"])
    ap.add_argument("--json", default="")
    ap.add_argument("--hits", default="", help="write every non-keep hit (file, line, family, word, rule, line text) as JSON")
    ap.add_argument("--list-residue", type=int, default=0)
    a = ap.parse_args(argv)
    res = scan(a.root, a.line)
    summ = summarize(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    if a.hits:
        rows = [[h.path, h.line_no, h.fam, h.word, rid, kind, h.line.strip()[:200]] for h, kind, rid, why in res if kind != "keep"]
        with open(a.hits, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
    print(json.dumps({k: summ[k] for k in ("total", "by_kind")}, indent=1))
    n = 0
    for h, kind, rid, why in res:
        if kind == "residue" and n < a.list_residue:
            print(f"RESIDUE {h.path}:{h.line_no} [{h.fam}] {h.tok[:70]} | {h.line.strip()[:110]}")
            n += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
