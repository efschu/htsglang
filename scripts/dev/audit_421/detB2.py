"""DETECTOR B2 -- "degenerate automatic derivation".

Companion to Detector B. Detector B answers "does anybody ever supply the
feature parameter?". B2 answers the other half: when the code DERIVES the value
automatically, is that derivation capable of producing a non-trivial value on a
real machine, or is it constant by construction?

The failure shape it is built for (#394 pre-fix):

    derive_link_weights()  -> one weight per rank
      <- _pcie_link_gbps_by_uuid()
           <- nvmlDeviceGetMaxPcieLinkGeneration / nvmlDeviceGetMaxPcieLinkWidth

Those two NVML calls return the card's NAMEPLATE maximum (gen4 x16 for all
three cards on the reference rig), not the link the card is actually trained
at. Every rank therefore gets the same number, the vector normalizes to
(1/n, ..., 1/n), and the `is_equal` short-circuit downstream turns the whole
feature off. The derivation runs, succeeds, and yields nothing.

Rule:
  a "vector producer" whose only value sources are CAPABILITY / NAMEPLATE
  queries or module-level literal tables, with no MEASUREMENT source anywhere
  in its transitive body, AND whose result feeds an all-equal-means-no-op
  short-circuit,
  ==> DEGENERATE-RISK: constant on any rig whose devices share a nameplate.

Static analysis cannot decide "constant on THIS rig" -- that needs the actual
NVML values. What it can decide is "every input is a per-device constant that
identical devices share", which is the precondition. Confirming it is a
read-the-code / run-nvidia-smi step, and is reported as such.

Usage:  python3 /tmp/a421/detB2.py <tree_root> [--paths p1,p2]
"""

import argparse
import ast
import re
import sys

sys.path.insert(0, "/tmp/a421")
from astlib import Index, is_fork_file, is_test_path  # noqa: E402

PRODUCER_RE = re.compile(
    r"(derive|resolve|compute|_?plan)_.*(weight|ratio|share|split|cost|rate)"
    r"|.*_(weights|ratios|shares|rates)$",
    re.I,
)

# Value sources that are the same for two identical devices.
NAMEPLATE_RE = re.compile(
    r"GetMax[A-Z]|nvmlDeviceGetMax|MaxPcieLink|max_pcie|nameplate|"
    r"total_memory|GetMemoryInfo|\.total\b|Supported|Capability|"
    r"get_device_capability|device_count|multi_processor_count|"
    r"LinkWidth|LinkGeneration|_LANE_GBPS|_GBPS\b|NOMINAL|SPEC_",
    re.I,
)
# Value sources that come from timing something on this machine.
MEASURE_RE = re.compile(
    r"probe|measur|elapsed|perf_counter|monotonic|time\.time|Event\(|"
    r"synchronize|benchmark|timed|observed|sampled|h2d_gb|gbs\b|"
    r"CurrPcieLink|GetCurr|nvmlDeviceGetPcieThroughput|utilization",
    re.I,
)
# The short-circuit that makes a constant vector a no-op.
EQUAL_SHORTCIRCUIT_RE = re.compile(
    r"is_equal|all_equal|len\(set\(|== *1/|is_uniform|is_trivial|"
    r"all\(.*==|math\.isclose",
    re.I,
)


def func_defs(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, []).append(n)
    return out


def names_in(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


class _StripDocstrings(ast.NodeTransformer):
    """Prose is not a data source. Docstrings and comments must not decide the
    verdict -- #394's pre-fix resolve_host_shard_ratio says "measured" in its
    docstring while every actual source in the chain is a nameplate."""

    def _strip(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


def code_only(fn):
    """Source of `fn` with docstrings and comments removed."""
    import copy

    f = _StripDocstrings().visit(copy.deepcopy(fn))
    ast.fix_missing_locations(f)
    try:
        return ast.unparse(f)
    except Exception:
        return ""


def transitive_body(src, tree, fn, defs, depth=3):
    """Concatenated CODE of fn plus module-local callees, up to `depth`."""
    seen, frontier, chunks = set(), [fn], []
    for _ in range(depth):
        nxt = []
        for f in frontier:
            if id(f) in seen:
                continue
            seen.add(id(f))
            chunks.append(code_only(f))
            for callee in names_in(f):
                for g in defs.get(callee, []):
                    if id(g) not in seen:
                        nxt.append(g)
        frontier = nxt
    return "\n".join(chunks)


def run(root, prefixes, fork_only=True):
    idx = Index(root)
    hits = []
    for rel, tree in idx.trees.items():
        if prefixes and not any(rel.startswith(p) for p in prefixes):
            continue
        if is_test_path(rel) or (fork_only and not is_fork_file(rel)):
            continue
        src = idx.files[rel]
        defs = func_defs(tree)
        module_has_shortcircuit = bool(EQUAL_SHORTCIRCUIT_RE.search(src))
        _ = src
        for name, fns in defs.items():
            if not PRODUCER_RE.search(name):
                continue
            for fn in fns:
                body = transitive_body(src, tree, fn, defs)
                nameplate = sorted(set(m.group(0) for m in NAMEPLATE_RE.finditer(body)))
                measured = sorted(set(m.group(0) for m in MEASURE_RE.finditer(body)))
                if not nameplate:
                    continue
                verdict = "DEGENERATE-RISK" if not measured else "has-measured-source"
                hits.append(
                    dict(
                        rel=rel,
                        line=fn.lineno,
                        fn=name,
                        verdict=verdict,
                        nameplate=nameplate[:8],
                        measured=measured[:8],
                        shortcircuit=module_has_shortcircuit,
                    )
                )
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--paths", default="python/sglang/srt")
    ap.add_argument("--any-file", action="store_true")
    a = ap.parse_args()
    hits = run(a.root, [p for p in a.paths.split(",") if p], not a.any_file)
    print(f"# DETECTOR B2 on {a.root} paths={a.paths}")
    for v in ("DEGENERATE-RISK", "has-measured-source"):
        sel = [h for h in hits if h["verdict"] == v]
        print(f"\n===== {v} ({len(sel)}) =====")
        for h in sel:
            flag = " +equal-shortcircuit" if h["shortcircuit"] else ""
            print(f"  {h['fn']}  {h['rel']}:{h['line']}{flag}")
            print(f"      nameplate-only sources: {h['nameplate']}")
            if h["measured"]:
                print(f"      measured sources:       {h['measured']}")


if __name__ == "__main__":
    main()
