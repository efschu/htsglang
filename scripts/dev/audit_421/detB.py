"""DETECTOR B -- "always-default parameter".

A feature parameter exists on a function / method / dataclass constructor, but
no PRODUCTION call site ever supplies a real value for it: every call either
omits it or passes literal None (or the declared default). The feature is then
inert by construction, no matter how well the parameter itself is tested.

Usage:
    python3 /tmp/a421/detB.py <tree_root> [--paths p1,p2] [--all] [--json out]

  <tree_root>   an exported tree, e.g. /tmp/a421/tree_HEAD
  --paths       comma-separated relative-path prefixes to restrict DEFINITIONS
                to (call sites are always searched over the whole tree)
  --all         also report parameters whose name is not "feature-ish"
  --json        dump the raw rows

Definition scope by default: fork-added (not present in upstream/main),
non-test .py files under python/.

Classification of a (callable, parameter):
    DEAD       -- the callable has no call site at all
    INERT      -- callable is called, but no call site anywhere passes a real
                  value for the parameter
    TEST-ONLY  -- only test/bench call sites pass a real value
    WIRED      -- at least one production call site passes a real value
Anything with `**kwargs` / `*args` forwarding at a call site is counted as
UNKNOWN for that site and reported, because static resolution stops there.
"""

import argparse
import ast
import json
import re
import sys

sys.path.insert(0, "/tmp/a421")
from astlib import Index, call_name, is_none_literal, is_fork_file, is_test_path  # noqa: E402

# Parameter names that carry a feature rather than plumbing.
FEATURE_RE = re.compile(
    r"(ratio|weight|policy|strateg|profile|mode|plan|context|shard|tier|"
    r"budget|override|hook|schedule|placement|layout|split|apportion|"
    r"assignment|mapping|selector|predicate|callback|provider|backend|"
    r"quota|limit|priority|affinit|topolog|link|cost)",
    re.I,
)
# Plumbing names that are default-None everywhere by convention: pure noise.
STOP_PARAMS = {
    "prefix",
    "name",
    "device",
    "dtype",
    "out",
    "output",
    "logger",
    "seed",
    "stream",
    "tp_rank",
    "rank",
    "world_size",
    "config",
    "kwargs",
    "args",
    "self",
    "cls",
    "quant_config",
    "prefix_",
    "tag",
    "msg",
    "reason",
}


def _param_list(fn):
    """[(name, index, default_node_or_MISSING)] in positional order."""
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args)
    ndef = len(a.defaults)
    out = []
    for i, arg in enumerate(pos):
        d = None
        has = False
        j = i - (len(pos) - ndef)
        if j >= 0:
            d = a.defaults[j]
            has = True
        out.append((arg.arg, i, d if has else "MISSING", "pos"))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append((arg.arg, None, d if d is not None else "MISSING", "kwonly"))
    return out


def _dataclass_fields(cls):
    """[(name, index, default_or_MISSING)] for a @dataclass body."""
    out = []
    i = 0
    for st in cls.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            out.append(
                (
                    st.target.id,
                    i,
                    st.value if st.value is not None else "MISSING",
                    "pos",
                )
            )
            i += 1
    return out


def _is_dataclass(cls):
    for d in cls.decorator_list:
        n = (
            d.func.attr
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            else d.func.id
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
            else d.attr
            if isinstance(d, ast.Attribute)
            else d.id
            if isinstance(d, ast.Name)
            else None
        )
        if n == "dataclass":
            return True
    return False


def collect_definitions(idx, path_prefixes, defs_scope_fork_only=True):
    """-> list of dicts: callable_name, kind, params, rel, lineno, self_offset."""
    defs = []
    for rel, tree in idx.trees.items():
        if path_prefixes and not any(rel.startswith(p) for p in path_prefixes):
            continue
        if is_test_path(rel):
            continue
        if defs_scope_fork_only and not is_fork_file(rel):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                fields = _dataclass_fields(node)
                if fields:
                    seam = set()
                    for m in node.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            seam |= _seam_params(m)
                    defs.append(
                        dict(
                            callable=node.name,
                            kind="dataclass",
                            rel=rel,
                            lineno=node.lineno,
                            params=fields,
                            self_offset=0,
                            seam=seam,
                        )
                    )
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls = idx.enclosing_class(node)
                params = _param_list(node)
                off = 0
                if params and params[0][0] in ("self", "cls"):
                    off = 1
                if node.name == "__init__" and cls is not None:
                    if _is_dataclass(cls):
                        continue
                    defs.append(
                        dict(
                            callable=cls.name,
                            kind="ctor",
                            rel=rel,
                            lineno=node.lineno,
                            params=params,
                            self_offset=off,
                            seam=_seam_params(node),
                        )
                    )
                else:
                    defs.append(
                        dict(
                            callable=node.name,
                            kind="method" if cls is not None else "func",
                            rel=rel,
                            lineno=node.lineno,
                            params=params,
                            self_offset=off,
                            seam=_seam_params(node),
                        )
                    )
    return defs


def string_literal_keys(idx):
    """Every string constant in PROD code.

    A dataclass field can be set without ever appearing as a keyword in a Call:
    `dataclasses.replace(cfg, **changes)` with `changes["tier_ratio"] = ...`,
    `Class(**parsed_yaml)`, `setattr(obj, key, v)`. Those are invisible to a
    name-based call-site scan, so any parameter whose name also occurs as a
    string literal in production code is downgraded to UNVERIFIABLE rather than
    reported. This is the #381 lesson wired into the detector: it was the
    barlink CollectiveConfig false positive.
    """
    out = {}
    for rel, tree in idx.trees.items():
        if is_test_path(rel):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if n.value.isidentifier():
                    out.setdefault(n.value, []).append(f"{rel}:{n.lineno}")
    return out


def build_callsite_index(idx):
    sites = {}
    for rel, tree in idx.trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                n = call_name(node)
                if n:
                    sites.setdefault(n, []).append((rel, node))
    return sites


def classify_site(node, param, pos_in_call):
    """-> 'real' | 'none' | 'omit' | 'unknown'"""
    for kw in node.keywords:
        if kw.arg == param:
            return "none" if is_none_literal(kw.value) else "real"
        if kw.arg is None:  # **kwargs forwarding
            return "unknown"
    if any(isinstance(a, ast.Starred) for a in node.args):
        return "unknown"
    if pos_in_call is not None and pos_in_call >= 0 and len(node.args) > pos_in_call:
        a = node.args[pos_in_call]
        return "none" if is_none_literal(a) else "real"
    return "omit"


def arg_expr_for(node, param, pos_in_call):
    for kw in node.keywords:
        if kw.arg == param:
            return kw.value
    if pos_in_call is not None and pos_in_call >= 0 and len(node.args) > pos_in_call:
        if not any(isinstance(a, ast.Starred) for a in node.args):
            return node.args[pos_in_call]
    return None


SEAM_HINT = re.compile(r"\bis None\b|\bis not None\b|\bnot \w+\b")


def is_injection_seam(idx, defnode_rel, defnode_line, param, idx_trees=None):
    """True when the body explicitly substitutes a real default for the param.

    `def f(x, lookup=None): lookup = real_impl if lookup is None else lookup`
    is a TEST SEAM, not an unwired feature: leaving it at the default IS the
    production behaviour. Suppressing these is what keeps Detector B's
    precision usable -- the fork is full of `link_gbps=None` / `identity_map=
    None` injection points.
    """
    return None  # replaced below by the body-aware implementation


def _seam_params(fn):
    """Parameters the body explicitly SUBSTITUTES a real default for.

    A seam is a substitution, not a guard:
        seam   : `lookup = real_impl if lookup is None else lookup`
                 `if lookup is None: lookup = real_impl`
                 `lookup = lookup or real_impl`
        NOT    : `if cold_shard is None: return default_plan`   <- feature guard
    Only the substitution form means "leaving it at the default IS production
    behaviour". A feature guard means the opposite, and must stay reportable.
    """
    out = set()

    def none_test_names(test, want_is=True):
        """Names compared `x is None` (want_is) or `x is not None`."""
        got = set()
        for n in ast.walk(test):
            if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name):
                for op, cmp in zip(n.ops, n.comparators):
                    if not is_none_literal(cmp):
                        continue
                    if want_is and isinstance(op, ast.Is):
                        got.add(n.left.id)
                    if (not want_is) and isinstance(op, ast.IsNot):
                        got.add(n.left.id)
        return got

    for n in ast.walk(fn):
        # if <p> is None: <assign>          -> substitution
        # if <p> is not None: ... else: <assign>  -> substitution in the else
        # `if <p> is not None: <assign>` is the FEATURE branch, never a seam.
        if isinstance(n, ast.If):
            if any(isinstance(b, (ast.Assign, ast.AnnAssign)) for b in n.body):
                out |= none_test_names(n.test, want_is=True)
            if any(isinstance(b, (ast.Assign, ast.AnnAssign)) for b in n.orelse):
                out |= none_test_names(n.test, want_is=False)
        # <target> = <expr> if <p> is None else <p>
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.IfExp):
            out |= none_test_names(n.value.test)
        # <target> = <p> or <real expr>
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.BoolOp)
            and isinstance(n.value.op, ast.Or)
        ):
            for v in n.value.values:
                if isinstance(v, ast.Name):
                    out.add(v.id)
        # f(x=<p> or <real expr>) / f(x=<real> if <p> is None else <p>)
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if isinstance(kw.value, ast.BoolOp) and isinstance(kw.value.op, ast.Or):
                    for v in kw.value.values:
                        if isinstance(v, ast.Name):
                            out.add(v.id)
                if isinstance(kw.value, ast.IfExp):
                    out |= none_test_names(kw.value.test)
    return out


def enclosing_callable_key(idx, node):
    """(callable_name, {param names}) of the function that contains `node`."""
    fn = idx.enclosing_func(node)
    if fn is None:
        return None, set()
    cls = idx.enclosing_class(fn)
    name = cls.name if (fn.name == "__init__" and cls is not None) else fn.name
    params = {p[0] for p in _param_list(fn)}
    return name, params


def run(root, path_prefixes, only_feature=True, fork_only=True):
    idx = Index(root)
    defs = collect_definitions(idx, path_prefixes, fork_only)
    sites = build_callsite_index(idx)
    strkeys = string_literal_keys(idx)

    # ---- pass 1: per-candidate raw site classification, remembering the
    #      FORWARDING provenance of every "real" site (arg is a bare Name that
    #      is itself a parameter of the enclosing callable).
    cands = []
    for d in defs:
        calls = sites.get(d["callable"], [])
        for pname, pidx, default, _k in d["params"]:
            if pname in STOP_PARAMS or pname.startswith("_"):
                continue
            default_is_none = isinstance(default, ast.AST) and is_none_literal(default)
            featureish = bool(FEATURE_RE.search(pname))
            if not (default_is_none or (featureish and default != "MISSING")):
                continue
            if only_feature and not featureish and not default_is_none:
                continue
            pos_in_call = None if pidx is None else pidx - d["self_offset"]
            sitelist = []
            for rel, node in calls:
                k = classify_site(node, pname, pos_in_call)
                fwd = None
                if k == "real":
                    e = arg_expr_for(node, pname, pos_in_call)
                    if isinstance(e, ast.Name):
                        encl, eparams = enclosing_callable_key(idx, node)
                        if encl is not None and e.id in eparams:
                            fwd = (encl, e.id)
                sitelist.append(
                    dict(
                        rel=rel,
                        line=node.lineno,
                        kind=k,
                        test=is_test_path(rel),
                        fwd=fwd,
                    )
                )
            cands.append(
                dict(
                    callable=d["callable"],
                    param=pname,
                    defkind=d["kind"],
                    site=f"{d['rel']}:{d['lineno']}",
                    calls=len(calls),
                    feature=featureish,
                    default_none=default_is_none,
                    seam=(pname in d.get("seam", set())),
                    strkey=strkeys.get(pname, [])[:3],
                    sites=sitelist,
                )
            )

    by_key = {}
    for c in cands:
        by_key.setdefault((c["callable"], c["param"]), []).append(c)

    # ---- pass 2: fixed point over forwarding taint.
    # A "real" site whose argument is forwarded from an INERT/TEST-ONLY
    # parameter of the caller does not prove the feature is wired; it only
    # moves the question one frame up.
    def classify(c):
        real_prod = [
            s
            for s in c["sites"]
            if s["kind"] == "real" and not s["test"] and not s.get("tainted")
        ]
        real_test = [
            s
            for s in c["sites"]
            if s["kind"] == "real" and (s["test"] or s.get("tainted_test"))
        ]
        if c["calls"] == 0:
            return "DEAD", real_prod, real_test
        if real_prod:
            return "WIRED", real_prod, real_test
        if real_test:
            return "TEST-ONLY", real_prod, real_test
        return "INERT", real_prod, real_test

    cls_map = {}
    for _ in range(10):
        changed = False
        for c in cands:
            cl, rp, rt = classify(c)
            k = (c["callable"], c["param"])
            if cls_map.get(id(c)) != cl:
                cls_map[id(c)] = cl
                changed = True
        # propagate taint
        keyclass = {}
        for k, lst in by_key.items():
            cs = {cls_map[id(c)] for c in lst}
            keyclass[k] = (
                "WIRED"
                if "WIRED" in cs
                else "TEST-ONLY"
                if "TEST-ONLY" in cs
                else "INERT"
                if "INERT" in cs
                else "DEAD"
            )
        for c in cands:
            for s in c["sites"]:
                if s["kind"] != "real" or not s["fwd"]:
                    continue
                src = keyclass.get(s["fwd"])
                if src in ("INERT", "DEAD"):
                    if not s.get("tainted"):
                        s["tainted"] = True
                        s["taint_src"] = f"{s['fwd'][0]}.{s['fwd'][1]}={src}"
                        changed = True
                elif src == "TEST-ONLY" and not s["test"]:
                    if not s.get("tainted"):
                        s["tainted"] = True
                        s["tainted_test"] = True
                        s["taint_src"] = f"{s['fwd'][0]}.{s['fwd'][1]}=TEST-ONLY"
                        changed = True
        if not changed:
            break

    rows = []
    for c in cands:
        cl, rp, rt = classify(c)
        rows.append(
            dict(
                cls=cl,
                callable=c["callable"],
                param=c["param"],
                defkind=c["defkind"],
                site=c["site"],
                calls=c["calls"],
                feature=c["feature"],
                default_none=c["default_none"],
                seam=c["seam"],
                strkey=c["strkey"],
                real_prod=[f"{s['rel']}:{s['line']}" for s in rp],
                real_test=[f"{s['rel']}:{s['line']}" for s in rt],
                tainted=[
                    f"{s['rel']}:{s['line']} <- {s.get('taint_src')}"
                    for s in c["sites"]
                    if s.get("tainted")
                ],
                none_prod=sum(
                    1 for s in c["sites"] if s["kind"] == "none" and not s["test"]
                ),
                omit_prod=sum(
                    1 for s in c["sites"] if s["kind"] == "omit" and not s["test"]
                ),
                unknown_prod=[
                    f"{s['rel']}:{s['line']}"
                    for s in c["sites"]
                    if s["kind"] == "unknown" and not s["test"]
                ],
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--paths", default="python/sglang/srt")
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--any-file",
        action="store_true",
        help="do not restrict defs to fork-added files",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--show", default="DEAD,INERT,TEST-ONLY")
    ap.add_argument(
        "--with-strkeys",
        action="store_true",
        help="also print params whose name occurs as a string literal in prod "
        "code (settable through a **dict builder; unverifiable statically)",
    )
    ap.add_argument(
        "--with-seams",
        action="store_true",
        help="also print params the body explicitly defaults to a real impl "
        "(deliberate test-injection seams; suppressed by default)",
    )
    a = ap.parse_args()
    prefixes = [p for p in a.paths.split(",") if p]
    rows = run(a.root, prefixes, only_feature=not a.all, fork_only=not a.any_file)
    show = set(a.show.split(","))
    counts, seamcounts, strcounts = {}, {}, {}
    for r in rows:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
        if r["seam"]:
            seamcounts[r["cls"]] = seamcounts.get(r["cls"], 0) + 1
        if r["strkey"] and not r["seam"]:
            strcounts[r["cls"]] = strcounts.get(r["cls"], 0) + 1
    print(
        f"# DETECTOR B on {a.root} paths={prefixes} fork_only={not a.any_file} feature_filter={not a.all}"
    )
    print("# raw counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(
        "# of which injection seams (suppressed): "
        + ", ".join(f"{k}={v}" for k, v in sorted(seamcounts.items()))
    )
    print(
        "# of which **dict-builder reachable / UNVERIFIABLE (suppressed): "
        + ", ".join(f"{k}={v}" for k, v in sorted(strcounts.items()))
    )
    for cls in ("DEAD", "INERT", "TEST-ONLY", "WIRED"):
        sel = [r for r in rows if r["cls"] == cls and cls in show]
        if not sel:
            continue
        print(f"\n===== {cls} ({len(sel)}) =====")
        for r in sorted(sel, key=lambda x: (x["seam"], x["callable"], x["param"])):
            if r["seam"] and not a.with_seams:
                continue
            if r["strkey"] and not a.with_strkeys:
                continue
            print(f"  {r['callable']}.{r['param']:24s} [{r['defkind']}] {r['site']}")
            print(
                f"      calls={r['calls']} real_prod={len(r['real_prod'])} real_test={len(r['real_test'])} "
                f"none_prod={r['none_prod']} omit_prod={r['omit_prod']} unknown_prod={len(r['unknown_prod'])}"
            )
            if r["real_test"][:3]:
                print(f"      test-real: {r['real_test'][:3]}")
            if r["unknown_prod"][:3]:
                print(f"      UNKNOWN-prod(**kwargs/*args): {r['unknown_prod'][:3]}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
