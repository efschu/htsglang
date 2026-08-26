#!/usr/bin/env python3
"""#864 -- test doubles that can no longer accept the call they stand in for.

THE CLASS, not the instance. A test replaces a production callable with a local
stub (``mock.patch.object(mod, "f", side_effect=fake)``). Later the production
CALL SITE grows an argument. The stub does not. From that moment the test can
only fail -- but it fails with a ``TypeError`` from inside ``mock``'s dispatch,
before any assertion of its own runs, so the red says nothing about the
behaviour the test was written to protect.

WHY IT SURVIVES. The drift is invisible everywhere the test does not actually
execute. The 16 standing failures that motivated this tool live in a file whose
device-bound tests SKIP on any desk machine (`mem_cache/conftest.py` converts a
missing accelerator into `unittest.SkipTest`), so the only machine that can see
them is one with cards -- and the desk gate, which is where a red would be
noticed, is exactly the machine that cannot.

THE CHECK, and the point of writing it as a tool rather than a fix. Signature
compatibility is decidable WITHOUT running the test and WITHOUT a device: parse
the stub, parse the production call site, and ask whether the stub's signature
can bind the call's arguments. A GPU-only red becomes a CPU-detectable one.

Usage:
    gate_double_drift.py --check <module.py>:<func> --call k1,k2,k3
    gate_double_drift.py --sweep <test dir>      # every patched double found
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sig_from_ast(fn: ast.FunctionDef) -> inspect.Signature:
    """An inspect.Signature for a function we parsed but never imported.

    Parsing rather than importing is the whole point: importing the test module
    would need the device the test is skipped for.
    """
    P = inspect.Parameter
    params: list[inspect.Parameter] = []
    a = fn.args
    for arg in a.posonlyargs:
        params.append(P(arg.arg, P.POSITIONAL_ONLY))
    ndef = len(a.defaults)
    npos = len(a.args)
    for i, arg in enumerate(a.args):
        default = P.empty if i < npos - ndef else None
        params.append(P(arg.arg, P.POSITIONAL_OR_KEYWORD, default=default))
    if a.vararg:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append(P(arg.arg, P.KEYWORD_ONLY,
                        default=P.empty if d is None else None))
    if a.kwarg:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))
    return inspect.Signature(params)


def find_func(path: Path, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(path.read_text(errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def can_bind(sig: inspect.Signature, npos: int, kwargs: list[str]) -> tuple[bool, str]:
    try:
        sig.bind(*([object()] * npos), **{k: object() for k in kwargs})
        return True, ""
    except TypeError as e:
        return False, str(e)


# ---- the sweep ------------------------------------------------------------
PATCH = re.compile(
    r"mock\.patch(?:\.object)?\(\s*([A-Za-z_][\w.]*)\s*,\s*[\"'](\w+)[\"']"
    r"(?:.*?(?:side_effect|new)\s*=\s*(\w+))?",
    re.S,
)


def sweep(testdir: Path) -> int:
    """Report every patched double alongside the stub it installs.

    Deliberately a REPORT and not a verdict: deciding automatically which
    production call site a given double stands in for needs import-time
    resolution this tool refuses to do. What it removes is the "nobody knew
    the double existed" failure -- the list is finite and reviewable.
    """
    rows = []
    for p in sorted(testdir.rglob("test_*.py")):
        try:
            src = p.read_text(errors="replace")
        except Exception:
            continue
        for m in PATCH.finditer(src):
            target, attr, stub = m.group(1), m.group(2), m.group(3)
            if not stub:
                continue
            fn = find_func(p, stub)
            if fn is None:
                continue
            sig = _sig_from_ast(fn)
            rows.append((p.relative_to(ROOT).as_posix(), f"{target}.{attr}", stub, str(sig)))
    print(f"# patched doubles with a named stub under {testdir.relative_to(ROOT)}")
    print(f"# {len(rows)} found\n")
    for f, tgt, stub, sig in rows:
        print(f"  {tgt}\n    stub {stub}{sig}\n    in   {f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="<file.py>:<stub function name>")
    ap.add_argument("--call", default="",
                    help="comma-separated keyword names the production call passes")
    ap.add_argument("--pos", type=int, default=1,
                    help="positional args the production call passes")
    ap.add_argument("--sweep", help="test directory to enumerate doubles in")
    args = ap.parse_args()

    if args.sweep:
        return sweep(Path(args.sweep) if Path(args.sweep).is_absolute()
                     else ROOT / args.sweep)

    if not args.check:
        ap.error("one of --check or --sweep is required")
    fpath, _, fname = args.check.rpartition(":")
    path = Path(fpath) if Path(fpath).is_absolute() else ROOT / fpath
    fn = find_func(path, fname)
    if fn is None:
        print(f"stub {fname!r} not found in {path}", file=sys.stderr)
        return 2
    sig = _sig_from_ast(fn)
    kwargs = [k for k in args.call.split(",") if k]
    ok, err = can_bind(sig, args.pos, kwargs)
    print("# double-drift check")
    print(f"  stub          {fname}{sig}")
    print(f"  in            {path.relative_to(ROOT)}")
    print(f"  production call: {args.pos} positional + {kwargs}")
    if ok:
        print("  VERDICT: the double can accept the production call.")
        return 0
    print("  VERDICT: DRIFTED -- the double CANNOT accept the production call.")
    print(f"           {err}")
    print("           Every test using this double fails on the first call, from")
    print("           inside mock's dispatch, before its own assertions run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
