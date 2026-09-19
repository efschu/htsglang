"""fn4m 19.09.: NameError 'get_spec' in EAGLEWorkerV2._configure_qsa_mtp_index_share
killed the NEXTN boot after the pools were sized. Every module-level name the
worker's methods reference must resolve at import time."""

import ast
import builtins
import pathlib


def test_eagle_worker_v2_has_no_undefined_module_names():
    path = pathlib.Path(__file__).resolve().parents[2] / "python/sglang/srt/speculative/eagle_worker_v2.py"
    tree = ast.parse(path.read_text())
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            for arg in getattr(node, "args", None).args if hasattr(node, "args") else []:
                defined.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            pass
    # names that are CALLED at module level of a method body: get_spec was one
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    missing = sorted(c for c in called if c not in defined)
    assert missing == [], missing
    assert "get_spec" in defined
