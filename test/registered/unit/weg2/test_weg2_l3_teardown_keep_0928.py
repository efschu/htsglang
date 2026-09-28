"""L3P-TEARDOWN (27B rc12z7b b1 08:01Z): the launcher teardown removed the
persistent L3 store (6.5 GB, 172795 files) and the next boot found
``identity=new files=0``. A persistent store survives the teardown; only the
pre-L3P per-boot store is removed."""

import ast
import inspect
import os

from sglang.srt.weg2 import launcher as L


def test_an_l3_named_dir_is_persistent(tmp_path):
    d = tmp_path / "l3-qwen27b-Qwen3.8-27B-INT8-a32fcecf07"
    d.mkdir()
    assert L.l3_persistent_store(str(d))
    assert L.l3_persistent_store(str(d) + "/")


def test_an_identity_file_makes_any_dir_persistent(tmp_path):
    d = tmp_path / "somename"
    d.mkdir()
    assert not L.l3_persistent_store(str(d))
    (d / L.L3_IDENTITY_FILE).write_text("{}")
    assert L.l3_persistent_store(str(d))


def test_a_per_boot_store_is_not_persistent(tmp_path):
    d = tmp_path / "dkr27bxyz"
    d.mkdir()
    assert not L.l3_persistent_store(str(d))
    assert not L.l3_persistent_store("")


def test_teardown_never_rmtrees_without_the_l3_guard():
    """Every rmtree of the store dir in teardown() sits in a branch that the
    L3 guard already excluded (the removal is the ``elif`` after it)."""
    src = inspect.getsource(L.teardown)
    tree = ast.parse(src.replace("\n    ", "\n", 0))
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "l3_persistent_store" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "shutil.rmtree":
                    # the rmtree must be in the orelse, never in the guarded body
                    assert all(sub is not b for n in node.body for b in ast.walk(n))
                    guarded = True
    assert guarded, "teardown's store rmtree is not behind l3_persistent_store"
