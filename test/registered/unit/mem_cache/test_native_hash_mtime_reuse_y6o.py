"""Boot y6o (2026-10-01 21:36Z): the first request after boot waited 6.3 s on
a rebuild of ``hicache_hash_cpp_avx2`` although the cache volume held a build
of byte-identical source.

Cause: ``load()`` handed ninja the tree's own ``hash_binding.cpp``; a delta
image extracts its tree with fresh mtimes, so every NEW image looked newer
than the object the previous image left on the persistent torch_extensions
volume, and ninja rebuilt. The loader now stages the source into the build
directory and rewrites it only when its bytes change.

The integration test fails on the old loader: tree B carries the same bytes
with a NEWER mtime, exactly what a new delta image presents.
"""

import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.mem_cache.cpp_utils import native_hash as nh

_SRC_DIR = os.path.dirname(os.path.abspath(nh.__file__))
_PY_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(sys.modules["flliper"].__file__))
)


def test_stage_writes_once_and_keeps_mtime_for_identical_bytes(tmp_path):
    tree = tmp_path / "tree"
    build = tmp_path / "build"
    tree.mkdir()
    build.mkdir()
    src = tree / "hash_binding.cpp"
    src.write_bytes(b"int x;\n")

    stage = lambda: nh._stage_sources(sources=[str(src)], build_dir=str(build))
    staged, written = stage()
    assert staged == [str(build / "hash_binding.cpp")] and written == staged
    dst = build / "hash_binding.cpp"
    before = dst.stat().st_mtime_ns

    # a new image: same bytes, tree mtime far newer than the staged copy
    future = time.time() + 3600
    os.utime(src, (future, future))
    staged, written = stage()
    assert written == [] and dst.stat().st_mtime_ns == before

    # a real source change is staged, so ninja rebuilds
    src.write_bytes(b"int y;\n")
    staged, written = stage()
    assert written == staged and dst.read_bytes() == b"int y;\n"
    assert not [p for p in os.listdir(build) if p.endswith(".tmp")]


_CHILD = r"""
import importlib.util, json, logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
spec = importlib.util.spec_from_file_location("nh_copy", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
mod = m._load_via_torch()
assert hasattr(mod, "get_hash")
print(json.dumps({"so": mod.__file__}))
"""


def _tree(root, name, extra=b""):
    """A fresh copy of the loader + source: what a new image's tree is."""
    d = root / name
    d.mkdir()
    shutil.copy(os.path.join(_SRC_DIR, "native_hash.py"), d / "native_hash.py")
    data = open(os.path.join(_SRC_DIR, "hash_binding.cpp"), "rb").read() + extra
    (d / "hash_binding.cpp").write_bytes(data)
    return d


def _load(tree, ext_dir):
    env = dict(os.environ)
    env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    env["PYTHONPATH"] = _PY_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tree / "native_hash.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    so = json.loads(r.stdout.strip().splitlines()[-1])["so"]
    return so, r.stderr


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_new_image_with_identical_source_does_not_rebuild(tmp_path):
    ext = tmp_path / "torch_extensions"
    so, log = _load(_tree(tmp_path, "image_a"), ext)
    assert "staged=written" in log
    built = os.stat(so).st_mtime_ns

    time.sleep(1.1)  # mtime granularity: the new tree is strictly newer
    tree_b = _tree(tmp_path, "image_b")
    so_b, log = _load(tree_b, ext)
    assert so_b == so
    assert "staged=unchanged" in log
    assert os.stat(so).st_mtime_ns == built, "identical source was rebuilt"

    # control: a real change still rebuilds
    time.sleep(1.1)
    so_c, log = _load(_tree(tmp_path, "image_c", extra=b"\n// changed\n"), ext)
    assert "staged=written" in log
    assert os.stat(so_c).st_mtime_ns != built
