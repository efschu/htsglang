"""#1409 (boot xsn156): the native hash module is loaded once per process,
under a lock, and VERIFIED -- a first import without `get_hash` is retried
from a private copy of the artifact, a second miss raises.
"""

import os
import shutil
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.mem_cache.cpp_utils import native_hash as nh

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


@pytest.fixture(autouse=True)
def _reset():
    nh._MODULE = None
    yield
    nh._MODULE = None


def test_loaded_once_and_shared_across_threads():
    seen = []

    def w():
        seen.append(id(nh._load_native_hash_module()))

    ts = [threading.Thread(target=w) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(seen)) == 1
    assert hasattr(nh._MODULE, "get_hash")


def test_a_module_without_get_hash_is_reloaded_from_a_private_copy(monkeypatch, caplog):
    real = nh._load_native_hash_module()
    nh._MODULE = None
    dead = types.ModuleType("hicache_hash_cpp_avx2")
    dead.__file__ = real.__file__
    monkeypatch.setattr(nh, "_load_via_torch", lambda: dead)
    m = nh._load_native_hash_module()
    assert hasattr(m, "get_hash") and m is not dead
    assert "#1409 NATIVE-HASH" in caplog.text
    # the reloaded module hashes exactly like the torch-loaded one
    a = nh.get_native_hash([1, 2, 3], None, None)
    nh._MODULE = real
    assert nh.get_native_hash([1, 2, 3], None, None) == a


def test_a_module_without_artifact_path_raises(monkeypatch):
    monkeypatch.setattr(nh, "_load_via_torch", lambda: types.ModuleType("x"))
    with pytest.raises(RuntimeError, match="#1409"):
        nh._load_native_hash_module()
    assert nh._MODULE is None
