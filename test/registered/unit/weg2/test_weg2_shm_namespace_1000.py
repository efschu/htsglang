"""Auftrag 1000: a pytest process's /dev/shm objects carry the TEST NAMESPACE and the
launcher's #1217 sweep does not count them as holders.

Boot A (03.10.2026, window 4dzm46) stayed empty because a broad agent pytest run held
real ``weg2-xchg-*`` objects the launcher counts as its own ("#1217 LIVE HOLDER").  The
fix puts ``test-<id>-`` IN FRONT of the whole name under pytest (``weg2/shm_namespace``);
this file pins (1) the prefix on every name family the exchange modules create,
(2) that the sweep ignores such a held entry while still refusing a held RELEASE-named
one, and (3) that the release spelling is byte-for-byte what it was, in a pytest-free
subprocess.
"""

import json
import os
import subprocess
import sys

import pytest

from sglang.srt.weg2 import launcher
from sglang.srt.weg2 import shm_namespace as shm_ns
from sglang.srt.weg2 import weight_exchange_bounce as wb
from sglang.srt.weg2 import weight_exchange_region as xr

ENV = "SGLANG_WEG2_SHM_NAMESPACE"
NONCE = "ns1000"

#: The release spelling of every name, written out LITERALLY (taken from the tree at
#: 5444cf8cde, before the namespace existed).  Not derived from the code under test.
RELEASE_NAMES = {
    "region_dir": "/dev/shm/weg2-xchg-ns1000",
    "region_path": "/dev/shm/weg2-xchg-ns1000/xchg.bin",
    "sem_cross": "/weg2-xchg-ns1000-0-1-0-empty",
    "sem_diag": "/weg2-xchg-ns1000-card2-1-full",
    "sem_permit": "/weg2-xchg-ns1000-lanes-permit",
    "bounce_path": "/dev/shm/weg2-xchg-ns1000/bounce.bin.c0",
    "bounce_slots": "/dev/shm/weg2-xchg-bnc-ns1000",
    "seq_buffer": "/dev/shm/weg2-seq-ns1000/c0_unit_buffer.bin",
    "seq_digest": "/dev/shm/weg2-seq-ns1000/c0_unit_digests.json",
    "seq_ring": "/dev/shm/weg2-seq-ns1000/c0_ring.bin",
}

_PROBE = """
import json
from sglang.srt.weg2 import weight_exchange_bounce as wb
from sglang.srt.weg2 import weight_exchange_region as xr
n = "ns1000"
print(json.dumps({
    "region_dir": xr.region_dir(n),
    "region_path": xr.region_path(n),
    "sem_cross": xr.sem_name(n, 0, 0, "empty"),
    "sem_diag": xr.diagonal_sem_name(n, 2, 1, "full"),
    "sem_permit": xr.lane_permit_sem_name(n),
    "bounce_path": wb.bounce_path(n, lane="c0"),
    "bounce_slots": wb.bounce_slots_path(n),
    "seq_buffer": wb.sequential_buffer_path(n, lane="c0"),
    "seq_digest": wb.sequential_digest_path(n, lane="c0"),
    "seq_ring": wb.seq_ring_path(n, lane="c0"),
}))
"""


def _names_here():
    n = NONCE
    return {
        "region_dir": xr.region_dir(n),
        "region_path": xr.region_path(n),
        "sem_cross": xr.sem_name(n, 0, 0, "empty"),
        "sem_diag": xr.diagonal_sem_name(n, 2, 1, "full"),
        "sem_permit": xr.lane_permit_sem_name(n),
        "bounce_path": wb.bounce_path(n, lane="c0"),
        "bounce_slots": wb.bounce_slots_path(n),
        "seq_buffer": wb.sequential_buffer_path(n, lane="c0"),
        "seq_digest": wb.sequential_digest_path(n, lane="c0"),
        "seq_ring": wb.seq_ring_path(n, lane="c0"),
    }


def _probe(env_patch):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTEST_CURRENT_TEST", ENV)}
    env.update(env_patch)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, check=True,
                         capture_output=True, text=True, timeout=180).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_under_pytest_every_name_carries_the_test_prefix(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    prefix = f"test-{os.getpid()}-"
    assert shm_ns.shm_prefix() == prefix
    got = _names_here()
    for key, rel in RELEASE_NAMES.items():
        if key.startswith("sem_"):
            want = "/" + prefix + rel[1:]
        else:
            want = "/dev/shm/" + prefix + rel[len("/dev/shm/"):]
        assert got[key] == want, (key, got[key], want)
    # the namespace was published so forked / spawned children compute the SAME token
    assert os.environ[ENV] == f"test-{os.getpid()}"
    # and the launcher's filter recognises the file names glibc / the region create
    assert shm_ns.is_test_shm_name(os.path.basename(got["region_dir"]))
    assert shm_ns.is_test_shm_name("sem." + got["sem_cross"].lstrip("/"))


def test_a_private_shm_root_keeps_its_historical_spelling(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    root = str(tmp_path)
    assert xr.region_dir(NONCE, root) == os.path.join(root, f"weg2-xchg-{NONCE}")
    assert wb.bounce_slots_path(NONCE, root) == os.path.join(root, f"weg2-xchg-bnc-{NONCE}")
    assert wb.sequential_buffer_path(NONCE, root, lane="c0") == (
        f"{root}/weg2-seq-{NONCE}/c0_unit_buffer.bin")


def test_namespace_value_rules(monkeypatch):
    monkeypatch.setenv(ENV, "abc/../x y")
    assert shm_ns.namespace() == "test-abc_.._x_y"       # forced marker, no path separators
    monkeypatch.setenv(ENV, "test-77")
    assert shm_ns.shm_prefix() == "test-77-"
    assert shm_ns.posix_sem_name("/weg2-xchg-a") == "/test-77-weg2-xchg-a"
    monkeypatch.setenv(ENV, "")
    assert shm_ns.shm_prefix() == ""                      # explicit OFF wins over pytest
    assert shm_ns.posix_sem_name("/weg2-xchg-a") == "/weg2-xchg-a"


def test_release_names_are_byte_identical_outside_pytest():
    """No env, no pytest: the literal names of the tree before the namespace existed."""
    assert _probe({}) == RELEASE_NAMES


def test_explicit_empty_namespace_is_release_even_with_pytest_marker():
    assert _probe({"PYTEST_CURRENT_TEST": "x::y (call)", ENV: ""}) == RELEASE_NAMES


def test_in_process_release_spelling_matches_the_literals(monkeypatch):
    monkeypatch.setenv(ENV, "")
    assert _names_here() == RELEASE_NAMES


def test_a_child_of_the_pytest_process_computes_the_same_token(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    here = _names_here()               # publishes test-<pid> into os.environ
    child = _probe({"PYTEST_CURRENT_TEST": "x::y (call)", ENV: os.environ[ENV]})
    assert child == here


def test_real_semaphores_land_in_the_test_namespace(monkeypatch):
    """The real syscalls, not just the strings: the glibc files of the semaphores appear
    under the test prefix and not under the release name."""
    monkeypatch.delenv(ENV, raising=False)
    nonce = f"ns1000real{os.getpid()}"
    prefix = shm_ns.shm_prefix()
    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    try:
        listing = os.listdir("/dev/shm")
        mine = [n for n in listing if nonce in n and n.startswith("sem.")]
        assert mine, "no semaphore file was created"
        assert all(n.startswith(f"sem.{prefix}{xr.REGION_PREFIX}") for n in mine), mine
        assert not [n for n in listing if n.startswith(f"sem.{xr.REGION_PREFIX}{nonce}")]
    finally:
        xr.unlink_semaphores(nonce)
    assert not [n for n in os.listdir("/dev/shm") if nonce in n]


def _fake_proc(tmp_path, pid, held_path):
    """A /proc tree with ONE process holding ``held_path`` open (fd symlink)."""
    fd_dir = tmp_path / "proc" / str(pid) / "fd"
    fd_dir.mkdir(parents=True)
    os.symlink(held_path, fd_dir / "3")
    return str(tmp_path / "proc")


def _sweep(tmp_path, proc_root, shm_dir):
    logs = []
    res = launcher.shm_residue_sweep(
        logs.append, "tagx", "20261003T000000Z", False, shm_dir=str(shm_dir),
        proc_root=proc_root, archive_root=str(tmp_path / "archive"),
        sem_unlink=lambda name: 0)
    return res, logs


def test_1217_sweep_ignores_a_held_test_namespace_entry(tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir()
    held_dir = shm / f"test-4242-{xr.REGION_PREFIX}{NONCE}"
    held_dir.mkdir()
    held = held_dir / "xchg.bin"
    held.write_bytes(b"\0" * 64)
    (shm / f"sem.test-4242-{xr.REGION_PREFIX}{NONCE}-0-1-0-empty").write_bytes(b"\0" * 32)
    proc = _fake_proc(tmp_path, 4242, str(held))
    # the holder scan, pointed at the entry, DOES see the holder ...
    assert launcher.shm_holder_pids(str(held_dir), proc) == [4242]
    # ... and the sweep still neither refuses nor touches anything
    res, logs = _sweep(tmp_path, proc, shm)
    assert res["swept"] == [] and res["refused"] == {}
    assert held.exists() and any("test-namespace" in line for line in logs), logs


def test_1217_sweep_still_refuses_a_held_release_named_entry(tmp_path):
    """Control (the red state of the boot-A incident): the same holder on the RELEASE
    spelling refuses the boot -- the namespace is what makes the difference."""
    shm = tmp_path / "shm"
    shm.mkdir()
    held_dir = shm / f"{xr.REGION_PREFIX}{NONCE}"
    held_dir.mkdir()
    held = held_dir / "xchg.bin"
    held.write_bytes(b"\0" * 64)
    proc = _fake_proc(tmp_path, 4242, str(held))
    with pytest.raises(launcher.Weg2LaunchRefused, match="LIVE HOLDER"):
        _sweep(tmp_path, proc, shm)


def test_1217_sweep_mixed_dir_sweeps_only_the_orphan_release_entry(tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / f"{xr.REGION_PREFIX}deadboot").mkdir()
    (shm / f"{xr.REGION_PREFIX}deadboot" / "xchg.bin").write_bytes(b"\0" * 64)
    t = shm / f"test-4242-{xr.REGION_PREFIX}{NONCE}"
    t.mkdir()
    (t / "xchg.bin").write_bytes(b"\0" * 64)
    proc = _fake_proc(tmp_path, 4242, str(t / "xchg.bin"))
    res, _logs = _sweep(tmp_path, proc, shm)
    assert res["swept"] == [f"{xr.REGION_PREFIX}deadboot"]
    assert (t / "xchg.bin").exists()
