"""W57 (28.09.): the store disk check measures the disk the store IS on.

Metal: a NF launcher dry run refused 'W57 Weg2StoreDiskRefused' with 171 GiB
free against 171.70 GiB needed (max_size 139.70 + min_free 32). 171 GiB is the
ZFS pool under /spinning (93 % full); the NF store is a bind of XFS [/l3/nf]
into the container at /var/lib/htsglang/hicache-weg2 (1.3 TB free, 106 GiB
used). The check measured statvfs(STORE_ROOT) -- the rig default
/spinning/hicache-weg2 when SGLANG_WEG2_STORE_ROOT is not set.

WHAT MUST HOLD.
(1) statvfs is taken on the store's own directory when it exists (a store
    directory can be a mount of its own), else on its root.
(2) The line names path, fs type and mount (/proc/mounts), free, reused,
    max_size, growth and needed; the refusal carries the same line.
(3) The persistent store's own bytes are credited: needed = max_size - reused
    + min_free; the free figure is the disk's, not free + reused.
(4) Not told where the store lives (SGLANG_WEG2_STORE_ROOT unset, no root
    passed): refused by name, nothing measured.
(5) fs_of picks the longest mount point above the path (the bind, not '/').
Nothing in a store is deleted by any of this.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger  # noqa: E402
from sglang.srt.weg2 import launcher  # noqa: E402

GIB = host_ledger.GIB
POOL_TOKENS = 262144
ATTN = (12,)
KV_MIB = 0.0078125  # 8 KiB/token/layer -> 96 KiB/token -> 24 GiB pool
DIR = "l3-nextflash-model-fe3b0e1fce"


class _Statvfs:
    def __init__(self, free, total=2300 * GIB):
        self.f_frsize = 4096
        self.f_bavail = int(free) // 4096
        self.f_blocks = int(total) // 4096


def _store(tmp_path, nbytes):
    d = tmp_path / DIR
    d.mkdir()
    (d / "page0.bin").write_bytes(os.urandom(nbytes))
    return d


def _plan(tmp_path, free, *, calls=None, **kw):
    def fake(p):
        if calls is not None:
            calls.append(str(p))
        return _Statvfs(free)

    with mock.patch.object(os, "statvfs", side_effect=fake):
        return launcher.plan_store("tag", POOL_TOKENS, ATTN, KV_MIB, root=str(tmp_path),
                                   min_free_gib=0.0, directory_name=DIR, **kw)


def test_the_reused_store_is_credited_and_free_is_the_disks(tmp_path):
    # the store's allocated bytes, as _tree_bytes reads them (a ZFS /tmp
    # reports st_blocks only after its txg sync -- the figure is fixed here)
    reused = 60 * GIB
    _store(tmp_path, 4096)
    probe = _plan(tmp_path, 1000 * GIB)
    max_size = probe.max_size_bytes
    # a disk that holds the growth but not max_size: passes, by the growth
    free = max_size - reused // 2
    with mock.patch.object(launcher, "_tree_bytes", return_value=(reused, reused, 1)):
        plan = _plan(tmp_path, free)
        with pytest.raises(launcher.Weg2StoreDiskRefused, match="reused=60.00 GiB"):
            _plan(tmp_path, max_size - reused - 8192)
    assert plan.reused_bytes == reused
    assert plan.fs_free_bytes == (free // 4096) * 4096  # the disk's free, not + reused
    assert plan.growth_bytes == max_size - plan.reused_bytes
    assert plan.needed_bytes == plan.growth_bytes + plan.min_free_bytes


def test_statvfs_is_taken_on_the_store_directory_when_it_exists(tmp_path):
    _store(tmp_path, 4096)
    calls = []
    _plan(tmp_path, 1000 * GIB, calls=calls)
    assert calls == [str(tmp_path / DIR)]
    fresh = tmp_path / "other"
    fresh.mkdir()
    calls = []
    with mock.patch.object(os, "statvfs", side_effect=lambda p: calls.append(str(p)) or _Statvfs(1000 * GIB)):
        launcher.plan_store("tag", POOL_TOKENS, ATTN, KV_MIB, root=str(fresh), min_free_gib=0.0)
    assert calls == [str(fresh)]


def test_the_line_names_path_fs_and_the_arithmetic(tmp_path):
    _store(tmp_path, 4096)
    with mock.patch.object(launcher, "fs_of",
                           lambda p, mounts_text=None: (str(tmp_path), "xfs", "/dev/nvme0n1p2")):
        plan = _plan(tmp_path, 1000 * GIB)
    ln = plan.disk_line("ok")
    for part in ("W57 STORE-DISK ok", "path=%s" % (tmp_path / DIR), "fs=xfs",
                 "mount=%s" % tmp_path, "free=1000.00 GiB", "reused=", "max_size=",
                 "growth=", "needed="):
        assert part in ln, part


def test_the_refusal_names_where_it_measured(tmp_path):
    _store(tmp_path, 4096)
    with mock.patch.object(launcher, "fs_of", lambda p, mounts_text=None: ("/", "zfs", "spinning/subvol")):
        with pytest.raises(launcher.Weg2StoreDiskRefused) as e:
            _plan(tmp_path, 1 * GIB)
    msg = str(e.value)
    assert "W57 STORE-DISK REFUSED" in msg and "fs=zfs" in msg
    assert "path=%s" % (tmp_path / DIR) in msg


def test_an_untold_store_root_is_refused_by_name(tmp_path):
    with mock.patch.object(launcher, "STORE_ROOT_TOLD", False), \
            mock.patch.object(launcher, "STORE_ROOT", str(tmp_path / "rigdefault")), \
            mock.patch.object(os, "statvfs", return_value=_Statvfs(1000 * GIB)) as sv:
        with pytest.raises(launcher.Weg2StoreDiskRefused, match="no store root told"):
            launcher.plan_store("tag", POOL_TOKENS, ATTN, KV_MIB, min_free_gib=0.0)
        assert not sv.called
        assert not (tmp_path / "rigdefault").exists()
    # told: the told root is measured
    with mock.patch.object(launcher, "STORE_ROOT_TOLD", True), \
            mock.patch.object(launcher, "STORE_ROOT", str(tmp_path / "told")), \
            mock.patch.object(os, "statvfs", return_value=_Statvfs(1000 * GIB)):
        plan = launcher.plan_store("tag", POOL_TOKENS, ATTN, KV_MIB, min_free_gib=0.0)
    assert plan.fs_path == str(tmp_path / "told")


def test_fs_of_takes_the_longest_mount_above_the_path(tmp_path):
    d = tmp_path / "store" / DIR
    d.mkdir(parents=True)
    mounts = ("spinning/subvol-999-disk-0 / zfs rw 0 0\n"
              "/dev/nvme0n1p2 %s xfs rw 0 0\n"
              "tmpfs %s/other tmpfs rw 0 0\n" % (tmp_path / "store", tmp_path))
    assert launcher.fs_of(str(d), mounts) == (str(tmp_path / "store"), "xfs", "/dev/nvme0n1p2")
    # a path that does not exist yet: its nearest existing ancestor decides
    assert launcher.fs_of(str(d / "not" / "yet"), mounts)[1] == "xfs"
    assert launcher.fs_of(str(tmp_path), mounts)[1] == "zfs"
    # an escaped mount point (space = \\040)
    sp = tmp_path / "a b"
    sp.mkdir()
    assert launcher.fs_of(str(sp), "x / zfs rw 0 0\nsrc %s xfs rw 0 0\n"
                          % str(sp).replace(" ", "\\040"))[1] == "xfs"


def test_the_launcher_logs_the_line_before_the_store_line():
    src = open(launcher.__file__).read()
    k = src.index("store_cfg = store_plan.extra_config()")
    assert src.index('log(store_plan.disk_line("ok"))', k) < src.index('f"WEG2-STORE: dir=', k)
