# SPDX-License-Identifier: Apache-2.0
"""L15-10 S3b: D's held extents travel to P as fds (wire + export contract).

The metal part (cuMemExportToShareableHandle / import / map) needs a GPU; the
desk pins what can be pinned without one: the export loop's contract (every
extent exported or none -- fds closed on a refusal, the refusal named) and
the socket transport (JSON header + SCM_RIGHTS fds arrive intact, here with
ordinary files standing in for VMM handles).
"""

from __future__ import annotations

import ctypes
import os
import socket
import tempfile

from flliper.srt.pdflip import l15_hold_share as hs


def _fake_export(fail_at=None):
    opened = []

    def fn(ptr, off, fd_p, size_p):
        if fail_at is not None and int(off.value) == fail_at:
            return 1
        f = os.open(tempfile.mkstemp()[1], os.O_RDONLY)
        opened.append(f)
        fd_p._obj.value = f
        size_p._obj.value = 4096
        return 0
    return fn, opened


def test_export_returns_one_fd_per_extent():
    fn, opened = _fake_export()
    got = hs.export_hold_extents(0x1000, [(0, 10), (10, 20)], export=fn)
    assert [(o, s) for o, s, _f in got] == [(0, 4096), (10, 4096)]
    for _o, _s, f in got:
        os.close(f)


def test_a_refused_extent_closes_the_fds_already_exported():
    fn, opened = _fake_export(fail_at=10)
    try:
        hs.export_hold_extents(0x1000, [(0, 10), (10, 20)], export=fn)
    except hs.L15ShareError as exc:
        assert "FLLIPER_PDFLIP_VMM_EXPORTABLE" in str(exc)
    else:
        raise AssertionError("a refused export must raise")
    for f in opened:
        try:
            os.fstat(f)
        except OSError:
            continue
        raise AssertionError("fd %d left open after the refusal" % f)


def test_header_and_fds_cross_the_socket():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    paths = []
    fds = []
    for i in range(2):
        fd_, p = tempfile.mkstemp()
        os.write(fd_, b"x%d" % i)
        paths.append(p)
        fds.append(fd_)
    hs.send_hold(a, {"rank": 1, "n_fds": 2, "extents": [[0, 2], [2, 4]]}, fds)
    header, got = hs.recv_hold(b)
    assert header["extents"] == [[0, 2], [2, 4]] and len(got) == 2
    for i, f in enumerate(got):
        os.lseek(f, 0, 0)
        assert os.read(f, 2) == b"x%d" % i
        os.close(f)
    for f in fds:
        os.close(f)
    a.close()
    b.close()
