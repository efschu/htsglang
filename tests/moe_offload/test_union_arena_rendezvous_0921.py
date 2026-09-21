"""The union arena's cross-process rendezvous, slice 2 (2026-09-21).

The fd handoff is what lets the peer group map the owner's pages. Its failure
modes are quiet ones -- a stale socket, a short read, a peer that arrives
before the owner -- so they are pinned here. Hermetic: real unix sockets and
real fds (pipes), no CUDA.
"""

import os
import socket
import threading

import pytest

from sglang.srt.weg2.union_arena import UnionShareError
from sglang.srt.weg2.union_arena_vmm import (
    UnionRendezvousServer,
    fetch_union,
    socket_path,
)


def _pipe_fd_with(payload: bytes) -> int:
    r, w = os.pipe()
    os.write(w, payload)
    os.close(w)
    return r


def test_manifest_and_fds_survive_the_handoff(tmp_path):
    path = socket_path(str(tmp_path), "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d")
    fd = _pipe_fd_with(b"page-bytes")
    server = UnionRendezvousServer(path, '{"version": 1}', [fd])
    try:
        text, fds = fetch_union(path, timeout_s=10)
        assert text == '{"version": 1}'
        assert len(fds) == 1
        # a DIFFERENT descriptor number that reads the same open file
        assert fds[0] != fd
        assert os.read(fds[0], 32) == b"page-bytes"
        os.close(fds[0])
        # repeatable: a peer group that restarts must still be able to attach
        text2, fds2 = fetch_union(path, timeout_s=10)
        assert text2 == text and len(fds2) == 1
        os.close(fds2[0])
    finally:
        server.stop()
        os.close(fd)


def test_the_socket_path_stays_inside_the_unix_limit():
    """The boot's own directory fits; an over-long one is a NAMED refusal,
    not a bare OSError out of bind()."""
    path = socket_path("/dev/shm/weg2-union-fnFL2", "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d")
    assert len(path.encode()) <= 107, path
    assert path.endswith("u-e773fd938f6d.sock")
    with pytest.raises(UnionShareError, match="at most 107"):
        socket_path("/tmp/" + "d" * 120, "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d")


def test_a_missing_owner_is_a_named_refusal_not_a_hang(tmp_path):
    path = socket_path(str(tmp_path), "GPU-absent")
    with pytest.raises(UnionShareError, match="no union arena owner"):
        fetch_union(path, timeout_s=1.0)


def test_the_peer_may_arrive_before_the_owner(tmp_path):
    """The peer group routinely reaches the attach before the owner has
    finished loading; that is a wait, not an error."""
    path = socket_path(str(tmp_path), "GPU-late")
    fd = _pipe_fd_with(b"late")
    box = {}

    def start_owner_later():
        import time

        time.sleep(1.0)
        box["server"] = UnionRendezvousServer(path, "late-manifest", [fd])

    t = threading.Thread(target=start_owner_later)
    t.start()
    try:
        text, fds = fetch_union(path, timeout_s=20)
        assert text == "late-manifest"
        os.close(fds[0])
    finally:
        t.join()
        box["server"].stop()
        os.close(fd)


def test_a_truncated_owner_message_is_refused(tmp_path):
    """A half-written header must not be read as a valid (short) manifest."""
    path = str(tmp_path / "half.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    def serve_half():
        conn, _ = srv.accept()
        conn.sendall(b"\x01\x00")  # two bytes of an eight-byte header
        conn.close()

    t = threading.Thread(target=serve_half)
    t.start()
    try:
        with pytest.raises(UnionShareError):
            fetch_union(path, timeout_s=3)
    finally:
        t.join()
        srv.close()
