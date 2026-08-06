"""Unit tests for BarlinkBar1Transport._abort_peer_flag_snapshot (#616c).

Stubs replace the real transport class so the ctypes host-address reads
exercise the exact same unbound method without touching GPU hardware.
"""

import ctypes
import re


from sglang.srt.distributed.device_communicators import barlink_bar1 as B


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fn():
    """Unbound method to test."""
    return B.BarlinkBar1Transport._abort_peer_flag_snapshot


class _Mapping:
    """Minimal stand-in for PeerTarget.flag (Mapping)."""

    def __init__(self, host_address: int = 0, length: int = 0):
        self.host_address = host_address
        self.length = length


class _Peer:
    """Minimal stand-in for PeerTarget."""

    def __init__(self, flag):
        self.flag = flag


def _stub(rank: int = 0, world: int = 3, group: str = "tp:0"):
    class _S:
        pass

    s = _S()
    s.rank = rank
    s.world = world
    s.group = group
    return s


def _buf_with_first_dwords(dwords):
    """Return a ctypes buffer so each 256-byte line starts with the given
    little-endian uint32 dword."""
    total = len(dwords) * 256
    buf = (ctypes.c_ubyte * total)()
    for idx, val in enumerate(dwords):
        buf[idx * 256 : idx * 256 + 4] = val.to_bytes(4, "little")
    return buf


def _count_dword_entries(text, peer_rank):
    """Count N:VALUE tokens in the section for the given peer rank."""
    marker = "peer %d [" % peer_rank
    start = text.index(marker)
    section = text[start:]
    after_bracket = section.split("]: ", 1)[1]
    tokens = after_bracket.split()
    count = 0
    for t in tokens:
        stripped = t.rstrip(".")
        if re.fullmatch(r"\d+:\d+", stripped):
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestPeerFlagSnapshot:
    def test_none_when_no_peers_attr(self):
        """Missing _peers -> None."""
        s = _stub()
        result = _fn()(s)
        assert result is None

    def test_none_when_empty_peers(self):
        """_peers is empty dict -> None."""
        s = _stub()
        s._peers = {}
        result = _fn()(s)
        assert result is None

    def test_two_peers_both_visible(self):
        """Two peers with real buffers: output contains both sections
        and correct first-dword values."""
        buf1 = _buf_with_first_dwords([0xDEAD0001])
        buf2 = _buf_with_first_dwords([0xDEAD0002])

        s = _stub(rank=0, world=3, group="tp:0")
        s._peers = {
            1: _Peer(_Mapping(ctypes.addressof(buf1), 256)),
            2: _Peer(_Mapping(ctypes.addressof(buf2), 256)),
        }
        result = _fn()(s)

        assert result is not None
        assert "peer 1" in result
        assert "peer 2" in result
        assert "0:3735879681" in result  # 0xDEAD0001
        assert "0:3735879682" in result  # 0xDEAD0002

    def test_little_endian_correctness(self):
        """Bytes 01 02 00 00 at offset 0 should report 513."""
        buf = _buf_with_first_dwords([513])
        s = _stub(rank=5, world=2)
        s._peers = {
            0: _Peer(_Mapping(ctypes.addressof(buf), 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert "0:513" in result

    def test_skip_zero_host_address(self):
        """Peer with host_address=0 is silently skipped."""
        buf = _buf_with_first_dwords([999])
        s = _stub()
        s._peers = {
            1: _Peer(_Mapping(host_address=0, length=256)),
            2: _Peer(_Mapping(ctypes.addressof(buf), 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert "peer 1" not in result
        assert "peer 2" in result
        assert "0:999" in result

    def test_skip_zero_length(self):
        """Peer with length=0 is silently skipped."""
        buf = _buf_with_first_dwords([777])
        s = _stub()
        s._peers = {
            1: _Peer(_Mapping(host_address=ctypes.addressof(buf), length=0)),
            2: _Peer(_Mapping(ctypes.addressof(buf), 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert "peer 1" not in result
        assert "peer 2" in result

    def test_all_peers_unreadable_returns_none(self):
        """When every peer is skipped, the overall result is None."""
        s = _stub()
        s._peers = {
            1: _Peer(_Mapping(host_address=0, length=256)),
            2: _Peer(_Mapping(host_address=1000, length=0)),
        }
        assert _fn()(s) is None

    def test_max_lines_caps_entries(self):
        """max_lines limits the number of lines dumped per peer."""
        buf = _buf_with_first_dwords(list(range(10)))  # 10 lines worth
        s = _stub()
        s._peers = {
            3: _Peer(_Mapping(ctypes.addressof(buf), 10 * 256)),
        }
        result = _fn()(s, max_lines=3)
        assert result is not None
        assert _count_dword_entries(result, 3) == 3

    def test_peers_reported_in_ascending_order(self):
        """Peers are dumped in sorted rank order regardless of dict order."""
        buf_a = _buf_with_first_dwords([100])
        buf_b = _buf_with_first_dwords([200])
        buf_c = _buf_with_first_dwords([300])

        s = _stub()
        s._peers = {
            5: _Peer(_Mapping(ctypes.addressof(buf_c), 256)),
            2: _Peer(_Mapping(ctypes.addressof(buf_a), 256)),
            8: _Peer(_Mapping(ctypes.addressof(buf_b), 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert result.index("peer 2") < result.index("peer 5") < result.index("peer 8")

    def test_observing_rank_and_group_in_output(self):
        """The header line contains rank R/W and the group name."""
        buf = _buf_with_first_dwords([1])
        s = _stub(rank=7, world=11, group="dp:3")
        s._peers = {
            0: _Peer(_Mapping(ctypes.addressof(buf), 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert "rank 7/11" in result
        assert "group dp:3" in result

    def test_peer_with_negative_length_skipped(self):
        """Peer with length < 0 is treated as unreadable."""
        buf = _buf_with_first_dwords([42])
        s = _stub()
        s._peers = {
            1: _Peer(_Mapping(ctypes.addressof(buf), length=-1)),
        }
        assert _fn()(s) is None

    def test_multiple_lines_per_peer(self):
        """Two lines per peer produce two dword entries."""
        buf = _buf_with_first_dwords([10, 20])
        s = _stub()
        s._peers = {
            0: _Peer(_Mapping(ctypes.addressof(buf), 2 * 256)),
        }
        result = _fn()(s)
        assert result is not None
        assert "0:10" in result
        assert "1:20" in result

    def test_unmapped_flag_returns_none(self):
        """Peer whose flag attribute is missing -> skipped -> None if alone."""

        class _PeerNoFlag:
            pass

        s = _stub()
        s._peers = {1: _PeerNoFlag()}
        assert _fn()(s) is None
