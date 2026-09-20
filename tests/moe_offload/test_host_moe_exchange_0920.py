# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for Form A's host-centric MoE exchange (slice 3).

No CUDA, no BAR1, no second process. The transport is a FAKE: an in-process
object that implements the same three operations barlink offers
(``broadcast``, ``put``, ``barrier``) against shared Python dicts, so the
whole three-rank round trip -- host broadcasts the MoE input, every rank
computes its own experts, the partials are reduced back into the host --
runs in one thread and is checked against the all-reduce it replaces.

What is pinned:

  * the ANSWER: the directed reduce equals the sum of the partials, i.e.
    exactly what today's post-MoE all-reduce produces on the host. If Form A
    changed the number, it would not be an optimisation;
  * the DIRECTION: a worker gets None back, not the total. Returning the
    total anyway is how the all-reduce shape sneaks back in through a
    variable name;
  * DETERMINISM: the summation order is a pure function of the geometry
    (ascending rank, host skipped), not of arrival order, so the sum is
    byte-identical across runs -- checked by reducing the same partials in
    two different push orders;
  * the WINDOW REFUSAL, by name and with its numbers, because a window too
    small is the one failure that a posted BAR1 write does not report;
  * the DEGRADATION is named: a transport with neither put+barrier nor
    reduce falls back to all_reduce and SAYS so through `reduce_mode`,
    rather than quietly costing every rank a result only the host reads;
  * every shape/dtype mismatch against the window geometry refuses before a
    byte moves.
"""

import pytest
import torch

from sglang.srt.layers.moe.host_moe_exchange import (
    ExchangeGeometry,
    HostMoEExchange,
    HostMoEShapeMismatch,
    HostMoETransportUnusable,
    HostMoEWindowTooSmall,
)

HIDDEN = 2560  # Qwen3.8-Flash-Next
ROWS = 4  # k=3 -> 4 verify rows; the decode case Form A is sized for
WORLD = 3
HOST_RANK = 0
DT = torch.bfloat16


# --------------------------------------------------------------------------
# The fake transport: three ranks in one process, sharing a "BAR1 window"
# --------------------------------------------------------------------------
class FakeFabric:
    """The shared medium. One inbox per rank, addressed by byte offset."""

    def __init__(self, world: int):
        self.world = world
        self.windows: dict = {}
        self.puts: list = []
        self.barriers = 0
        self.broadcasts = 0
        self._bcast_value = None

    def register(self, rank: int, tensor: torch.Tensor) -> None:
        self.windows[rank] = tensor


class FakeTransport:
    """One rank's view of the fabric. Mirrors barlink's operation names."""

    def __init__(self, fabric: FakeFabric, rank: int, *, directed: bool = True):
        self.fabric = fabric
        self.rank = rank
        self._directed = directed
        if directed:
            self.put = self._put
            self.barrier = self._barrier

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        if self.rank == src:
            self.fabric._bcast_value = tensor.clone()
            self.fabric.broadcasts += 1
        else:
            tensor.copy_(self.fabric._bcast_value)
        return tensor

    def _put(self, dst: int, source_ptr: int, nbytes: int, offset: int = 0) -> None:
        # The real put is a posted DMA from a device pointer. Here the
        # payload travels by the registered source tensor, and what is
        # CHECKED is the addressing: who wrote how many bytes where.
        self.fabric.puts.append((self.rank, dst, offset, nbytes))
        window = self.fabric.windows[dst]
        flat = window.view(-1)
        itemsize = window.element_size()
        assert offset % itemsize == 0
        start = offset // itemsize
        payload = self._pending
        flat[start : start + payload.numel()].copy_(payload.reshape(-1))

    def _barrier(self) -> None:
        self.fabric.barriers += 1

    def stage(self, tensor: torch.Tensor) -> None:
        """Stand-in for "the bytes the put will move"."""
        self._pending = tensor


class AllReduceOnlyTransport:
    """A transport with no directed path at all -- the degraded case."""

    def __init__(self, fabric: FakeFabric, rank: int):
        self.fabric = fabric
        self.rank = rank
        self.all_reduces = 0

    def broadcast(self, tensor, src):
        return FakeTransport.broadcast(self, tensor, src)

    def all_reduce(self, tensor):
        self.all_reduces += 1
        return tensor


def _geometry(**kw) -> ExchangeGeometry:
    base = dict(
        world=WORLD,
        host_rank=HOST_RANK,
        hidden_size=HIDDEN,
        max_rows=ROWS,
        dtype_bytes=2,
        window_bytes=(WORLD - 1) * ROWS * HIDDEN * 2,
    )
    base.update(kw)
    return ExchangeGeometry(**base)


def _round_trip(partials, push_order=None):
    """Run one full host-centric MoE exchange over the fake fabric."""
    geo = _geometry()
    fabric = FakeFabric(WORLD)
    transports = {r: FakeTransport(fabric, r) for r in range(WORLD)}
    ex = {
        r: HostMoEExchange(transports[r], geo, r, dtype=DT) for r in range(WORLD)
    }
    fabric.register(HOST_RANK, ex[HOST_RANK]._inbox)

    # 1. the host broadcasts the MoE input
    moe_input = torch.randn(ROWS, HIDDEN, dtype=torch.float32).to(DT)
    received = {}
    ex[HOST_RANK].broadcast_rows(moe_input.clone())
    for r in range(WORLD):
        buf = moe_input.clone() if r == HOST_RANK else torch.zeros_like(moe_input)
        received[r] = ex[r].broadcast_rows(buf)

    # 2. every rank computes its own experts (here: the injected partials)
    # 3. the partials go back to the host
    order = push_order or [r for r in range(WORLD) if r != HOST_RANK]
    for r in order:
        transports[r].stage(partials[r])
        assert ex[r].reduce_to_host(partials[r]) is None
    total = ex[HOST_RANK].reduce_to_host(partials[HOST_RANK])
    return ex, fabric, received, moe_input, total


# ==========================================================================
# 1. The answer, the direction, the determinism
# ==========================================================================
def test_the_directed_reduce_equals_the_all_reduce_it_replaces():
    partials = {
        r: torch.randn(ROWS, HIDDEN, dtype=torch.float32).to(DT) for r in range(WORLD)
    }
    _, _, received, moe_input, total = _round_trip(partials)

    # every rank got the same MoE input
    for r in range(WORLD):
        assert torch.equal(received[r], moe_input)
    # and the host holds the sum
    expected = partials[0].clone()
    expected += partials[1]
    expected += partials[2]
    assert torch.equal(total, expected)


def test_a_worker_gets_nothing_back():
    """Returning the total to a worker is how the all-reduce shape sneaks
    back in through a variable name."""
    partials = {r: torch.ones(ROWS, HIDDEN, dtype=DT) for r in range(WORLD)}
    ex, _, _, _, total = _round_trip(partials)
    assert total is not None and ex[HOST_RANK].is_host
    for r in (1, 2):
        assert not ex[r].is_host
        assert ex[r].reduce_to_host(partials[r]) is None


def test_the_sum_does_not_depend_on_who_pushed_first():
    """The reduction order is a pure function of the geometry, so the same
    partials give byte-identical sums whatever order the posted writes
    happened to land in."""
    partials = {
        r: torch.randn(ROWS, HIDDEN, dtype=torch.float32).to(DT) for r in range(WORLD)
    }
    _, _, _, _, a = _round_trip(partials, push_order=[1, 2])
    _, _, _, _, b = _round_trip(partials, push_order=[2, 1])
    assert torch.equal(a, b)


def test_each_worker_lands_in_its_own_slot_and_nobody_overlaps():
    partials = {r: torch.full((ROWS, HIDDEN), float(r), dtype=DT) for r in range(WORLD)}
    _, fabric, _, _, _ = _round_trip(partials)
    payload = ROWS * HIDDEN * 2
    assert [(src, dst, off, n) for src, dst, off, n in fabric.puts] == [
        (1, 0, 0, payload),
        (2, 0, payload, payload),
    ]
    # one barrier per rank per reduce -- the host included
    assert fabric.barriers == WORLD


def test_partial_rows_use_only_the_front_of_the_slot():
    """The window is mapped for max_rows; a shorter decode round must not
    read the stale tail of the slot."""
    geo = _geometry()
    fabric = FakeFabric(WORLD)
    transports = {r: FakeTransport(fabric, r) for r in range(WORLD)}
    ex = {r: HostMoEExchange(transports[r], geo, r, dtype=DT) for r in range(WORLD)}
    fabric.register(HOST_RANK, ex[HOST_RANK]._inbox)
    ex[HOST_RANK]._inbox.fill_(99.0)  # stale bytes from a previous round

    short = {r: torch.full((2, HIDDEN), float(r + 1), dtype=DT) for r in range(WORLD)}
    for r in (1, 2):
        transports[r].stage(short[r])
        ex[r].reduce_to_host(short[r])
    total = ex[HOST_RANK].reduce_to_host(short[0])
    assert total.shape == (2, HIDDEN)
    assert torch.equal(total, torch.full((2, HIDDEN), 1.0 + 2.0 + 3.0, dtype=DT))


# ==========================================================================
# 2. The window refusal
# ==========================================================================
def test_a_window_too_small_refuses_by_name_with_its_numbers():
    payload = ROWS * HIDDEN * 2
    geo = _geometry(window_bytes=payload)  # room for one worker, not two
    fabric = FakeFabric(WORLD)
    with pytest.raises(HostMoEWindowTooSmall) as e:
        HostMoEExchange(FakeTransport(fabric, 0), geo, 0, dtype=DT)
    msg = str(e.value)
    assert str(2 * payload) in msg and str(payload) in msg
    assert "--barlink-bar1-window-mib" in msg
    assert "2 rows" not in msg and f"{ROWS} rows" in msg


def test_a_window_exactly_large_enough_is_accepted():
    payload = ROWS * HIDDEN * 2
    geo = _geometry(window_bytes=2 * payload)
    HostMoEExchange(FakeTransport(FakeFabric(WORLD), 0), geo, 0, dtype=DT)


def test_the_required_window_grows_with_the_worker_count_not_the_rows_alone():
    small = _geometry(world=3).required_window_bytes
    big = _geometry(world=5, window_bytes=10**9).required_window_bytes
    assert big == 2 * small


# ==========================================================================
# 3. The degradation is NAMED, never silent
# ==========================================================================
def test_a_transport_without_a_directed_path_says_so():
    geo = _geometry()
    fabric = FakeFabric(WORLD)
    ex = HostMoEExchange(AllReduceOnlyTransport(fabric, 0), geo, 0, dtype=DT)
    assert ex.reduce_mode == "all_reduce"
    partial = torch.ones(ROWS, HIDDEN, dtype=DT)
    assert ex.reduce_to_host(partial) is not None
    worker = HostMoEExchange(AllReduceOnlyTransport(fabric, 1), geo, 1, dtype=DT)
    assert worker.reduce_to_host(partial) is None  # still directed in MEANING


def test_a_transport_with_a_native_reduce_uses_it():
    class WithReduce(FakeTransport):
        def __init__(self, fabric, rank):
            super().__init__(fabric, rank, directed=False)
            self.reduced = 0

        def reduce(self, tensor, dst):
            self.reduced += 1
            return tensor

    ex = HostMoEExchange(WithReduce(FakeFabric(WORLD), 0), _geometry(), 0, dtype=DT)
    assert ex.reduce_mode == "reduce"
    ex.reduce_to_host(torch.ones(ROWS, HIDDEN, dtype=DT))
    assert ex.transport.reduced == 1


def test_a_transport_that_cannot_reduce_at_all_refuses_at_construction():
    class BroadcastOnly:
        def broadcast(self, tensor, src):
            return tensor

    with pytest.raises(HostMoETransportUnusable, match="no way to get the partial"):
        HostMoEExchange(BroadcastOnly(), _geometry(), 0, dtype=DT)


def test_a_transport_without_broadcast_refuses_and_names_barlinks():
    class Nothing:
        pass

    with pytest.raises(HostMoETransportUnusable, match="barlink.py:1572"):
        HostMoEExchange(Nothing(), _geometry(), 0, dtype=DT)


# ==========================================================================
# 4. Geometry and shape refusals -- before a byte moves
# ==========================================================================
def test_geometry_refuses_a_world_of_one_and_a_host_outside_it():
    with pytest.raises(HostMoEShapeMismatch, match="at least two ranks"):
        _geometry(world=1)
    with pytest.raises(HostMoEShapeMismatch, match="outside world"):
        _geometry(host_rank=7)
    with pytest.raises(HostMoEShapeMismatch, match="max_rows must be positive"):
        _geometry(max_rows=0)


def test_shape_and_dtype_mismatches_refuse():
    ex = HostMoEExchange(FakeTransport(FakeFabric(WORLD), 0), _geometry(), 0, dtype=DT)
    with pytest.raises(HostMoEShapeMismatch, match="expected a \\(rows, 2560\\)"):
        ex.broadcast_rows(torch.ones(ROWS, 128, dtype=DT))
    with pytest.raises(HostMoEShapeMismatch, match="exceed the 4"):
        ex.broadcast_rows(torch.ones(ROWS + 1, HIDDEN, dtype=DT))
    with pytest.raises(HostMoEShapeMismatch, match="dtype"):
        ex.broadcast_rows(torch.ones(ROWS, HIDDEN, dtype=torch.float32))


def test_a_geometry_whose_dtype_bytes_lie_refuses():
    """A window sized for 2-byte rows that is handed 4-byte ones is a fault
    that only shows up as corrupt rows, so it is caught at construction."""
    with pytest.raises(HostMoEShapeMismatch, match="the window would be"):
        HostMoEExchange(
            FakeTransport(FakeFabric(WORLD), 0),
            _geometry(),
            0,
            dtype=torch.float32,
        )


def test_the_host_has_no_slot_of_its_own():
    geo = _geometry()
    assert geo.slot_offset(1) == 0
    assert geo.slot_offset(2) == ROWS * HIDDEN * 2
    with pytest.raises(HostMoEShapeMismatch, match="is the host"):
        geo.slot_offset(HOST_RANK)
    # ... and with a host in the middle the seats stay packed
    mid = _geometry(host_rank=1)
    assert mid.slot_offset(0) == 0
    assert mid.slot_offset(2) == ROWS * HIDDEN * 2
    assert mid.worker_ranks() == [0, 2]


def test_a_worker_keeps_no_inbox():
    geo = _geometry()
    worker = HostMoEExchange(FakeTransport(FakeFabric(WORLD), 1), geo, 1, dtype=DT)
    with pytest.raises(HostMoETransportUnusable, match="keeps no inbox"):
        worker.inbox_view(1, ROWS)


def test_the_decode_payload_is_the_small_one_form_a_promised():
    """Sanity on the size that makes the whole layout worth it: the MoE
    input of a decode round is kilobytes, not megabytes."""
    geo = _geometry(max_rows=3)
    assert geo.payload_bytes == 3 * 2560 * 2 == 15360  # ~15 KB, k=2
    assert _geometry(max_rows=4).payload_bytes == 20480  # ~20 KB, k=3
