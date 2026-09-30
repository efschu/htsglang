# SPDX-License-Identifier: Apache-2.0
"""Form A's MoE exchange: one host broadcasts the rows, everyone reduces back.

Under Form A exactly one rank (the attention HOST) has tokens. Per MoE layer
it broadcasts the MoE input -- in decode 3-4 rows of 2560 bf16, i.e. 15-20 KB
-- every rank computes the experts it OWNS, and the partial sums are reduced
back into the host. That is a fundamentally cheaper shape than the all-to-all
a2a the general EP path needs, and it is NOT what `Bar1EPDispatcher`
(``token_dispatcher/bar1ep.py:302``) is built for: that one routes rows to
experts across all ranks and requires an EQUAL expert split to do it
(``bar1ep.py:364-369``, which is precisely the assumption Form A breaks).

So this is a second, much smaller exchange next to it, and the split of work
is deliberate: bar1ep stays the general dispatcher, this is the host-centric
one. Two things it does that a plain ``all_reduce`` does not:

  * the REDUCE is directed. Today the MoE ends in an all-reduce, so every
    rank pays for a result only the host will use. barlink's facade has
    ``broadcast`` (``barlink.py:1572``) but no ``reduce``; the directed seam
    underneath is ``barlink_bar1.put`` (``barlink_bar1.py:3270``), which is
    put-only on purpose ("everyone pushes for themselves" -- a foreign-BAR
    read is non-posted and measured at a third of the write rate). This
    module builds the reduce out of that put.
  * the degradation is NAMED. If the transport cannot do a directed reduce,
    the fallback to all_reduce is reported through `reduce_mode` and logged
    once -- it is not allowed to be a silent loss of the whole point.

Transport contract (duck-typed on purpose -- the real one is a
``GroupCoordinator``-owned barlink communicator, the test one is 30 lines):

    broadcast(tensor, src)          required
    barrier()                       required for the directed path
    put(dst, source_ptr, nbytes, offset=0)      -> reduce_mode "put"
    reduce(tensor, dst)             -> reduce_mode "reduce"
    all_reduce(tensor)              -> reduce_mode "all_reduce" (degraded)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "HostMoEExchangeError",
    "HostMoEWindowTooSmall",
    "HostMoETransportUnusable",
    "HostMoEShapeMismatch",
    "ExchangeGeometry",
    "HostMoEExchange",
]


class HostMoEExchangeError(RuntimeError):
    """Base: the host-centric MoE exchange cannot run as configured."""


class HostMoEWindowTooSmall(HostMoEExchangeError):
    """The BAR1 receive window cannot hold what the exchange must land in it."""


class HostMoETransportUnusable(HostMoEExchangeError):
    """The transport lacks an operation this exchange cannot do without."""


class HostMoEShapeMismatch(HostMoEExchangeError):
    """A tensor handed in does not match the geometry the window was sized for."""


@dataclass(frozen=True)
class ExchangeGeometry:
    """Everything the window size and the refusals are computed from.

    `max_rows` is the CAPACITY the window is sized for, not the row count of
    any one call: the window is mapped once and re-mapping on the hot path is
    "exactly the expensive part" (``barlink_bar1.py:3285-3288``). Under
    Form A with k=2 that is 3 verify rows, with k=3 four.
    """

    world: int
    host_rank: int
    hidden_size: int
    max_rows: int
    dtype_bytes: int
    window_bytes: int

    def __post_init__(self) -> None:
        if self.world < 2:
            raise HostMoEShapeMismatch(
                f"a host-centric exchange needs at least two ranks, got "
                f"world={self.world}. With one rank there is nothing to "
                "broadcast to and nothing to reduce from."
            )
        if not 0 <= self.host_rank < self.world:
            raise HostMoEShapeMismatch(
                f"host_rank {self.host_rank} is outside world {self.world}."
            )
        for field_name in ("hidden_size", "max_rows", "dtype_bytes"):
            if getattr(self, field_name) <= 0:
                raise HostMoEShapeMismatch(
                    f"{field_name} must be positive, got "
                    f"{getattr(self, field_name)}."
                )

    @property
    def payload_bytes(self) -> int:
        """One full MoE-input block: `max_rows` x hidden, in its dtype."""
        return self.max_rows * self.hidden_size * self.dtype_bytes

    @property
    def required_window_bytes(self) -> int:
        """What the HOST's window must hold: one partial sum per worker,
        landing concurrently, because every worker pushes for itself and
        nobody waits for a turn."""
        return (self.world - 1) * self.payload_bytes

    def slot_offset(self, rank: int) -> int:
        """Where rank `rank` pushes its partial in the host's window.

        Ranks are packed in ascending rank order with the host skipped, so
        the reduction order is a pure function of the geometry and the sum
        is byte-identical on every run. That determinism is the reason the
        offset is computed here and not by the caller.
        """
        if rank == self.host_rank:
            raise HostMoEShapeMismatch(
                f"rank {rank} is the host; it does not push into its own "
                "window, it sums what the workers pushed."
            )
        seat = rank if rank < self.host_rank else rank - 1
        return seat * self.payload_bytes

    def worker_ranks(self) -> List[int]:
        return [r for r in range(self.world) if r != self.host_rank]


class HostMoEExchange:
    """Broadcast the MoE input from the host, reduce the partials back to it."""

    def __init__(
        self,
        transport: Any,
        geometry: ExchangeGeometry,
        rank: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not 0 <= rank < geometry.world:
            raise HostMoEShapeMismatch(
                f"rank {rank} is outside world {geometry.world}."
            )
        if dtype.itemsize != geometry.dtype_bytes:
            raise HostMoEShapeMismatch(
                f"dtype {dtype} is {dtype.itemsize} bytes but the geometry "
                f"was sized for {geometry.dtype_bytes}; the window would be "
                "the wrong size, which is a fault that only shows up as "
                "corrupt rows."
            )
        if not hasattr(transport, "broadcast"):
            raise HostMoETransportUnusable(
                "the transport has no broadcast(tensor, src); the host "
                "cannot hand its MoE input to the workers at all. barlink's "
                "is barlink.py:1572."
            )
        self.transport = transport
        self.geometry = geometry
        self.rank = rank
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._reduce_mode = self._resolve_reduce_mode()
        self._inbox: Optional[torch.Tensor] = None
        if self._reduce_mode == "put":
            self._check_window()
            if self.is_host:
                self._inbox = torch.zeros(
                    (geometry.world - 1, geometry.max_rows, geometry.hidden_size),
                    dtype=dtype,
                    device=self.device,
                )

    # -- capability resolution --------------------------------------------
    @property
    def is_host(self) -> bool:
        return self.rank == self.geometry.host_rank

    @property
    def reduce_mode(self) -> str:
        """"put" | "reduce" | "all_reduce". The last one is the DEGRADED
        mode: correct, but every rank pays for a result only the host uses,
        which is the cost Form A exists to remove."""
        return self._reduce_mode

    def _resolve_reduce_mode(self) -> str:
        t = self.transport
        if hasattr(t, "put") and hasattr(t, "barrier"):
            return "put"
        if hasattr(t, "reduce"):
            return "reduce"
        if hasattr(t, "all_reduce"):
            logger.warning(
                "Form A host MoE exchange: the transport offers neither a "
                "directed put (+ barrier) nor a reduce, so the partial sums "
                "go back through all_reduce. That is CORRECT but it is the "
                "collective Form A exists to replace -- every rank pays for "
                "a result only rank %d reads. Directed seam: "
                "barlink_bar1.py:3270 (put).",
                self.geometry.host_rank,
            )
            return "all_reduce"
        raise HostMoETransportUnusable(
            "the transport offers no way to get the partial sums back to "
            "the host: it has neither put(+barrier), nor reduce, nor "
            "all_reduce."
        )

    def _check_window(self) -> None:
        need = self.geometry.required_window_bytes
        have = self.geometry.window_bytes
        if have < need:
            raise HostMoEWindowTooSmall(
                f"the directed reduce needs {need} bytes of BAR1 receive "
                f"window on rank {self.geometry.host_rank} "
                f"({self.geometry.world - 1} worker(s) x "
                f"{self.geometry.payload_bytes} bytes: "
                f"{self.geometry.max_rows} rows x "
                f"{self.geometry.hidden_size} x {self.geometry.dtype_bytes} "
                f"B), but the window is {have} bytes. Raise "
                f"--barlink-bar1-window-mib to at least "
                f"{-(-need // 2**20)}, or lower max_rows. Re-mapping the "
                "window on the hot path is excluded by design "
                "(barlink_bar1.py:3285-3288)."
            )

    # -- the two operations -----------------------------------------------
    def broadcast_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Host -> every rank. On a worker, `rows` is the receive buffer."""
        self._require_shape(rows, "broadcast_rows")
        return self.transport.broadcast(rows, self.geometry.host_rank)

    def reduce_to_host(self, partial: torch.Tensor) -> Optional[torch.Tensor]:
        """Every rank's expert partial sum -> the host.

        Returns the summed tensor on the host and None on a worker: a worker
        has no use for the total, and returning it anyway is how the
        all-reduce shape sneaks back in.
        """
        self._require_shape(partial, "reduce_to_host")
        if self._reduce_mode == "all_reduce":
            total = self.transport.all_reduce(partial)
            return total if self.is_host else None
        if self._reduce_mode == "reduce":
            total = self.transport.reduce(partial, self.geometry.host_rank)
            return total if self.is_host else None
        return self._reduce_via_put(partial)

    def _reduce_via_put(self, partial: torch.Tensor) -> Optional[torch.Tensor]:
        g = self.geometry
        src = partial.contiguous()
        if not self.is_host:
            self.transport.put(
                g.host_rank,
                src.data_ptr(),
                src.numel() * src.element_size(),
                g.slot_offset(self.rank),
            )
        # Every rank waits: the workers so their posted writes are retired
        # before the host reads, the host so it reads only landed bytes.
        self.transport.barrier()
        if not self.is_host:
            return None
        assert self._inbox is not None
        rows = partial.shape[0]
        total = partial.clone()
        # Fixed ascending rank order -> the same sum on every run.
        for seat, _ in enumerate(g.worker_ranks()):
            total += self._inbox[seat, :rows, :]
        return total

    def _require_shape(self, t: torch.Tensor, where: str) -> None:
        g = self.geometry
        if t.dim() != 2 or t.shape[1] != g.hidden_size:
            raise HostMoEShapeMismatch(
                f"{where}: expected a (rows, {g.hidden_size}) tensor, got "
                f"{tuple(t.shape)}."
            )
        if t.shape[0] > g.max_rows:
            raise HostMoEShapeMismatch(
                f"{where}: {t.shape[0]} rows exceed the {g.max_rows} the "
                "window was mapped for. The window is sized once at "
                "construction; growing it on the hot path is excluded."
            )
        if t.dtype != self.dtype:
            raise HostMoEShapeMismatch(
                f"{where}: dtype {t.dtype} is not the {self.dtype} the "
                "window was sized for."
            )

    # -- what a worker's receive buffer looks like ------------------------
    def inbox_view(self, worker_rank: int, rows: int) -> torch.Tensor:
        """The host's landing slot for `worker_rank` -- for tests and for
        the transport setup that registers it in the BAR1 window."""
        if self._inbox is None:
            raise HostMoETransportUnusable(
                f"rank {self.rank} keeps no inbox: it is either a worker or "
                f"the reduce mode is {self._reduce_mode!r}, not 'put'."
            )
        seat = self.geometry.worker_ranks().index(worker_rank)
        return self._inbox[seat, :rows, :]
