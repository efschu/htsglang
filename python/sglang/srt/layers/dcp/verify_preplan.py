# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Host-known FlashInfer plan metadata for the DFLASH decode round.

SGLANG_DFLASH_PLAN_SYNC_FREE (default off). Measured on the 27B D group
(xsn421/xsn422, D TP0, bs 1, DFLASH): the host waited inside every decode round
for the draft forward to finish -- ``build_dcp_weighted_kv_indices``
(``compact[owned]``, owner.py:566, 83-105 py-spy samples) during the verify
prep -- and FlashInfer's ``plan()`` read ``qo_indptr`` / ``kv_indptr`` /
``last_page_len`` back from the device on every draft and verify plan
(prefill.py 1963/1974/1975/3054/3055). Each of those reads drains the stream,
so the GPU stood still from the end of one forward until the host had
finished planning and launched the next.

FlashInfer's ``plan()`` only needs those three vectors on the HOST to build
its schedule; when they are already host tensors its ``.to("cpu")`` is a no-op
and it copies them into the graph buffers with a non-blocking H2D. The one
vector the host cannot know from the scheduler's mirrors is the verify's
per-request OWNED-slot count under weighted DCP: ownership is a property of
the slot ids (``L % cp_S``), which only the device holds.

That count does not depend on the draft. The verify reads the committed
prefix ``[0, seq_len)``, whose slot ids were written rounds ago, so the index
build can run BEFORE the draft forward in stream order, followed by a
non-blocking D2H of ``kv_indptr`` into pinned memory and an event.
``DcpVerifyPrebuilt`` carries that. When the host later plans the verify, the
event has fired long ago (it sits in front of the draft on the same stream),
so reading the exact counts costs no stall; the GPU runs the draft meanwhile.

EXACTNESS IS THE CONTRACT, NOT AN APPROXIMATION. FlashInfer's FA2 paged
kernel derives its split-kv output layout (``num_kv_chunks``) from the DEVICE
kv length, the plan's ``merge_indptr`` / ``o_indptr`` from the HOST one. An
over-estimated host length that changes a request's chunk count therefore
mis-addresses the partial outputs -- it is not a safe "upper bound". Every
host vector handed to ``plan()`` on this path is the exact device value: the
verify's owned counts come from the readback, the draft's lengths from the
exact published ``seq_lens_cpu`` (page_size 1), and the draft path checks the
equality on the device (``torch._assert_async``) before its graph replays.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch

_PINNED_CONSTS: Dict[Tuple[str, int, int], torch.Tensor] = {}


def _pin(t: torch.Tensor) -> torch.Tensor:
    try:
        if torch.cuda.is_available():
            return t.pin_memory()
    except RuntimeError:
        pass
    return t


def host_arange_indptr(bs: int, stride: int) -> torch.Tensor:
    """``[0, stride, 2*stride, ..., bs*stride]`` as an int32 HOST tensor.

    Cached per (bs, stride) and never written after creation, so it is safe
    as the source of any number of in-flight non-blocking H2D copies.
    """
    key = ("arange", int(bs), int(stride))
    t = _PINNED_CONSTS.get(key)
    if t is None:
        t = _pin(
            torch.arange(
                0, (int(bs) + 1) * int(stride), step=int(stride), dtype=torch.int32
            )
        )
        _PINNED_CONSTS[key] = t
    return t


def host_ones(n: int) -> torch.Tensor:
    """``n`` int32 ones on the host (the page_size-1 ``last_page_len``)."""
    key = ("ones", int(n), 1)
    t = _PINNED_CONSTS.get(key)
    if t is None:
        t = _pin(torch.ones(int(n), dtype=torch.int32))
        _PINNED_CONSTS[key] = t
    return t


def host_indptr_from_lens(lens_cpu: torch.Tensor, add: int = 0) -> torch.Tensor:
    """``[0, cumsum(lens + add)]`` as a fresh int32 host tensor."""
    lens = lens_cpu.to(dtype=torch.int64, device="cpu") + int(add)
    out = torch.zeros(lens.numel() + 1, dtype=torch.int32)
    if lens.numel() > 0:
        out[1:] = torch.cumsum(lens, dim=0).to(torch.int32)
    return out


class HostPlanMeta:
    """The three host vectors FlashInfer's paged ``plan()`` schedules from.

    ``check_device``: the host vectors are exact by an ASSUMPTION about the
    scheduler's mirror (the draft path), not by construction (the verify
    path reads them back from the device), so the caller re-checks them on
    the device after the plan.
    """

    __slots__ = ("qo_indptr", "kv_indptr", "last_page_len", "check_device")

    def __init__(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        check_device: bool = False,
    ):
        self.qo_indptr = qo_indptr
        self.kv_indptr = kv_indptr
        self.last_page_len = last_page_len
        self.check_device = bool(check_device)


class DcpVerifyPrebuilt:
    """The verify's owned-slot index, built ahead of the draft.

    ``kv_indptr`` is a view of the backend's persistent ``kv_indptr[0]``
    buffer (the same storage the verify graph's paged wrapper reads), already
    holding the exact owned prefix counts; ``kv_indices`` is the packed owned
    slot list (``total_tokens_bound + pad`` long, meaningful up to
    ``kv_indptr[bs]``). ``host_kv_indptr()`` returns the same counts on the
    host.
    """

    def __init__(
        self,
        *,
        bs: int,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        host_buf: Optional[torch.Tensor],
        event,
    ):
        self.bs = int(bs)
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self._host_buf = host_buf
        self._event = event
        self._host: Optional[torch.Tensor] = None

    @classmethod
    def launch(
        cls,
        *,
        bs: int,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        host_buf: torch.Tensor,
    ) -> "DcpVerifyPrebuilt":
        """Stage the D2H of ``kv_indptr`` behind the index build, no wait."""
        n = int(bs) + 1
        if kv_indptr.is_cuda:
            host_buf[:n].copy_(kv_indptr[:n], non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host_buf[:n].copy_(kv_indptr[:n])
            event = None
        return cls(
            bs=bs,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            host_buf=host_buf,
            event=event,
        )

    def host_kv_indptr(self) -> torch.Tensor:
        """The exact owned prefix counts on the host (int32, ``bs + 1``).

        Waits on the event recorded right after the index build. On the
        stream that event sits IN FRONT of the draft forward, so by the time
        the verify is planned it has normally long resolved (measured
        xsn423: the host reaches the verify prep ~1.5 ms of host work after
        the round's publish, the event needs ~0.7 ms of device work) and the
        first ``query()`` returns True. A wait here never covers the draft.

        Polled, not ``Event.synchronize()``: the old read this replaces
        (owner.py ``compact[owned]``) was an unbounded wait inside the
        collective window (the #616c/#649 wedge class). A Python-level poll
        keeps the thread interruptible and lets the barlink watchdog threads
        run; it does not add a deadline -- a stream that never reaches the
        event is reported by those threads, not guessed at here.
        """
        if self._host is None:
            event = self._event
            if event is not None:
                while not event.query():
                    time.sleep(5e-5)
            self._host = self._host_buf[: self.bs + 1].clone()
            self._host_buf = None
        return self._host
