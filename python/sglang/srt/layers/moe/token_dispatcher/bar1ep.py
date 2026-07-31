# SPDX-License-Identifier: Apache-2.0
"""MoE token dispatcher over the BAR1 direct path.

WHY THIS FILE EXISTS AT ALL
----------------------------
Building ``all_to_all`` into the barlink seam was correct and **useless**:
the MoE dispatchers never call in there. Checked, not assumed --
``deepep.py:578`` ``buffer.dispatch(...)``, ``flashinfer.py:259``
``moe_a2a.dispatch(...)``, ``mooncake.py:236``, ``nixl.py:293``,
``moriep.py:724`` all bypass ``torch.distributed`` for their own library.
Whoever wants the MoE load has to **write a dispatcher**, not a collective.

THE CONTRACT THIS CLASS FULFILLS -- backed by file and line
-------------------------------------------------------------
``token_dispatcher/base.py``:

* ``:279`` ``dispatch(hidden_states, topk_output) -> DispatchOutput``
  (abstract), ``:304`` ``combine(combine_input) -> torch.Tensor``
  (abstract). ``BaseDispatcher`` demands nothing more.
* ``:187`` ``DispatchOutput`` is a ``Protocol`` with the field
  ``hidden_states`` and the property ``format``; ``:242``
  ``CombineInput`` only with ``format``.
* ``:161`` ``DispatchOutputFormat`` and ``:235`` ``CombineInputFormat``
  are **closed** enumerations. A new value would be a format no runner
  knows -- the dispatcher would run, and nobody could compute its result.
* ``:361`` ``set_quant_config(dict)``, ``:364`` ``set_overlap_args``,
  ``:370`` ``clear_overlap_args`` -- the framework calls them.
* ``:285``/``:308`` hang hooks before and after both directions; they
  operate on ``self.dispatch``/``self.combine`` and need nothing further
  from a subclass.

That is why this class does **not** deliver its own format, but
``DEEPEP_NORMAL``: the same ``NamedTuple`` classes from ``deepep.py:95``
(``DeepEPNormalDispatchOutput``) and ``deepep.py:128``
(``DeepEPNormalCombineInput``). Whoever unpacks them has been checked:

* ``moe_runner/deep_gemm.py:779`` ``pre_permute_deepep_normal_to_deep_gemm``
  unpacks the 5-tuple ``(hidden_states, hidden_states_scale, topk_ids,
  topk_weights, num_recv_tokens_per_expert)``, forms
  ``all_tokens = sum(num_recv_tokens_per_expert)`` -- so a **Python list on
  the CPU** -- and calls ``ep_scatter``.
* ``ep_moe/kernels.py:1108`` ``ep_scatter`` reads ``recv_topk`` as a
  **local** expert number in ``[0, num_local_experts)`` and ``-1`` for
  every slot that does not belong here (``_fwd_kernel_ep_scatter_2``:
  ``if expert_id >= 0``), and indexes ``expert_start_loc`` with it.
  ``m_indices.shape[0] % 128 == 0`` is checked -- hence the alignment of
  the counts to 128, exactly like ``deepep.py:589``
  ``expert_alignment=128 if ENABLE_JIT_DEEPGEMM``.
* ``ep_moe/kernels.py:1234`` ``ep_gather`` only weights where
  ``expert_id >= 0`` -- so the weights travel **unmasked**, as with DeepEP.
* ``moe_runner/deep_gemm.py:867`` ``post_permute_deep_gemm_to_deepep_normal``
  builds ``DeepEPNormalCombineInput(hidden_states, topk_ids,
  topk_weights)`` from it, with ``hidden_states`` in **bf16** and in the
  row count of the received tokens.
* ``ep_moe/layer.py:207`` and ``quantization/unquant.py:837`` distinguish
  the cases via ``DispatchOutputChecker.format_is_deepep_normal`` -- so via
  exactly this format.

WHAT DEEPEP EXCHANGES BEFORE THE DATA, AND IN WHAT ORDER
-----------------------------------------------------------
``deepep.py:559`` ``buffer.get_dispatch_layout(topk_ids, num_experts)``
computes **purely locally** from ``topk_ids``:

1. ``num_tokens_per_rank`` -- how many tokens go to each rank,
2. ``num_tokens_per_rdma_rank`` -- the same per RDMA node (moot here,
   there is one node),
3. ``num_tokens_per_expert`` -- how many tokens per **global** expert,
4. ``is_token_in_rank`` -- the bitmask [T, R].

Only after that (``deepep.py:578``) does ``buffer.dispatch(...)`` run, and
**inside it** is the actual collective over the counts (DeepEP's
``notify_dispatch``), because the receiver cannot know its buffer size
before the sender has counted. Exactly this order is used here: local
decomposition, then **one** ``all_gather`` of the counts over the CPU
group, then the data.

The count exchange is a host collective and needs the numbers on the CPU.
That is the reason this path is **not CUDA-graph-capable** -- the same
reason ``barlink.py:525`` exempts the unevenly split ``all_to_all_single``
there. ``server_args`` therefore turns the graphs off for
``--moe-a2a-backend bar1ep``, the same way it already does for
``deepep_mode=normal``.

WHAT IS ON THE WIRE
--------------------
Two calls per direction, not four:

* **Payload** -- ``[Token, hidden_size]``, handled as ``uint8``. It stays
  untouched and lands row by row exactly where the runner expects it.
* **Metadata** -- ``[Token, topk*8 + topk*4 + scale row]``: local expert
  numbers (int64), weights (float32) and, when fp8 is running, the scale
  row. Three small fields in one block.

Four calls would be four locks; one would mean unpacking the whole payload.
Two is the middle ground, and the small extra copy only hits the metadata.

If a block is larger than an a2a slot, it runs over several rounds. The
round count follows from the **group-wide** maximum over all R*R blocks;
every rank thereby counts the same number of rounds. If two ranks counted
differently, that would be a hang, not an error.

FP8
---
The kernel moves **bytes** (``barlink_bar1_ext.py:1136``, "no dtype, no
reduction"). ``torch.float8_e4m3fn`` therefore needs no special case -- the
payload is handled as ``uint8`` anyway, so no ``index_select`` on an fp8
tensor is needed either. What does **not** travel along on its own is the
scale factors: ``deepep.py:512`` quantizes with
``sglang_per_token_group_quant_fp8(hidden_states, 128, ...)`` and lets
DeepEP carry the pair ``(x_q, x_s)``; here ``x_s`` travels per token in the
metadata block, row for row next to ``topk_ids`` and ``topk_weights``. The
switch settings are taken from ``deepep.py:512``, not reimagined.

WHAT THIS DISPATCHER CANNOT DO
--------------------------------
* **Low-latency form.** ``DEEPEP_LL`` has a different layout (fixed bucket
  size per expert, ``masked_m``, ``expected_m``) and a different runner
  path. Only the normal form is built here. ``--deepep-mode auto`` or
  ``low_latency`` is therefore **rejected**, not silently bent into shape:
  otherwise ``DeepEPMoE`` would compute with LL assumptions and get
  normal-form tensors.
* **NVFP4.** The scales are interleaved there and not contiguous per
  token. Not built, so not offered.
* **More than one node.** The direct path is BAR1 to BAR1 over PCIe.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DeepEPOutputDtype,
    get_deepep_output_dtype,
)

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)


class Bar1EPUnavailable(RuntimeError):
    """The direct path does not carry this dispatcher here.

    Explicitly **no** silent fallback: whoever chose ``bar1ep`` gets either
    BAR1 or an error message with a reason. A fallback that does something
    else and looks like BAR1 would be the worst of all answers -- the
    measurement would then say something about a path nobody chose.
    """


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "no", "off", "false")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

#: The transport methods this dispatcher calls, probed by name before the
#: first call. The names are the ones ``BarlinkBar1Transport`` publishes
#: (``barlink_bar1.py``) -- and they are ONLY here as strings, which is what
#: makes them dangerous: a ``hasattr`` probe survives every rename tool and
#: every import check, so a stale spelling here closes the gate silently and
#: forever. Task #295 renamed ``traegt_a2a``/``a2a_schlitz_bytes`` to
#: ``supports_a2a``/``a2a_slot_bytes``, this probe was not renamed with them,
#: and the BAR1 EP dispatch was unreachable until task #361.
#: ``test_bar1ep_transport_gate.py`` pins every name against the real class.
TRANSPORT_A2A_ATTRS = ("barlink_all_to_all_single", "supports_a2a", "a2a_slot_bytes")

#: Decline reasons already announced in this process. The gate is asked once
#: per MoE layer and once per dispatcher, so an unconditional log line would
#: repeat dozens of times per boot; a set keeps it loud without being noise.
_DECLINE_ANNOUNCED: set = set()


def _declined(reason: str):
    """Announce a closed gate exactly once, then return ``(None, reason)``.

    A gate that declines without a word is how this path died: the condition
    was false on every rank, and nothing in the log said so. The caller that
    asked for ``bar1ep`` explicitly still gets an exception carrying the same
    text (``create_moe_dispatcher``); this line is for the boot where the
    reason would otherwise only exist as a return value nobody prints.
    """
    if reason not in _DECLINE_ANNOUNCED:
        _DECLINE_ANNOUNCED.add(reason)
        logger.warning("bar1ep: BAR1 dispatch path not available -- %s", reason)
    return None, reason


def bar1ep_transport(group_coordinator=None):
    """This group's BAR1 transport, or ``(None, reason)``.

    Every condition is rank-uniform: it hangs off group-wide agreed state
    (environment variables, ``_a2a_proof`` from an ``all_gather_object`` in
    ``barlink_bar1.byte_proof_a2a``, geometry from rank-uniform sizes). Two
    ranks must never answer differently here -- one would run into the
    collective, the other would not, and the result would be a hang instead
    of an error.

    Every decline goes through ``_declined`` and thereby ends up in the log.
    """
    if group_coordinator is None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group_coordinator = get_tp_group()

    comm = getattr(group_coordinator, "barlink_comm", None)
    if comm is None:
        return _declined(
            "barlink is not active (SGLANG_BARLINK=0 or world_size==1). The "
            "BAR1 direct path hangs off the BarlinkCommunicator; without it "
            "there is neither a peer pointer table nor slots."
        )
    if getattr(comm, "disabled", False):
        return _declined("BarlinkCommunicator is disabled (world_size == 1).")
    t = getattr(comm, "transport", None)
    if t is None:
        return _declined(
            "barlink runs on the gloo level -- no transport. "
            "SGLANG_BARLINK_TRANSPORT=bar1 or =matrix selects the direct path."
        )
    missing = [n for n in TRANSPORT_A2A_ATTRS if not hasattr(t, n)]
    if missing:
        return _declined(
            f"Transport {type(t).__name__} has no all_to_all "
            f"({', '.join(missing)} missing). That is not a BAR1 transport."
        )
    slot = int(t.a2a_slot_bytes())
    if slot <= 0:
        return _declined(
            "The BAR1 transport is up, but its a2a byte proof has not "
            "passed (or SGLANG_BARLINK_BAR1_A2A=0). Without a passed proof, "
            "all_to_all opts out -- see barlink_bar1.byte_proof_a2a."
        )
    return t, ""


def bar1ep_available(group_coordinator=None) -> Tuple[bool, str]:
    """``(True, "")`` exactly when the ``bar1ep`` choice may be offered.

    Checks what is checkable **without** the model geometry. The questions
    that can only be answered once ``hidden_size``/``topk`` are known (does
    a row fit into a slot?) and the byte proof live in the dispatcher's
    constructor, because they need the numbers.
    """
    t, reason = bar1ep_transport(group_coordinator)
    return (t is not None), reason


#: A passed self-test per (CPU group, geometry). The test is the
#: prerequisite for the dispatcher offering itself; but it costs startup
#: time, and with TBO active two dispatchers of the same geometry are built,
#: once more per MoE layer. The key contains everything the test actually
#: checks.
_SELFTEST_STATE: dict = {}


def _slice(source: torch.Tensor, off: int, width: int,
           dtype: torch.dtype, columns: int) -> torch.Tensor:
    """A column slice out of a ``uint8`` block, reinterpreted.

    Deliberately via a **fresh** buffer and ``copy_`` instead of
    ``.contiguous().view(dtype)``: with one row (or zero rows), PyTorch
    already considers the column slice contiguous, ``contiguous()`` then
    returns the view with its memory offset, and ``view(dtype)`` thereby
    hangs off an alignment condition that depends on ``topk``. A fresh
    buffer starts at offset 0 -- the condition drops out instead of holding
    almost always.
    """
    n = source.shape[0]
    dest = torch.empty((n, width), dtype=torch.uint8, device=source.device)
    dest.copy_(source[:, off : off + width])
    return dest.view(dtype).reshape(n, columns)


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


class Bar1EPDispatcher(BaseDispatcher):
    """Dispatch/combine over ``bar1_all_to_all``, normal form.

    The state between ``dispatch`` and ``combine`` (sort index, counts,
    round count) hangs off the instance, not off the ``DispatchOutput`` --
    the same solution as ``deepep.py:566`` ("`handle` should be transmitted
    with tokens ... keeping `handle` as a member variable works"), and for
    the same reason: the tuple format is closed, a sixth field would be a
    new format.
    """

    def __init__(
        self,
        group: Optional[torch.distributed.ProcessGroup] = None,
        router_topk: int = None,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        **_unused,
    ):
        super().__init__()

        from sglang.srt.distributed.parallel_state import get_tp_group

        self.group = get_tp_group()
        self.comm = getattr(self.group, "barlink_comm", None)
        self.transport, reason = bar1ep_transport(self.group)
        if self.transport is None:
            raise Bar1EPUnavailable(
                f"--moe-a2a-backend bar1ep selected, but: {reason}"
            )

        if group is not None and group is not getattr(
            self.group, "device_group", None
        ):
            # create_moe_dispatcher passes get_tp_group().device_group in
            # here (fused_moe_triton/layer.py:96/125). If a different group
            # arrived here, the count exchange would run over a different
            # set of ranks than the peer pointer table -- a hang, not an
            # error.
            raise Bar1EPUnavailable(
                "bar1ep only runs on the TP group: the BAR1 transport hangs "
                "off get_tp_group().barlink_comm, and a second group would "
                "have neither peer pointers nor slots."
            )

        self.cpu_group = self.comm.cpu_group
        self.world = int(self.comm.world_size)
        self.rank = int(self.comm.rank)

        self.router_topk = int(router_topk)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.hidden_size = int(hidden_size)
        self.params_dtype = params_dtype or torch.bfloat16
        self.device = torch.device("cuda", torch.cuda.current_device())

        if self.num_experts != self.num_local_experts * self.world:
            raise Bar1EPUnavailable(
                f"num_experts {self.num_experts} is not "
                f"{self.num_local_experts} * {self.world}. bar1ep maps "
                f"expert e onto rank e // num_local_experts -- the same "
                f"mapping as DeepEP; without an equal split, there is none."
            )
        if self.hidden_size % 128 != 0 and self._scales_possible():
            raise Bar1EPUnavailable(
                f"hidden_size {self.hidden_size} is not a multiple of 128; "
                f"the fp8 path's 128-element block quantization "
                f"(deepep.py:512) does not exist for it."
            )

        self.deepep_mode = deepep_mode
        if deepep_mode is not None and not deepep_mode.is_normal():
            raise Bar1EPUnavailable(
                f"bar1ep only builds the normal form (DEEPEP_NORMAL), but "
                f"deepep_mode is {deepep_mode}. The low-latency form has a "
                f"different output format (masked_m/expected_m) and a "
                f"different runner path; silently replacing it with the "
                f"normal form here would mean giving DeepEPMoE normal-form "
                f"tensors under LL assumptions. Please use "
                f"--deepep-mode normal."
            )

        # DeepEP/Mooncake/Nixl mark invalid topk slots with -1; the AITER
        # pre_permute redirects them to a sink slot. Without AITER there is
        # nothing to mask here -- but MaybeTboDeepEPDispatcher reads the
        # field unconditionally (two_batch_overlap.py:1097).
        self.expert_mask_gpu = None

        self.quant_config: Optional[dict] = None
        self.use_fp8 = False
        self._slot = int(self.transport.a2a_slot_bytes())
        self._set_output_dtype()

        # State between dispatch and combine.
        self._send_rows: List[int] = []
        self._recv_rows: List[int] = []
        self._send_index: Optional[torch.Tensor] = None
        self._token_count = 0
        self._max_rows = 0
        self._output_dtype = self.params_dtype
        self._dispatch_state = None
        self._combine_state = None

        self._bar1_dispatch_hooks = DeepEPPDispatchHooks()

        self._check_window()
        self._selftest_if_needed()

    # -- Capability ----------------------------------------------------

    def _scales_possible(self) -> bool:
        return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    def _set_output_dtype(self) -> None:
        dtype = get_deepep_output_dtype(self)
        if dtype == DeepEPOutputDtype.NVFP4:
            raise Bar1EPUnavailable(
                "bar1ep does not carry nvfp4: its scales are interleaved "
                "and not contiguous per token. Not built means not offered."
            )
        if dtype == DeepEPOutputDtype.INT8:
            raise Bar1EPUnavailable("bar1ep does not carry int8 dispatch (NPU path).")
        # Like deepep.py:510: quantization only happens when DeepGEMM
        # actually computes the fp8 path. Without it, DeepEP also runs bf16.
        self.use_fp8 = (
            dtype == DeepEPOutputDtype.FP8 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        )

    def _scale_dtype(self) -> torch.dtype:
        return torch.int32 if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 else torch.float32

    def _scale_columns(self) -> int:
        """Columns of the scale tensor per token -- 0 without fp8.

        The numbers are taken from ``fp8_kernel.py:488-511``: with ue8m0 the
        quantizer packs four scales into one ``int32`` and aligns to four,
        otherwise it is one ``float32`` per 128-element block. ``ep_scatter``
        checks exactly this column count (``kernels.py:1104``).
        """
        if not self.use_fp8:
            return 0
        s = self.hidden_size // 128
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            return -(-s // 4)
        return s

    def _payload_row_bytes(self) -> int:
        e = 1 if self.use_fp8 else (torch.finfo(self.params_dtype).bits // 8)
        return self.hidden_size * e

    def _meta_row_bytes(self) -> int:
        # topk_ids int64, topk_weights float32, plus the scale row (int32
        # or float32 -- both four bytes).
        return self.router_topk * 8 + self.router_topk * 4 + self._scale_columns() * 4

    def _check_window(self) -> None:
        """Does even ONE row fit into a slot?

        The question is not academic: the slot is ``chunk_max`` from
        ``barlink_bar1.geometry`` and coincides with the window size. A row
        that does not fit cannot be split into rounds either -- the split
        divides rows, not row contents. Then the dispatcher declines instead
        of failing later on the hot path.
        """
        self._slot = int(self.transport.a2a_slot_bytes())
        for name, rb in (
            ("payload", self._payload_row_bytes()),
            ("metadata", self._meta_row_bytes()),
        ):
            if rb > self._slot:
                raise Bar1EPUnavailable(
                    f"A {name} row is {rb} bytes and does not fit into the "
                    f"a2a slot of {self._slot} bytes. A larger window "
                    f"(SGLANG_BARLINK_BAR1_WINDOW_MIB) is needed -- a "
                    f"fallback here would not be a solution, but a "
                    f"different measurement under the same name."
                )

    def _rows_per_round(self, row_bytes: int) -> int:
        return max(1, int(self._slot // max(1, row_bytes)))

    # -- The data path ---------------------------------------------------

    def _a2a_rows(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        send_rows: List[int],
        recv_rows: List[int],
        row_bytes: int,
        max_rows: int,
        rows_per_round: Optional[int] = None,
    ) -> torch.Tensor:
        """``all_to_all`` with uneven blocks, over several rounds if needed.

        ``inp``/``out`` are contiguous ``uint8`` tensors of shape
        ``[rows, row_bytes]``. ``max_rows`` is the **group-wide** maximum
        over all R*R blocks; the round count follows from it, and because
        that number is the same group-wide, every rank counts the same
        number of rounds.

        The offsets are computed and **passed in**, instead of letting the
        seam guess them as a prefix sum: one round moves only one piece out
        of each block, and the blocks stay put where they are. This is
        exactly why ``barlink_bar1.barlink_all_to_all_single`` has taken
        ``send_offsets``/``recv_offsets`` since that change; the kernel has
        always kept offsets and lengths separate anyway.
        """
        R = self.world
        rpr = rows_per_round or self._rows_per_round(row_bytes)
        rounds = max(1, -(-int(max_rows) // rpr))

        s_base, acc = [], 0
        for n in send_rows:
            s_base.append(acc)
            acc += int(n)
        e_base, acc = [], 0
        for n in recv_rows:
            e_base.append(acc)
            acc += int(n)

        inp_flat = inp.reshape(-1)
        out_flat = out.reshape(-1)
        # An empty tensor has data_ptr() == 0. If BOTH sides are empty, they
        # point at the same address, and the extension rejects that ("in and
        # out must not be the same", barlink_bar1_ext.py:1209) -- rightly
        # so, since it cannot know that there is nothing to move here.
        # Bailing out is still not allowed: the other ranks wait on my flag
        # in the same lock, even if I send zero bytes. So two placeholders;
        # all lengths are 0, nothing is read from them and nothing is
        # written into them.
        if inp_flat.numel() == 0:
            inp_flat = torch.zeros(16, dtype=torch.uint8, device=inp.device)
        if out_flat.numel() == 0:
            out_flat = torch.zeros(16, dtype=torch.uint8, device=out.device)

        for k in range(rounds):
            s_off, s_len, e_off, e_len = [], [], [], []
            for j in range(R):
                a = min(k * rpr, int(send_rows[j]))
                b = min((k + 1) * rpr, int(send_rows[j]))
                s_off.append((s_base[j] + a) * row_bytes)
                s_len.append((b - a) * row_bytes)
                a = min(k * rpr, int(recv_rows[j]))
                b = min((k + 1) * rpr, int(recv_rows[j]))
                e_off.append((e_base[j] + a) * row_bytes)
                e_len.append((b - a) * row_bytes)
            largest = max(s_len + e_len)
            if not self.transport.supports_a2a(largest):
                raise Bar1EPUnavailable(
                    f"Round {k}: largest block {largest} bytes does not fit "
                    f"into the slot of {self._slot} bytes. The round split "
                    f"should have prevented this -- this line is proof that "
                    f"it did not."
                )
            self.transport.barlink_all_to_all_single(
                self.comm, out_flat, inp_flat, s_len, e_len, s_off, e_off,
            )
        return out

    def _decompose(self, topk_ids: torch.Tensor):
        """The local decomposition -- the counterpart to ``get_dispatch_layout``.

        Returns ``(is_token_in_rank, num_tokens_per_rank,
        num_tokens_per_expert)``. ``topk_ids`` may contain ``-1`` (invalid
        slots, the way the DeepEP family marks them); those do not count
        anywhere.
        """
        T = topk_ids.shape[0]
        R, nle = self.world, self.num_local_experts
        valid = topk_ids >= 0
        dest = torch.where(
            valid,
            torch.div(topk_ids, nle, rounding_mode="floor"),
            torch.zeros_like(topk_ids),
        )
        # scatter_add_ instead of scatter_: if several topk slots land on
        # the same rank, scatter_ would overwrite in an unspecified order,
        # and an invalid slot (dest 0) could erase a real hit.
        counts = torch.zeros((T, R), dtype=torch.int32, device=topk_ids.device)
        counts.scatter_add_(1, dest, valid.to(torch.int32))
        in_rank = counts > 0
        ntpr = in_rank.sum(dim=0).to(torch.int64)
        ntpe = torch.bincount(
            topk_ids[valid].reshape(-1), minlength=self.num_experts
        ).to(torch.int64)[: self.num_experts]
        return in_rank, ntpr, ntpe

    def _exchange_counts(self, ntpr: torch.Tensor, ntpe: torch.Tensor):
        """An ``all_gather`` over the CPU group. The only host sync.

        Exactly the step DeepEP runs in ``notify_dispatch``, for the same
        reason: the receiver cannot know its buffer size before the sender
        has counted. It happens **before** the data path, not inside it.
        Row ``i`` is ``[num_tokens_per_rank (R), num_tokens_per_expert
        (num_experts)]`` of rank ``i``.
        """
        flat = torch.cat([ntpr, ntpe]).to("cpu")
        gathered = [torch.empty_like(flat) for _ in range(self.world)]
        dist.all_gather(gathered, flat, group=self.cpu_group)
        return [t.tolist() for t in gathered]

    # -- Outbound ----------------------------------------------------------

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: "TopKOutput"
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._bar1_dispatch_hooks is not None:
            self._bar1_dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: "TopKOutput"):
        self._dispatch_state = self._dispatch_prepare(hidden_states, topk_output)

    def dispatch_b(self, *state) -> DispatchOutput:
        if not state:
            state = self._dispatch_state
            self._dispatch_state = None
        return self._dispatch_core(*state)

    def _dispatch_prepare(
        self, hidden_states: torch.Tensor, topk_output: "TopKOutput"
    ):
        """Everything up to the data path: quantize, decompose, exchange counts."""
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        self._output_dtype = hidden_states.dtype
        hidden_states = hidden_states.contiguous()

        scale = None
        if self.use_fp8:
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            # The same switches as deepep.py:512 -- not similar ones.
            hidden_states, scale = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            # With ue8m0, the quantizer lays out the scale tensor
            # column-major (fp8_kernel.py:499 `.transpose(-1,-2)`). For the
            # transport the row needs to be contiguous; ep_scatter reads
            # over strides anyway and accepts both forms.
            scale = scale.contiguous()

        in_rank, ntpr, ntpe = self._decompose(topk_ids)
        matrix = self._exchange_counts(ntpr, ntpe)
        return (hidden_states, scale, topk_ids, topk_weights, in_rank, matrix)

    def _dispatch_core(
        self, hidden_states, scale, topk_ids, topk_weights, in_rank, matrix
    ) -> DispatchOutput:
        R, nle, K = self.world, self.num_local_experts, self.router_topk
        T = hidden_states.shape[0]
        device = hidden_states.device

        send_rows = [int(matrix[self.rank][j]) for j in range(R)]
        recv_rows = [int(matrix[i][self.rank]) for i in range(R)]
        max_rows = max(int(matrix[i][j]) for i in range(R) for j in range(R))
        S, N = sum(send_rows), sum(recv_rows)

        # Send order: per destination rank, own token numbers ascending, the
        # destination ranks ascending one after another. The receiver
        # thereby knows them without a single transmitted index byte -- and
        # the return path finds the same order.
        if T:
            positions = torch.nonzero(
                in_rank.t().reshape(-1), as_tuple=False
            ).reshape(-1)
            send_index = torch.remainder(positions, T)
        else:
            send_index = torch.zeros(0, dtype=torch.int64, device=device)

        # -- Payload: handled as bytes, so fp8 needs no special case and no
        #    index_select on an fp8 tensor.
        prb = self._payload_row_bytes()
        x_bytes = hidden_states.view(torch.uint8).reshape(T, prb)
        send_x = x_bytes[send_index]
        recv_x_bytes = torch.empty((N, prb), dtype=torch.uint8, device=device)

        # -- Metadata: local expert numbers, weights, scale row.
        owner = torch.repeat_interleave(
            torch.arange(R, device=device, dtype=torch.int64),
            torch.tensor(send_rows, device=device, dtype=torch.int64),
        )
        ids_raw = topk_ids[send_index]
        matches = (
            torch.div(ids_raw, nle, rounding_mode="floor") == owner.unsqueeze(1)
        )
        ids_local = torch.where(
            (ids_raw >= 0) & matches,
            ids_raw - owner.unsqueeze(1) * nle,
            torch.full_like(ids_raw, -1),
        ).contiguous()
        weights = topk_weights[send_index].contiguous()

        parts = [
            ids_local.view(torch.uint8).reshape(S, K * 8),
            weights.view(torch.uint8).reshape(S, K * 4),
        ]
        scale_bytes = self._scale_columns() * 4
        if scale_bytes:
            parts.append(
                scale[send_index].contiguous().view(torch.uint8).reshape(S, scale_bytes)
            )
        mrb = self._meta_row_bytes()
        send_meta = torch.cat(parts, dim=1)
        recv_meta = torch.empty((N, mrb), dtype=torch.uint8, device=device)

        self._a2a_rows(recv_x_bytes, send_x, send_rows, recv_rows, prb, max_rows)
        self._a2a_rows(
            recv_meta, send_meta, send_rows, recv_rows, mrb, max_rows
        )

        # -- Unpacking.
        recv_ids = _slice(recv_meta, 0, K * 8, torch.int64, K)
        recv_weights = _slice(recv_meta, K * 8, K * 4, torch.float32, K)
        recv_scale = (
            _slice(
                recv_meta, K * 12, scale_bytes, self._scale_dtype(), self._scale_columns()
            )
            if scale_bytes
            else None
        )
        recv_x = recv_x_bytes.view(
            torch.float8_e4m3fn if self.use_fp8 else self._output_dtype
        ).reshape(N, self.hidden_size)

        # -- Counts per local expert. A CPU list, the way the runner needs
        #    it (moe_runner/deep_gemm.py:797 `sum(...)`).
        raw_counts = [0] * nle
        for i in range(R):
            row = matrix[i]
            for e in range(nle):
                raw_counts[e] += int(row[R + self.rank * nle + e])
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            # ep_scatter checks m_indices.shape[0] % 128 == 0 -- the same
            # alignment that deepep.py:589 passes as expert_alignment.
            num_recv_tokens_per_expert = [(-(-c // 128)) * 128 for c in raw_counts]
        else:
            num_recv_tokens_per_expert = raw_counts

        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=torch.tensor(send_rows, dtype=torch.int64),
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=torch.tensor(
                [int(x) for x in matrix[self.rank][R:]], dtype=torch.int64
            ),
        )

        self._send_rows = send_rows
        self._recv_rows = recv_rows
        self._send_index = send_index
        self._token_count = T
        self._max_rows = max_rows

        return DeepEPNormalDispatchOutput(
            recv_x, recv_scale, recv_ids, recv_weights, num_recv_tokens_per_expert
        )

    # -- Return path ---------------------------------------------------------

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(self, combine_input: CombineInput):
        hidden_states, topk_ids, topk_weights = combine_input
        self._combine_state = (hidden_states,)

    def combine_b(self, *state) -> torch.Tensor:
        if not state:
            state = self._combine_state
            self._combine_state = None
        return self._combine_core(*state)

    def _combine_core(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._send_index is None:
            raise RuntimeError(
                "bar1ep: combine without a preceding dispatch. The return "
                "path's sort index is created in the outbound path."
            )
        T = self._token_count
        H = hidden_states.shape[1]
        device, dtype = hidden_states.device, hidden_states.dtype
        row_bytes = H * (torch.finfo(dtype).bits // 8)

        # Return path: the same machinery in the opposite direction. What
        # was received on the outbound path is now sent -- block by block,
        # in the same order, so without a single index byte on the wire.
        send_rows = list(self._recv_rows)
        recv_rows = list(self._send_rows)
        S = sum(recv_rows)

        inp = hidden_states.contiguous().view(torch.uint8).reshape(-1, row_bytes)
        out = torch.empty((S, row_bytes), dtype=torch.uint8, device=device)
        self._a2a_rows(out, inp, send_rows, recv_rows, row_bytes, self._max_rows)
        returned = out.view(dtype).reshape(S, H)

        # The reduction: a token that had experts on several ranks gets one
        # contribution per rank. index_add_ over the outbound path's sort
        # index sums them. In float32, because DeepEP's combine kernel also
        # does so -- summing in bf16 would be cheaper and a different number.
        if _env_flag("SGLANG_BAR1EP_COMBINE_FP32", "1"):
            acc = torch.zeros((T, H), dtype=torch.float32, device=device)
            acc.index_add_(0, self._send_index, returned.to(torch.float32))
            result = acc.to(dtype)
        else:
            result = torch.zeros((T, H), dtype=dtype, device=device)
            result.index_add_(0, self._send_index, returned)

        self._send_index = None
        return result

    # -- Framework -----------------------------------------------------------

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        self.quant_config = quant_config
        self._set_output_dtype()
        self._check_window()

    def set_overlap_args(
        self, combine_overlap_args: "CombineOverlapArgs", meta_overlap_args: dict
    ) -> None:
        # The direct path has no second queue and no receive hook:
        # dispatch/combine are each one kernel launch with one lock. The
        # overlap arguments are accepted (the framework sets them
        # unconditionally) and deliberately unused -- taking them and
        # ignoring them is more honest than a stub that looks like overlap.
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)

    def register_deepep_dispatch_hook(self, hook):
        return self._bar1_dispatch_hooks.register_hook(hook)

    # -- Byte proof ------------------------------------------------------

    def _selftest_if_needed(self) -> None:
        if not _env_flag("SGLANG_BAR1EP_SELFTEST", "1"):
            logger.warning(
                "bar1ep: byte proof skipped via SGLANG_BAR1EP_SELFTEST=0. "
                "With that, no number from this run carries any statement "
                "about whether the bytes actually arrive."
            )
            return
        key = (
            id(self.cpu_group),
            self.hidden_size,
            self.router_topk,
            self.num_experts,
            bool(self.use_fp8),
            str(self.params_dtype),
        )
        state = _SELFTEST_STATE.get(key)
        if state is True:
            return
        if state is False:
            raise Bar1EPUnavailable(
                "bar1ep: the byte proof already failed in this process."
            )
        ok, reason = self.byte_proof()
        _SELFTEST_STATE[key] = ok
        if not ok:
            raise Bar1EPUnavailable(f"bar1ep: byte proof failed -- {reason}")

    def byte_proof(self) -> Tuple[bool, str]:
        """The proof without which the dispatcher does not offer itself.

        Three passes, all over the REAL path:

        1. **Raw bytes, uneven and unaligned, over several rounds.** Block
           lengths ``97*(1+((q+z)%3)) + ((q*5+z*3)%7)`` -- the factor makes
           the blocks uneven (the MoE normal case), the summand makes them
           non-multiples of 16 and thereby pushes every following offset
           out of alignment. The round count is forced to at least three,
           so that double buffering (``round & 1``) and the offset
           computation actually run and do not merely exist. Checked per
           sender individually, byte by byte, on the RECEIVING card -- even
           the sender's own block, which does not go over the aperture at
           all.
        2. **Dispatch, structured, per directed pair.** The token -> expert
           mapping follows a rule that every rank can recompute for every
           other rank. That way every receiver knows in advance which row
           with what content must come from whom -- so the check does not
           hang off the same bookkeeping it is supposed to check. The
           content depends on (source rank, token number, column): a
           swapped row and a shifted column both show up. Runs in the
           **configured** form; with fp8 it is compared byte for byte
           against the locally quantized expected form, scales included.
        3. **Combine, structured.** The return path gets back exactly what
           the outbound path brought. Then ``combine(dispatch(x))`` must
           equal ``x * (number of ranks responsible for this token)`` -- a
           closed formula, not a second rebuild of the same bookkeeping.

        If any of this fails, **bar1ep** declines. ``all_reduce`` and the
        barlink seam's a2a are unaffected: they have their own proofs, and a
        failed dispatcher implies nothing about them.
        """
        ok, reason = True, ""
        try:
            ok, reason = self._proof_raw_bytes()
            if ok:
                ok, reason = self._proof_structure()
        except Exception as ex:  # noqa: BLE001 -- reason goes into the log
            ok, reason = False, repr(ex)
            logger.warning("bar1ep: byte proof aborted: %r", ex)

        # From here on group-wide, IN EVERY CASE. A rank that bails out
        # before the all_gather_object leaves the others standing in it --
        # a failed proof would turn into a hang, and a hang does not say
        # what is broken.
        carrier: list = [None] * self.world
        dist.all_gather_object(carrier, (bool(ok), str(reason)), group=self.cpu_group)
        bad = [i for i, (o, _) in enumerate(carrier) if not o]
        if bad:
            reasons = "; ".join(f"rank {i}: {carrier[i][1]}" for i in bad)
            logger.warning("bar1ep: byte proof failed group-wide -- %s", reasons)
            return False, reasons
        logger.info(
            "bar1ep: byte proof passed (raw bytes uneven/unaligned over "
            "several rounds; dispatch per directed pair against the "
            "routing rule; combine against the closed formula) -- %d "
            "ranks, hidden=%d, topk=%d, fp8=%s.",
            self.world, self.hidden_size, self.router_topk, self.use_fp8,
        )
        return True, ""

    @staticmethod
    def _mark(source: int, dest: int) -> int:
        """A byte that differs per directed pair, never 0x00 and never 0xFF.

        0xFF is the output buffer's preset, 0x00 the receive slot's; both
        are thereby distinguishable from the pattern, and a block that was
        NOT written shows up as such, instead of randomly looking like a
        hit.
        """
        return 0x40 | ((source * 8 + dest) & 0x3F)

    def _proof_raw_bytes(self) -> Tuple[bool, str]:
        R, r = self.world, self.rank

        def length(q: int, z: int) -> int:
            return 97 * (1 + ((q + z) % 3)) + ((q * 5 + z * 3) % 7)

        send = [length(r, z) for z in range(R)]
        recv = [length(q, r) for q in range(R)]
        max_rows = max(length(q, z) for q in range(R) for z in range(R))
        # Row width 1 byte: then block lengths equal row counts and the
        # offsets are consistently unaligned. The round count is forced,
        # instead of falling out of a slot that on this rig would be enough
        # for everything at once.
        rpr = max(1, max_rows // 3)

        inp = torch.empty((sum(send), 1), dtype=torch.uint8, device=self.device)
        o = 0
        for z in range(R):
            inp[o : o + send[z]] = self._mark(r, z)
            o += send[z]
        out = torch.full((sum(recv), 1), 0xFF, dtype=torch.uint8, device=self.device)

        dist.barrier(group=self.cpu_group)
        self._a2a_rows(out, inp, send, recv, 1, max_rows, rows_per_round=rpr)
        torch.cuda.synchronize(self.device)

        back = out.reshape(-1).cpu()
        o = 0
        for q in range(R):
            expected = self._mark(q, r)
            chunk = back[o : o + recv[q]]
            bad = int((chunk != expected).sum().item())
            if bad:
                return False, (
                    f"raw bytes {q}->{r}: {bad} of {recv[q]} bytes wrong"
                )
            o += recv[q]
        return True, ""

    def _probe_routes(self, q: int) -> List[List[int]]:
        """Rank ``q``'s topk rows in the probe. Purely computed.

        Two blocks:

        * Block 1 makes the **pairs** uneven and none empty: for each
          destination ``z`` exactly ``1 + ((q*3+z*5) % 5)`` tokens that go
          ONLY there. That way every one of the R*R directed pairs carries
          bytes -- a proof over an empty pair would be no proof.
        * Block 2 makes the **multiple assignment**: ``R`` tokens, each of
          which hits ``min(topk, R)`` different ranks. Without this block,
          every token would get exactly one contribution, and the combine
          test would be blind to the sum.
        """
        R, nle, K = self.world, self.num_local_experts, self.router_topk
        rows: List[List[int]] = []
        for z in range(R):
            for _ in range(1 + ((q * 3 + z * 5) % 5)):
                row = [-1] * K
                for k in range(min(K, nle)):
                    row[k] = z * nle + k
                rows.append(row)
        for j in range(R):
            row = [-1] * K
            for k in range(min(K, R)):
                z = (j + k) % R
                row[k] = z * nle + ((j + k) % nle)
            rows.append(row)
        return rows

    @staticmethod
    def _probe_value(q: int, t: int, columns: int, device) -> torch.Tensor:
        """A probe row's content -- different per (rank, token, column).

        Integers below 128: exact in bf16, exact in float32, and after the
        128-element block quantization the same bit pattern on every rank,
        because the quantization is independent per row and per block.
        """
        s = torch.arange(columns, device=device, dtype=torch.int32)
        return (q * 131 + t * 17 + s) % 113

    def _proof_structure(self) -> Tuple[bool, str]:
        R, r, nle, K = self.world, self.rank, self.num_local_experts, self.router_topk
        H, device = self.hidden_size, self.device

        routes = {q: self._probe_routes(q) for q in range(R)}
        mine = routes[r]
        T = len(mine)

        topk_ids = torch.tensor(mine, dtype=torch.int64, device=device)
        topk_weights = (
            torch.arange(T * K, device=device, dtype=torch.float32).reshape(T, K) % 7.0
        ) + 1.0
        pattern = torch.stack(
            [self._probe_value(r, t, H, device) for t in range(T)], dim=0
        )
        x = pattern.to(torch.bfloat16 if self.use_fp8 else self.params_dtype)

        class _ProbeTopK:
            pass

        tk = _ProbeTopK()
        tk.topk_ids = topk_ids
        tk.topk_weights = topk_weights

        dist.barrier(group=self.cpu_group)
        state = self._dispatch_prepare(x, tk)
        out = self._dispatch_core(*state)
        torch.cuda.synchronize(self.device)

        # What must have arrived? Every rank recomputes every rank's send
        # order from the rule -- independent of the bookkeeping currently
        # under test.
        expected: List[Tuple[int, int]] = []  # (source rank, token number)
        for q in range(R):
            for t, row in enumerate(routes[q]):
                if any(e >= 0 and e // nle == r for e in row):
                    expected.append((q, t))
        N = out.hidden_states.shape[0]
        if N != len(expected):
            return False, f"dispatch: {N} rows received, expected {len(expected)}"

        recv_ids = out.topk_ids.cpu()
        for p, (q, t) in enumerate(expected):
            expected_ids = [
                (e - r * nle) if (e >= 0 and e // nle == r) else -1
                for e in routes[q][t]
            ]
            if recv_ids[p].tolist() != expected_ids:
                return False, (
                    f"dispatch: row {p} (from rank {q}, token {t}) carries "
                    f"topk_ids {recv_ids[p].tolist()}, expected {expected_ids}"
                )

        # Payload and scales, byte for byte, against the locally built
        # expected form. This is not a second rebuild of the bookkeeping:
        # only the EXPECTED input of the foreign ranks is built (from the
        # rule), quantized with the same kernel, and the bytes are compared.
        expected_x, expected_scale = self._probe_expected(expected)
        actual_x = out.hidden_states.view(torch.uint8).reshape(N, -1)
        bad = int((actual_x != expected_x).sum().item())
        if bad:
            return False, (
                f"dispatch: {bad} of {actual_x.numel()} payload bytes wrong"
            )
        if (expected_scale is None) != (out.hidden_states_scale is None):
            return False, (
                f"dispatch: scale expected={expected_scale is not None}, "
                f"arrived={out.hidden_states_scale is not None}"
            )
        if expected_scale is not None:
            actual_scale = out.hidden_states_scale.contiguous().view(torch.uint8)
            bad = int((actual_scale != expected_scale.view(torch.uint8)).sum().item())
            if bad:
                return False, (
                    f"dispatch: {bad} of {actual_scale.numel()} scale bytes wrong"
                )

        # -- Combine: back exactly what came in.
        returned = (
            torch.stack(
                [
                    self._probe_value(q, t, H, device).to(torch.bfloat16)
                    for (q, t) in expected
                ],
                dim=0,
            )
            if N
            else torch.zeros((0, H), dtype=torch.bfloat16, device=device)
        )
        result = self.combine(
            DeepEPNormalCombineInput(returned, out.topk_ids, out.topk_weights)
        )
        torch.cuda.synchronize(self.device)

        ranks_per_token = torch.tensor(
            [len({e // nle for e in row if e >= 0}) for row in mine],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        expected_result = (pattern.to(torch.float32) * ranks_per_token).to(result.dtype)
        bad = int((result != expected_result).sum().item())
        if bad:
            return False, (
                f"combine: {bad} of {expected_result.numel()} values wrong "
                f"(expected input * number of responsible ranks)"
            )
        return True, ""

    def _probe_expected(self, expected):
        """The expected form of the received payload, built locally.

        Without fp8, that is the rule itself. With fp8, the rule is run
        through the SAME quantizer, with the same switches -- it operates
        independently per row and per 128-element block, so the result is
        the same bit pattern on every rank as on the sender's.
        """
        H, device = self.hidden_size, self.device
        if not expected:
            empty_x = torch.empty(
                (0, self._payload_row_bytes()), dtype=torch.uint8, device=device
            )
            if not self.use_fp8:
                return empty_x, None
            return empty_x, torch.empty(
                (0, self._scale_columns()), dtype=self._scale_dtype(), device=device
            )
        raw = torch.stack(
            [self._probe_value(q, t, H, device) for (q, t) in expected], dim=0
        )
        if not self.use_fp8:
            w = raw.to(self.params_dtype).contiguous()
            return w.view(torch.uint8).reshape(len(expected), -1), None

        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        q8, s8 = sglang_per_token_group_quant_fp8(
            raw.to(torch.bfloat16).contiguous(),
            128,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        return (
            q8.contiguous().view(torch.uint8).reshape(len(expected), -1),
            s8.contiguous(),
        )
