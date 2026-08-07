"""Boot-time preconditions for a token-sharded (DCP) decode server.

Background (task #631, Route A: a PP prefill group handing KV to a
TP+NEXTN decode group).

The decode-side owner rule is not read from the process groups on every
use. ``attn_dcp_size`` / ``attn_dcp_rank`` are read ONCE in the attention
backend constructor (``layers/attention/triton_backend.py``, and the same
pattern in the flashinfer backend) and cached on the backend instance. The
values they read come from ``ParallelContext`` in
``sglang/srt/runtime_context.py``, whose ``dcp_enabled`` getter is::

    if get_dcp_group_no_assert() is None:
        return False
    return self.dcp_size > 1

and ``attn_dcp_size`` is ``self.dcp_size if self.dcp_enabled else 1``.

So if the DCP process group has not been built at the moment a backend is
constructed, the backend does not raise and does not hang. It caches
``dcp_size=1, dcp_rank=0``. Downstream, ``uneven_dcp_owner_bounds()``
returns ``None`` on EVERY rank, the owner rule is bypassed, and every rank
treats every global slot as one of its own local rows: all ranks write the
same token to the same row (last write wins) and read the whole sequence as
local. The result is silently wrong output. There is no error and no hang
to follow.

The same read also decides which branch the PD decode path takes in
``disaggregation/decode.py``: with the bounds ``None`` and ``dcp_enabled``
False, neither the ``owned_ordinals`` branch nor the "stock head-sharded
DCP receive is not supported" refusal fires, so the handover proceeds
unfiltered and equally silently.

The guard below turns that into a boot-time refusal. It compares the
RESOLVED request (``server_args.dcp_size``, already normalized by the
uneven-TP resolution in ServerArgs) against the value a backend
constructed right now would actually cache. It is a no-op whenever the two
agree, which is every configuration where the ordinary
``init_torch_distributed`` -> ``initialize_model_parallel`` ->
``init_attention_backends`` order holds. In particular a PP prefill group
runs with ``dcp_size == 1`` and no DCP group, so both sides read 1 and the
guard passes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _requested_dcp_size(server_args: Any) -> int:
    try:
        return max(1, int(getattr(server_args, "dcp_size", 1) or 1))
    except (TypeError, ValueError):
        return 1


def assert_dcp_group_formed(server_args: Any, *, where: str) -> None:
    """Refuse to build attention backends against an unformed DCP group.

    ``where`` names the call site and is echoed in the message so the
    failure points at the construction step that is out of order rather
    than at this function.

    Raises ``RuntimeError`` when the DCP size a backend would cache differs
    from the size the boot recipe asked for. Returns None otherwise.
    """
    from sglang.srt.distributed import parallel_state
    from sglang.srt.runtime_context import get_parallel

    requested = _requested_dcp_size(server_args)
    parallel = get_parallel()
    # Read exactly what the backend constructor reads. Do NOT read
    # ``parallel.dcp_size`` here: that one asserts the group exists, so it
    # would report an unformed group as a bare AssertionError from
    # parallel_state instead of naming the ordering problem.
    effective = parallel.attn_dcp_size

    if effective == requested:
        return

    group_built = parallel_state.get_dcp_group_no_assert() is not None
    raise RuntimeError(
        f"DCP group precondition failed at {where}: the boot recipe asked "
        f"for dcp_size={requested}, but an attention backend constructed "
        f"now would cache attn_dcp_size={effective} "
        f"(dcp process group built={group_built}). "
        "The attention backend reads attn_dcp_size/attn_dcp_rank once in "
        "its constructor and caches them, so building a backend in this "
        "state does not fail -- it silently disables the DCP owner rule "
        "(uneven_dcp_owner_bounds() returns None on every rank) and every "
        "rank then writes every token to the same KV row and reads the "
        "whole sequence as local. Build the DCP process group "
        "(initialize_model_parallel with decode_context_parallel_size="
        f"{requested}) before constructing attention backends."
    )


def _worker_page_size(tp_worker: Any) -> Any:
    """Resolved allocator page_size of a worker, or None if not reachable.

    ``TpModelWorker.get_memory_pool()`` returns ``(req_to_token_pool,
    token_to_kv_pool_allocator)``. Best-effort by design: a guard must not
    be the thing that breaks a boot, and None simply leaves the check to
    the existing request-path refusal.
    """
    try:
        return tp_worker.get_memory_pool()[1].page_size
    except Exception:  # noqa: BLE001 - any failure means "cannot check"
        return None


def assert_pd_decode_dcp_supported(server_args: Any, *, page_size: Any = None) -> None:
    """Fail a token-sharded PD decode server at boot, not at first request.

    Three combinations are rejected by name deep in the request path today:

    * ``disaggregation/decode.py`` raises "Stock head-sharded DCP receive is
      not supported" when a decode server has ``dcp_enabled`` but is not on
      the uneven-TP replicated-KV layout.
    * only the mooncake sender implements the ``owned_ordinals`` filter;
      the nixl and mori senders raise on a non-None ``owned_ordinals``.
    * the same block requires ``page_size == 1``, because the owner rule is
      defined on single-token slots. ``--page-size`` resolves late (it
      defaults to None and is filled in per backend/model), so this one
      cannot be read off the command line and is passed in resolved: the
      caller supplies ``token_to_kv_pool_allocator.page_size``, which is
      the exact attribute ``disaggregation/decode.py`` reads.

    All are correct refusals, but all arrive on the first transferred
    request, after weights are loaded and the server reports healthy. This
    hoists them to boot.

    Only ever fires for ``--disaggregation-mode decode`` with
    ``dcp_size > 1``; a NULL-mode server and a PP prefill server both
    return immediately, so the default path is untouched.
    """
    from sglang.srt.distributed.utils import get_tp_partition_ratios

    if getattr(server_args, "disaggregation_mode", "null") != "decode":
        return
    requested = _requested_dcp_size(server_args)
    if requested <= 1:
        return

    if get_tp_partition_ratios() is None:
        raise RuntimeError(
            f"PD decode server with dcp_size={requested} requires the "
            "uneven-TP replicated-KV layout (--rank-tp-ratio with "
            "dcp_size == tp_size). Stock head-sharded DCP receive is "
            "refused by disaggregation/decode.py once the first request "
            "is transferred; refusing at boot instead."
        )

    backend = getattr(server_args, "disaggregation_transfer_backend", "mooncake")
    if backend != "mooncake":
        raise RuntimeError(
            f"PD decode server with dcp_size={requested} requires "
            "--disaggregation-transfer-backend mooncake: only the mooncake "
            "sender implements the owned_ordinals row filter that a "
            "token-sharded decode pool needs. The nixl and mori senders "
            f"refuse it by name on the first transfer. Got "
            f"'{backend}'."
        )

    if page_size is not None and int(page_size) != 1:
        raise RuntimeError(
            f"PD decode server with dcp_size={requested} requires "
            f"page_size == 1, got {int(page_size)}. The DCP owner rule "
            "(L % S in [lo, hi)) is defined on single-token slots, so a "
            "paged allocator has no owner for a page. disaggregation/"
            "decode.py refuses this on the first transferred request; "
            "refusing at boot instead. Pass --page-size 1."
        )

    logger.info(
        "PD decode DCP preconditions satisfied: dcp_size=%d, uneven-TP "
        "replicated-KV layout installed, transport=mooncake, page_size=%s.",
        requested,
        page_size if page_size is not None else "unchecked",
    )
