from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

# Pool keyed by (lane, name, numel, dtype, device); see share_input_buffer.
_PoolKey = Tuple[Optional[int], str, int, torch.dtype, torch.device]
_forward_input_buffer_pool: Dict[_PoolKey, torch.Tensor] = {}


def _pool_debug() -> bool:
    return os.environ.get("SGLANG_DEBUG_INPUT_BUFFER_POOL", "0") == "1"


def _pool_ignores_lane() -> bool:
    """Escape hatch that restores the pre-#274 process-wide key.

    It exists so the defect below stays reproducible on demand: it is the
    falsifier for the fix, not an operating mode. Turning it on with a
    CONCURRENT lane re-arms the aliasing crash.
    """
    return os.environ.get("SGLANG_LANE_SHARED_INPUT_BUFFERS", "0") == "1"


def share_input_buffer(name: str, new_buffer: torch.Tensor) -> torch.Tensor:
    """Coalesce a buffer by ``(lane, name, size, dtype, device)`` into the
    process-wide input-buffer pool.

    Distinct callers that request the same field ``name`` with the same
    size/dtype/device share one physical allocation (and therefore one
    ``data_ptr``): the first registrant's buffer becomes canonical and every
    later identical request is returned as a view aliased onto it. Requests
    that differ in size get their own allocation — they never reuse or displace
    an existing entry — so the sharing *structure* is independent of
    registration order and no already-captured buffer is ever repointed.

    This pool governs *every* ``share_buffers()`` caller — including graph
    runners not yet on the registry (the speculative draft / draft-extend /
    frozen-kv-mtp / multi-layer-eagle runners), which register identically-named
    ``input_ids`` / ``positions`` / ``out_cache_loc`` / ``mrope_positions``.
    Cross-runner sharing rests on ONE invariant: the buffers are filled
    immediately before each replay, and the forwards that use them are
    sequential / mutually exclusive.

    KEYED BY LANE (#274), because a concurrent dual-group lane is exactly the
    case that breaks that invariant. The lane's runner is a second full set of
    graph runners in the SAME process on the SAME card, and its prefill tier
    ladder tops out at the serving group's `chunked_prefill_size` — so
    ``out_cache_loc`` / ``input_ids`` / ``positions`` match the serving
    breakable-prefill runner's key exactly (2048 x int64 x cuda:0 on the
    measured vehicle) and both graphs get CAPTURED against one address. Under
    ``--dual-group-lane-concurrent`` the lane replays its prefill graph on its
    own thread and stream while the serving group replays its own: two writers,
    one buffer, and the slot ids the loser's ``store_kvcache`` reads are the
    winner's — indices from a foreign pool, hence the
    ``index >= 0 && index < size_limit`` device assert (DESIGN_121 §13.1).

    ``current_lane_id()`` is the same key the graph memory pool, the GGUF
    dequant workspace and ``GraphSharedOutput`` already use: the serving group
    is ``None`` and keeps exactly one set of buffers, so the default path and
    the serial lane mode are unchanged, and a concurrent lane gets its own.
    """
    from sglang.srt.runtime_context import current_lane_id

    scope = current_lane_id()
    lane = None if _pool_ignores_lane() else scope
    key: _PoolKey = (
        lane,
        name,
        new_buffer.numel(),
        new_buffer.dtype,
        new_buffer.device,
    )
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    if _pool_debug():
        # Raw registration record, dumped before any analysis: which lane asked
        # for which key, and whether it landed on someone else's allocation.
        cross = [
            k[0]
            for k in _forward_input_buffer_pool
            if k[1:] == key[1:] and k[0] != lane
        ]
        logger.info(
            "input-buffer-pool: scope=%s lane=%s name=%s numel=%d dtype=%s "
            "device=%s ptr=0x%x new=%s same_key_other_lanes=%s",
            scope,
            lane,
            name,
            new_buffer.numel(),
            new_buffer.dtype,
            new_buffer.device,
            canonical.data_ptr(),
            canonical is new_buffer,
            cross,
        )
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())


def share_input_buffers_in(obj) -> None:
    """Pool every tensor buffer on ``obj`` (dataclass / ``SimpleNamespace``)
    through the process-wide pool, in place. No-op on NPU; recurses into dict /
    dataclass buffer fields (``pp_proxy_tensors`` / ``ngram_embedding_info``)."""
    if is_npu():
        return

    for name, buffer in list(vars(obj).items()):
        if buffer is None:
            continue
        if dataclasses.is_dataclass(buffer):
            buffer = vars(buffer)
        if isinstance(buffer, dict):
            for sub_name, sub_buffer in buffer.items():
                assert isinstance(
                    sub_buffer, torch.Tensor
                ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                buffer[sub_name] = share_input_buffer(f"{name}.{sub_name}", sub_buffer)
        else:
            assert isinstance(
                buffer, torch.Tensor
            ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
            setattr(obj, name, share_input_buffer(name, buffer))


@dataclass
class ForwardInputBuffers:

    def _share_one_buffer(self, name: str, new_buffer: torch.Tensor) -> torch.Tensor:
        return share_input_buffer(name, new_buffer)

    def share_buffers(self):
        # disable share input buffer on npu due to accuracy issue
        if is_npu():
            return

        for f in fields(self):
            name = f.name
            buffer = getattr(self, name)

            if buffer is None:
                continue

            if dataclasses.is_dataclass(buffer):
                buffer = vars(buffer)

            if isinstance(buffer, dict):
                for sub_name, sub_buffer in buffer.items():
                    assert isinstance(
                        sub_buffer, torch.Tensor
                    ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    new_buffer = self._share_one_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
                    buffer[sub_name] = new_buffer
            else:
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._share_one_buffer(name, buffer)
                setattr(self, name, new_buffer)
