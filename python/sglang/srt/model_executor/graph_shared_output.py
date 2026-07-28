from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class GraphSharedOutput:
    """``(max_rows, vocab)`` logits buffer, shared by every cuda-graph runner.

    Shared PER LANE, not per process (#274). The buffer is handed to captured
    decode graphs as their next-token-logits destination, so every graph that
    holds it writes to the same address on replay. That is safe while only one
    of them can be replaying -- the serving group's own runners are serial with
    each other -- and stops being safe the moment a dual-group lane replays its
    decode graph on its own thread and stream while the serving group replays
    its own: two concurrent writers, one buffer, and the logits the loser reads
    are the winner's.

    This is the Slice-C reasoning for the graph MEMORY pool ("graphs sharing a
    pool share their intermediates' buffers -- concurrent replay overwrites")
    applied to the OUTPUT buffer, which that pass keyed by lane while this one
    stayed process-wide. Keyed by ``current_lane_id()``: the serving group is
    ``None`` and keeps exactly one buffer, so the default path is unchanged,
    and a lane in scope gets its own.
    """

    _lane_shared: Dict[Optional[int], GraphSharedOutput] = {}

    def __init__(
        self,
        *,
        device: torch.device,
        max_rows: int,
    ) -> None:
        self.device = torch.device(device)
        self.max_rows = max_rows
        self._logits_buffers: Dict[int, torch.Tensor] = {}

    @classmethod
    def create_for_model_runner(
        cls, model_runner: ModelRunner
    ) -> Optional[GraphSharedOutput]:
        cuda_graph_config = model_runner.server_args.cuda_graph_config
        if cuda_graph_config is None:
            return None

        max_rows = 0
        decode = cuda_graph_config.decode
        if decode.backend != Backend.DISABLED and decode.bs:
            max_rows = max(max_rows, model_runner.max_decode_logits_rows())

        if max_rows <= 0:
            return None

        from sglang.srt.runtime_context import current_lane_id

        device = torch.device(model_runner.device)
        key = current_lane_id()
        shared = cls._lane_shared.get(key)
        if (
            shared is not None
            and shared.device == device
            and shared.max_rows >= max_rows
        ):
            return shared
        cls._lane_shared[key] = cls(device=device, max_rows=max_rows)
        return cls._lane_shared[key]

    def get_logits_buffer(self, vocab_size: int, *, rows: int) -> torch.Tensor:
        assert rows <= self.max_rows, (
            f"shared logits buffer holds {self.max_rows} rows but caller "
            f"needs {rows} (vocab_size={vocab_size})"
        )
        buffer = self._logits_buffers.get(vocab_size)
        if buffer is None:
            buffer = torch.zeros(
                (self.max_rows, vocab_size), dtype=torch.float, device=self.device
            )
            self._logits_buffers[vocab_size] = buffer
        return buffer[:rows]
