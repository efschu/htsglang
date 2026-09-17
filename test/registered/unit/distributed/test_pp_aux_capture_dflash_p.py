"""DFlash-family aux capture across PP stages (distributed/pp_aux_capture).

Hermetic: a fake typed channel (in-memory inbox keyed by (src, kind)) stands
in for the PP group. The cases pin the two properties the P-side DFlash
draft-KV producer depends on: every stage's captures reach the last stage,
and they arrive in LAYER-ID order regardless of which stage owns which
layer (group P's ownership is gapped).
"""

import pytest
import torch

from sglang.srt.distributed.pp_aux_capture import (
    AUX_CAPTURE_KIND,
    PpAuxCaptureError,
    exchange_captured_aux,
)


class _Group:
    def __init__(self, rank, world_size):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_last_rank = rank == world_size - 1


class _Channel:
    """The typed channel's contract, in memory: dicts land under (src, kind)."""

    def __init__(self):
        self.inbox = {}
        self.sent = []

    def send(self, group, payload, dst, kind):
        self.sent.append((group.rank_in_group, dst, kind, sorted(payload)))
        self.inbox.setdefault((group.rank_in_group, kind), []).append(dict(payload))

    def recv(self, group, kind, src=None):
        q = self.inbox.get((src, kind))
        return q.pop(0) if q else None


def _run(ownership, world_size, channel=None, hidden=4):
    """ownership: {rank: [layer ids]} -> the last stage's assembled list."""
    channel = channel or _Channel()
    tensors = {}
    out = {}
    # Non-last stages send first (they finish their loops first), the last
    # stage receives afterwards -- the order the real pipeline has.
    for rank in list(range(world_size - 1)) + [world_size - 1]:
        captured = {}
        for lid in ownership.get(rank, []):
            t = torch.full((3, hidden), float(lid))
            tensors[lid] = t
            captured[lid] = t
        out[rank] = exchange_captured_aux(
            captured=captured,
            pp_group=_Group(rank, world_size),
            send=channel.send,
            recv=channel.recv,
        )
    return out, tensors, channel


def test_gapped_ownership_assembles_in_layer_id_order():
    # Group P shape: PP0 owns 6/20/34, PP1 owns 48, PP2 (last) owns 62 -- and
    # an interleaved variant where the last stage owns a LOW layer too.
    out, tensors, ch = _run({0: [6, 20, 34], 1: [48], 2: [62]}, 3)
    assert out[0] is None and out[1] is None
    assert [float(t[0, 0]) for t in out[2]] == [6.0, 20.0, 34.0, 48.0, 62.0]
    assert all(out[2][i] is tensors[l] for i, l in enumerate([6, 20, 34, 48, 62]))
    # exactly one aux message per non-last stage, addressed to the last stage
    assert [(s, d, k) for s, d, k, _ in ch.sent] == [
        (0, 2, AUX_CAPTURE_KIND),
        (1, 2, AUX_CAPTURE_KIND),
    ]


def test_interleaved_ownership_sorts_by_layer_not_stage():
    out, _, _ = _run({0: [34], 1: [6, 62], 2: [20, 48]}, 3)
    assert [float(t[0, 0]) for t in out[2]] == [6.0, 20.0, 34.0, 48.0, 62.0]


def test_stage_without_capture_layers_still_announces_itself():
    out, _, ch = _run({0: [6, 20, 34, 48], 2: [62]}, 3)
    assert [float(t[0, 0]) for t in out[2]] == [6.0, 20.0, 34.0, 48.0, 62.0]
    # stage 1 sent a message carrying only the count
    assert ch.sent[1][0] == 1 and ch.sent[1][3] == ["aux_count"]


def test_single_stage_is_identity_without_channel():
    ch = _Channel()
    captured = {20: torch.zeros(2, 2), 6: torch.ones(2, 2)}
    got = exchange_captured_aux(
        captured=captured, pp_group=_Group(0, 1), send=ch.send, recv=ch.recv
    )
    assert got[0] is captured[6] and got[1] is captured[20]
    assert ch.sent == []


def test_missing_stage_message_is_refused():
    ch = _Channel()
    with pytest.raises(PpAuxCaptureError, match="no aux-capture message"):
        exchange_captured_aux(
            captured={62: torch.zeros(1, 1)},
            pp_group=_Group(2, 3),
            send=ch.send,
            recv=ch.recv,
        )


def test_duplicate_capture_layer_is_refused():
    ch = _Channel()
    exchange_captured_aux(
        captured={62: torch.zeros(1, 1)}, pp_group=_Group(0, 2),
        send=ch.send, recv=ch.recv,
    )
    with pytest.raises(PpAuxCaptureError, match="two stages claim"):
        exchange_captured_aux(
            captured={62: torch.ones(1, 1)}, pp_group=_Group(1, 2),
            send=ch.send, recv=ch.recv,
        )


def test_declared_count_mismatch_is_refused():
    ch = _Channel()
    ch.inbox[(0, AUX_CAPTURE_KIND)] = [
        {"aux_layer_6": torch.zeros(1, 1), "aux_count": torch.tensor([2])}
    ]
    with pytest.raises(PpAuxCaptureError, match="declared 2"):
        exchange_captured_aux(
            captured={}, pp_group=_Group(1, 2), send=ch.send, recv=ch.recv
        )
