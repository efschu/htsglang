"""MAMBA-DONATION-DISJOINT (N2, 01.10.): the slot donated to the tree is never the overlap forward's track target.

NF's open point on MAMBA-SNAPSHOT-FENCE: the write op's start_event sits on
schedule_stream, and the WAR fast path makes schedule_stream wait only for the
previous forward's READ-done, not its state writes. Two forwards are in play
when a write op is issued in the overlap loop (Scheduler.event_loop_overlap):

* forward N, whose result is being processed: batch_result_processor syncs the
  host on N's ``copy_done`` (recorded after N's kernels, track writes included)
  BEFORE cache_(un)finished_req donates a slot or issues the write op. N is
  finished on the card when the write op is enqueued -- no race.
* forward N+1, enqueued before N's result is processed: nothing orders the
  write stream against it. It is harmless only if it never writes a slot that
  the processing of N hands to the tree. That holds by construction under
  overlap: the ping-pong buffer has TWO slots (memory_pool:
  ``2 if enable_overlap_schedule else 1``), prepare_for_extend swaps
  ``mamba_next_track_idx`` at SCHEDULE time, so N+1 is built against
  ``buf[next]`` while ``donate_mamba_ping_pong_slot`` takes
  ``buf[keep] = buf[other(next)]`` -- the slot N just wrote.

These tests pin that invariant on the real pool methods. With a ONE-slot
buffer (overlap off) keep == next: the donated slot IS the next track target,
which is only safe because no N+1 is in flight -- the test names that too, so
a pool built with size 1 under an overlap loop is caught here, not on metal."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool  # noqa: E402


class _Req:
    def __init__(self, buf, nxt):
        self.mamba_ping_pong_track_buffer = torch.tensor(buf, dtype=torch.int64)
        self.mamba_next_track_idx = nxt
        self.req_pool_idx = 0
        self.mamba_pingpong_clear_indices = None
        self.rid = "needle"
        self.last_node = None


def _pool(size):
    p = object.__new__(HybridReqToTokenPool)
    p.mamba_ping_pong_track_buffer_size = size
    p.enable_mamba_extra_buffer_lazy = False
    p.req_index_to_mamba_ping_pong_track_buffer_mapping = {}
    return p


def test_overlap_donation_takes_the_slot_the_extend_wrote_not_the_next_target():
    p = _pool(2)
    # prepare_for_extend: the extend tracks into buf[0], then next := other(0) = 1
    req = _Req([11, 12], 0)
    extend_target = int(req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx])
    req.mamba_next_track_idx = p.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)
    next_target = int(req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx])   # what N+1 is built with
    donated = int(p.donate_mamba_ping_pong_slot(req, torch.tensor([99]))[0])
    assert donated == extend_target == 11
    assert donated != next_target == 12
    # N+1's target slot is still the request's, the donated one is gone from it
    assert int(req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]) == next_target
    assert donated not in req.mamba_ping_pong_track_buffer.tolist()


def test_one_slot_buffer_donates_the_next_target_so_it_needs_overlap_off():
    p = _pool(1)
    req = _Req([11, -1], 0)
    req.mamba_next_track_idx = p.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)
    next_target = int(req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx])
    donated = int(p.donate_mamba_ping_pong_slot(req, torch.tensor([99]))[0])
    assert donated == next_target   # safe ONLY without an in-flight N+1


def test_the_buffer_size_follows_the_overlap_switch():
    import inspect

    src = inspect.getsource(HybridReqToTokenPool.__init__)
    assert "self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1" in src
