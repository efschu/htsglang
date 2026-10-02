"""WI: a Form A worker's eager prefill is bracketed by the per-rank prefill timer.

Metal (NF D, every boot y3u 00:31:46 / y3v 01:04:05 / y3w 01:35:15 / y3y
01:59:43): TP1 and TP2 log ``Prefill rank timing DISABLED on this rank: the
record and duration queues drifted -9 apart`` a few minutes after the boot,
and from then on their ``Prefill rank batch`` lines carry no gpu-ms, no
compute/wait and no wait-by-family -- flushed in bulk, late (y3w: nine lines
at 01:35:15 for batches TP0 had logged since 01:34:08). The scheduler records
every plain prefill on every rank; ``ModelRunner._forward_raw`` sent a Form A
worker's forward to ``_forward_form_a_worker`` with only the decode-round
bracket -- the prefill timer (eager runner / PCG sites) never wrapped it, so
the worker's record queue ran ahead of its duration queue until the #691
guard refused the pairing. What the lines before the refusal showed was
paired with OTHER forwards (a draft extend's span).

Hermetic: ``_forward_raw`` on a bare runner, the worker body stubbed.
RED on a332187f28 (no wrap), GREEN with WI.
"""
from __future__ import annotations

import contextlib
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.model_runner import ModelRunner  # noqa: E402


class _Timer:
    def __init__(self):
        self.wrapped = []

    @contextlib.contextmanager
    def wrap(self, metadata):
        self.wrapped.append(dict(metadata))
        yield


def _runner(timer):
    r = object.__new__(ModelRunner)
    r._forward_peak = None
    r._layer_fingerprint = None
    r.attn_backend = None
    r.is_weightless_head = False
    r.is_weightless_worker = False
    r.is_form_a_worker = True
    r.device = "cuda"
    r.decode_cuda_graph_runner = None
    r.decode_round_log = None
    r.prefill_rank_timer = timer
    r._run_form_a_boot_gate = lambda: None
    r._forward_form_a_worker = lambda fb: ("worker-forward", fb.forward_mode)
    return r


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_the_worker_prefill_is_timed_once(mode):
    """RED on a332187f28: wrapped == [] -- the record without a duration."""
    timer = _Timer()
    out = _runner(timer)._forward_raw(types.SimpleNamespace(forward_mode=mode), None)
    assert out == ("worker-forward", mode)
    assert timer.wrapped == [{"category": "extend"}]


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.IDLE])
def test_decode_and_idle_stay_unwrapped(mode):
    """The prefill line pairs only plain prefills (is_plain_prefill): a
    decode or idle forward on the worker must not feed its duration queue."""
    timer = _Timer()
    _runner(timer)._forward_raw(types.SimpleNamespace(forward_mode=mode), None)
    assert timer.wrapped == []


def test_without_a_timer_nothing_changes():
    out = _runner(None)._forward_raw(types.SimpleNamespace(forward_mode=ForwardMode.EXTEND), None)
    assert out[0] == "worker-forward"
