# Copyright 2023-2024 SGLang Team
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
"""#824 W5: the watchdog must name the arm that tripped it.

WHAT BOOT_827 ESTABLISHED, AND WHAT IT DID NOT
----------------------------------------------
The watchdog worked. It fired on PP0 at 09:17:58 against a 300 s timeout,
dumped py-spy for every scheduler, and SIGQUIT at 09:18:03. Loud death is
not the defect.

Two things were still wrong, and this file pins both:

  (a) THE LINE NAMED NO ARM. It read "Scheduler watchdog timeout
      (self.watchdog_timeout=300, self.soft=False)" and nothing else, so
      which wait had stopped the ring had to be reconstructed afterwards
      from the py-spy dump. Worse, the accompanying dump_info line asserted
      "this rank is parked in a blocking PP dict receive" for EVERY firing,
      because _pp_recv_typed_dict was the only site that set the marker --
      and PP0 was blocked in a SEND, not in a typed-dict receive. The one
      line an operator reads first named the wrong channel.

  (b) THE PATH THAT WEDGED WAS NOT INSTRUMENTED AT ALL. #821 stamps
      _pp_blocked_recv_since inside _pp_recv_typed_dict only. Two of three
      ranks wedged in pp_chain_receiver's request-relay receive, which set
      nothing, so "#821's arm covers this" was never established by that
      boot.

CPU-only, no gloo, no CUDA.
"""

import time
import types

import pytest


# ---------------------------------------------------------------------------
# (a) the trigger line
# ---------------------------------------------------------------------------


def test_trigger_line_names_the_arm_when_one_is_known():
    from sglang.srt.utils.watchdog import compose_timeout_line

    line = compose_timeout_line(
        "Scheduler", 300, False, "blocked-recv[chain-recv/size<-0] waited=31.4s"
    )
    assert "Scheduler watchdog timeout" in line
    assert "tripped_by=" in line, (
        "the trigger line still names no arm -- this is exactly the "
        f"boot_827 line an operator could not act on: {line!r}"
    )
    assert "chain-recv/size<-0" in line
    assert "31.4s" in line


def test_trigger_line_is_worded_exactly_as_before_without_a_describer():
    """No arm known -> no new noise. The old wording is load-bearing for
    every log grep and dashboard that already matches on it."""
    from sglang.srt.utils.watchdog import compose_timeout_line

    line = compose_timeout_line("Scheduler", 300, False, "")
    assert line == (
        "Scheduler watchdog timeout "
        "(self.watchdog_timeout=300, self.soft=False)"
    )
    assert "tripped_by" not in line


# ---------------------------------------------------------------------------
# (b) the marker on the funnel that actually wedged
# ---------------------------------------------------------------------------


class _NeverArrives:
    """A Work whose wait() never returns, i.e. the boot_827 size recv."""

    def __init__(self, *_a, **_kw):
        self._blocked = True

    def wait(self):
        while True:
            time.sleep(0.05)

    def is_completed(self):
        return False


class _SilentDist:
    """torch.distributed stand-in whose upstream never sends."""

    def irecv(self, tensor, src=None, group=None):
        return _NeverArrives()


def test_chain_receive_stamps_the_arm_it_is_blocked_in(monkeypatch):
    """THE W5(b) GAP. pp_chain_receiver had no marker at all."""
    from sglang.srt.managers import pp_chain_receiver as mod

    monkeypatch.setattr(mod, "dist", _SilentDist())

    seen = []
    rx = mod.PpChainReceiver(
        group=None,
        src=0,
        dst=1,
        stall_timeout_s=0.5,
        on_blocked=lambda arm, since: seen.append((arm, since)),
    )

    with pytest.raises(mod.PpChainRecvStalled) as exc:
        rx.recv()

    assert seen, (
        "the chain receive blocked and told nobody -- this is the blind "
        "spot that made two of three wedged ranks invisible on boot_827"
    )
    arm, since = seen[0]
    assert arm == "size", f"the marker must name WHICH arm, got {arm!r}"
    assert isinstance(since, float)
    assert rx.blocked_recv_arm == "size"
    # The named branch, not a silent park.
    assert "arm=size" in str(exc.value)
    assert "src=0" in str(exc.value)


def test_marker_clears_when_the_message_finally_lands(monkeypatch):
    """A stale timestamp reads as a wedge later, so clearing matters as
    much as stamping."""
    import pickle

    import torch

    from sglang.srt.managers import pp_chain_receiver as mod

    payload = pickle.dumps([{"rid": "r0"}])

    class _SlowThenArrives:
        def __init__(self, tensor, source, delay_s):
            self._tensor = tensor
            self._source = source
            self._deadline = time.monotonic() + delay_s

        def wait(self):
            while time.monotonic() < self._deadline:
                time.sleep(0.02)
            self._tensor.copy_(self._source)

        def is_completed(self):
            return False

    class _SlowDist:
        def irecv(self, tensor, src=None, group=None):
            if tensor.dtype == torch.long:
                return _SlowThenArrives(
                    tensor, torch.tensor([len(payload)], dtype=torch.long), 1.2
                )
            return _SlowThenArrives(
                tensor, torch.frombuffer(bytearray(payload), dtype=torch.uint8), 0.0
            )

    monkeypatch.setattr(mod, "dist", _SlowDist())
    seen = []
    rx = mod.PpChainReceiver(
        group=None,
        src=0,
        dst=1,
        on_blocked=lambda arm, since: seen.append((arm, since)),
    )
    got = rx.recv()

    assert got == [{"rid": "r0"}]
    assert rx.blocked_recv_arm is None, "marker left stale after the message landed"
    assert rx.blocked_recv_since is None
    assert ("size", pytest.approx(seen[0][1])) == seen[0]
    assert seen[-1] == (None, None), f"marker was never cleared: {seen}"


# ---------------------------------------------------------------------------
# the two halves meeting: the scheduler's describer reads the marker
# ---------------------------------------------------------------------------


def test_describer_reports_the_chain_arm_the_watchdog_used_to_miss():
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        create_scheduler_watchdog,
    )

    scheduler = types.SimpleNamespace(
        is_initializing=False,
        cur_batch_for_debug=None,
        forward_ct=0,
        _pp_blocked_recv_since=time.monotonic() - 31.0,
        _pp_blocked_recv_arm="chain-recv/size<-0",
    )
    # A huge timeout so the watchdog thread only ever sleeps.
    wd = create_scheduler_watchdog(scheduler, watchdog_timeout=10**6, soft=True)

    described = wd.describe_arm()
    assert "chain-recv/size<-0" in described, (
        "the describer cannot name the request-relay arm, so the trigger "
        f"line would repeat boot_827's silence: {described!r}"
    )
    assert "blocked-recv[" in described
    assert "waited=3" in described  # ~31s

    # And the typed-dict arm still reports, unchanged.
    scheduler._pp_blocked_recv_arm = "typed-dict/admission"
    assert "typed-dict/admission" in wd.describe_arm()
