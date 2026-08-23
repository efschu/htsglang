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
"""#824 W4a: the chain wait aborts on STATE, never on a clock.

W4a made the chain receive bounded, named and resumable, but left its
automatic abort off: SGLANG_PP_CHAIN_RECV_STALL_S defaults to 0, and that
default stays. It has to. An idle PP rank legitimately blocks in this
receive until a request arrives, so there is NO duration that separates
"idle" from "wedged" -- a wall-clock default would SIGQUIT a healthy idle
server, which is the same trap #821's marker documents for its own arm.

So the recovery is armed by EVIDENCE instead.

THE EVIDENCE. ``bump_attempted`` publishes that a rank has ENTERED a send
BEFORE it posts it (phase_flip_counters.py:220-226) -- the only counter
whose timing can witness a peer parked INSIDE a send rather than one that
has finished. When this rank's CHAN_DICT upstream has entered more dict
sends than this rank has taken off that wire, the peer is parked in a send
only this rank can drain, while this rank is parked in a receive that peer
will never feed. That is boot_827's ring, stated in counters:

    PP0  _pp_commit_admission_send_work   blocked in the typed-dict SEND
    PP1  _advance (pp_chain_receiver.py)  blocked in the chain RECEIVE
    PP2  _advance (pp_chain_receiver.py)  blocked in the chain RECEIVE

THE FALSE-POSITIVE DIRECTION IS THE SAFE ONE, the same argument
``_pp_wait_for_dict_readiness`` makes for the mirror gate (#789). A
spurious fire costs one drain turn and a resumed receive; the receive stays
posted and framed throughout, so the late message still arrives intact.
Missing a real one costs the boot.

CPU-only. No gloo, no CUDA, no scheduler construction.
"""

from types import SimpleNamespace

import pytest


class _Counters:
    """Only the three readers the predicate uses."""

    def __init__(self, attempted=0, consumed=0):
        self._attempted = attempted
        self._consumed = consumed

    def attempted(self, chan, rank):
        return self._attempted

    def local_consumed(self, chan):
        return self._consumed


def _holder(attempted=0, consumed=0, counters=True, upstream=0):
    from sglang.srt.managers.scheduler import Scheduler

    s = SimpleNamespace()
    s.pp_flip_counters = _Counters(attempted, consumed) if counters else None
    s._pp_flip_upstream = lambda: upstream
    s._pp_chain_abort_check = Scheduler._pp_chain_abort_check.__get__(s, SimpleNamespace)
    return s


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


def test_a_peer_parked_in_a_dict_send_aborts_the_chain_wait():
    """The boot_827 ring, in counters."""
    s = _holder(attempted=3, consumed=2)
    reason = s._pp_chain_abort_check("size", 12.0)
    assert reason, (
        "the upstream has entered a dict send this rank has not taken, so "
        "the ring is closed and the chain wait must not be permanent"
    )
    assert "CHAN_DICT" in reason
    assert "attempted=3" in reason and "local_consumed=2" in reason


def test_an_idle_rank_is_never_aborted():
    """THE REGRESSION THAT MATTERS. An idle PP rank blocks here for as long
    as no request arrives, and no elapsed time may change that."""
    s = _holder(attempted=5, consumed=5)
    assert s._pp_chain_abort_check("size", 0.0) is None
    assert s._pp_chain_abort_check("size", 3600.0) is None, (
        "an idle rank was aborted for waiting -- this is the wall-clock "
        "default the design refuses"
    )


def test_a_consumed_ahead_of_attempted_reading_never_aborts():
    """Counters are read without a lock, so the skew both ways must be
    inert rather than clever."""
    s = _holder(attempted=2, consumed=7)
    assert s._pp_chain_abort_check("size", 30.0) is None


def test_no_counters_means_no_opinion():
    """A boot without the flip has no counters and must behave exactly as
    it did before this predicate existed."""
    s = _holder(counters=False)
    assert s._pp_chain_abort_check("size", 30.0) is None


def test_a_broken_counter_read_never_breaks_the_loop():
    s = _holder(attempted=3, consumed=1)

    def _boom():
        raise RuntimeError("shm gone")

    s._pp_flip_upstream = _boom
    assert s._pp_chain_abort_check("size", 30.0) is None


# ---------------------------------------------------------------------------
# the ring-cut: stall -> drain -> resume the SAME receive
# ---------------------------------------------------------------------------


class _StallingChain:
    """A chain receiver that reports a closed ring until it is serviced."""

    def __init__(self, stalls=1, payload=("req",)):
        self.remaining = stalls
        self.payload = list(payload)
        self.recv_calls = 0

    def recv(self):
        from sglang.srt.managers.pp_chain_receiver import PpChainRecvStalled

        self.recv_calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise PpChainRecvStalled("closed ring (test)")
        return self.payload


def _receiver(chain, service=None):
    from sglang.srt.managers.scheduler_components.request_receiver import (
        SchedulerRequestReceiver,
    )

    rr = object.__new__(SchedulerRequestReceiver)
    object.__setattr__(rr, "chain_receiver", chain)
    object.__setattr__(rr, "pp_chain_stall_service", service)
    return rr


def test_a_stall_is_serviced_once_and_the_receive_resumes():
    chain = _StallingChain(stalls=1)
    turns = []
    rr = _receiver(chain, service=lambda: turns.append(1))

    got = rr._recv_chain_breaking_closed_rings()

    assert got == ["req"], "the resumed receive must return the real payload"
    assert len(turns) == 1, f"expected exactly one drain turn, got {len(turns)}"
    assert chain.recv_calls == 2


def test_a_stall_that_servicing_cannot_clear_is_re_raised():
    """Spinning here would replace a loud wedge with a silent one."""
    from sglang.srt.managers.pp_chain_receiver import PpChainRecvStalled

    chain = _StallingChain(stalls=99)
    turns = []
    rr = _receiver(chain, service=lambda: turns.append(1))

    with pytest.raises(PpChainRecvStalled):
        rr._recv_chain_breaking_closed_rings()

    assert len(turns) == rr.MAX_STALL_SERVICE_TURNS


def test_without_a_service_hook_the_stall_propagates_unchanged():
    """A boot without the flip installs no hook and must keep the pre-#824
    behaviour exactly."""
    from sglang.srt.managers.pp_chain_receiver import PpChainRecvStalled

    chain = _StallingChain(stalls=1)
    rr = _receiver(chain, service=None)

    with pytest.raises(PpChainRecvStalled):
        rr._recv_chain_breaking_closed_rings()
    assert chain.recv_calls == 1


def test_the_healthy_path_never_calls_the_service_hook():
    chain = _StallingChain(stalls=0)
    turns = []
    rr = _receiver(chain, service=lambda: turns.append(1))

    assert rr._recv_chain_breaking_closed_rings() == ["req"]
    assert turns == []
    assert chain.recv_calls == 1
