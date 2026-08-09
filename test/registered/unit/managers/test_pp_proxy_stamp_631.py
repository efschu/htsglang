"""#631 variant B: the proxy stream carries its own identity.

THE DEFECT THESE PIN (root cause, specimen
/spinning/evidence-631/pp_proxy_mispair_20260809T0626Z). A flip abandon is
RANK-LOCAL: each rank times out on its own clock. The first rank to disarm
resumes launching and sends its proxy hidden states; its downstream is
still armed, still withholding, so that rank has no ``cur_batch`` and the
proxy recv -- guarded by THIS rank's batch, never by whether the upstream
sent -- is not made. The message strands in ``_pp_tensor_dict_inbox``.

``PPProxyTensors`` carried NO identity, so the pairing was purely
positional: "whatever came off the wire this slot iteration" met "whatever
batch I have this slot iteration". ONE stranded message therefore put every
later receive off by one, SILENTLY, for the rest of the loop's life.

The fix stamps every proxy send with ``(mb_id, monotone seqno, rows)`` and
matches on the stamp at receive. A leftover names a slot that is not this
one, matches nothing, and is dropped LOUDLY instead of being computed on.

WHAT THESE TESTS DELIBERATELY ALSO RECORD: the match is on ``mb_id``
ALONE, and ``mb_id`` is cyclic modulo ``pp_loop_size`` (3 on this rig). A
leftover whose slot happens to coincide is NOT caught. The seqno is
stamped and currently unused. ``test_a_coinciding_slot_is_the_residual_hole``
pins that limit as MEASURED rather than leaving it as a surprise; do not
delete it to make the suite look greener.

CPU-only.
"""

import logging
from collections import deque

import pytest
import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class _FakeGroup:
    def __init__(self, is_first_rank=False):
        self.is_first_rank = is_first_rank
        self.sent = []

    def send_tensor_dict(self, tensor_dict, all_gather_group=None, async_send=True):
        # The wire carries the dict AS GIVEN -- that is the whole reason the
        # stamp can ride inside it without a header.
        self.sent.append(dict(tensor_dict))
        return []


class _FakeResult:
    def __init__(self, rows):
        self.pp_hidden_states_proxy_tensors = PPProxyTensors(
            {"hidden_states": torch.zeros(rows, 4)}
        )


class _Rank:
    """The smallest object on which the two methods under test are real.

    They are taken unbound off the mixin, so this exercises the SHIPPING
    code, not a transcription of it.
    """

    _pp_proxy_stamp = SchedulerPPMixin._pp_proxy_stamp
    _pp_recv_proxy_tensors = SchedulerPPMixin._pp_recv_proxy_tensors
    _pp_send_dict_to_next_stage = SchedulerPPMixin._pp_send_dict_to_next_stage

    def __init__(self, wire=(), is_first_rank=False):
        self.pp_group = _FakeGroup(is_first_rank)
        self.attn_tp_group = None
        self.require_attn_tp_allgather = False
        self._wire = deque(wire)
        self.recv_calls = 0

    # stands in for _pp_recv_typed_dict, which is the demultiplexer, not
    # the thing under test.
    def _pp_recv_typed_dict(self, expected_kind="default", all_gather_group=None):
        self.recv_calls += 1
        if not self._wire:
            raise AssertionError(
                "recv called with an empty wire: the drain is unbounded"
            )
        return self._wire.popleft()

    def _pp_boundary_stats(self):
        return None

    def _pp_flip_bump_sent(self, chan):
        pass


def _msg(mb_id, seq, rows, tag=None):
    d = {
        "hidden_states": torch.zeros(rows, 4),
        "__msg_type__": "proxy",
        "__stamp__": (mb_id, seq, rows),
    }
    if tag is not None:
        d["tag"] = tag
    return d


# -- the stamp itself ----------------------------------------------------------


def test_stamp_carries_slot_monotone_seqno_and_rows():
    r = _Rank()
    assert r._pp_proxy_stamp(1, _FakeResult(24)) == (1, 1, 24)
    assert r._pp_proxy_stamp(2, _FakeResult(1)) == (2, 2, 1)
    assert r._pp_proxy_stamp(0, _FakeResult(7)) == (0, 3, 7)


def test_the_seqno_never_resets_when_the_slot_repeats():
    """Two messages for the SAME slot is exactly the pair a strand creates.

    The slot cannot tell them apart; only the seqno can.
    """
    r = _Rank()
    first = r._pp_proxy_stamp(1, _FakeResult(24))
    second = r._pp_proxy_stamp(1, _FakeResult(24))
    assert first[0] == second[0]
    assert second[1] > first[1]


def test_a_stamp_never_breaks_a_send_even_if_the_result_is_malformed():
    class _Broken:
        pp_hidden_states_proxy_tensors = None

    r = _Rank()
    assert r._pp_proxy_stamp(1, _Broken()) == (1, 1, -1)


# -- round trip through the wire -----------------------------------------------


def test_the_stamp_rides_in_the_dict_that_crosses_the_wire():
    sender = _Rank()
    payload = {"hidden_states": torch.zeros(24, 4)}
    sender._pp_send_dict_to_next_stage(
        payload, async_send=True, msg_type="proxy", stamp=(1, 9, 24)
    )
    on_wire = sender.pp_group.sent[0]
    assert on_wire["__stamp__"] == (1, 9, 24)
    assert on_wire["__msg_type__"] == "proxy"
    assert on_wire["hidden_states"].shape[0] == 24


def test_round_trip_a_matching_message_is_delivered():
    sender = _Rank()
    payload = {"hidden_states": torch.zeros(24, 4)}
    sender._pp_send_dict_to_next_stage(
        payload, async_send=True, msg_type="proxy", stamp=sender._pp_proxy_stamp(
            1, _FakeResult(24)
        )
    )
    receiver = _Rank(wire=sender.pp_group.sent)
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert isinstance(got, PPProxyTensors)
    assert got["hidden_states"].shape[0] == 24
    assert receiver.recv_calls == 1
    assert getattr(receiver, "_pp_proxy_drops", 0) == 0


def test_an_unstamped_message_is_accepted_unchanged():
    """Backward compatibility: nothing else on this wire is stamped."""
    receiver = _Rank(wire=[{"hidden_states": torch.zeros(3, 4), "__msg_type__": "proxy"}])
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert got["hidden_states"].shape[0] == 3


def test_the_first_rank_receives_nothing():
    r = _Rank(is_first_rank=True)
    assert r._pp_recv_proxy_tensors(mb_id=0) is None
    assert r.recv_calls == 0


# -- THE DEFECT: a planted leftover --------------------------------------------


def test_a_planted_leftover_is_dropped_loudly_and_the_next_message_is_taken(caplog):
    """THE CAN-FAIL. Under the old positional pairing the FIRST message --
    the leftover -- was returned and computed on.
    """
    leftover = _msg(mb_id=2, seq=41, rows=1, tag="LEFTOVER")
    mine = _msg(mb_id=1, seq=42, rows=24, tag="MINE")
    receiver = _Rank(wire=[leftover, mine])

    with caplog.at_level(logging.ERROR):
        got = receiver._pp_recv_proxy_tensors(mb_id=1)

    assert got["tag"] == "MINE", "the leftover was computed on -- the defect"
    assert got["hidden_states"].shape[0] == 24
    assert receiver._pp_proxy_drops == 1
    text = caplog.text
    assert "PROXY LEFTOVER DROPPED" in text
    # the identity must be IN the log, or the next investigation has nothing
    assert "mb_id=2" in text and "seq=41" in text and "rows=1" in text


def test_forcing_the_match_open_reproduces_the_defect():
    """MUTATION PROOF that the pin above can fail.

    Accept-everything is the pre-fix behaviour; it must hand back the
    leftover, which is precisely the 1-row-vs-24-token mispairing the
    specimen recorded.
    """
    leftover = _msg(mb_id=2, seq=41, rows=1, tag="LEFTOVER")
    mine = _msg(mb_id=1, seq=42, rows=24, tag="MINE")
    receiver = _Rank(wire=[leftover, mine])

    # mb_id < 0 is the module's own "no slot clock" escape hatch, i.e. the
    # match forced open without editing the file.
    got = receiver._pp_recv_proxy_tensors(mb_id=-1)
    assert got["tag"] == "LEFTOVER"
    assert got["hidden_states"].shape[0] == 1
    assert getattr(receiver, "_pp_proxy_drops", 0) == 0


def test_several_leftovers_are_drained_and_all_are_counted(caplog):
    wire = [_msg(mb_id=2, seq=s, rows=1) for s in (10, 11, 12)]
    wire.append(_msg(mb_id=0, seq=13, rows=24, tag="MINE"))
    receiver = _Rank(wire=wire)
    with caplog.at_level(logging.ERROR):
        got = receiver._pp_recv_proxy_tensors(mb_id=0)
    assert got["tag"] == "MINE"
    assert receiver._pp_proxy_drops == 3


def test_the_drain_is_bounded_and_gives_up_loudly(caplog):
    """A persistent disagreement must not spin for ever on the wire."""
    wire = [_msg(mb_id=2, seq=s, rows=1) for s in range(50)]
    receiver = _Rank(wire=wire)
    with caplog.at_level(logging.ERROR):
        got = receiver._pp_recv_proxy_tensors(mb_id=0)
    assert got is None
    assert receiver.recv_calls == 8, "the drain bound moved"
    assert "gave up draining proxy leftovers" in caplog.text


# -- THE STAMP MUST NOT REACH MODEL COMPUTE ------------------------------------


def test_the_stamp_is_stripped_before_the_model_sees_it():
    """A delivered proxy must contain TENSORS ONLY.

    ``PPProxyTensors.__getitem__``'s slice path maps ``v[key]`` over EVERY
    entry, and cuda-graph buffer copies iterate the dict; a stray tuple
    there is silent nonsense rather than an error, which is the worst
    available outcome. The identity has done its whole job by the time the
    message is accepted, so it is removed at the boundary.
    """
    receiver = _Rank(wire=[_msg(mb_id=1, seq=42, rows=24)])
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert "__stamp__" not in got.tensors

    # ``__msg_type__`` is a PRE-EXISTING non-tensor entry that has always
    # travelled into PPProxyTensors in production. It is named here rather
    # than quietly tolerated: this fix does not widen that exposure, and the
    # assertion below fails the moment a THIRD kind of non-tensor appears.
    non_tensors = {
        k for k, v in got.tensors.items() if not isinstance(v, torch.Tensor)
    }
    assert non_tensors == {"__msg_type__"}, (
        f"a new non-tensor entry reached model compute: {non_tensors}"
    )


def test_a_delivered_proxy_survives_the_slice_path():
    """The concrete consumer named as the risk, exercised for real."""
    receiver = _Rank(wire=[_msg(mb_id=1, seq=42, rows=24)])
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    sliced = got[0:4]
    assert sliced["hidden_states"].shape[0] == 4


# -- the honest limit ----------------------------------------------------------


def test_a_coinciding_slot_is_the_residual_hole():
    """MEASURED LIMIT, not an aspiration.

    ``mb_id`` is cyclic modulo ``pp_loop_size`` (3 on this rig), so a
    leftover that is a whole cycle stale names THIS slot and is accepted.
    The seqno that would settle it is stamped and not consulted. Metal has
    to say whether this case occurs before more machinery is justified --
    the ``model_runner.forward`` shape check is the standing tripwire for
    it.
    """
    stale_but_coinciding = _msg(mb_id=1, seq=41, rows=1, tag="STALE")
    mine = _msg(mb_id=1, seq=44, rows=24, tag="MINE")
    receiver = _Rank(wire=[stale_but_coinciding, mine])
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert got["tag"] == "STALE"
    assert getattr(receiver, "_pp_proxy_drops", 0) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
