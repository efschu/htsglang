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

THE FIX HAS TWO HALVES, and they are pinned separately because metal
proved one and falsified the other's first cut.

PREVENTION -- ``pp_flip_drain_tensor_dicts``. While ARMED a rank consumes
and discards whatever its upstream published, so nothing strands. The
blocking recv is made ONLY when the upstream's counter exceeds this rank's
consumed count, so the message provably exists; discarding is right here
and only here, because an armed rank launches nothing and the message
names a pass it never ran.

DETECTION -- the stamp. Every proxy send carries ``(mb_id, monotone seqno,
rows)`` and the receive matches on it. A leftover names a slot that is not
this one and is REFUSED, loudly and by identity.

REFUSED, NOT DROPPED-AND-RETRIED. The first cut dropped the message and
looped to take the next one; that wedged the instance on metal (corpse R,
specimen stamp_drop_wedge_20260809T0719Z) because the wire owes exactly
one message per pass and the surplus recv closes a cycle with the peers.
``test_a_planted_leftover_is_REFUSED_and_the_wire_is_not_touched_again``
pins both halves of that: not computed on, and not replaced.

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

    # #789 HARNESS REPAIR (interface drift, no assertion touched):
    # _pp_send_dict_to_next_stage now publishes an "entered the send" count
    # BEFORE the post, so a downstream can tell a RENDEZVOUS sender from an
    # idle one. This rank counts nothing, so a no-op restores this file's
    # previous behaviour rather than changing it.
    def _pp_flip_bump_attempted(self, chan):
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
        payload,
        async_send=True,
        msg_type="proxy",
        stamp=sender._pp_proxy_stamp(1, _FakeResult(24)),
    )
    receiver = _Rank(wire=sender.pp_group.sent)
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert isinstance(got, PPProxyTensors)
    assert got["hidden_states"].shape[0] == 24
    assert receiver.recv_calls == 1
    assert getattr(receiver, "_pp_proxy_drops", 0) == 0


def test_an_unstamped_message_is_accepted_unchanged():
    """Backward compatibility: nothing else on this wire is stamped."""
    receiver = _Rank(
        wire=[{"hidden_states": torch.zeros(3, 4), "__msg_type__": "proxy"}]
    )
    got = receiver._pp_recv_proxy_tensors(mb_id=1)
    assert got["hidden_states"].shape[0] == 3


def test_the_first_rank_receives_nothing():
    r = _Rank(is_first_rank=True)
    assert r._pp_recv_proxy_tensors(mb_id=0) is None
    assert r.recv_calls == 0


# -- THE DEFECT: a planted leftover --------------------------------------------


def test_a_planted_leftover_is_REFUSED_and_the_wire_is_not_touched_again(caplog):
    """THE CAN-FAIL, and it pins BOTH halves of corpse R.

    The leftover must not be computed on (the original defect), and the
    function must not make a second blocking call to replace it (the
    disposal that wedged the instance on metal: the wire owes exactly one
    message per pass, so a second recv closes a cycle with the peers).
    """
    leftover = _msg(mb_id=2, seq=41, rows=1, tag="LEFTOVER")
    never = _msg(mb_id=1, seq=42, rows=24, tag="MUST_NOT_BE_TAKEN")
    receiver = _Rank(wire=[leftover, never])

    with pytest.raises(RuntimeError, match="PROXY LEFTOVER REFUSED"):
        receiver._pp_recv_proxy_tensors(mb_id=1)

    assert receiver.recv_calls == 1, (
        "corpse R: the receive took a SECOND message off the wire. The wire "
        "owes one message per pass; the surplus recv is what wedged PP1/PP2 "
        "against PP0 on metal."
    )


def test_the_refusal_names_the_message(caplog):
    receiver = _Rank(wire=[_msg(mb_id=2, seq=41, rows=1)])
    with pytest.raises(RuntimeError) as exc:
        receiver._pp_recv_proxy_tensors(mb_id=1)
    text = str(exc.value)
    assert "mb_id=2" in text and "seq=41" in text and "rows=1" in text
    assert "mb_id=1" in text


def test_forcing_the_match_open_reproduces_the_defect():
    """MUTATION PROOF that the pin above can fail.

    Accept-everything is the pre-fix behaviour; it hands back the leftover,
    which is precisely the 1-row-vs-24-token mispairing the specimen
    recorded.
    """
    leftover = _msg(mb_id=2, seq=41, rows=1, tag="LEFTOVER")
    mine = _msg(mb_id=1, seq=42, rows=24, tag="MINE")
    receiver = _Rank(wire=[leftover, mine])

    # mb_id < 0 is the module's own "no slot clock" escape hatch, i.e. the
    # match forced open without editing the file.
    got = receiver._pp_recv_proxy_tensors(mb_id=-1)
    assert got["tag"] == "LEFTOVER"
    assert got["hidden_states"].shape[0] == 1


# -- THE PREVENTION HALF: the armed drain -------------------------------------


class _FakeCounters:
    def __init__(self, posted=0):
        self.posted = posted
        self.taken = 0

    def sent(self, chan, rank):
        return self.posted

    def local_consumed(self, chan):
        return self.taken


class _ArmedRank(_Rank):
    pp_flip_drain_tensor_dicts = SchedulerPPMixin.pp_flip_drain_tensor_dicts

    def __init__(self, wire=(), posted=0, upstream=0):
        super().__init__(wire=wire)
        self.pp_flip_counters = _FakeCounters(posted)
        self._upstream = upstream
        self.pp_group = _FakeGroup(False)
        self.pp_group.wire = list(wire)

        def _recv_tensor_dict(all_gather_group=None):
            self.recv_calls += 1
            if not self.pp_group.wire:
                raise AssertionError("armed drain blocked on an empty wire")
            return self.pp_group.wire.pop(0)

        self.pp_group.recv_tensor_dict = _recv_tensor_dict

    def _pp_flip_upstream(self):
        return self._upstream

    def _pp_flip_bump_consumed(self, chan):
        self.pp_flip_counters.taken += 1


def test_the_armed_drain_takes_exactly_what_the_upstream_published():
    """THE SAFETY ARGUMENT: it blocks only on a message proved to exist."""
    r = _ArmedRank(wire=[_msg(0, 1, 1), _msg(1, 2, 1), _msg(2, 3, 1)], posted=2)
    assert r.pp_flip_drain_tensor_dicts() == 2
    assert r.recv_calls == 2, "it took more than the counter accounted for"


def test_the_armed_drain_makes_no_call_when_the_upstream_is_not_ahead():
    """counter-lags-send is the only permitted skew, and it under-reports.

    If the drain ever called on an equal count it would block for ever on a
    message that does not exist -- the failure the publish-after-post
    ordering exists to make impossible.
    """
    r = _ArmedRank(wire=[], posted=0)
    assert r.pp_flip_drain_tensor_dicts() == 0
    assert r.recv_calls == 0


def test_the_armed_drain_is_bounded():
    r = _ArmedRank(wire=[_msg(0, i, 1) for i in range(200)], posted=10_000)
    assert r.pp_flip_drain_tensor_dicts() == 64
    assert r.recv_calls == 64


def test_the_armed_drain_TAKES_an_output_off_the_wire_but_KEEPS_it():
    """#757 REPLACES the "both kinds are equally void" pin.

    That sentence was falsified ON METAL and the production code records the
    refutation next to the function: "the upstream wire MULTIPLEXES the proxy
    forward and the output return, and an output belongs to work launched
    BEFORE the arm. This function ate one (kind=output, PP1, 07:33:30Z) and
    PP1 then blocked for ever waiting for it." The drain was disabled because
    of that -- which removed the prevention half and let #757's
    PROXY LEFTOVER REFUSED fire under load on comp4.

    So the invariant this file pins is unchanged in the part that matters --
    the message still leaves the WIRE, because the upstream's blocking commit
    waits on exactly that -- and corrected in the part metal disproved: it is
    stashed for its consumer instead of destroyed.
    """
    out = {"next_token_ids": torch.zeros(2), "__msg_type__": "output"}
    r = _ArmedRank(wire=[out], posted=1)
    assert r.pp_flip_drain_tensor_dicts() == 1
    assert r.recv_calls == 1
    # ...and it is still reachable by the consumer that is owed it.
    assert list(r._pp_tensor_dict_inbox["output"]) == [out]


def test_the_armed_drain_still_DROPS_a_void_proxy():
    """The #757 half: a proxy for a pass this rank never ran is void.

    Leaving it on the wire is what strands it and puts every later receive off
    by one -- the specimen's mb_id=2 seq=151 rows=512 arriving on a rank at
    mb_id=1.
    """
    proxy = {"hidden_states": torch.zeros(2), "__msg_type__": "proxy",
             "__stamp__": (2, 151, 512)}
    r = _ArmedRank(wire=[proxy], posted=1)
    assert r.pp_flip_drain_tensor_dicts() == 1
    assert r.recv_calls == 1
    assert not list(getattr(r, "_pp_tensor_dict_inbox", {}).get("proxy", []))


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
    non_tensors = {k for k, v in got.tensors.items() if not isinstance(v, torch.Tensor)}
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
