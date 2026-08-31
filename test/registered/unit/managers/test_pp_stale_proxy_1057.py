"""#1057: a leftover pass of the SAME epoch must be un-takeable, not merely detected.

RED-FIRST AGAINST THE BOOT-26 SPECIMEN
(boot_855_1056acc_0840f82601_0831_150350.log, 15:15:03Z, PP2):

    #631 PP proxy/batch mismatch: received hidden_states with 976 row(s) for a
    1 batch of 558 token(s) (bs=2); sender stamp (mb_id=1 seq=39 rows=976
    epoch=8 fwd_ct=118 sender_geom=('2cc5b5d2', 32768, 33325)), receiver
    fwd_ct=117, receiver_geom=('2cc5b5d2', 32768, 33325).

Identical geometry stamps, sender rows == received rows, mb_id and epoch
matching -- so `pp_proxy_stamp_names_pass` accepts, and the width check 30
layers down is the only thing that catches it, by killing the group.

`pp_proxy_stamp_names_pass` names this case itself, as its HONEST RESIDUAL:
"within ONE epoch a leftover a whole ring-cycle stale still names this slot
and is still accepted". `pp_loop_size` is 3 on that boot form, so a pass three
laps older lands on the same slot in the same epoch. That is the case below.

THE SECOND TEST IS THE DANGER-DIRECTION MUTANT and it is the one that matters
more. #995's first version died of a refusal on the live path with no way
onward: 175 refusals on one rid and a dead window. So it is not enough that
the leftover be refused -- the receive must CONTINUE and deliver the real
message. A guard that merely raises would pass a naive "was it refused?" test
and reproduce the window death exactly.
"""

import pytest

from sglang.srt.distributed.pp_typed_channel import (
    _INBOX_ATTR,
    stash_typed,
    take_typed,
)


class _Group:
    """The minimum `take_typed` touches: an object that can hold an attribute."""

    def __init__(self):
        self.rank_in_group = 2
        self.world_size = 3


def _proxy(seq: int, epoch: int = 8, mb_id: int = 1, rows: int = 976):
    """A raw tensor-dict shaped like the ones on the wire, boot-26 stamp layout."""
    return {
        "__msg_type__": "proxy",
        "__stamp__": (mb_id, seq, rows, epoch, -1, ("2cc5b5d2", 32768, 33325)),
        "hidden_states": f"payload-seq-{seq}",
    }


def _accept_factory(high_water):
    """The scheduler's predicate, reduced to the arithmetic under test.

    Mirrors `Scheduler._pp_proxy_is_not_stale`: pure read, per-epoch mark,
    unreadable means True (today's behaviour, never a new refusal).
    """

    def accept(message):
        stamp = message.get("__stamp__")
        if not isinstance(stamp, (tuple, list)) or len(stamp) < 2:
            return True
        seq, epoch = int(stamp[1]), stamp[3]
        prev = high_water.get((epoch,))
        return prev is None or seq > prev

    return accept


def test_a_three_lap_older_pass_of_the_same_epoch_is_not_taken():
    """The boot-26 specimen: same epoch, same slot, three laps stale."""
    group = _Group()
    # This rank has already consumed seq=42 in epoch 8 -- three laps of a
    # 3-slot ring past the leftover's seq=39.
    high_water = {(8,): 42}
    dropped = []

    stash_typed(group, None, "proxy", _proxy(seq=39))

    served = take_typed(
        group,
        None,
        "proxy",
        accept=_accept_factory(high_water),
        on_reject=dropped.append,
    )

    assert served is None, (
        "the leftover was served; before #1057 this reached model compute and "
        "#631 killed the group"
    )
    assert len(dropped) == 1, "the loss must be counted, never silent (#800 rule)"
    assert dropped[0]["__stamp__"][1] == 39


def test_the_receive_continues_past_the_leftover_and_delivers_the_real_message():
    """THE DANGER DIRECTION: refuse without a way onward == the #995 v1 death.

    A leftover ahead of the genuine message in the same queue must not strand
    the genuine one behind it. The drain has to walk past the leftover and
    hand back the real pass in the same call.
    """
    group = _Group()
    high_water = {(8,): 42}
    dropped = []

    stash_typed(group, None, "proxy", _proxy(seq=39))  # leftover, three laps old
    stash_typed(group, None, "proxy", _proxy(seq=43))  # the pass actually owed

    served = take_typed(
        group,
        None,
        "proxy",
        accept=_accept_factory(high_water),
        on_reject=dropped.append,
    )

    assert served is not None, "the receive refused with no way onward -- #995 v1"
    assert served["__stamp__"][1] == 43
    assert served["hidden_states"] == "payload-seq-43"
    assert len(dropped) == 1


def test_an_empty_queue_after_dropping_falls_through_to_the_wire():
    """Draining the inbox must terminate in None, so the caller reaches the wire."""
    group = _Group()
    high_water = {(8,): 42}
    dropped = []

    for seq in (37, 38, 39):
        stash_typed(group, None, "proxy", _proxy(seq=seq))

    served = take_typed(
        group, None, "proxy", accept=_accept_factory(high_water), on_reject=dropped.append
    )

    assert served is None, "must fall through to the wire, not spin on the inbox"
    assert len(dropped) == 3
    inbox = getattr(group, _INBOX_ATTR)
    assert all(len(q) == 0 for q in inbox.values()), "the queue must be drained"


def test_a_newer_pass_is_never_refused():
    """The guard may only remove what is PROVABLY older. Everything else passes."""
    group = _Group()
    high_water = {(8,): 42}
    stash_typed(group, None, "proxy", _proxy(seq=43))
    served = take_typed(group, None, "proxy", accept=_accept_factory(high_water))
    assert served is not None and served["__stamp__"][1] == 43


def test_a_different_epoch_has_its_own_mark():
    """Epoch is the namespace: a mark from epoch 8 may not refuse epoch 9."""
    group = _Group()
    high_water = {(8,): 42}
    stash_typed(group, None, "proxy", _proxy(seq=3, epoch=9))
    served = take_typed(group, None, "proxy", accept=_accept_factory(high_water))
    assert served is not None, "an epoch-8 mark must not judge an epoch-9 pass"


@pytest.mark.parametrize(
    "stamp",
    [None, "not-a-tuple", (1,)],
    ids=["absent", "wrong-type", "too-short"],
)
def test_an_unreadable_stamp_is_todays_behaviour_never_a_new_refusal(stamp):
    """The safe direction: this guard may not invent a refusal it cannot justify."""
    group = _Group()
    high_water = {(8,): 42}
    message = {"__msg_type__": "proxy", "hidden_states": "x"}
    if stamp is not None:
        message["__stamp__"] = stamp
    stash_typed(group, None, "proxy", message)
    served = take_typed(group, None, "proxy", accept=_accept_factory(high_water))
    assert served is not None, "an unreadable stamp must pass, exactly as before"


def test_without_the_accept_predicate_the_old_behaviour_is_bit_for_bit():
    """CAN-FAIL EVIDENCE: with no predicate the leftover IS served.

    This is the pre-#1057 behaviour and the reason the first test is red
    before the fix -- it pins that the guard, not the harness, is what changes
    the outcome.
    """
    group = _Group()
    stash_typed(group, None, "proxy", _proxy(seq=39))
    served = take_typed(group, None, "proxy")
    assert served is not None and served["__stamp__"][1] == 39
