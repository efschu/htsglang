"""#1493: successor of the #795 cross-epoch mispair test, on today's PP wire.

#795 (boot instr15, 2026-08-21, /spinning/evidence-665-f1/
SPECIMEN_795_proxy_batch_mismatch.txt): a proxy stranded across a cutover
carried a slot number that COINCIDED with the slot the victim resumed on, so a
slot-only comparison accepted it and only the width check 30 layers down
caught the mispair. The cure is ``pp_proxy_stamp_names_pass``: the pass
identity is (flip epoch, slot), and a stamp from another epoch names no pass
of this rank.

WHY THIS FILE EXISTS. The old file (test_pp_proxy_cross_epoch_mispair_795.py)
spawns three gloo processes around a holder that predates #1015/#1233 and now
dies on AttributeError before it reaches any logic (4 of 4 red on base
702d5e90c9). The neighbours do not cover the epoch half either:
test_pp_stale_proxy_1057.py runs a hand-written MIRROR of the predicate (not
the shipped function) and only the same-epoch high-water mark;
test_pdflip_s0_unweave_1233.py pins ``_pp_flip_epoch`` as None and the receive
path with epoch-less stamps, so the epoch branch of the shipped predicate is
never taken by a green test.

WHAT IS PINNED, all hermetic (no gloo, no torch.distributed):
  1. RUNTIME: the shipped predicate with a real epoch on both sides.
  2. RUNTIME: the shipped receive ``_pp_recv_proxy_tensors`` refuses a proxy
     whose slot coincides and whose epoch is foreign (named stop, drop
     counted, payload never returned), and still delivers the same-epoch and
     epoch-less ones.
  3. RUNTIME: the shipped disarm drain ``pp_flip_drain_leftover_dicts`` drops
     the foreign-epoch proxy, keeps the owed one, never eats a non-proxy
     message, and keeps both when the epoch did not move (abandoned flip).
  4. RUNTIME: the sender writes its epoch where the readers look.
  5. SOURCE ORDER: every call of the predicate in the mixin takes its epoch
     from ``pp_flip_epoch_of(self)`` read in the lines before it.
"""

import re
import types
import unittest
from unittest import mock

from flliper.srt.distributed.pp_typed_channel import typed_inbox
from flliper.srt.managers import scheduler_pp_mixin as ppm
from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from flliper.srt.model_executor.forward_batch_info import PPProxyTensors
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

EPOCH_BEFORE = 7
EPOCH_AFTER = 8
LIVE_MB = 1


def _stamp(mb_id, epoch, seq=17, rows=3):
    # (mb_id, seq, rows, epoch): the layout of `_pp_proxy_stamp`.
    return (mb_id, seq, rows, epoch)


class TestStampNamesPassUsesTheEpoch(unittest.TestCase):
    def test_same_slot_foreign_epoch_names_no_pass(self):
        # The #795 specimen: slot coincides, generation does not.
        self.assertFalse(
            ppm.pp_proxy_stamp_names_pass(
                _stamp(LIVE_MB, EPOCH_BEFORE), LIVE_MB, EPOCH_AFTER
            )
        )

    def test_same_slot_same_epoch_names_the_pass(self):
        self.assertTrue(
            ppm.pp_proxy_stamp_names_pass(
                _stamp(LIVE_MB, EPOCH_AFTER), LIVE_MB, EPOCH_AFTER
            )
        )

    def test_other_slot_names_no_pass_even_in_the_same_epoch(self):
        self.assertFalse(
            ppm.pp_proxy_stamp_names_pass(
                _stamp(LIVE_MB + 1, EPOCH_AFTER), LIVE_MB, EPOCH_AFTER
            )
        )

    def test_absent_epoch_on_either_side_is_the_slot_only_test(self):
        # Stamp says -1 (sender had no runtime), receiver knows an epoch.
        self.assertTrue(
            ppm.pp_proxy_stamp_names_pass(_stamp(LIVE_MB, -1), LIVE_MB, EPOCH_AFTER)
        )
        # Stamp carries an epoch, receiver has none (today's S0 state).
        self.assertTrue(
            ppm.pp_proxy_stamp_names_pass(_stamp(LIVE_MB, EPOCH_BEFORE), LIVE_MB, None)
        )
        # A short legacy tuple has no epoch element at all.
        self.assertTrue(
            ppm.pp_proxy_stamp_names_pass((LIVE_MB, 5, 3), LIVE_MB, EPOCH_AFTER)
        )

    def test_unreadable_stamp_names_no_pass(self):
        self.assertFalse(ppm.pp_proxy_stamp_names_pass("junk", LIVE_MB, EPOCH_AFTER))


class _RecvHolder:
    """Hermetic holder for the REAL `_pp_recv_proxy_tensors` (cf. 1233)."""

    require_attn_tp_allgather = False
    attn_tp_group = None
    forward_ct = 7
    mbs = None

    def __init__(self, raw, epoch):
        self.pp_group = types.SimpleNamespace(is_first_rank=False)
        self._pp_gapped_wire = False
        self._pp_proxy_drops = 0
        self._raw = raw
        self._epoch = epoch

    def _pp_flip_epoch(self):
        return self._epoch

    def _pp_wait_for_proxy_readiness(self, mb_id):
        pass

    def _pp_recv_typed_dict(self, expected_kind=None, all_gather_group=None):
        assert expected_kind == "proxy"
        return dict(self._raw)


class TestReceiveRefusesAForeignEpochProxy(unittest.TestCase):
    def _recv(self, stamp, epoch):
        holder = _RecvHolder({"hidden_states": "HS", "__stamp__": stamp}, epoch)
        return holder, SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=LIVE_MB)

    def test_foreign_epoch_with_coinciding_slot_is_never_delivered(self):
        holder = _RecvHolder(
            {"hidden_states": "HS", "__stamp__": _stamp(LIVE_MB, EPOCH_BEFORE)},
            EPOCH_AFTER,
        )
        with self.assertRaises(RuntimeError) as cm:
            SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=LIVE_MB)
        # Named stop (#1004), not a silent pairing and not a nameless None.
        self.assertIn("#1004 SLOT DISAGREEMENT", str(cm.exception))
        self.assertEqual(holder._pp_proxy_drops, 1)

    def test_same_epoch_is_delivered_with_its_payload(self):
        holder, out = self._recv(_stamp(LIVE_MB, EPOCH_AFTER), EPOCH_AFTER)
        self.assertIsInstance(out, PPProxyTensors)
        self.assertEqual(out.tensors["hidden_states"], "HS")
        self.assertEqual(holder._pp_proxy_drops, 0)

    def test_epochless_stamp_is_delivered_as_before_the_field_existed(self):
        holder, out = self._recv(_stamp(LIVE_MB, -1), EPOCH_AFTER)
        self.assertIsInstance(out, PPProxyTensors)
        self.assertEqual(holder._pp_proxy_drops, 0)


class _Counters:
    def __init__(self, posted):
        self._posted = posted
        self._consumed = 0

    def sent(self, chan, upstream):
        return self._posted

    def local_consumed(self, chan):
        return self._consumed

    def bump_consumed(self, chan):
        # The per-KIND sub-channels are bumped through the same method; only
        # the wire counter itself feeds `local_consumed`.
        if chan == ppm.CHAN_DICT:
            self._consumed += 1


class _Group:
    rank_in_group = 2
    world_size = 3

    def __init__(self, wire):
        self._wire = list(wire)

    def recv_tensor_dict(self, all_gather_group=None):
        return self._wire.pop(0)


def _drain_holder(wire, epoch):
    h = types.SimpleNamespace(
        pp_flip_counters=_Counters(len(wire)),
        pp_group=_Group(wire),
        attn_tp_group=None,
        require_attn_tp_allgather=False,
        _pp_flip_epoch=lambda: epoch,
        _pp_flip_upstream=lambda: 1,
    )
    h._pp_flip_bump_consumed = types.MethodType(
        SchedulerPPMixin._pp_flip_bump_consumed, h
    )
    return h


def _proxy(epoch, mb_id=LIVE_MB, tag="p"):
    return {
        "__msg_type__": "proxy",
        "__stamp__": _stamp(mb_id, epoch),
        "hidden_states": tag,
    }


def _inbox_tags(h, kind):
    q = typed_inbox(h.pp_group).get((1, kind), [])
    return [m["hidden_states"] for m in q]


class TestDisarmDrainDropsAForeignEpochProxy(unittest.TestCase):
    def _drain(self, wire, epoch):
        h = _drain_holder(wire, epoch)
        # No settle wait: the wire is fully counted before the drain starts.
        with mock.patch.object(ppm, "DRAIN_SETTLE_BUDGET_S", 0.0):
            discarded = SchedulerPPMixin.pp_flip_drain_leftover_dicts(h, LIVE_MB)
        return h, discarded

    def test_foreign_epoch_dropped_owed_kept(self):
        wire = [
            _proxy(EPOCH_BEFORE, tag="stranded"),
            _proxy(EPOCH_AFTER, tag="owed"),
        ]
        h, discarded = self._drain(wire, EPOCH_AFTER)
        self.assertEqual(discarded, 1)
        self.assertEqual(_inbox_tags(h, "proxy"), ["owed"])

    def test_abandoned_flip_keeps_both(self):
        # An abandoned flip does not advance the epoch: pre-arm proxies are
        # still owed, so nothing may be dropped.
        wire = [
            _proxy(EPOCH_AFTER, tag="a"),
            _proxy(EPOCH_AFTER, tag="b"),
        ]
        h, discarded = self._drain(wire, EPOCH_AFTER)
        self.assertEqual(discarded, 0)
        self.assertEqual(_inbox_tags(h, "proxy"), ["a", "b"])

    def test_a_non_proxy_message_is_never_eaten(self):
        # Corpse S: an output is owed to a real consumer whatever epoch the
        # stamp-bearing proxies around it are from.
        out = {"__msg_type__": "output", "hidden_states": "result"}
        wire = [_proxy(EPOCH_BEFORE, tag="stranded"), out]
        h, discarded = self._drain(wire, EPOCH_AFTER)
        self.assertEqual(discarded, 1)
        self.assertEqual(_inbox_tags(h, "output"), ["result"])
        self.assertEqual(_inbox_tags(h, "proxy"), [])


class TestSenderWritesTheEpochWhereReadersLook(unittest.TestCase):
    def test_stamp_carries_the_epoch_at_the_documented_index(self):
        h = types.SimpleNamespace(_pp_flip_epoch=lambda: EPOCH_AFTER)
        with mock.patch.object(ppm, "_999_geom", return_value=None):
            stamp = SchedulerPPMixin._pp_proxy_stamp(h, LIVE_MB, types.SimpleNamespace())
        self.assertEqual(stamp[0], LIVE_MB)
        self.assertEqual(stamp[ppm.PP_PROXY_STAMP_EPOCH_INDEX], EPOCH_AFTER)
        self.assertEqual(ppm.pp_proxy_stamp_epoch(stamp), EPOCH_AFTER)

    def test_no_runtime_writes_minus_one_which_readers_map_to_none(self):
        h = types.SimpleNamespace(_pp_flip_epoch=lambda: None)
        with mock.patch.object(ppm, "_999_geom", return_value=None):
            stamp = SchedulerPPMixin._pp_proxy_stamp(h, LIVE_MB, types.SimpleNamespace())
        self.assertEqual(stamp[ppm.PP_PROXY_STAMP_EPOCH_INDEX], -1)
        self.assertIsNone(ppm.pp_proxy_stamp_epoch(stamp))


class TestSourceOrderEveryReaderTakesItsEpoch(unittest.TestCase):
    """SOURCE-ORDER INVARIANT (text read, not executed)."""

    def test_each_predicate_call_is_fed_by_pp_flip_epoch_of_before_it(self):
        src = open(ppm.__file__, encoding="utf-8").read()
        calls = [
            m.start() for m in re.finditer(r"(?<!def )pp_proxy_stamp_names_pass\(stamp, ", src)
        ]
        # drain, frame-presence probe, receive: the three stamp-reading sites.
        self.assertEqual(len(calls), 3)
        for pos in calls:
            window = src[max(0, pos - 6000) : pos]
            self.assertIn("epoch = pp_flip_epoch_of(self)", window)
            call = src[pos : src.index(")", pos)]
            self.assertTrue(call.endswith(", epoch"), call)


if __name__ == "__main__":
    unittest.main()
