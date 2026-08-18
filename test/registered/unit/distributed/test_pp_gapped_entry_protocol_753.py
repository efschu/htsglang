# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#753: the two defects boot v7pp4 found, held down as tests.

Boot v7pp4 (2026-08-18) built every pool and then wedged with no crash. The
py-spy dump named the cycle exactly: PP0 blocked in the crossing wire's receive
for layer 4, PP1 and PP2 blocked in the scheduler's stage-boundary receive
waiting for a ``proxy`` message that PP0 could not send until its own forward
finished -- which required layer 3, on PP1.

Two separate defects sit behind that, and each gets a test that FAILS against
the pre-fix code:

1. THE ENTRY PROTOCOL. Under a gapped set a stage's entry activations arrive
   over the wire, not as a stage-boundary handoff. Nothing expressed that, so
   the ranks queued behind a handoff that was structurally unsendable.

2. THE SHARED CHANNEL. ``output`` dicts ring from the last rank to rank 0 --
   the same directed pair as PP2 -> PP0's crossings. With the wire receiving
   untyped and the scheduler stashing into a private inbox, either consumer
   could take the other's message off the wire.
"""

import os
import unittest
from collections import deque
from unittest import mock

from sglang.srt.distributed.pp_crossing_wire import (
    NoCrossingWire,
    build_crossing_wire,
)
from sglang.srt.distributed.pp_typed_channel import (
    CROSSING_KIND,
    recv_typed_tensor_dict,
    resolve_src,
    send_typed_tensor_dict,
    stash_typed,
    take_typed,
)
from sglang.srt.distributed.utils import (
    PP_CROSSING_WIRE_ENV,
    PP_LAYER_SET_ENV,
    pp_gapped_ownership_active,
)

#: The #735 cut this rig actually boots: 48 GDN layers on the 5090 (PP0), 8
#: full-attention layers on each 3080. Written out rather than generated so a
#: change to the layout is a visible change to the test.
GAPPED_SET = (
    "0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,"
    "48-50,52-54,56-58,60-62;3,7,11,15,19,23,27,31;35,39,43,47,51,55,59,63"
)
CONTIGUOUS_SET = "0-21;22-42;43-63"


def _owned_from(raw: str, num_layers: int, world: int):
    from sglang.srt.distributed.utils import parse_pp_layer_sets

    return parse_pp_layer_sets(raw, num_layers, world, allow_gapped=True)


class _FakeGroup:
    """Enough GroupCoordinator to exercise the channel, and nothing more.

    ``inbox`` is deliberately NOT provided: the point of the fix is that the
    store is created on the group and shared, so the test must not hand one in.
    """

    def __init__(self, rank_in_group: int, world_size: int):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        #: Messages queued per source rank, in wire order.
        self.wire = {}
        self.sent = []

    def queue_from(self, src: int, message):
        self.wire.setdefault(src, deque()).append(message)

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        resolved = resolve_src(self, src)
        queue = self.wire.get(resolved)
        if not queue:
            raise AssertionError(
                f"blocking receive from rank {resolved} with nothing queued -- "
                f"in a live run this is the wedge, here it is the failure"
            )
        return queue.popleft()

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        self.sent.append((dst, dict(tensor_dict)))
        return []


class TestGappedOwnershipDetection(unittest.TestCase):
    """Which runs need the new protocol, decided from the set itself."""

    def _active(self, layer_set, wire, world=3):
        env = {}
        if layer_set is not None:
            env[PP_LAYER_SET_ENV] = layer_set
        if wire is not None:
            env[PP_CROSSING_WIRE_ENV] = wire
        with mock.patch.dict(os.environ, env, clear=True):
            return pp_gapped_ownership_active(world)

    def test_gapped_set_with_wire_is_active(self):
        self.assertTrue(self._active(GAPPED_SET, "1"))

    def test_contiguous_set_is_not_gapped(self):
        """Step 1 of the design's ladder must keep the ordinary handoff.

        A contiguous split expressed through the set mechanism needs no wire
        and no protocol change; calling it gapped would disable the very
        transport that carries it.
        """
        self.assertFalse(self._active(CONTIGUOUS_SET, "1"))

    def test_gapped_set_without_the_wire_is_not_active(self):
        self.assertFalse(self._active(GAPPED_SET, "0"))

    def test_no_layer_set_is_never_active(self):
        self.assertFalse(self._active(None, "1"))

    def test_single_stage_is_never_active(self):
        self.assertFalse(self._active(GAPPED_SET, "1", world=1))


class TestStageEntryComesFromTheWire(unittest.TestCase):
    """Defect 1: who supplies a stage's first hidden states."""

    def setUp(self):
        self.owned = _owned_from(GAPPED_SET, 64, 3)

    def _wire(self, rank):
        return build_crossing_wire(self.owned, 64, rank, link=object())

    def test_rank_owning_layer_zero_does_not_take_entry_from_the_wire(self):
        """PP0 starts from the embedding, as it always did."""
        wire = self._wire(0)
        self.assertFalse(wire.provides_entry_activations)

    def test_downstream_gapped_stages_take_entry_from_the_wire(self):
        """PP1 and PP2 have no stage-boundary handoff to wait for.

        This is the assertion the v7pp4 wedge violates: before the fix nothing
        reported that these ranks' entry activations arrive mid-loop, so the
        scheduler kept them blocked on a ``proxy`` receive.
        """
        for rank in (1, 2):
            with self.subTest(rank=rank):
                wire = self._wire(rank)
                self.assertTrue(
                    wire.provides_entry_activations,
                    f"rank {rank}'s first owned layer must be a crossing target",
                )

    def test_entry_layer_is_the_first_owned_layer(self):
        """The flag tracks the FIRST owned layer, not merely 'some crossing'."""
        for rank in (1, 2):
            with self.subTest(rank=rank):
                wire = self._wire(rank)
                first_owned = min(self.owned[rank])
                self.assertIn(first_owned, wire.recv_before)

    def test_every_gapped_crossing_is_carried_exactly_once(self):
        """Sends and receives pair up across the three ranks.

        The schedule includes the ordinary stage-boundary crossings as well as
        the mid-loop ones -- which is precisely why the proxy handoff must be
        switched OFF under a gapped set rather than left to run alongside: both
        would carry the same boundary, and the second would strand.
        """
        sends = {}
        receives = {}
        for rank in range(3):
            wire = self._wire(rank)
            for after_layer, (dst, _slot) in wire.send_after.items():
                sends[(rank, dst, after_layer)] = True
            for before_layer, (src, _slot) in wire.recv_before.items():
                receives[(src, rank, before_layer - 1)] = True
        self.assertEqual(
            sorted(sends), sorted(receives), "a send with no matching receive wedges"
        )

    def test_the_last_layers_owner_sends_nothing_onward(self):
        """PP2 owns layer 63, so its forward ends the pass."""
        wire = self._wire(2)
        self.assertEqual(wire.send_after.get(63), None)

    def test_contiguous_ownership_keeps_the_ordinary_protocol(self):
        """The null object comes from the model's entry point, with the wire off.

        ``build_crossing_wire`` describes crossings for any split, boundary
        ones included; what decides the PROTOCOL is
        ``pp_gapped_ownership_active``, and for a contiguous set it says no.
        """
        with mock.patch.dict(
            os.environ,
            {PP_LAYER_SET_ENV: CONTIGUOUS_SET, PP_CROSSING_WIRE_ENV: "1"},
            clear=True,
        ):
            self.assertFalse(pp_gapped_ownership_active(3))

    def test_wire_is_the_null_object_when_switched_off(self):
        from sglang.srt.distributed.pp_crossing_wire import build_wire_for_model

        class _Cfg:
            num_hidden_layers = 64

        class _Grp:
            world_size = 3
            rank_in_group = 1

        with mock.patch.dict(
            os.environ, {PP_LAYER_SET_ENV: GAPPED_SET}, clear=True
        ):  # wire env absent
            wire = build_wire_for_model(_Cfg(), _Grp())
        self.assertIsInstance(wire, NoCrossingWire)
        self.assertFalse(wire.provides_entry_activations)


class TestCrossingsAndOutputsShareOneChannel(unittest.TestCase):
    """Defect 2: two consumers, one wire, and no way to confuse them."""

    def test_output_ring_and_crossing_use_the_same_directed_pair(self):
        """The premise. If this stops holding the collision is gone with it."""
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        # An output receive names no source and resolves to the previous rank.
        self.assertEqual(resolve_src(pp0, None), 2)
        # PP2 owns layer 35, PP0 owns 36: that crossing is PP2 -> PP0.
        owned = _owned_from(GAPPED_SET, 64, 3)
        wire = build_crossing_wire(owned, 64, 0, link=object())
        self.assertIn((2, 0), [(src, 0) for src, _ in wire.recv_before.values()])

    def test_a_crossing_receive_does_not_swallow_an_output_dict(self):
        """The silent-corruption case, stated as an assertion.

        An ``output`` dict carries sampled token ids. Handing it to a decoder
        layer as hidden states does not raise -- it produces confident wrong
        text, which is the #753 defect in a new costume.
        """
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        output_msg = {"__msg_type__": "output", "next_token_ids": "TOKENS"}
        crossing_msg = {"__msg_type__": CROSSING_KIND, "hidden_states": "HIDDEN"}
        # The output arrives FIRST, which is what makes the untyped receive wrong.
        pp0.queue_from(2, output_msg)
        pp0.queue_from(2, crossing_msg)

        got = recv_typed_tensor_dict(pp0, CROSSING_KIND, src=2)
        self.assertEqual(got["hidden_states"], "HIDDEN")

        # And the output was held, not destroyed: its real consumer still gets it.
        held = recv_typed_tensor_dict(pp0, "output", src=None)
        self.assertEqual(held["next_token_ids"], "TOKENS")

    def test_an_output_receive_does_not_swallow_a_crossing(self):
        """The same collision from the other side."""
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        crossing_msg = {"__msg_type__": CROSSING_KIND, "hidden_states": "HIDDEN"}
        output_msg = {"__msg_type__": "output", "next_token_ids": "TOKENS"}
        pp0.queue_from(2, crossing_msg)
        pp0.queue_from(2, output_msg)

        got = recv_typed_tensor_dict(pp0, "output", src=None)
        self.assertEqual(got["next_token_ids"], "TOKENS")

        held = recv_typed_tensor_dict(pp0, CROSSING_KIND, src=2)
        self.assertEqual(held["hidden_states"], "HIDDEN")

    def test_crossings_from_two_peers_are_not_interchangeable(self):
        """Why the key is ``(src, kind)`` and not ``kind``.

        PP0 receives crossings from BOTH PP1 (after layer 31) and PP2 (after
        layer 35). Keyed by kind alone, a crossing stashed from one peer would
        be served to a receive that named the other -- pairing one stage's
        activations with another's place in the schedule.
        """
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        pp0.queue_from(2, {"__msg_type__": CROSSING_KIND, "hidden_states": "FROM_PP2"})
        pp0.queue_from(1, {"__msg_type__": CROSSING_KIND, "hidden_states": "FROM_PP1"})

        # Force a stash from PP2 by asking for something else on that pair.
        pp0.queue_from(2, {"__msg_type__": "output", "next_token_ids": "TOKENS"})
        self.assertEqual(
            recv_typed_tensor_dict(pp0, "output", src=None)["next_token_ids"],
            "TOKENS",
        )
        # PP2's crossing is now stashed. A receive naming PP1 must NOT take it.
        from_pp1 = recv_typed_tensor_dict(pp0, CROSSING_KIND, src=1)
        self.assertEqual(from_pp1["hidden_states"], "FROM_PP1")
        from_pp2 = recv_typed_tensor_dict(pp0, CROSSING_KIND, src=2)
        self.assertEqual(from_pp2["hidden_states"], "FROM_PP2")

    def test_the_inbox_is_one_store_on_the_group(self):
        """Two private stashes would be a race, not a demultiplexer."""
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        stash_typed(pp0, 2, CROSSING_KIND, {"hidden_states": "X"})
        # Reached through the group by any consumer, not through one owner.
        self.assertIsNotNone(take_typed(pp0, 2, CROSSING_KIND))
        self.assertIsNone(take_typed(pp0, 2, CROSSING_KIND))

    def test_send_stamps_the_crossing_kind(self):
        pp0 = _FakeGroup(rank_in_group=0, world_size=3)
        send_typed_tensor_dict(pp0, {"hidden_states": "H"}, 1, CROSSING_KIND)
        dst, payload = pp0.sent[-1]
        self.assertEqual(dst, 1)
        self.assertEqual(payload["__msg_type__"], CROSSING_KIND)


class TestGappedCorridorHoldback(unittest.TestCase):
    """The reserve a gapped rank keeps back, and who it must not touch.

    v7pp8 priced PP2 at weights 6.248 + mamba 0.384 + pool 11.923 GiB against
    an 18.55 GiB budget, left 0.15 GB free, and died on a 32 MiB decode
    allocation. ``rank_user_reserve_mib`` existed for exactly this and had one
    consumer, on the phase-flip path.
    """

    def _call(self, rest, *, layer_set=GAPPED_SET, wire="1", reserve=1024, pp=3):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        class _SA:
            pp_size = pp
            rank_user_reserve_mib = reserve

        class _Stub:
            server_args = _SA()

        env = {PP_CROSSING_WIRE_ENV: wire}
        if layer_set is not None:
            env[PP_LAYER_SET_ENV] = layer_set
        with mock.patch.dict(os.environ, env, clear=True):
            return ModelRunnerKVCacheMixin._gapped_corridor_holdback(_Stub(), rest)

    def test_gapped_rank_holds_the_reserve_back(self):
        rest, post = self._call(11.923)
        self.assertAlmostEqual(rest, 10.923, places=6)
        self.assertIsNotNone(post)
        self.assertAlmostEqual(post[1], 1.0, places=6)

    def test_contiguous_boot_sizing_is_untouched(self):
        """Every shipped configuration must price exactly as before."""
        rest, post = self._call(11.923, layer_set=CONTIGUOUS_SET)
        self.assertEqual(rest, 11.923)
        self.assertIsNone(post)

    def test_boot_without_a_layer_set_is_untouched(self):
        rest, post = self._call(11.923, layer_set=None)
        self.assertEqual(rest, 11.923)
        self.assertIsNone(post)

    def test_reserve_never_drives_the_pool_negative(self):
        """A budget tighter than the reserve is reported, not disguised."""
        rest, post = self._call(0.5)
        self.assertEqual(rest, 0.5)
        self.assertIsNone(post)

    def test_reserve_can_be_switched_off(self):
        rest, post = self._call(11.923, reserve=0)
        self.assertEqual(rest, 11.923)
        self.assertIsNone(post)


if __name__ == "__main__":
    unittest.main()
