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

import inspect
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
        """Every shipped configuration must price exactly as before.

        WITHDRAWAL (#774). These two cases used to leave `reserve` at the
        helper's 1024 default while asserting "untouched", which held only
        while the holdback was gated on the gapped path ALONE. `5ea4500466`
        ("An explicitly set --rank-user-reserve-mib is never silently
        dropped") widened the gate to

            if configured is None and not pp_gapped_ownership_active(pp_size):
                return rest_memory, None

        so a reserve the operator actually asked for is honoured on a
        contiguous boot too. Under that rule the old fixture was asserting
        that an EXPLICIT request gets ignored -- which is the defect #774
        exists to fix, not the property these cases are named for.

        `reserve=None` is the correction, and it makes the test finally match
        its own sentence: a shipped configuration is precisely one that does
        not pass the flag, as the production comment says in as many words
        ("they never set this flag, so they are unaffected either way").
        The explicit-flag behaviour is pinned in its own case below rather
        than dropped.
        """
        rest, post = self._call(11.923, layer_set=CONTIGUOUS_SET, reserve=None)
        self.assertEqual(rest, 11.923)
        self.assertIsNone(post)

    def test_an_explicit_reserve_is_honoured_on_a_contiguous_boot(self):
        """#774: asked for and ignored is worse than absent.

        The other half of the widened gate, and the half the two "untouched"
        cases above used to assert the opposite of. A contiguous boot that
        DOES pass --rank-user-reserve-mib must have it priced in.
        """
        rest, post = self._call(11.923, layer_set=CONTIGUOUS_SET, reserve=1024)
        self.assertAlmostEqual(rest, 10.923, places=6)
        self.assertIsNotNone(post)
        self.assertAlmostEqual(post[1], 1.0, places=6)

    def test_an_explicit_zero_reserve_switches_the_holdback_off(self):
        """0 means "no holdback", and must not be read as "unset".

        The production code calls this out at its own gate: "None means
        'unset, take the default'; 0 means 'the operator asked for no
        holdback'. The ``or`` idiom conflates the two and would make the
        reserve impossible to switch off." Pinned here on the gapped path,
        where the default would otherwise apply.
        """
        rest, post = self._call(11.923, reserve=0)
        self.assertEqual(rest, 11.923)
        self.assertIsNone(post)

    def test_boot_without_a_layer_set_is_untouched(self):
        """No layer set and no explicit reserve: the shipped shape, unmoved.

        `reserve=None` for the reason given on the contiguous case above
        (#774 widened the gate; a shipped configuration is one that does not
        pass the flag). This is also the boot shape #774's own measurement
        note describes -- driven by --pp-stage-ratio, so no PP layer set in
        the environment -- which is where an explicit reserve used to be
        discarded without a word.
        """
        rest, post = self._call(11.923, layer_set=None, reserve=None)
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


class TestOutputRingSymmetricGating(unittest.TestCase):
    """#753: send and receive must ask ONE question of ONE slot.

    SPECIMEN v7pp12: the iteration barrier timed out after 120s with 'no peer
    could be proven dead'. The peers were not dead -- they had decided the
    output exchange was not due, because the send gate read
    ``mbs[(mb_id + pp_size) % pp_loop_size]`` and the receive gate read
    ``mbs[(mb_id + 1) % pp_loop_size]``. Those name the same batch only while
    the pipeline stagger puts the ranks on different slots at the same instant.
    A gapped forward drives the offsets to zero, and the two gates then name
    different batches.
    """

    def _slots(self, pp_size, pp_loop_size, mb_id=0):
        """The two gate indices, exactly as the event loop computes them."""
        send_gate = (mb_id + pp_size) % pp_loop_size
        recv_gate = (mb_id + 1) % pp_loop_size
        return send_gate, recv_gate

    def test_the_staggered_ring_splits_the_gates(self):
        """The premise. At the ordinary ring size the two gates DIFFER."""
        send_gate, recv_gate = self._slots(pp_size=3, pp_loop_size=3)
        self.assertNotEqual(
            send_gate,
            recv_gate,
            "if these ever coincide at loop size 3 the v7pp12 diagnosis is wrong",
        )

    def test_one_slot_collapses_both_gates(self):
        """The fix, stated as arithmetic rather than as hope."""
        for pp_size in (2, 3, 4):
            with self.subTest(pp_size=pp_size):
                send_gate, recv_gate = self._slots(pp_size=pp_size, pp_loop_size=1)
                self.assertEqual(send_gate, 0)
                self.assertEqual(recv_gate, 0)

    def test_gapped_boot_selects_one_slot(self):
        """A gapped run must actually take the one-slot path.

        Guards the specific regression this commit undoes: d139c463cc removed
        the pp_loop_size=1 selection on a wrong reading of why v7pp5 starved.
        """
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.SchedulerPPMixin.init_pp_loop_state)
        self.assertIn("self.pp_loop_size = 1", src)

    def test_exchange_predicate_is_one_expression(self):
        """Both sides call the same function, so they cannot drift apart."""
        from sglang.srt.managers.scheduler_pp_mixin import _pp_output_exchange_due

        class _Mode:
            def __init__(self, prebuilt):
                self._p = prebuilt

            def is_prebuilt(self):
                return self._p

        class _Batch:
            def __init__(self, prebuilt=False, n=2, last_chunk=True):
                self.forward_mode = _Mode(prebuilt)
                self.reqs = [object()] * n
                self.contains_last_prefill_chunk = last_chunk
                self.return_logprob = False

        self.assertFalse(_pp_output_exchange_due(None))
        self.assertFalse(_pp_output_exchange_due(_Batch(prebuilt=True)))
        self.assertTrue(_pp_output_exchange_due(_Batch()))

    def test_send_side_uses_the_shared_predicate(self):
        """The last rank's gate is the function, not a re-spelling of it."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.SchedulerPPMixin._pp_send_output_to_next_stage)
        self.assertIn("_pp_output_exchange_due(target)", src)

    def test_split_slots_are_refused_on_the_gapped_path(self):
        """A future ring-size change must fail loudly, not starve silently."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors)
        self.assertIn("next_first_rank_mb_id != next_mb_id", src)


class TestIdleTickSendsNothing(unittest.TestCase):
    """#753: receiving nothing must mean sending nothing.

    SPECIMEN v7pp17. The server fired up, warmed up symmetrically on all three
    ranks, went idle, and the iteration barrier then timed out after 120s with
    'no peer could be proven dead'. No request was ever scheduled. The peer was
    not dead: a middle rank still held pp_outputs from the last pass that DID
    exchange, and on an idle tick it sent them to a peer whose receive had
    early-returned for want of a batch. Gapped sends are synchronous, so that
    unmatched send blocked for ever and the rank never reached the barrier.
    """

    def test_none_is_distinguishable_from_not_supplied(self):
        """The sentinel exists and is not None.

        A None default cannot express both 'no value given' and 'the value is
        nothing', which is precisely the conflation that stalled v7pp17.
        """
        from sglang.srt.managers.scheduler_pp_mixin import _NOT_SUPPLIED

        self.assertIsNotNone(_NOT_SUPPLIED)

    def test_do_send_defaults_to_the_sentinel(self):
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(
            m.SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors
        )
        self.assertIn("forward_now=_NOT_SUPPLIED", src)
        self.assertIn("forward_now is _NOT_SUPPLIED", src)

    def test_stale_outputs_are_not_reachable_when_nothing_was_received(self):
        """Forwarding an explicit None must not fall back to pp_outputs."""
        sentinel_src = inspect.getsource(
            __import__(
                "sglang.srt.managers.scheduler_pp_mixin", fromlist=["x"]
            ).SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors
        )
        # The fallback must be keyed on the sentinel, never on `is None`.
        self.assertNotIn("pp_outputs if forward_now is None else", sentinel_src)


class TestGappedForwardIsRefused(unittest.TestCase):
    """#753: the known-wrong gapped forward may not be served by accident.

    The probe that condemns it: 'The capital of France is' at temperature 0,
    seed 735000001, returns '\\n\\n' on the gapped cut and 'Paris' on the same
    checkpoint under a contiguous layout.
    """

    def test_refusal_fires_by_default(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            PP_GAPPED_KNOWN_WRONG_ENV,
            _refuse_known_wrong_gapped_forward,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _refuse_known_wrong_gapped_forward()
        msg = str(ctx.exception)
        # The refusal must NAME the defect, not merely decline.
        self.assertIn("REFUSED", msg)
        self.assertIn("Paris", msg)
        self.assertIn(PP_GAPPED_KNOWN_WRONG_ENV, msg)

    def test_escape_hatch_allows_debugging(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            PP_GAPPED_KNOWN_WRONG_ENV,
            _refuse_known_wrong_gapped_forward,
        )

        with mock.patch.dict(
            os.environ, {PP_GAPPED_KNOWN_WRONG_ENV: "1"}, clear=True
        ):
            _refuse_known_wrong_gapped_forward()  # must not raise

    def test_falsey_values_do_not_open_the_hatch(self):
        """'0' and '' must keep the refusal shut."""
        from sglang.srt.managers.scheduler_pp_mixin import (
            PP_GAPPED_KNOWN_WRONG_ENV,
            _refuse_known_wrong_gapped_forward,
        )

        for value in ("0", "", "false", "False"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {PP_GAPPED_KNOWN_WRONG_ENV: value}, clear=True
                ):
                    with self.assertRaises(ValueError):
                        _refuse_known_wrong_gapped_forward()

    def test_the_gapped_init_path_calls_the_refusal(self):
        """The gate must sit on the path a gapped boot actually takes."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.SchedulerPPMixin.init_pp_loop_state)
        self.assertIn("_refuse_known_wrong_gapped_forward()", src)

    def test_contiguous_boots_never_reach_the_gate(self):
        """A contiguous layout must be entirely unaffected."""
        with mock.patch.dict(
            os.environ,
            {PP_LAYER_SET_ENV: CONTIGUOUS_SET, PP_CROSSING_WIRE_ENV: "1"},
            clear=True,
        ):
            self.assertFalse(pp_gapped_ownership_active(3))


if __name__ == "__main__":
    unittest.main()
