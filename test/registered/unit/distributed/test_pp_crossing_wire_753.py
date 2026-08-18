"""#753: the mid-loop crossing wire moves activations, and provably.

``qwen3_5.py``'s loop exchanges ``pp_proxy_tensors`` once per rank. For a
gapped set that skips the peer's layers in silence. #735 delivered the desk
half (which crossings exist, what they cost, which transport carries them) and
moved no tensor. This is the wire.

The load-bearing test here is the BYTE GATE: a full simulated forward driven
through the wire must equal the same layers computed straight through, exactly.
A wire that dropped, duplicated or reordered a crossing changes the number, so
the gate fails rather than "looks close".

The layer function is deliberately order-sensitive
(``h = h * 3 + layer_id``): skipping a layer, applying it twice, or applying
two layers in the wrong order all produce different results. A commutative
stand-in would let a real defect pass.
"""

import os
import types
import unittest

from sglang.srt.distributed.pp_crossing_schedule import crossing_schedule
from sglang.srt.distributed.pp_crossing_wire import (
    CrossingWire,
    NoCrossingWire,
    PpCrossingWireError,
    build_crossing_wire,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

N = 64
FA = [i for i in range(N) if i % 4 == 3]
GDN = [i for i in range(N) if i % 4 != 3]

#: The user's target layout: 48 GDN on the 5090, the 16 FA split across the
#: two 3080s. Gapped by construction.
TARGET = [frozenset(GDN), frozenset(FA[:8]), frozenset(FA[8:])]

CONTIGUOUS = [frozenset(range(0, 31)), frozenset(range(31, 48)), frozenset(range(48, N))]


class _Loopback:
    """One in-process mailbox for every (src, dst, slot). Not a transport."""

    def __init__(self):
        self.box = {}
        self.sends = []

    def send(self, dst, slot, payload, timeout_s):
        self.sends.append((dst, slot))
        self.box[(dst, slot)] = dict(payload)

    def recv(self, src, slot, timeout_s):
        # Keyed by the RECEIVER's view: the sender addressed it to us.
        for (dst, s), payload in list(self.box.items()):
            if s == slot:
                del self.box[(dst, s)]
                return payload
        raise KeyError(f"nothing queued from {src} slot {slot}")


def _owner_of(owned, layer):
    for rank, s in enumerate(owned):
        if layer in s:
            return rank
    raise AssertionError(f"layer {layer} owned by nobody")


def _layer(h, layer_id):
    """Order-sensitive on purpose; see the module docstring."""
    return h * 3 + layer_id


def _straight_through(num_layers, h0):
    h = h0
    for i in range(num_layers):
        h = _layer(h, i)
    return h


def _run_wired(owned, num_layers, h0):
    """Drive a full forward across all ranks through their wires.

    Walks layers in model order and hands the state between rank wires exactly
    as the real loop would: ``before_layer`` on the owner, compute, then
    ``after_layer``.
    """
    link = _Loopback()
    wires = {
        r: build_crossing_wire(owned, num_layers, r, link)
        for r in range(len(owned))
    }
    h, residual = h0, None
    for layer in range(num_layers):
        r = _owner_of(owned, layer)
        h, residual = wires[r].before_layer(layer, h, residual)
        h = _layer(h, layer)
        wires[r].after_layer(layer, h, residual)
    return h, wires


class TestTheContiguousPathIsUntouched(CustomTestCase):
    def test_a_single_stage_yields_the_null_wire(self):
        """Nothing to cross: one stage owns everything."""
        w = build_crossing_wire([frozenset(range(N))], N, 0, _Loopback())
        self.assertIsInstance(w, NoCrossingWire)
        self.assertFalse(w)

    def test_contiguous_ownership_carries_ONLY_the_stage_boundaries(self):
        """ONE mechanism, not two interleaved.

        A contiguous 3-stage split still has 2 crossings -- the stage
        boundaries the old ``pp_proxy_tensors`` return carried. The wire takes
        all of them rather than splitting the job with the old path: two
        mechanisms exchanging activations in the same loop is how they drift
        apart, and the boundary case is the one the old path already got right,
        so it is the cheapest place to prove the wire equals it.
        """
        schedule = crossing_schedule(CONTIGUOUS, N)
        self.assertEqual(len(schedule), 2)
        w0 = build_crossing_wire(CONTIGUOUS, N, 0, _Loopback())
        self.assertEqual(w0.expected_sends(), 1)
        self.assertEqual(w0.expected_receives(), 0)

    def test_the_null_wire_hooks_are_identity(self):
        w = NoCrossingWire()
        h, r = w.before_layer(7, "HIDDEN", "RESIDUAL")
        self.assertEqual((h, r), ("HIDDEN", "RESIDUAL"))
        self.assertIsNone(w.after_layer(7, "HIDDEN", "RESIDUAL"))

    def test_BYTE_GATE_contiguous_through_the_wire_equals_the_old_path(self):
        """The acceptance the task names: routing a contiguous set through the
        wire path must equal the old path exactly."""
        expected = _straight_through(N, 1)
        got, wires = _run_wired(CONTIGUOUS, N, 1)
        self.assertEqual(got, expected)
        # ...and it must have equalled it by actually carrying the two stage
        # boundaries, not by short-circuiting them.
        self.assertEqual(sum(w.crossings_sent for w in wires.values()), 2)
        self.assertEqual(sum(w.crossings_received for w in wires.values()), 2)


class TestTheGappedPathCarriesEveryCrossing(CustomTestCase):
    def test_BYTE_GATE_the_gapped_target_layout_equals_straight_through(self):
        """The whole point: the user's fast layout must compute the same model."""
        expected = _straight_through(N, 1)
        got, _ = _run_wired(TARGET, N, 1)
        self.assertEqual(got, expected)

    def test_the_crossing_count_equals_the_schedule_length(self):
        schedule = crossing_schedule(TARGET, N)
        _, wires = _run_wired(TARGET, N, 1)
        sent = sum(w.crossings_sent for w in wires.values())
        received = sum(w.crossings_received for w in wires.values())
        self.assertEqual(sent, len(schedule))
        self.assertEqual(received, len(schedule))

    def test_the_target_layout_has_the_31_crossings_735_priced(self):
        self.assertEqual(len(crossing_schedule(TARGET, N)), 31)

    def test_the_wire_agrees_with_the_schedule_per_rank(self):
        schedule = crossing_schedule(TARGET, N)
        for r in range(len(TARGET)):
            w = build_crossing_wire(TARGET, N, r, _Loopback())
            self.assertEqual(
                w.expected_sends(), sum(1 for c in schedule if c.src == r)
            )
            self.assertEqual(
                w.expected_receives(), sum(1 for c in schedule if c.dst == r)
            )

    def test_CAN_FAIL_a_dropped_crossing_changes_the_answer(self):
        """Proof the byte gate can fail.

        If the gate passed for a wire that skipped a crossing, it would be
        measuring nothing. Drop one send and the result must diverge.
        """
        link = _Loopback()
        wires = {
            r: build_crossing_wire(TARGET, N, r, link) for r in range(len(TARGET))
        }
        # Silently lose the first crossing this rank would send.
        victim = wires[0]
        self.assertTrue(victim.send_after)
        victim.send_after.pop(sorted(victim.send_after)[0])
        with self.assertRaises((PpCrossingWireError, KeyError)):
            h, residual = 1, None
            for layer in range(N):
                r = _owner_of(TARGET, layer)
                h, residual = wires[r].before_layer(layer, h, residual)
                h = _layer(h, layer)
                wires[r].after_layer(layer, h, residual)


class TestFailuresAreNamedNeverSilent(CustomTestCase):
    def _wire(self, link):
        return build_crossing_wire(TARGET, N, 1, link)

    def test_a_failed_receive_refuses_by_name(self):
        class _Dead:
            def send(self, *a):
                pass

            def recv(self, *a):
                raise OSError("peer gone")

        w = self._wire(_Dead())
        layer = sorted(w.recv_before)[0]
        with self.assertRaises(PpCrossingWireError) as ctx:
            w.before_layer(layer, 1, None)
        msg = str(ctx.exception)
        self.assertIn("#753", msg)
        self.assertIn(str(layer), msg)

    def test_a_none_payload_is_refused_rather_than_used(self):
        class _Empty:
            def send(self, *a):
                pass

            def recv(self, *a):
                return None

        w = self._wire(_Empty())
        layer = sorted(w.recv_before)[0]
        with self.assertRaises(PpCrossingWireError):
            w.before_layer(layer, 1, None)

    def test_a_failed_send_refuses_by_name(self):
        class _Dead:
            def send(self, *a):
                raise OSError("peer gone")

            def recv(self, *a):
                return {"hidden_states": 1, "residual": None}

        w = build_crossing_wire(TARGET, N, 0, _Dead())
        layer = sorted(w.send_after)[0]
        with self.assertRaises(PpCrossingWireError) as ctx:
            w.after_layer(layer, 1, None)
        self.assertIn(str(layer), str(ctx.exception))


class TestTheObservable(CustomTestCase):
    """log_routing had ZERO callers, so '31 crossings observed' was a claim no
    run could evidence. The wire is now its caller."""

    def test_building_with_a_peer_map_routes_and_logs(self):
        import logging

        from sglang.srt.distributed.device_communicators.barlink_peer_transport import (
            resolve_peer_transports,
        )

        uuids = ["GPU-a", "GPU-b", "GPU-c"]
        lanes = {"GPU-a": 8, "GPU-b": 8, "GPU-c": 4}
        pmap = resolve_peer_transports(
            uuids, lanes_by_uuid=lambda u: lanes[u], bar1_p2p_available=True
        )
        log = logging.getLogger("test.753.observable")
        with self.assertLogs(log, level="INFO") as caught:
            build_crossing_wire(TARGET, N, 0, _Loopback(), log=log, peer_map=pmap)
        joined = "\n".join(caught.output)
        self.assertIn("pp crossing routing", joined)
        self.assertIn("31 crossings", joined)


class TestTheModelEntryPoint(CustomTestCase):
    """``build_wire_for_model`` is what the model calls; the default must be
    the null object, so the shipped path is inert."""

    class _Cfg:
        num_hidden_layers = N

    class _Group:
        world_size = 3
        rank_in_group = 0

    def _build(self, env):
        import os
        from unittest.mock import patch

        from sglang.srt.distributed.pp_crossing_wire import build_wire_for_model

        with patch.dict(os.environ, env, clear=False):
            for k in ("SGLANG_PP_LAYER_SET", "SGLANG_PP_CROSSING_WIRE"):
                if k not in env:
                    os.environ.pop(k, None)
            return build_wire_for_model(self._Cfg(), self._Group())

    def test_no_layer_set_gives_the_null_wire(self):
        self.assertIsInstance(self._build({}), NoCrossingWire)

    def test_a_layer_set_WITHOUT_the_wire_flag_gives_the_null_wire(self):
        """The flag is the declaration. Without it the gate refuses gapped sets
        anyway, so building a wire here would be answering a question nobody
        asked."""
        raw = ";".join(
            (",".join(map(str, GDN)), ",".join(map(str, FA[:8])), ",".join(map(str, FA[8:])))
        )
        self.assertIsInstance(
            self._build({"SGLANG_PP_LAYER_SET": raw}), NoCrossingWire
        )

    def test_a_layer_set_WITH_the_wire_flag_builds_the_real_wire(self):
        raw = ";".join(
            (",".join(map(str, GDN)), ",".join(map(str, FA[:8])), ",".join(map(str, FA[8:])))
        )
        w = self._build(
            {"SGLANG_PP_LAYER_SET": raw, "SGLANG_PP_CROSSING_WIRE": "1"}
        )
        self.assertIsInstance(w, CrossingWire)
        schedule = crossing_schedule(TARGET, N)
        self.assertEqual(w.expected_sends(), sum(1 for c in schedule if c.src == 0))


class TestThePpGroupLink(CustomTestCase):
    def test_it_delegates_to_the_group_tensor_dict_path(self):
        from sglang.srt.distributed.pp_crossing_wire import PpGroupLink

        seen = {}

        class _G:
            def send_tensor_dict(self, payload, dst):
                seen["sent"] = (dict(payload), dst)

            def recv_tensor_dict(self, src):
                seen["recv_from"] = src
                return {"hidden_states": 7, "residual": None}

        link = PpGroupLink(_G())
        link.send(2, 0, {"hidden_states": 5, "residual": None}, 1.0)
        self.assertEqual(seen["sent"][1], 2)
        self.assertEqual(seen["sent"][0]["hidden_states"], 5)
        got = link.recv(1, 0, 1.0)
        self.assertEqual(seen["recv_from"], 1)
        self.assertEqual(got["hidden_states"], 7)


if __name__ == "__main__":
    unittest.main()


class TestWireIsScopedToRealPipelines(CustomTestCase):
    """#754 seam, second call site.

    SGLANG_PP_LAYER_SET is process-wide, but a phase flip builds a SECOND model
    -- the TP stack -- in the same process with pp world size 1. get_pp_layer_set
    already answers None there; build_wire_for_model called parse_pp_layer_sets
    raw and inherited its by-stage-count refusal instead, killing a gapped boot
    at 163s inside build_phase_flip_tp_stack.
    """

    GAPPED = (
        "0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,"
        "48-50,52-54,56-58,60-62;3,7,11,15,19,23,27,31;35,39,43,47,51,55,59,63"
    )

    def setUp(self):
        self._env = dict(os.environ)
        os.environ["SGLANG_PP_LAYER_SET"] = self.GAPPED
        os.environ["SGLANG_PP_CROSSING_WIRE"] = "1"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    @staticmethod
    def _cfg():
        cfg = types.SimpleNamespace()
        cfg.num_hidden_layers = 64
        return cfg

    def test_tp_stack_gets_the_null_object_not_a_refusal(self):
        from sglang.srt.distributed.pp_crossing_wire import (
            NoCrossingWire,
            build_wire_for_model,
        )

        group = types.SimpleNamespace(world_size=1, rank_in_group=0)
        wire = build_wire_for_model(self._cfg(), group)
        self.assertIsInstance(wire, NoCrossingWire)

    def test_a_real_pipeline_still_gets_a_real_wire(self):
        """The guard must not disarm the wire it was added next to."""
        from sglang.srt.distributed.pp_crossing_wire import (
            NoCrossingWire,
            build_wire_for_model,
        )

        group = types.SimpleNamespace(world_size=3, rank_in_group=0)
        wire = build_wire_for_model(self._cfg(), group)
        self.assertNotIsInstance(wire, NoCrossingWire)
