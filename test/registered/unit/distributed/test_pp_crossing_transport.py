# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#735: the crossing schedule driven over the real per-peer transport map.

The first point where the two desk-proven halves touch. Until now the schedule
was driven through a ``LoopbackLink`` double -- the ``Link`` SHAPE, with no
transport behind it -- and the transport map was resolved against no schedule.

The fixture is the actual family plan, not a toy: 64 layers, full attention
every 4th layer at [3, 7, ..., 63], GDN on the 5090 and the 16 FA layers split
8/8 across the two 3080s, with layer 63 (the TERMINAL full-attention layer,
which costs one crossing rather than two) on the x4 card per the #732
placement lever. That map yields 31 crossings, and the rig it runs on has one
x4 edge and one x8 edge -- so no single transport is right for all of them,
which is the whole point.

Rank -> card, fixed by the fixture:

    rank 0 = 5090        (GDN host, x8 slot)
    rank 1 = 3080 x8     -> the fast edge, where BAR1 is measured to LOSE
    rank 2 = 3080 x4     -> the slow edge, where BAR1 wins, and where the
                            terminal layer is placed
"""

import logging
import unittest

from sglang.srt.distributed.device_communicators.barlink_peer_transport import (
    PeerTransport,
    resolve_peer_transports,
)
from sglang.srt.distributed.pp_crossing_schedule import (
    Crossing,
    LoopbackLink,
    crossing_schedule,
    schedule_cost,
)
from sglang.srt.distributed.pp_crossing_transport import (
    MEASURED_GBPS_BY_LANES,
    RoutedLink,
    UnroutableCrossing,
    log_routing,
    per_pair_us_from_map,
    route_schedule,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_X8 = "GPU-cccccccc-0000-0000-0000-000000000003"
UUID_3080_X4 = "GPU-aaaaaaaa-0000-0000-0000-000000000001"

#: rank -> uuid. Rank 0 hosts GDN, ranks 1 and 2 split the FA layers.
WORLD = [UUID_5090, UUID_3080_X8, UUID_3080_X4]
LANES = {UUID_5090: 8, UUID_3080_X8: 8, UUID_3080_X4: 4}

NUM_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
FA_LAYERS = [i for i in range(NUM_LAYERS) if i % FULL_ATTENTION_INTERVAL == 3]

#: 5 MiB: one chunk of 512 tokens x hidden 5120 x 2 bytes.
PAYLOAD_BYTES = 5 * 1024 * 1024


def _family_owned():
    """The family map: GDN on rank 0, FA split 8/8 with layer 63 on rank 2."""
    assert len(FA_LAYERS) == 16, FA_LAYERS
    assert FA_LAYERS[-1] == 63, FA_LAYERS
    fa_first, fa_last = FA_LAYERS[:8], FA_LAYERS[8:]
    gdn = [i for i in range(NUM_LAYERS) if i not in set(FA_LAYERS)]
    return [frozenset(gdn), frozenset(fa_first), frozenset(fa_last)]


def _map(*, bar1_p2p_available=True, world=WORLD, lanes=None):
    lanes = lanes or LANES
    return resolve_peer_transports(
        world,
        lanes_by_uuid=lambda u: lanes.get(u),
        bar1_p2p_available=bar1_p2p_available,
    )


class TestTheFixtureIsTheRealPlan(CustomTestCase):
    """If the fixture is not the family plan, nothing below means anything."""

    def test_thirty_one_crossings(self):
        sched = crossing_schedule(_family_owned(), NUM_LAYERS)
        self.assertEqual(len(sched), 31)

    def test_the_terminal_layer_sits_on_the_x4_card(self):
        owned = _family_owned()
        self.assertIn(63, owned[2])

    def test_the_rig_has_both_an_x4_and_an_x8_edge(self):
        """No single transport can be right for this schedule."""
        m = _map()
        self.assertEqual(m.for_ranks(0, 1).lanes, 8)
        self.assertEqual(m.for_ranks(0, 2).lanes, 4)


class TestEveryCrossingResolves(CustomTestCase):
    def test_a_gapped_schedule_resolves_every_crossing_to_a_named_transport(self):
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        self.assertEqual(len(routes), 31)
        for r in routes:
            self.assertIn(
                r.transport,
                (PeerTransport.NCCL_SENDRECV, PeerTransport.BAR1_P2P),
                msg=r.describe(),
            )
            self.assertFalse(r.refused, msg=r.describe())

    def test_the_split_follows_the_732_recommendation(self):
        """x8 edges take NCCL, x4 edges take BAR1 -- per edge, not per group."""
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        for r in routes:
            expect = (
                PeerTransport.NCCL_SENDRECV
                if r.binding.lanes >= 8
                else PeerTransport.BAR1_P2P
            )
            self.assertEqual(r.transport, expect, msg=r.describe())

    def test_both_transports_are_actually_used(self):
        """CAN-FAIL guard: a uniform answer would satisfy the test above."""
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        self.assertEqual(
            {r.transport for r in routes},
            {PeerTransport.NCCL_SENDRECV, PeerTransport.BAR1_P2P},
        )

    def test_the_x4_edge_is_the_minority_by_count_and_the_majority_by_cost(self):
        """The #732 finding, stated precisely -- it is about TIME, not count.

        The terminal layer sits on the x4 card, so that edge carries 15 of the
        31 crossings and the x8 edge carries 16: a MINORITY by count. But each
        x4 crossing is 1.78x slower, so the slow edge takes ~62 % of the pass's
        crossing time. Conflating the two is easy and this test exists to stop
        it -- the transport recommendation is driven by the cost share, and an
        assertion on the count would have been false.
        """
        sched = crossing_schedule(_family_owned(), NUM_LAYERS)
        routes = route_schedule(sched, _map())
        on_x4 = [r for r in routes if r.binding.lanes == 4]
        self.assertEqual(len(on_x4), 15)
        self.assertLess(len(on_x4), len(routes) / 2)

        per_pair = per_pair_us_from_map(_map(), PAYLOAD_BYTES)
        x4_us = sum(per_pair[c.pair] for c in sched if per_pair[c.pair] > 900)
        total_us = schedule_cost(sched, per_pair, 0.0)
        self.assertAlmostEqual(x4_us / total_us, 0.625, delta=0.02)

    def test_a_schedule_the_map_does_not_cover_is_refused_by_name(self):
        sched = [Crossing(after_layer=0, src=0, dst=9, slot=0)]
        with self.assertRaises(UnroutableCrossing) as ctx:
            route_schedule(sched, _map())
        self.assertIn("after layer 0", str(ctx.exception))
        self.assertIn("no transport binding", str(ctx.exception))


class TestTheLoudFallback(CustomTestCase):
    def test_absent_kernel_degrades_the_x4_edge_loudly(self):
        routes = route_schedule(
            crossing_schedule(_family_owned(), NUM_LAYERS),
            _map(bar1_p2p_available=False),
        )
        degraded = [r for r in routes if r.degraded]
        self.assertTrue(degraded)
        for r in degraded:
            self.assertEqual(r.transport, PeerTransport.NCCL_SENDRECV)
            self.assertEqual(r.binding.preferred, PeerTransport.BAR1_P2P)

    def test_the_warning_is_per_pair_and_carries_the_crossing_count(self):
        """Per pair, not per crossing: 15 identical lines would bury it."""
        routes = route_schedule(
            crossing_schedule(_family_owned(), NUM_LAYERS),
            _map(bar1_p2p_available=False),
        )
        log = logging.getLogger("test.pp.routing")
        with self.assertLogs(log, level="WARNING") as caught:
            log_routing(routes, log)
        warnings = [r for r in caught.records if r.levelno >= logging.WARNING]
        pairs = {r.crossing.pair for r in routes if r.degraded}
        self.assertEqual(len(warnings), len(pairs))
        joined = "\n".join(caught.output)
        self.assertIn("FALLBACK", joined)
        self.assertIn("of 31 crossings", joined)

    def test_no_warning_when_nothing_degrades(self):
        """CAN-FAIL guard: the warning must be caused by the degradation."""
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        log = logging.getLogger("test.pp.routing.clean")
        with self.assertLogs(log, level="INFO") as caught:
            log_routing(routes, log)
        self.assertFalse([r for r in caught.records if r.levelno >= logging.WARNING])


class TestPricingComesFromTheSameMap(CustomTestCase):
    def test_priced_pairs_use_the_measured_rows(self):
        per_pair = per_pair_us_from_map(_map(), PAYLOAD_BYTES)
        expect_x8 = PAYLOAD_BYTES / (MEASURED_GBPS_BY_LANES[8] * 1e9) * 1e6
        expect_x4 = PAYLOAD_BYTES / (MEASURED_GBPS_BY_LANES[4] * 1e9) * 1e6
        self.assertAlmostEqual(per_pair[(0, 1)], expect_x8, places=3)
        self.assertAlmostEqual(per_pair[(0, 2)], expect_x4, places=3)
        self.assertGreater(per_pair[(0, 2)], per_pair[(0, 1)])

    def test_an_unmeasured_lane_count_is_omitted_not_invented(self):
        """The gap is handed over intact; schedule_cost owns the fallback."""
        odd = _map(lanes={UUID_5090: 8, UUID_3080_X8: 8, UUID_3080_X4: 2})
        per_pair = per_pair_us_from_map(odd, PAYLOAD_BYTES)
        self.assertIn((0, 1), per_pair)
        self.assertNotIn((0, 2), per_pair)

    def test_an_unpriced_pair_falls_back_rather_than_costing_zero(self):
        odd = _map(lanes={UUID_5090: 8, UUID_3080_X8: 8, UUID_3080_X4: 2})
        sched = crossing_schedule(_family_owned(), NUM_LAYERS)
        per_pair = per_pair_us_from_map(odd, PAYLOAD_BYTES)
        default = 12345.0
        cost = schedule_cost(sched, per_pair, default)
        unpriced = [c for c in sched if c.pair not in per_pair]
        self.assertTrue(unpriced)
        self.assertGreaterEqual(cost, default * len(unpriced))

    def test_the_pass_cost_matches_the_732_arithmetic(self):
        """~24.7 ms for the 8/8 map with the terminal layer on the x4 card."""
        sched = crossing_schedule(_family_owned(), NUM_LAYERS)
        per_pair = per_pair_us_from_map(_map(), PAYLOAD_BYTES)
        total_ms = schedule_cost(sched, per_pair, 0.0) / 1000.0
        self.assertAlmostEqual(total_ms, 24.68, delta=0.15)


class TestRoutedLinkDispatch(CustomTestCase):
    """The junction proper: per-pair dispatch through the Link shape."""

    def _backends(self):
        return {
            PeerTransport.NCCL_SENDRECV: LoopbackLink(),
            PeerTransport.BAR1_P2P: LoopbackLink(),
        }

    def test_send_uses_the_transport_of_its_own_outgoing_edge(self):
        backends = self._backends()
        link = RoutedLink(0, _map(), backends)
        link.send(1, 0, "payload-x8", 1.0)
        link.send(2, 0, "payload-x4", 1.0)
        self.assertEqual(len(backends[PeerTransport.NCCL_SENDRECV].sent), 1)
        self.assertEqual(len(backends[PeerTransport.BAR1_P2P].sent), 1)

    def test_recv_resolves_the_incoming_direction(self):
        """MUTATION TARGET, and it needs a genuinely ASYMMETRIC map.

        ``recv(src)`` must look up ``src -> self.rank``. A junction that used
        ``self.rank -> src`` would route the REVERSE edge.

        The default policy cannot catch that: the edge width is the bottleneck
        of the two cards, ``min(a, b)``, so both directions of a pair always
        resolve to the same transport and a direction-blind lookup is
        invisible. An earlier version of this test claimed otherwise and a
        mutation survived it.

        The override makes the map asymmetric on purpose -- 0 -> 2 forced to
        NCCL while 2 -> 0 keeps the policy's BAR1 -- which is exactly the
        asymmetry the directed algebra exists to support.
        """
        asymmetric = resolve_peer_transports(
            WORLD,
            lanes_by_uuid=lambda u: LANES.get(u),
            bar1_p2p_available=True,
            override={(0, 2): PeerTransport.NCCL_SENDRECV},
        )
        self.assertEqual(
            asymmetric.for_ranks(0, 2).transport, PeerTransport.NCCL_SENDRECV
        )
        self.assertEqual(asymmetric.for_ranks(2, 0).transport, PeerTransport.BAR1_P2P)

        backends = self._backends()
        nccl, bar1 = (
            backends[PeerTransport.NCCL_SENDRECV],
            backends[PeerTransport.BAR1_P2P],
        )
        bar1._src, bar1._dst = 2, 0
        bar1.send(0, 7, "from-x4", 1.0)

        link = RoutedLink(0, asymmetric, backends)
        got = link.recv(2, 7, 1.0)
        self.assertEqual(got, "from-x4")
        self.assertEqual(
            len(nccl.received), 0, "an incoming x4 crossing must not hit the NCCL backend"
        )

    def test_a_missing_backend_refuses_by_name(self):
        link = RoutedLink(0, _map(), {PeerTransport.NCCL_SENDRECV: LoopbackLink()})
        link.send(1, 0, "fine", 1.0)  # x8 edge has a backend
        with self.assertRaises(UnroutableCrossing) as ctx:
            link.send(2, 0, "no backend", 1.0)
        msg = str(ctx.exception)
        self.assertIn("bar1_p2p", msg)
        self.assertIn("no backend is registered", msg)
        self.assertIn("nobody chose", msg)

    def test_a_refused_binding_is_not_sent_anyway(self):
        forced = resolve_peer_transports(
            WORLD,
            lanes_by_uuid=lambda u: LANES.get(u),
            bar1_p2p_available=False,
            override={(0, 2): PeerTransport.BAR1_P2P},
        )
        self.assertTrue(forced.for_ranks(0, 2).refused)
        link = RoutedLink(0, forced, self._backends())
        with self.assertRaises(UnroutableCrossing) as ctx:
            link.send(2, 0, "must not go", 1.0)
        self.assertIn("REFUSED", str(ctx.exception))

    def test_driving_the_whole_schedule_touches_both_backends(self):
        """End to end: 31 crossings over a routed link, nothing unrouted."""
        backends = self._backends()
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        for r in routes:
            link = RoutedLink(r.crossing.src, _map(), backends)
            backend = backends[r.transport]
            backend._src, backend._dst = r.crossing.src, r.crossing.dst
            link.send(r.crossing.dst, r.crossing.slot, r.describe(), 1.0)
        total = sum(len(b.sent) for b in backends.values())
        self.assertEqual(total, 31)
        self.assertTrue(backends[PeerTransport.NCCL_SENDRECV].sent)
        self.assertTrue(backends[PeerTransport.BAR1_P2P].sent)


class TestJunctionMutation(CustomTestCase):
    """Wrong-pair routing must fail, not merely look different."""

    def test_a_pair_blind_router_would_misroute_this_schedule(self):
        """The schedule admits no uniform transport, so blindness is visible."""
        routes = route_schedule(crossing_schedule(_family_owned(), NUM_LAYERS), _map())
        by_pair = {}
        for r in routes:
            by_pair.setdefault(r.crossing.pair, set()).add(r.transport)
        for pair, transports in by_pair.items():
            self.assertEqual(len(transports), 1, f"{pair} routed inconsistently")
        self.assertGreater(
            len({next(iter(t)) for t in by_pair.values()}),
            1,
            "fixture must contain pairs that route differently",
        )

    def test_swapping_the_direction_changes_the_transport(self):
        """Directedness is load-bearing, not decorative."""
        m = _map()
        self.assertEqual(m.for_ranks(0, 1).transport, PeerTransport.NCCL_SENDRECV)
        self.assertEqual(m.for_ranks(0, 2).transport, PeerTransport.BAR1_P2P)

    def test_routing_reads_the_crossings_own_pair(self):
        """A router returning the first binding for everything must fail."""
        sched = crossing_schedule(_family_owned(), NUM_LAYERS)
        routes = route_schedule(sched, _map())
        for r in routes:
            self.assertEqual(
                (r.binding.src_rank, r.binding.dst_rank),
                r.crossing.pair,
                msg=r.describe(),
            )


if __name__ == "__main__":
    unittest.main()
