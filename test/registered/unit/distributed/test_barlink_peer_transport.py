# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#732: the transport binds per PEER LINK, not per communicator.

``barlink.py:554`` ``_select(op, nbytes)`` picks one transport for the whole
group. The #732 survey found the right shape is per-edge, because the measured
BAR1 standing changes sign with link width: it LOSES on the fast x8 pair
(down to 0.81x, 1-8 MiB) and WINS everywhere on the x4 pair
(``FEATURES_VS_UPSTREAM.md:1349``). A rig whose crossings straddle that
boundary cannot be served by a single verdict.

Three things are pinned here, and each has a can-fail proof:

1. **The width canon** (``CanonTests``). Width must come from the CURRENT
   link, not the maximum. ``nvmlDeviceGetMaxPcieLinkWidth`` reports what the
   CARD can do (x16 for all three cards on the reference rig) while the SLOTS
   are wired x4 / x8 / x8. A resolver built on the max collapses every edge to
   "fast", picks NCCL everywhere, and silently disables this feature on
   exactly the box it exists for. That failure is asserted directly, so the
   canon cannot be quietly relaxed.

2. **Identity** (``IdentityTests``, the #392 falsifier pattern). Lanes are
   resolved by UUID through the ``IdentityMap``. Every rig here has the CUDA
   and NVML orders disagreeing in the reference shape (5090 at CUDA ordinal 0,
   NVML index 1), plus one control rig where they agree. Feeding a CUDA
   ordinal into NVML reads the neighbouring card, which on this fixture
   swaps a x4 edge for a x8 one -- i.e. it inverts the transport choice, not
   merely a log line.

3. **Per-pair dispatch** (``MutationTests``). A map that ignores which pair it
   was asked about must fail. The rig is built asymmetric on purpose so that
   any uniform answer is wrong for at least one edge.

No driver is touched: the NVML device list, the CUDA-ordinal bridge and
pynvml itself are injected.
"""

import logging
import sys
import unittest
from unittest.mock import patch

from sglang.srt.distributed.device_communicators.barlink_peer_transport import (
    FAST_EDGE_LANES,
    PeerBinding,
    PeerTransport,
    PeerTransportMap,
    PeerTransportRefused,
    lanes_by_uuid_via_nvml,
    parse_peer_map_override,
    resolve_peer_transports,
)
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024**2

UUID_3080_X4 = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_X8 = "GPU-cccccccc-0000-0000-0000-000000000003"

BDF_3080_X4 = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"
BDF_3080_X8 = "00000000:41:00.0"

TOTAL_3080 = 20480 * MIB
TOTAL_5090 = 32768 * MIB

#: The physical wiring of the reference rig, as ``CurrPcieLinkWidth`` reports
#: it. This is the ground truth every assertion below is measured against.
CURR_WIDTH = {UUID_3080_X4: 4, UUID_5090: 8, UUID_3080_X8: 8}

#: What ``MaxPcieLinkWidth`` reports: the CARD's capability, identical for all
#: three, and therefore useless for telling the slots apart.
MAX_WIDTH = {UUID_3080_X4: 16, UUID_5090: 16, UUID_3080_X8: 16}


class _Handle:
    def __init__(self, uuid, name):
        self.uuid = uuid
        self.name = name


class _FakePynvml:
    """Enough pynvml for the width read.

    Handles ARE NVML indices, so a caller that passes a CUDA ordinal reads the
    neighbouring card out of this -- which is the whole point of the #392
    fixture.
    """

    NVMLError = Exception

    def __init__(self, devices, *, curr=CURR_WIDTH, max_width=MAX_WIDTH, curr_raises=False):
        self._devices = [_Handle(d.uuid, d.name) for d in devices]
        self._curr = curr
        self._max = max_width
        self._curr_raises = curr_raises

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        return None

    def nvmlDeviceGetCount(self):
        return len(self._devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self._devices[index]

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        if self._curr_raises:
            raise RuntimeError("older binding has no CurrPcieLinkWidth")
        return self._curr[handle.uuid]

    def nvmlDeviceGetMaxPcieLinkWidth(self, handle):
        return self._max[handle.uuid]


def _nvml_bus_order():
    """NVML enumerates in BUS order: x4 3080, 5090, x8 3080."""
    return [
        DeviceInfo(0, UUID_3080_X4, "NVIDIA GeForce RTX 3080", TOTAL_3080, BDF_3080_X4),
        DeviceInfo(1, UUID_5090, "NVIDIA GeForce RTX 5090", TOTAL_5090, BDF_5090),
        DeviceInfo(2, UUID_3080_X8, "NVIDIA GeForce RTX 3080", TOTAL_3080, BDF_3080_X8),
    ]


def _cuda_fastest_first():
    """CUDA enumerates FASTEST_FIRST: the 5090 becomes ordinal 0.

    The divergence: 5090 is CUDA 0 / NVML 1, the x4 3080 is CUDA 1 / NVML 0.
    """
    return {BDF_5090: 0, BDF_3080_X4: 1, BDF_3080_X8: 2}


def _cuda_agrees():
    """The control rig: both orders identical, so conflating them is benign."""
    return {BDF_3080_X4: 0, BDF_5090: 1, BDF_3080_X8: 2}


class _RigCase(CustomTestCase):
    """CUDA and NVML disagreeing, injected at every seam."""

    cuda_bridge = staticmethod(_cuda_fastest_first)
    curr_raises = False

    def setUp(self):
        devices = _nvml_bus_order()
        self.pynvml = _FakePynvml(devices, curr_raises=type(self).curr_raises)
        self._patches = [
            patch.object(registry_nvml, "list_devices", lambda: list(devices)),
            patch.object(
                registry_nvml,
                "_cuda_ordinals_by_bus",
                lambda allow_cuda_init=False: type(self).cuda_bridge(),
            ),
            patch.dict(sys.modules, {"pynvml": self.pynvml}),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for p in self._patches:
            p.stop()


# ===========================================================================
# 1. The width canon: CURRENT link, not maximum
# ===========================================================================


class CanonTests(_RigCase):
    def test_current_width_is_what_the_slots_are(self):
        """The live resolver reports the wiring, 4 / 8 / 8."""
        got = {u: lanes_by_uuid_via_nvml(u) for u in CURR_WIDTH}
        self.assertEqual(got, {UUID_3080_X4: 4, UUID_5090: 8, UUID_3080_X8: 8})

    def test_max_width_would_disable_the_feature_entirely(self):
        """CAN-FAIL PROOF for the canon.

        Had the resolver been built on ``MaxPcieLinkWidth``, every card reads
        x16, every edge clears ``FAST_EDGE_LANES``, and the policy answers
        NCCL for all six directed pairs -- the feature silently off on the one
        rig it was written for. Asserting the damage here is what stops the
        canon from being relaxed to "the max is close enough".
        """
        uuids = [UUID_3080_X4, UUID_5090, UUID_3080_X8]
        by_max = resolve_peer_transports(
            uuids,
            lanes_by_uuid=lambda u: MAX_WIDTH[u],
            bar1_p2p_available=True,
        )
        self.assertEqual(len(by_max), 6)
        self.assertEqual(
            {b.transport for b in by_max},
            {PeerTransport.NCCL_SENDRECV},
            "max-width resolution must collapse every edge to the fast class",
        )
        # And the real canon does not.
        by_curr = resolve_peer_transports(
            uuids,
            lanes_by_uuid=lambda u: CURR_WIDTH[u],
            bar1_p2p_available=True,
        )
        self.assertIn(PeerTransport.BAR1_P2P, {b.transport for b in by_curr})

    def test_missing_current_falls_back_to_max_rather_than_none(self):
        """An older binding must still yield a usable, if unoptimised, map."""
        with patch.dict(
            sys.modules,
            {"pynvml": _FakePynvml(_nvml_bus_order(), curr_raises=True)},
        ):
            self.assertEqual(lanes_by_uuid_via_nvml(UUID_3080_X4), 16)


# ===========================================================================
# 2. Identity: #392 falsifier pattern
# ===========================================================================


class IdentityTests(_RigCase):
    def test_uuid_route_resolves_the_right_card(self):
        self.assertEqual(lanes_by_uuid_via_nvml(UUID_5090), 8)
        self.assertEqual(lanes_by_uuid_via_nvml(UUID_3080_X4), 4)

    def test_cuda_ordinal_fed_to_nvml_inverts_a_transport_choice(self):
        """CAN-FAIL PROOF for the identity route.

        The 5090 is CUDA ordinal 0; NVML index 0 is the x4 3080. A resolver
        that passed the ordinal straight through would read x4 for the 5090
        and x8 for the x4 3080 -- and because the policy switches on width,
        that is not a cosmetic error: the 5090<->x8-3080 edge would flip from
        NCCL to BAR1. Pinned as the exact wrong answer.
        """
        cuda_of = _cuda_fastest_first()
        nvml_order = _nvml_bus_order()
        bdf_to_uuid = {d.pci_bus_id: d.uuid for d in nvml_order}
        ordinal_of_uuid = {bdf_to_uuid[b]: o for b, o in cuda_of.items()}

        def wrong_lanes(uuid):
            # The bug: treat the CUDA ordinal as an NVML index.
            return CURR_WIDTH[nvml_order[ordinal_of_uuid[uuid]].uuid]

        self.assertEqual(wrong_lanes(UUID_5090), 4, "fixture must actually mislead")

        uuids = [UUID_5090, UUID_3080_X8]
        right = resolve_peer_transports(
            uuids, lanes_by_uuid=lambda u: CURR_WIDTH[u], bar1_p2p_available=True
        )
        wrong = resolve_peer_transports(
            uuids, lanes_by_uuid=wrong_lanes, bar1_p2p_available=True
        )
        self.assertEqual(
            right.for_pair(UUID_5090, UUID_3080_X8).transport,
            PeerTransport.NCCL_SENDRECV,
        )
        self.assertEqual(
            wrong.for_pair(UUID_5090, UUID_3080_X8).transport,
            PeerTransport.BAR1_P2P,
            "the ordinal/index conflation must invert this edge, not just log oddly",
        )


class ControlRigTests(_RigCase):
    """The control: orders agree, so the conflation is harmless here."""

    cuda_bridge = staticmethod(_cuda_agrees)

    def test_orders_agree_and_widths_are_unchanged(self):
        got = {u: lanes_by_uuid_via_nvml(u) for u in CURR_WIDTH}
        self.assertEqual(got, {UUID_3080_X4: 4, UUID_5090: 8, UUID_3080_X8: 8})


# ===========================================================================
# 3. Policy
# ===========================================================================


class PolicyTests(unittest.TestCase):
    def _map(self, uuids, **kw):
        kw.setdefault("lanes_by_uuid", lambda u: CURR_WIDTH[u])
        kw.setdefault("bar1_p2p_available", True)
        return resolve_peer_transports(uuids, **kw)

    def test_fast_edge_takes_nccl_and_slow_edge_takes_bar1(self):
        m = self._map([UUID_5090, UUID_3080_X8, UUID_3080_X4])
        self.assertEqual(
            m.for_pair(UUID_5090, UUID_3080_X8).transport, PeerTransport.NCCL_SENDRECV
        )
        self.assertEqual(
            m.for_pair(UUID_5090, UUID_3080_X4).transport, PeerTransport.BAR1_P2P
        )

    def test_edge_width_is_the_bottleneck_not_the_source_card(self):
        """An x8 card talking to an x4 card gets x4 whichever way bytes go."""
        m = self._map([UUID_5090, UUID_3080_X4])
        for a, b in ((UUID_5090, UUID_3080_X4), (UUID_3080_X4, UUID_5090)):
            self.assertEqual(m.for_pair(a, b).lanes, 4)
            self.assertEqual(m.for_pair(a, b).transport, PeerTransport.BAR1_P2P)

    def test_unresolved_width_picks_the_never_catastrophic_transport(self):
        m = self._map([UUID_5090, UUID_3080_X4], lanes_by_uuid=lambda u: None)
        b = m.for_pair(UUID_5090, UUID_3080_X4)
        self.assertIsNone(b.lanes)
        self.assertEqual(b.transport, PeerTransport.NCCL_SENDRECV)
        self.assertIn("unresolved", b.reason)

    def test_map_is_directed_and_complete(self):
        m = self._map([UUID_5090, UUID_3080_X8, UUID_3080_X4])
        self.assertEqual(len(m), 6)  # R*(R-1), no self edges
        for b in m:
            self.assertNotEqual(b.src_uuid, b.dst_uuid)

    def test_unknown_pair_raises_rather_than_defaulting(self):
        m = self._map([UUID_5090, UUID_3080_X4])
        with self.assertRaises(KeyError) as ctx:
            m.for_pair(UUID_5090, UUID_3080_X8)
        self.assertIn("no transport binding", str(ctx.exception))

    def test_two_ranks_on_one_card_is_refused_as_ambiguous(self):
        with self.assertRaises(ValueError) as ctx:
            self._map([UUID_5090, UUID_5090])
        self.assertIn("one card per rank", str(ctx.exception))

    def test_reason_cites_its_evidence(self):
        m = self._map([UUID_5090, UUID_3080_X4, UUID_3080_X8])
        for b in m:
            self.assertIn("FEATURES_VS_UPSTREAM.md:1349", b.reason)


# ===========================================================================
# 4. The loud fallback
# ===========================================================================


class FallbackTests(unittest.TestCase):
    def _map(self, uuids, **kw):
        kw.setdefault("lanes_by_uuid", lambda u: CURR_WIDTH[u])
        return resolve_peer_transports(uuids, **kw)

    def test_absent_kernel_degrades_bar1_edges_to_nccl(self):
        m = self._map([UUID_5090, UUID_3080_X4], bar1_p2p_available=False)
        b = m.for_pair(UUID_5090, UUID_3080_X4)
        self.assertEqual(b.preferred, PeerTransport.BAR1_P2P)
        self.assertEqual(b.transport, PeerTransport.NCCL_SENDRECV)
        self.assertTrue(b.degraded)

    def test_default_is_the_shipping_state(self):
        """``bar1_p2p_available`` defaults False, because today it IS false."""
        m = self._map([UUID_5090, UUID_3080_X4])
        self.assertTrue(m.for_pair(UUID_5090, UUID_3080_X4).degraded)

    def test_fallback_names_itself_and_marks_the_cost_unpriced(self):
        """No invented delta. The note must say the margin is UNMEASURED."""
        m = self._map([UUID_5090, UUID_3080_X4], bar1_p2p_available=False)
        note = m.for_pair(UUID_5090, UUID_3080_X4).note
        self.assertIn("BAR1 p2p kernel unavailable", note)
        self.assertIn("UNMEASURED", note)
        self.assertIn("NOTE_732_transport_selection.md", note)

    def test_fallback_is_logged_at_warning_not_swallowed(self):
        m = self._map([UUID_5090, UUID_3080_X4], bar1_p2p_available=False)
        log = logging.getLogger("test.peer.transport")
        with self.assertLogs(log, level="WARNING") as caught:
            m.log_decisions(log)
        joined = "\n".join(caught.output)
        self.assertIn("FALLBACK", joined)
        self.assertIn("bar1_p2p", joined)

    def test_no_warning_when_nothing_degraded(self):
        """CAN-FAIL guard: the WARNING must be caused by the fallback."""
        m = self._map([UUID_5090, UUID_3080_X8], bar1_p2p_available=False)
        log = logging.getLogger("test.peer.transport.clean")
        with self.assertLogs(log, level="INFO") as caught:
            m.log_decisions(log)
        self.assertFalse([r for r in caught.records if r.levelno >= logging.WARNING])

    def test_fallback_is_not_a_refusal(self):
        m = self._map([UUID_5090, UUID_3080_X4], bar1_p2p_available=False)
        self.assertEqual(m.refusals(), ())
        m.require_no_refusals()  # must not raise


# ===========================================================================
# 5. The override flag
# ===========================================================================


class OverrideParseTests(unittest.TestCase):
    def test_empty_is_empty(self):
        for spec in (None, "", "   "):
            self.assertEqual(parse_peer_map_override(spec), {})

    def test_all_and_pair(self):
        got = parse_peer_map_override("all=nccl_sendrecv,0>1=bar1_p2p")
        self.assertEqual(got[None], PeerTransport.NCCL_SENDRECV)
        self.assertEqual(got[(0, 1)], PeerTransport.BAR1_P2P)

    def test_unknown_transport_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            parse_peer_map_override("all=infiniband")
        self.assertIn("unknown transport", str(ctx.exception))

    def test_self_edge_refused(self):
        with self.assertRaises(ValueError) as ctx:
            parse_peer_map_override("2>2=bar1_p2p")
        self.assertIn("no edge to itself", str(ctx.exception))

    def test_malformed_keys_are_named(self):
        for spec in ("garbage", "0-1=bar1_p2p", "a>b=bar1_p2p"):
            with self.assertRaises(ValueError):
                parse_peer_map_override(spec)


class OverrideApplyTests(unittest.TestCase):
    UUIDS = [UUID_5090, UUID_3080_X8, UUID_3080_X4]

    def _map(self, spec, **kw):
        kw.setdefault("bar1_p2p_available", True)
        return resolve_peer_transports(
            self.UUIDS,
            lanes_by_uuid=lambda u: CURR_WIDTH[u],
            override=parse_peer_map_override(spec),
            **kw,
        )

    def test_all_overrides_every_edge(self):
        m = self._map("all=nccl_sendrecv")
        self.assertEqual({b.transport for b in m}, {PeerTransport.NCCL_SENDRECV})
        self.assertEqual(m.source, "override")

    def test_explicit_pair_beats_all_regardless_of_order(self):
        for spec in ("all=nccl_sendrecv,0>2=bar1_p2p", "0>2=bar1_p2p,all=nccl_sendrecv"):
            m = self._map(spec)
            self.assertEqual(
                m.for_pair(UUID_5090, UUID_3080_X4).transport, PeerTransport.BAR1_P2P
            )
            self.assertEqual(
                m.for_pair(UUID_5090, UUID_3080_X8).transport,
                PeerTransport.NCCL_SENDRECV,
            )

    def test_override_is_one_directed_edge_only(self):
        m = self._map("all=nccl_sendrecv,0>2=bar1_p2p")
        self.assertEqual(
            m.for_pair(UUID_3080_X4, UUID_5090).transport, PeerTransport.NCCL_SENDRECV
        )

    def test_forcing_an_unavailable_transport_refuses_instead_of_degrading(self):
        """An A/B that silently ran the other arm would be worse than a stop."""
        m = self._map("0>2=bar1_p2p", bar1_p2p_available=False)
        b = m.for_pair(UUID_5090, UUID_3080_X4)
        self.assertEqual(b.transport, PeerTransport.REFUSED)
        self.assertTrue(b.refused)
        self.assertIn("forced bar1_p2p", b.note)
        with self.assertRaises(PeerTransportRefused) as ctx:
            m.require_no_refusals()
        self.assertIn("rank 0 -> 2", str(ctx.exception))

    def test_refusal_is_logged_at_warning(self):
        m = self._map("0>2=bar1_p2p", bar1_p2p_available=False)
        log = logging.getLogger("test.peer.transport.refused")
        with self.assertLogs(log, level="WARNING") as caught:
            m.log_decisions(log)
        self.assertIn("REFUSED", "\n".join(caught.output))


# ===========================================================================
# 6. Mutation proof for per-pair dispatch
# ===========================================================================


class MutationTests(unittest.TestCase):
    """A map that ignores which pair it was asked about must fail.

    The rig is asymmetric by construction: with a x4 card and two x8 cards
    there is no single transport that is right for every edge, so any
    pair-blind implementation is wrong somewhere and one of these fails.
    """

    UUIDS = [UUID_5090, UUID_3080_X8, UUID_3080_X4]

    def setUp(self):
        self.m = resolve_peer_transports(
            self.UUIDS,
            lanes_by_uuid=lambda u: CURR_WIDTH[u],
            bar1_p2p_available=True,
        )

    def test_the_rig_admits_no_uniform_answer(self):
        transports = {b.transport for b in self.m}
        self.assertEqual(
            transports, {PeerTransport.NCCL_SENDRECV, PeerTransport.BAR1_P2P}
        )

    def test_swapping_the_queried_pair_changes_the_answer(self):
        fast = self.m.for_pair(UUID_5090, UUID_3080_X8).transport
        slow = self.m.for_pair(UUID_5090, UUID_3080_X4).transport
        self.assertNotEqual(
            fast, slow, "a pair-blind dispatch would return the same transport here"
        )

    def test_every_edge_matches_its_own_width(self):
        for b in self.m:
            expect = (
                PeerTransport.NCCL_SENDRECV
                if b.lanes >= FAST_EDGE_LANES
                else PeerTransport.BAR1_P2P
            )
            self.assertEqual(b.transport, expect, f"wrong transport on {b.describe()}")

    def test_duplicate_directed_pair_is_rejected(self):
        """The map must not silently keep one of two bindings for a pair."""
        b = PeerBinding(
            src_uuid=UUID_5090,
            dst_uuid=UUID_3080_X4,
            src_rank=0,
            dst_rank=2,
            lanes=4,
            preferred=PeerTransport.BAR1_P2P,
            transport=PeerTransport.BAR1_P2P,
            reason="x",
        )
        with self.assertRaises(ValueError):
            PeerTransportMap([b, b])

    def test_describe_names_every_pair_and_flags_the_degraded_ones(self):
        degraded = resolve_peer_transports(
            self.UUIDS,
            lanes_by_uuid=lambda u: CURR_WIDTH[u],
            bar1_p2p_available=False,
        )
        text = degraded.describe()
        self.assertEqual(text.count("rank "), 6)
        self.assertIn("WANTED bar1_p2p", text)


if __name__ == "__main__":
    unittest.main()
