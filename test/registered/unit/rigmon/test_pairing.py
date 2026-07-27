"""Task #214: coupling a second rig, driven entirely from the host.

No network and no second machine: :func:`pairing.fetch_remote` takes an
``opener`` and :class:`pairing.PairingStore` takes the same, which is the
injection seam the rest of rigmon already uses (``EngineScraper(opener=...)``,
``capabilities.ProbeEnv``). The store is driven synchronously here so the
assertions do not race the worker thread; the threading itself is covered by
:class:`TestAdvanceIsNonBlocking`.
"""

import json
import threading
import time
import unittest

from sglang.srt.rigmon import compat, pairing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


REMOTE_IDENTITY = {
    "node_id": "far-rig",
    "commit": "deadbeef",
    "branch": "main",
    "dirty": False,
    "torch_version": "2.11.0+cu130",
    "cuda_version": "13.0",
    "driver_version": "580.00",
    "gpu_archs": ["sm75"],
    "built_archs": ["sm75", "sm86"],
    "gpu_names": ["RTX 2080 Ti"],
    "models": {},
}

REMOTE_CAPS = {
    "engine_seen": False,
    "capabilities": [
        {"key": "nccl_colocation", "label": "NCCL multi-rank per GPU",
         "state": "available", "reason": None, "evidence": {}, "probe": ""},
        {"key": "mps", "label": "CUDA MPS", "state": "unavailable",
         "reason": "no MPS control directory", "evidence": {}, "probe": ""},
        {"key": "rdma", "label": "RDMA", "state": "unavailable",
         "reason": "no InfiniBand devices", "evidence": {}, "probe": ""},
    ],
}


def _opener(nodes=None, snapshot=None, fail=None):
    """A canned remote rigmon."""

    def open_(url, timeout):
        if fail is not None:
            raise fail
        if url.endswith("/api/nodes"):
            body = nodes if nodes is not None else {
                "nodes": [{"node_id": "far-rig", "identity": REMOTE_IDENTITY}]
            }
        else:
            body = snapshot if snapshot is not None else {
                "nodes": {
                    "far-rig": {
                        "state": {
                            "identity": REMOTE_IDENTITY,
                            "capabilities": REMOTE_CAPS,
                        }
                    }
                }
            }
        return json.dumps(body).encode()

    return open_


class TestFetchRemote(CustomTestCase):
    def test_reads_identity_and_capabilities(self):
        v = pairing.fetch_remote("far:8770", opener=_opener())
        self.assertTrue(v.reachable)
        self.assertEqual(v.node_ids, ["far-rig"])
        self.assertEqual(v.identity["node_id"], "far-rig")
        self.assertIsNotNone(v.capabilities)
        self.assertIsNotNone(v.rtt_ms)

    def test_scheme_is_optional(self):
        v = pairing.fetch_remote("far:8770", opener=_opener())
        self.assertTrue(v.url.startswith("http://"))

    def test_unreachable_is_a_state_not_an_exception(self):
        # "the other rig is off" is the most common answer this step gives;
        # it has to render like any other state, with a remedy attached.
        v = pairing.fetch_remote(
            "far:8770", opener=_opener(fail=OSError("connection refused"))
        )
        self.assertFalse(v.reachable)
        self.assertIn("connection refused", v.error)
        self.assertTrue(v.remedy)

    def test_reachable_but_silent_collector_is_not_an_error(self):
        v = pairing.fetch_remote(
            "far:8770", opener=_opener(snapshot={"nodes": {}})
        )
        self.assertTrue(v.reachable)
        # identity still comes off /api/nodes
        self.assertEqual(v.identity["node_id"], "far-rig")


class TestGate(CustomTestCase):
    def _local(self, **over):
        base = dict(REMOTE_IDENTITY, node_id="this-rig")
        base.update(over)
        return compat.NodeIdentity.from_json(base)

    def test_matching_rigs_pass(self):
        rows = pairing.gate_rows(
            self._local(), compat.NodeIdentity.from_json(REMOTE_IDENTITY),
            REMOTE_CAPS, REMOTE_CAPS,
        )
        self.assertFalse([r for r in rows if r.verdict == compat.BLOCK])

    def test_commit_mismatch_blocks_with_a_remedy(self):
        rows = pairing.gate_rows(
            self._local(commit="0000000"),
            compat.NodeIdentity.from_json(REMOTE_IDENTITY),
            REMOTE_CAPS, REMOTE_CAPS,
        )
        blocked = [r for r in rows if r.verdict == compat.BLOCK]
        self.assertTrue(blocked)
        for r in blocked:
            self.assertTrue(r.reason, r.key)
            self.assertTrue(r.remedy, f"{r.key} blocks without telling anyone how to fix it")

    def test_every_unmet_row_carries_reason_and_remedy(self):
        # The table rule: nothing is greyed out silently.
        rows = pairing.gate_rows(
            self._local(), compat.NodeIdentity.from_json(REMOTE_IDENTITY),
            REMOTE_CAPS, REMOTE_CAPS,
        )
        for r in rows:
            if r.verdict in (compat.BLOCK, compat.WARN):
                self.assertTrue(r.reason, r.key)
                self.assertTrue(r.remedy, r.key)

    def test_missing_capability_warns_rather_than_blocks(self):
        # A missing RDMA path is a slower pairing, not an impossible one;
        # blocking here would hide a usable transport.
        rows = pairing.gate_rows(
            self._local(), compat.NodeIdentity.from_json(REMOTE_IDENTITY),
            REMOTE_CAPS, REMOTE_CAPS,
        )
        rdma = [r for r in rows if r.key == "capability:rdma"][0]
        self.assertEqual(rdma.verdict, compat.WARN)
        self.assertIn("TCP", rdma.remedy)

    def test_no_remote_identity_blocks_clearly(self):
        rows = pairing.gate_rows(self._local(), None, REMOTE_CAPS, None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].verdict, compat.BLOCK)
        self.assertIn("collect", rows[0].remedy)


class TestTransportChoice(CustomTestCase):
    def test_nothing_measured_offers_the_probe(self):
        v = pairing.RemoteView(url="http://far:8770", reachable=True)
        out = pairing.transport_rows(None, v)
        self.assertFalse(out["measured"])
        self.assertIn("no transport can be recommended", out["note"])
        self.assertIsNotNone(out["offer"])
        # the offer states its cost, and states that it boots nothing
        self.assertIn("does not boot", out["offer"]["cost"])

    def test_local_only_measurements_are_not_reported_as_measured(self):
        # Having measured THIS rig says nothing about the link to the other
        # one. Calling that "measured" would be exactly the dishonesty the
        # transport module's unknown rows exist to prevent.
        profile = {
            "created": "2026-01-01 00:00:00",
            "probe_seconds": 12.0,
            "gpus": {
                "GPU-aaa": {"name": "RTX 5090", "total_mib": 32607,
                            "gemm_tflops": 100.0, "membw_gbs": 1500.0},
                "GPU-bbb": {"name": "RTX 3080", "total_mib": 20480,
                            "gemm_tflops": 50.0, "membw_gbs": 760.0},
            },
            "links": {"GPU-aaa|GPU-bbb": {"p2p_gbs": 5.0}},
        }
        v = pairing.RemoteView(url="http://far:8770", reachable=True)
        out = pairing.transport_rows(profile, v)
        self.assertFalse(out["measured"])
        self.assertEqual(out["cross_rig_pairs"], 0)
        self.assertIn("Only this rig has been measured", out["note"])
        self.assertIsNotNone(out["offer"])


class TestLaunchConfig(CustomTestCase):
    def test_ends_at_a_configuration_and_boots_nothing(self):
        v = pairing.RemoteView(url="http://far:8770", reachable=True)
        cfg = pairing.launch_config(v, {"pairs": []})
        self.assertFalse(cfg["ready"])
        joined = " ".join(cfg["notes"])
        self.assertIn("not a launch", joined)
        self.assertIn("Nothing has been started", joined)

    def test_no_real_environment_value_is_baked_in(self):
        # The block has to be safe to paste into a repository or an issue.
        v = pairing.RemoteView(url="http://far.example:8770", reachable=True)
        cfg = pairing.launch_config(v, {"pairs": []})
        for k, val in cfg["env"].items():
            self.assertTrue(
                val.startswith("${"), f"{k}={val} is not a placeholder"
            )
            self.assertNotIn("far.example", val)


class TestSessionFlow(CustomTestCase):
    def _store(self, **kw):
        st = pairing.PairingStore(opener=_opener(**kw))
        st.synchronous = True
        return st

    def _matching_store(self):
        """A far rig that mirrors THIS host's identity.

        The gate compares against the real local identity, so a walk that is
        meant to reach the end has to present a remote that genuinely
        matches. Anything else is testing the block path, which
        :meth:`test_blocked_step_stays_current` already covers.
        """
        mine = compat.local_identity("far-rig").to_json()
        st = pairing.PairingStore(
            opener=_opener(
                nodes={"nodes": [{"node_id": "far-rig", "identity": mine}]},
                snapshot={"nodes": {"far-rig": {"state": {
                    "identity": mine, "capabilities": REMOTE_CAPS}}}},
            )
        )
        st.synchronous = True
        return st

    def test_walks_all_four_steps(self):
        st = self._matching_store()
        s = st.create("far:8770")
        self.assertEqual(s.next_step, "reach")
        for _ in pairing.STEPS:
            st.advance(s.session_id)
        self.assertIsNone(s.next_step, [x.to_json() for x in s.steps.values()])
        self.assertTrue(s.to_json()["complete"])

    def test_a_mismatched_rig_stops_at_the_gate(self):
        # The canned remote differs from this host, which is the normal case
        # and must not be walked past.
        st = self._store()
        s = st.create("far:8770")
        for _ in pairing.STEPS:
            st.advance(s.session_id)
        self.assertEqual(s.next_step, "gate")
        self.assertEqual(s.steps["gate"].state, pairing.BLOCKED)
        self.assertTrue(s.steps["gate"].remedy)

    def test_blocked_step_stays_current(self):
        # The flow must not walk past a gate that said no.
        st = self._store(fail=OSError("refused"))
        s = st.create("far:8770")
        st.advance(s.session_id)
        self.assertEqual(s.steps["reach"].state, pairing.BLOCKED)
        self.assertEqual(s.next_step, "reach")
        self.assertTrue(s.blocked)
        self.assertTrue(s.steps["reach"].remedy)

    def test_later_steps_refuse_without_their_predecessor(self):
        st = self._store(fail=OSError("refused"))
        s = st.create("far:8770")
        st.advance(s.session_id, "gate")
        self.assertEqual(s.steps["gate"].state, pairing.BLOCKED)
        self.assertIn("reached", s.steps["gate"].error)

    def test_reset_clears_results_but_keeps_the_target(self):
        st = self._store()
        s = st.create("far:8770")
        st.advance(s.session_id)
        self.assertEqual(s.steps["reach"].state, pairing.OK)
        st.reset(s.session_id)
        self.assertEqual(s.steps["reach"].state, pairing.PENDING)
        self.assertEqual(s.target, "far:8770")

    def test_state_survives_and_is_readable_by_id(self):
        # This is what makes a browser reload resume rather than restart, and
        # what lets a curl-driven flow show up in the dashboard.
        st = self._store()
        s = st.create("far:8770")
        st.advance(s.session_id)
        again = st.get(s.session_id)
        self.assertIsNotNone(again)
        self.assertEqual(again.steps["reach"].state, pairing.OK)

    def test_unknown_session_and_step_are_rejected(self):
        st = self._store()
        with self.assertRaises(KeyError):
            st.advance("nope")
        s = st.create("far:8770")
        with self.assertRaises(ValueError):
            st.advance(s.session_id, "not-a-step")

    def test_empty_target_is_rejected(self):
        st = self._store()
        with self.assertRaises(ValueError):
            st.create("   ")


class TestAdvanceIsNonBlocking(CustomTestCase):
    def test_advance_returns_while_the_step_is_still_running(self):
        # A slow or unreachable far rig must never hold an HTTP request open.
        release = threading.Event()

        def slow(url, timeout):
            release.wait(5)
            return json.dumps({"nodes": []}).encode()

        st = pairing.PairingStore(opener=slow)
        s = st.create("far:8770")
        started = time.time()
        st.advance(s.session_id)
        elapsed = time.time() - started
        self.assertLess(elapsed, 1.0, "advance() blocked on the far rig")
        self.assertEqual(s.steps["reach"].state, pairing.RUNNING)
        release.set()
        for _ in range(100):
            if s.steps["reach"].state != pairing.RUNNING:
                break
            time.sleep(0.05)
        self.assertNotEqual(s.steps["reach"].state, pairing.RUNNING)


if __name__ == "__main__":
    unittest.main()
