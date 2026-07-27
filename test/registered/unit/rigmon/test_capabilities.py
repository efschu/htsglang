"""CPU unit tests for the probe-derived capability table.

Every probe takes an injected ``ProbeEnv``, so the container case, the
link-down case and the healthy case are all reproducible without the hardware
(or the container) they describe.
"""

import unittest

from sglang.srt.rigmon.capabilities import (
    ACTIVE,
    AVAILABLE,
    UNAVAILABLE,
    UNKNOWN,
    ProbeEnv,
    probe_all,
    probe_p2p,
    probe_python_module,
    probe_rdma,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def fake_fs(tree):
    """``tree``: path -> str (file contents) or list (directory entries)."""

    def exists(p):
        return p in tree

    def listdir(p):
        v = tree[p]
        if not isinstance(v, list):
            raise NotADirectoryError(p)
        return list(v)

    def read_text(p):
        v = tree[p]
        if isinstance(v, list):
            raise IsADirectoryError(p)
        return v

    return exists, listdir, read_text


#: The live layout measured on this host: two RoCE devices, one port ACTIVE at
#: 40 Gb/sec, verbs classes present in sysfs — and /dev/infiniband absent.
LXC_TREE = {
    "/sys/class/infiniband": ["rocep4s0f0", "rocep4s0f1"],
    "/sys/class/infiniband/rocep4s0f0/ports": ["1"],
    "/sys/class/infiniband/rocep4s0f0/ports/1/state": "1: DOWN\n",
    "/sys/class/infiniband/rocep4s0f0/ports/1/phys_state": "3: Disabled\n",
    "/sys/class/infiniband/rocep4s0f0/ports/1/rate": "100 Gb/sec (4X EDR)\n",
    "/sys/class/infiniband/rocep4s0f0/ports/1/link_layer": "Ethernet\n",
    "/sys/class/infiniband/rocep4s0f1/ports": ["1"],
    "/sys/class/infiniband/rocep4s0f1/ports/1/state": "4: ACTIVE\n",
    "/sys/class/infiniband/rocep4s0f1/ports/1/phys_state": "5: LinkUp\n",
    "/sys/class/infiniband/rocep4s0f1/ports/1/rate": "40 Gb/sec (4X QDR)\n",
    "/sys/class/infiniband/rocep4s0f1/ports/1/link_layer": "Ethernet\n",
    "/sys/class/infiniband_verbs": ["uverbs0", "uverbs1"],
}


class TestRdmaProbe(CustomTestCase):
    def test_container_without_dev_infiniband_is_named_not_silently_false(self):
        exists, listdir, read_text = fake_fs(LXC_TREE)
        cap = probe_rdma(
            ProbeEnv(exists=exists, listdir=listdir, read_text=read_text, env={})
        )
        self.assertEqual(cap.state, UNAVAILABLE)
        # The reason must contain the actionable facts, not a bare "no".
        self.assertIn("/dev/infiniband", cap.reason)
        self.assertIn("container", cap.reason)
        self.assertIn("40 Gb/sec", cap.reason)
        self.assertEqual(cap.evidence["ports_up"], 1)
        self.assertFalse(cap.evidence["dev_infiniband"])
        self.assertEqual(cap.evidence["uverbs_class"], ["uverbs0", "uverbs1"])

    def test_no_adapter_at_all(self):
        exists, listdir, read_text = fake_fs({"/sys/class/infiniband": []})
        cap = probe_rdma(
            ProbeEnv(exists=exists, listdir=listdir, read_text=read_text, env={})
        )
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("no RDMA devices", cap.reason)

    def test_link_down_is_distinguished_from_missing_device_nodes(self):
        tree = dict(LXC_TREE)
        tree["/dev/infiniband"] = ["uverbs0"]
        tree["/sys/class/infiniband/rocep4s0f1/ports/1/state"] = "1: DOWN\n"
        exists, listdir, read_text = fake_fs(tree)
        cap = probe_rdma(
            ProbeEnv(exists=exists, listdir=listdir, read_text=read_text, env={})
        )
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("no port is ACTIVE", cap.reason)
        self.assertNotIn("container", cap.reason)

    def test_usable_but_unused_is_available_not_active(self):
        tree = dict(LXC_TREE)
        tree["/dev/infiniband"] = ["uverbs0"]
        exists, listdir, read_text = fake_fs(tree)
        cap = probe_rdma(
            ProbeEnv(
                exists=exists,
                listdir=listdir,
                read_text=read_text,
                run=lambda cmd: (127, "not found"),
                env={},
            )
        )
        self.assertEqual(cap.state, AVAILABLE)
        self.assertIn("does not select an RDMA transport", cap.reason)
        # Missing tooling must not flip the verdict.
        self.assertIn("ibv_devinfo_note", cap.evidence)

    def test_active_when_the_run_selects_an_rdma_transport(self):
        tree = dict(LXC_TREE)
        tree["/dev/infiniband"] = ["uverbs0"]
        exists, listdir, read_text = fake_fs(tree)
        cap = probe_rdma(
            ProbeEnv(
                exists=exists,
                listdir=listdir,
                read_text=read_text,
                run=lambda cmd: (0, "hca_id: rocep4s0f1"),
                env={},
                engine_info={"disaggregation_transfer_backend": "mooncake-rdma"},
            )
        )
        self.assertEqual(cap.state, ACTIVE)

    def test_nccl_ib_disable_does_not_count_as_active(self):
        tree = dict(LXC_TREE)
        tree["/dev/infiniband"] = ["uverbs0"]
        exists, listdir, read_text = fake_fs(tree)
        cap = probe_rdma(
            ProbeEnv(
                exists=exists,
                listdir=listdir,
                read_text=read_text,
                run=lambda cmd: (127, ""),
                env={"NCCL_IB_DISABLE": "1"},
                engine_info={},
            )
        )
        self.assertEqual(cap.state, AVAILABLE)


class TestP2pProbe(CustomTestCase):
    def test_measured_bandwidth_outranks_topology(self):
        """This rig's measured pair bandwidths are single-digit GB/s, i.e.
        host-staged. A measurement present must decide the verdict."""
        env = ProbeEnv(
            hw_profile={
                "links": {
                    "a|b": {"p2p_gbs": 5.11},
                    "a|c": {"p2p_gbs": 9.06},
                    "__group__": {"ar_10kb_us": 31.4},
                }
            },
            run=lambda cmd: (0, "GPU0 GPU1\nOK OK"),
        )
        cap = probe_p2p(env)
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("host-staged", cap.reason)
        self.assertAlmostEqual(cap.evidence["best_pair_gbs"], 9.06)

    def test_fast_pairs_are_available(self):
        env = ProbeEnv(hw_profile={"links": {"a|b": {"p2p_gbs": 240.0}}})
        self.assertEqual(probe_p2p(env).state, AVAILABLE)

    def test_without_probe_topology_is_flagged_as_unconfirmed(self):
        env = ProbeEnv(hw_profile=None, run=lambda cmd: (0, "GPU0\tOK\n"))
        cap = probe_p2p(env)
        self.assertEqual(cap.state, AVAILABLE)
        self.assertIn("has been wrong", cap.reason)

    def test_unknown_when_nothing_can_be_read(self):
        env = ProbeEnv(hw_profile=None, run=lambda cmd: (127, "not found"))
        self.assertEqual(probe_p2p(env).state, UNKNOWN)


class TestModuleProbe(CustomTestCase):
    def test_import_error_text_is_the_reason(self):
        def boom(name):
            raise ImportError("undefined symbol: _ZN3c10 ... built for sm86")

        env = ProbeEnv(import_module=boom)
        cap = probe_python_module(env, "flashinfer", "FlashInfer", "flashinfer")
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("sm86", cap.reason)

    def test_installed_without_engine_cannot_claim_active(self):
        env = ProbeEnv(import_module=lambda n: type("M", (), {"__version__": "1"}))
        cap = probe_python_module(env, "flashinfer", "FlashInfer", "flashinfer")
        self.assertEqual(cap.state, AVAILABLE)
        self.assertIn("no engine reachable", cap.reason)

    def test_active_when_the_engine_selected_it(self):
        env = ProbeEnv(
            import_module=lambda n: type("M", (), {"__version__": "1"}),
            engine_info={"attention_backend": "flashinfer"},
        )
        cap = probe_python_module(
            env,
            "flashinfer",
            "FlashInfer",
            "flashinfer",
            active_when=lambda i: "flashinfer" in str(i.get("attention_backend")),
        )
        self.assertEqual(cap.state, ACTIVE)


class TestFullTable(CustomTestCase):
    def _env(self, **kw):
        exists, listdir, read_text = fake_fs(LXC_TREE)
        base = dict(
            exists=exists,
            listdir=listdir,
            read_text=read_text,
            run=lambda cmd: (127, "not found"),
            import_module=lambda n: (_ for _ in ()).throw(ImportError("absent")),
            env={},
        )
        base.update(kw)
        return ProbeEnv(**base)

    def test_every_capability_has_a_state_and_a_reason_when_negative(self):
        report = probe_all(self._env())
        self.assertTrue(report.capabilities)
        for cap in report.capabilities:
            self.assertIn(cap.state, (ACTIVE, AVAILABLE, UNAVAILABLE, UNKNOWN))
            if cap.state in (UNAVAILABLE, UNKNOWN):
                self.assertTrue(
                    cap.reason, f"{cap.key} is {cap.state} without a reason"
                )

    def test_without_engine_nothing_is_active(self):
        report = probe_all(self._env())
        self.assertFalse(report.engine_seen)
        self.assertIn("active", report.to_json()["note"])
        self.assertFalse([c for c in report.capabilities if c.state == ACTIVE])

    def test_engine_settings_resolve_to_active(self):
        report = probe_all(
            self._env(
                engine_info={
                    "kv_cache_dtype": "fp8_e4m3",
                    "dcp_size": 3,
                    "rank_tp_ratio": "6,1,1",
                    "disable_cuda_graph": False,
                    "speculative_algorithm": "EAGLE",
                }
            )
        )
        states = {c.key: c.state for c in report.capabilities}
        self.assertEqual(states["fp8_kv"], ACTIVE)
        self.assertEqual(states["dcp"], ACTIVE)
        self.assertEqual(states["uneven_tp"], ACTIVE)
        self.assertEqual(states["cuda_graphs"], ACTIVE)
        self.assertEqual(states["speculation"], ACTIVE)

    def test_explicitly_disabled_reports_the_flag_that_disabled_it(self):
        report = probe_all(self._env(engine_info={"disable_cuda_graph": True}))
        cap = report.by_key("cuda_graphs")
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("--disable-cuda-graph", cap.reason)


if __name__ == "__main__":
    unittest.main()
