"""CPU unit tests for the facility table (what can be measured / controlled,
and what it would take)."""

import unittest

from sglang.srt.rigmon.facilities import (
    CONTROL,
    MEASURE,
    HostEnvironment,
    detect_host_environment,
    facilities,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


LXC_STATUS = "Name:\tpython3\nCapEff:\t000001ffffffffff\nCapBnd:\t000001ffffffffff\n"


def env_reader(files):
    def read(path):
        if path in files:
            return files[path]
        raise FileNotFoundError(path)

    return read


class TestHostDetection(CustomTestCase):
    def test_lxc_with_full_caps_is_detected_as_container(self):
        env = detect_host_environment(
            read_text=env_reader(
                {
                    "/proc/1/environ": "container=lxc\0container_ttys=pts/1\0",
                    "/proc/self/status": LXC_STATUS,
                    "/proc/driver/nvidia/version": (
                        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  "
                        "595.58.03  Tue Jan  1 00:00:00 UTC 2026"
                    ),
                }
            ),
            exists=lambda p: False,
            run=lambda cmd: "lxc\n",
            getuid=lambda: 0,
        )
        self.assertEqual(env.container, "lxc")
        self.assertTrue(env.in_container)
        self.assertTrue(env.is_root)
        self.assertTrue(env.full_caps)
        self.assertEqual(env.driver_version, "595.58.03")

    def test_bare_metal(self):
        env = detect_host_environment(
            read_text=env_reader({"/proc/1/environ": "HOME=/root\0"}),
            exists=lambda p: False,
            run=lambda cmd: "none\n",
            getuid=lambda: 0,
        )
        self.assertFalse(env.in_container)

    def test_docker_via_marker_file(self):
        env = detect_host_environment(
            read_text=env_reader({"/proc/1/environ": ""}),
            exists=lambda p: p == "/.dockerenv",
            run=lambda cmd: None,
            getuid=lambda: 0,
        )
        self.assertEqual(env.container, "docker")


class TestFacilities(CustomTestCase):
    LXC = HostEnvironment(
        container="lxc",
        virt="lxc",
        is_root=True,
        cap_eff=0x1FFFFFFFFFF,
        full_caps=True,
        driver_version="595.58.03",
    )
    HOST = HostEnvironment(container="", is_root=True, driver_version="595.58.03")

    def test_control_in_container_is_blocked_and_says_privilege_will_not_help(self):
        """Full capabilities and root, and the driver still refuses. The remedy
        must not send the user chasing a permission problem."""
        f = {x.key: x for x in facilities(self.LXC, exists=lambda p: True)}
        pt = f["power_target"]
        self.assertFalse(pt.available)
        self.assertEqual(pt.kind, CONTROL)
        self.assertTrue(pt.impossible_in_container)
        self.assertIn("lxc", pt.reason)
        self.assertIn("full capability set", pt.reason)
        joined = " ".join(pt.remedy)
        self.assertIn("hypervisor host", joined)
        self.assertIn("privileged", joined)

    def test_blocked_controls_stay_visible(self):
        f = {x.key: x for x in facilities(self.LXC, exists=lambda p: True)}
        self.assertEqual(f["power_target"].to_json()["ui"], "visible_disabled")

    def test_control_on_host_as_root_is_available(self):
        import shutil

        if not shutil.which("nvidia-smi"):
            self.skipTest("nvidia-smi not on PATH")
        f = {x.key: x for x in facilities(self.HOST, exists=lambda p: True)}
        self.assertTrue(f["power_target"].available)

    def test_rdma_missing_device_node_has_a_container_remedy(self):
        f = {x.key: x for x in facilities(self.LXC, exists=lambda p: False)}
        rd = f["rdma"]
        self.assertFalse(rd.available)
        # A device node CAN be passed in, so this one is not impossible.
        self.assertFalse(rd.impossible_in_container)
        self.assertIn("/dev/infiniband", " ".join(rd.remedy))
        self.assertIn("--device", " ".join(rd.remedy))

    def test_gpm_is_impossible_on_consumer_cards(self):
        f = {
            x.key: x
            for x in facilities(
                self.LXC,
                exists=lambda p: True,
                gpm_supported=False,
                gpm_reason="isSupportedDevice=0",
            )
        }
        prof = f["profiling_counters"]
        self.assertFalse(prof.available)
        self.assertEqual(prof.kind, MEASURE)
        self.assertTrue(prof.impossible_in_container)
        self.assertIn("Hopper", " ".join(prof.remedy))

    def test_gpm_supported_is_available(self):
        f = {
            x.key: x
            for x in facilities(self.HOST, exists=lambda p: True, gpm_supported=True)
        }
        self.assertTrue(f["profiling_counters"].available)

    def test_every_blocked_facility_carries_a_remedy(self):
        for fac in facilities(self.LXC, exists=lambda p: False, gpm_supported=False):
            if not fac.available:
                self.assertTrue(
                    fac.remedy, f"{fac.key} is blocked without a remedy"
                )
                self.assertTrue(fac.reason)

    def test_every_facility_states_its_purpose(self):
        for fac in facilities(self.HOST):
            self.assertTrue(fac.purpose, f"{fac.key} has no stated purpose")


if __name__ == "__main__":
    unittest.main()
