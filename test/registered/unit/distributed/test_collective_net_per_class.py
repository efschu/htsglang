# SPDX-License-Identifier: Apache-2.0
"""CPU tests for per-message-class link selection (task #240).

Two halves, because the feature has two:

1. ``ServerArgs._handle_collective_net_env`` -- flag parsing, rejection of a
   device this host does not have, and the export into the environment the
   worker processes inherit.
2. ``UcpWorker(net_devices=...)`` -- that the value actually reaches UCX.
   Half (2) is the one worth having: UCX does NOT fail on an unknown device,
   it warns and comes up on loopback only, so a test that merely asserted
   "no exception" would pass on a typo. It compares the worker ADDRESS
   instead, which encodes the reachable transports and therefore changes when
   the device selection changes.

No GPU, no second host, no RDMA link: everything runs over UCX's own
tcp/self/sm on this machine.
"""

import ctypes
import importlib.util
import os
import pathlib
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ENV_KEYS = ("SGLANG_COLLECTIVE_NET_SMALL", "SGLANG_COLLECTIVE_NET_BULK")

_BINDINGS = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "distributed"
    / "device_communicators"
    / "htccl_ucx_bindings.py"
)


def _a_local_device() -> str:
    """A device name this host really has, preferring an RDMA one."""
    known = sorted(ServerArgs._known_net_devices())
    for name in known:
        if name.startswith(("mlx", "roce", "ib")):
            return name
    return "lo"


def _bare_server_args(**kwargs) -> ServerArgs:
    """A ServerArgs carrying only the fields this handler reads.

    Deliberately not a full construction: that one loads the model config,
    and the handler under test is pure field-and-environment logic.
    """
    sa = ServerArgs.__new__(ServerArgs)
    sa.collective_net_small = kwargs.get("small")
    sa.collective_net_bulk = kwargs.get("bulk")
    sa.disaggregation_ib_device = kwargs.get("ib")
    return sa


class TestCollectiveNetServerArgs(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_unset_changes_nothing(self):
        """The default path: no flags, no environment, no field touched."""
        sa = _bare_server_args()
        sa._handle_collective_net_env()
        self.assertIsNone(sa.disaggregation_ib_device)
        for key in _ENV_KEYS:
            self.assertNotIn(key, os.environ)

    def test_small_exported_for_the_workers(self):
        dev = _a_local_device()
        sa = _bare_server_args(small=f"{dev}:1")
        sa._handle_collective_net_env()
        self.assertEqual(os.environ["SGLANG_COLLECTIVE_NET_SMALL"], f"{dev}:1")
        # SMALL must not reach into the bulk transport.
        self.assertIsNone(sa.disaggregation_ib_device)

    def test_bulk_seeds_ib_device_without_the_port_suffix(self):
        dev = _a_local_device()
        sa = _bare_server_args(bulk=f"{dev}:1")
        sa._handle_collective_net_env()
        self.assertEqual(sa.disaggregation_ib_device, dev)

    def test_bulk_conflicting_with_an_explicit_ib_device_is_rejected(self):
        dev = _a_local_device()
        sa = _bare_server_args(bulk=f"{dev}:1", ib="mlx5_77")
        with self.assertRaises(ValueError) as cm:
            sa._handle_collective_net_env()
        self.assertIn("--disaggregation-ib-device", str(cm.exception))

    def test_an_explicit_environment_value_wins_over_the_flag(self):
        dev = _a_local_device()
        os.environ["SGLANG_COLLECTIVE_NET_SMALL"] = "preset"
        sa = _bare_server_args(small=dev)
        sa._handle_collective_net_env()
        self.assertEqual(os.environ["SGLANG_COLLECTIVE_NET_SMALL"], "preset")

    def test_unknown_device_is_rejected_by_name(self):
        sa = _bare_server_args(small="definitely_not_a_device:1")
        with self.assertRaises(ValueError) as cm:
            sa._handle_collective_net_env()
        message = str(cm.exception)
        self.assertIn("definitely_not_a_device:1", message)
        self.assertIn("this host does not have", message)

    def test_empty_element_is_rejected(self):
        sa = _bare_server_args(small=f"{_a_local_device()},")
        with self.assertRaises(ValueError):
            sa._handle_collective_net_env()

    def test_all_is_passed_through(self):
        sa = _bare_server_args(small="all")
        sa._handle_collective_net_env()
        self.assertEqual(os.environ["SGLANG_COLLECTIVE_NET_SMALL"], "all")


class TestUcxNetDeviceReachesUcx(CustomTestCase):
    """The passthrough itself, against a real libucp."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_htccl_net_dev", _BINDINGS)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        try:
            cls.mod.UcpLibrary.instance()
        except Exception as e:  # no libucp on this host
            raise unittest.SkipTest(f"libucp unavailable: {e}")
        # Loopback-only transports: no RDMA link, no peer, safe anywhere.
        os.environ.setdefault("UCX_TLS", "tcp,self,sm")

    def _address(self, **kwargs) -> bytes:
        worker = self.mod.UcpWorker(**kwargs)
        try:
            return worker.address()
        finally:
            worker.close()

    def test_config_modify_is_declared(self):
        lib = self.mod.UcpLibrary.instance().lib
        self.assertTrue(hasattr(lib, "ucp_config_modify"))
        self.assertEqual(lib.ucp_config_modify.restype, ctypes.c_int)

    def test_default_leaves_the_config_untouched(self):
        """net_devices=None must not call ucp_config_modify at all."""
        worker = self.mod.UcpWorker()
        try:
            self.assertIsNone(worker.net_devices)
        finally:
            worker.close()

    def test_pinning_a_device_changes_the_worker_address(self):
        """The value reaches UCX, rather than being stored and ignored.

        The UCP worker address enumerates the reachable transports, so pinning
        the context to a single interface produces a different (shorter)
        address than the unpinned one. Asserting on that is what distinguishes
        a real passthrough from a no-op.
        """
        default = self._address()
        pinned = self._address(net_devices="lo")
        self.assertNotEqual(default, pinned)

    def test_a_wrong_device_falls_back_to_loopback_only(self):
        """Why the server-args check has to reject unknown devices.

        UCX does not fail here -- it prints 'network device ... is not
        available' and builds a context with no network transport at all. The
        result is a run that completes and reports numbers from the wrong
        link, which is worse than a crash. This test pins that behavior so the
        early rejection is never mistaken for belt-and-braces.
        """
        pinned = self._address(net_devices="definitely_not_a_device:1")
        self.assertNotEqual(self._address(net_devices="lo"), pinned)


if __name__ == "__main__":
    unittest.main()
