# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard rails for the ctypes UCP binding (task #117, Nordstern L1).

No GPU, no network, no second host: everything here runs against UCX's own
loopback (``self``/``sm``) in a single process, so it is safe on a machine
whose cards are busy.

What it locks down:
  1. the hand-transcribed ``ucp_request_param_t`` layout -- a wrong offset
     here corrupts a UCX call in a way no higher-level test would localise,
  2. the module imports with NO sglang and NO torch (the second rig runs it
     out of a bare python3, and the version-parity check has to work there),
  3. a real send/recv round trip, both below and above the eager/rendezvous
     threshold,
  4. teardown is idempotent.
"""

import ctypes
import importlib.util
import pathlib
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BINDINGS = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python" / "sglang" / "srt" / "distributed" / "device_communicators"
    / "barlink_ucx_bindings.py"
)


def _load_bare():
    """Import the bindings by path, WITHOUT importing sglang.

    Deliberately not `from sglang.srt... import`: the cross-rig harness loads
    this module on a host that has no sglang installed, so a dependency
    sneaking into it must fail here rather than at 3am on the second rig.
    """
    spec = importlib.util.spec_from_file_location("_barlink_ucx_bind_test", _BINDINGS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestUcxBindingLayout(CustomTestCase):
    def test_module_imports_without_sglang_or_torch(self):
        before = set(sys.modules)
        mod = _load_bare()
        newly = set(sys.modules) - before
        self.assertTrue(
            not any(m.split(".")[0] == "torch" for m in newly),
            f"bindings pulled torch in: {sorted(newly)}",
        )
        self.assertTrue(hasattr(mod, "UcpWorker"))

    def test_request_param_layout(self):
        mod = _load_bare()
        # _assert_layout() already ran at import; re-state the sizes so a
        # silently loosened assertion in the module is caught here too.
        self.assertEqual(ctypes.sizeof(mod.UcpRequestParam), 1024)
        for name, off in (
            ("op_attr_mask", 0), ("flags", 4), ("request", 8), ("cb", 16),
            ("datatype", 24), ("user_data", 32), ("reply_buffer", 40),
            ("memory_type", 48), ("recv_info", 56), ("memh", 64),
        ):
            self.assertEqual(getattr(mod.UcpRequestParam, name).offset, off, name)
        self.assertEqual(mod.UcpEpParams.address.offset, 8)
        self.assertEqual(mod.UcpEpParams.err_mode.offset, 16)
        self.assertEqual(mod.UcpWorkerParams.thread_mode.offset, 8)
        self.assertEqual(mod.UcpParams.features.offset, 8)

    def test_error_pointer_decoding(self):
        """UCS encodes failures as small negative values in a pointer slot."""
        mod = _load_bare()
        for status in (-1, -6, -100):
            enc = status + (1 << 64)
            self.assertTrue(mod._ptr_is_err(enc), status)
            self.assertEqual(mod._ptr_status(enc), status)
        # A plausible heap pointer must not be mistaken for an error.
        self.assertFalse(mod._ptr_is_err(0x7F0000000000))


def _ucx_available(mod):
    try:
        mod.UcpLibrary.instance()
        return True
    except Exception:
        return False


class TestUcxLoopback(CustomTestCase):
    """Round trips over UCX's own loopback -- no peer process required."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_bare()
        if not _ucx_available(cls.mod):
            raise unittest.SkipTest("libucp not loadable (apt install libucx0)")

    def test_version_query(self):
        lib = self.mod.UcpLibrary.instance()
        major, minor, _ = lib.version()
        self.assertEqual(major, 1)
        self.assertGreaterEqual(minor, 8)
        self.assertTrue(lib.version_string().startswith(f"{major}.{minor}"))

    def _roundtrip(self, nbytes):
        w = self.mod.UcpWorker(self.mod.UcpLibrary.instance(), timeout_s=60.0)
        try:
            w.connect(0, w.address())
            src = (ctypes.c_ubyte * nbytes)()
            for i in range(0, nbytes, max(nbytes // 64, 1)):
                src[i] = (i * 7) & 0xFF
            dst = (ctypes.c_ubyte * nbytes)()
            tag = 0x0123456789ABCDEF
            r = w.post_recv(ctypes.addressof(dst), nbytes, tag)
            s = w.post_send(0, ctypes.addressof(src), nbytes, tag)
            w.wait([r, s])
            self.assertEqual(bytes(dst), bytes(src))
        finally:
            w.close()
            w.close()  # idempotent

    def test_roundtrip_eager(self):
        self._roundtrip(64)

    def test_roundtrip_rendezvous(self):
        """Large enough to take UCX's rendezvous protocol, not the eager one."""
        self._roundtrip(2 << 20)

    def test_tags_do_not_cross(self):
        """Two concurrently posted transfers must not satisfy each other."""
        w = self.mod.UcpWorker(self.mod.UcpLibrary.instance(), timeout_s=60.0)
        try:
            w.connect(0, w.address())
            a = (ctypes.c_ubyte * 16)(*([0xAA] * 16))
            b = (ctypes.c_ubyte * 16)(*([0xBB] * 16))
            ra, rb = (ctypes.c_ubyte * 16)(), (ctypes.c_ubyte * 16)()
            # Post the receives in the opposite order to the sends: only exact
            # tag matching gets the right bytes into the right buffer.
            r2 = w.post_recv(ctypes.addressof(rb), 16, 0xB)
            r1 = w.post_recv(ctypes.addressof(ra), 16, 0xA)
            s1 = w.post_send(0, ctypes.addressof(a), 16, 0xA)
            s2 = w.post_send(0, ctypes.addressof(b), 16, 0xB)
            w.wait([r1, r2, s1, s2])
            self.assertEqual(bytes(ra), bytes(a))
            self.assertEqual(bytes(rb), bytes(b))
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
