"""#1321: the Weg-2 front serves on the LAN (user order 2026-09-10).

Three facts are pinned: the front argv carries ``--host`` with the launcher's
``--front-host`` (default 0.0.0.0), the flag parses with that default, and the
pre-spawn bind probe refuses BY NAME while another listener holds the port
(the temporary LAN forwarder is the case that exists today) and stays quiet
on a free port and on a dry run.
"""

import socket
import unittest
from types import SimpleNamespace

from sglang.srt.weg2 import launcher as L


def _ns(**kw):
    base = dict(
        tag="t",
        fairness_w_s=45.0,
        drain_deadline_s=90.0,
        min_dwell_ms=None,
        d_admit_max_tokens=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _argv(**kw):
    return L.front_argv_for(
        "py", "/store", 1, 2, {}, [], _ns(), 0, 0, 8, 8, 22000, 22000, "D", **kw
    )


class TestFrontHostArgv(unittest.TestCase):
    def test_the_front_binds_the_lan_by_default(self):
        argv = _argv()
        self.assertEqual(L.DEFAULT_FRONT_HOST, "0.0.0.0")
        i = argv.index("--host")
        self.assertEqual(argv[i + 1], "0.0.0.0")
        # the port is still there and still the front's port
        self.assertEqual(argv[argv.index("--port") + 1], str(L.PORT_FRONT))

    def test_the_flag_overrides_the_bind_address(self):
        argv = _argv(front_host="127.0.0.1")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")

    def test_the_parser_defaults_front_host_to_the_lan(self):
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t"])
        self.assertEqual(ns.front_host, "0.0.0.0")
        ns2 = L.build_parser().parse_args(
            ["--tree", "/x", "--tag", "t", "--front-host", "127.0.0.1"]
        )
        self.assertEqual(ns2.front_host, "127.0.0.1")


class TestFrontBindProbe(unittest.TestCase):
    def test_an_occupied_port_is_refused_by_name_before_any_spawn(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        lines = []
        try:
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.refuse_if_front_unbindable(lines.append, "0.0.0.0", port, dry=False)
        finally:
            holder.close()
        msg = str(cm.exception)
        self.assertIn(f"0.0.0.0:{port}", msg)
        self.assertIn("weg2-lan-forward.socket", msg)
        self.assertIn("systemctl disable --now", msg)

    def test_a_free_port_passes_and_is_logged(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        lines = []
        L.refuse_if_front_unbindable(lines.append, "0.0.0.0", port, dry=False)
        self.assertTrue(any("bindable" in ln for ln in lines), lines)

    def test_a_dry_run_never_probes(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        lines = []
        try:
            L.refuse_if_front_unbindable(lines.append, "0.0.0.0", port, dry=True)
        finally:
            holder.close()
        self.assertTrue(any("skipped on dry run" in ln for ln in lines), lines)


if __name__ == "__main__":
    unittest.main()
