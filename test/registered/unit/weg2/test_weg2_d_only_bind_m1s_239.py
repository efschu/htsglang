"""#239 M1s side finding: a --d-only boot bound D to 127.0.0.1 only.

rc12z30g D-only (28.09. ~22:00Z): with --d-only there is no front, so D on
:30032 is the client surface; common_flags binds every group to 127.0.0.1 and
the container's Docker port forward could not reach it (host_acceptance
serve_d waited forever, probes went in by nsenter). D-only now binds where the
front would (--front-host, default 0.0.0.0); a flip boot is unchanged.
RED on ca2a9706ec/7bd3541c4f, GREEN with the fix.
"""

from __future__ import annotations

import argparse
import inspect

from sglang.srt.weg2 import launcher as L


def test_d_only_binds_the_front_host():
    assert L.d_only_bind_argv(L.DEFAULT_FRONT_HOST) == ["--host", "0.0.0.0"]
    assert L.d_only_bind_argv("127.0.0.1") == ["--host", "127.0.0.1"]  # an explicit --front-host wins


def test_only_the_d_only_spec_appends_it():
    src = inspect.getsource(L.main)
    specs = [ln for ln in src.splitlines() if 'spec_d = GroupSpec("D", PORT_D' in ln]
    assert len(specs) >= 2
    with_bind = [ln for ln in specs if "d_only_bind_argv(ns.front_host)" in ln]
    assert len(with_bind) == 1  # the flip D specs keep loopback (the front is the surface)
    d_only = src[src.index("if ns.d_only:"):]
    assert d_only.index("d_only_bind_argv(ns.front_host)") < d_only.index("else:")


def test_extra_is_the_tail_of_argv_d_so_the_last_host_wins():
    src = inspect.getsource(L.argv_d)
    assert src.rstrip().endswith("+ extra")
    from sglang.srt.server_args import ServerArgs

    ap = argparse.ArgumentParser()
    ServerArgs.add_cli_args(ap)
    ns = ap.parse_args(["--model-path", "m", "--host", "127.0.0.1"] + L.d_only_bind_argv("0.0.0.0"))
    assert ns.host == "0.0.0.0"
