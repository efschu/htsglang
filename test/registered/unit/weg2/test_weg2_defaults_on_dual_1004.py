# SPDX-License-Identifier: Apache-2.0
"""Default-on audit 1300 (04.10.2026): --dual-unified-kv. Every dual profile since dual1g carries
`--dual-unified-kv on`; the dual boots show the card KV ledger at work (front "DUAL-KV-PRESSURE ... freed=").
The default is ON only under --dual-share, i.e. only behind the dual gate; `--dual-unified-kv off` still
switches it off, and without --dual-share/--dual-layout nothing changes.

DANGER DIRECTIONS guarded here: the dual default must not leak into a non-dual boot; an explicit
`--dual-unified-kv off` must win under --dual-share; `on` without --dual-share stays refused.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _ns(*extra):
    return L.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])


def test_dual_unified_kv_switch_default_on_under_dual_share():
    ns = _ns("--dual-share")
    L.resolve_dual_layout(ns)
    assert ns.dual_unified_kv == "on"
    assert ns.dual_layout is True


def test_dual_unified_kv_default_does_not_leak_into_a_non_dual_boot():
    ns = _ns()
    L.resolve_dual_layout(ns)
    assert ns.dual_unified_kv == "off"
    ns = _ns("--dual-layout")  # dual layout without --dual-share: no card KV ledger, unified stays off
    L.resolve_dual_layout(ns)
    assert ns.dual_unified_kv == "off"


def test_dual_unified_kv_explicit_off_wins_under_dual_share():
    ns = _ns("--dual-share", "--dual-unified-kv", "off")
    L.resolve_dual_layout(ns)
    assert ns.dual_unified_kv == "off"


def test_dual_unified_kv_explicit_on_without_share_is_still_refused():
    with pytest.raises(L.Weg2DualLayoutRefused):
        L.resolve_dual_layout(_ns("--dual-unified-kv", "on"))


def test_dual_unified_kv_explicit_on_is_what_the_profiles_pass():
    ns = _ns("--dual-share", "--dual-unified-kv", "on")
    L.resolve_dual_layout(ns)
    assert ns.dual_unified_kv == "on"
