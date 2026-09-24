# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H47 -- the HICACHE-JIT warning for an element that spills to LMEM.

fnFL2x151: the arena run write built the pointer/stride module with a
229376-B element (unroll 1): 7168 B of LocalStorage per thread, the driver
grew PP0's stack 1248 -> 7104 B (1769 MiB on the 5090) and kept it. The build
of such a module now says so, with the reservation per card."""

from __future__ import annotations

from sglang.jit_kernel import hicache as hc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THREADS_5090 = 170 * 1536
THREADS_3080 = 68 * 1536


def test_local_bytes_follow_the_kernel_layout():
    # kNumThreads = 32 / unroll share one element
    assert hc.local_bytes_per_thread(229376, 1) == 7168
    assert hc.local_bytes_per_thread(1024, hc._default_unroll(1024)) == 64
    assert hc.local_bytes_per_thread(512, hc._default_unroll(512)) == 64
    assert hc.local_bytes_per_thread(8192, 1) == 256


def test_the_warning_names_element_unroll_bytes_and_the_card_reservation():
    line = hc.lmem_warning_line(229376, 1, THREADS_5090)
    assert line.startswith(
        "HICACHE-JIT element=229376 unroll=1 local_bytes_per_thread=7168 > 256: "
        "LMEM-Reservierung 1785 MiB je Karte")
    assert "98304 unroll=1 local_bytes_per_thread=3072" in hc.lmem_warning_line(98304, 1, THREADS_3080)
    assert "LMEM-Reservierung n/a MiB" in hc.lmem_warning_line(65536, 1, None)


def test_no_warning_at_or_below_the_threshold():
    assert hc.lmem_warning_line(1024, 2, THREADS_5090) is None
    assert hc.lmem_warning_line(8192, 1, THREADS_5090) is None     # exactly 256 B
    assert hc.lmem_warning_line(8320, 1, THREADS_5090) is not None  # 260 B


def test_the_build_emits_it_once_per_module(monkeypatch):
    warned, built = [], []
    monkeypatch.setattr(hc, "_resident_threads", lambda: THREADS_5090)
    monkeypatch.setattr(hc, "load_jit", lambda *a, **kw: built.append(a) or object())
    monkeypatch.setattr(hc.logging.getLogger(hc.__name__), "warning",
                        lambda fmt, *a: warned.append(fmt % a))
    fn = getattr(hc._jit_hicache_module, "__wrapped__", None)
    assert fn is not None, "cache_once_per_arch must keep functools.wraps' __wrapped__"
    fn(element_size=229376, unroll=1, block_quota=16)
    fn(element_size=1024, unroll=2, block_quota=16)
    assert len(built) == 2
    assert len(warned) == 1 and "element=229376" in warned[0]
