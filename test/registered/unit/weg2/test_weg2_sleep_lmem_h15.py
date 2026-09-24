# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H15 -- das Local-Memory-Polster eines schlafenden Kontexts.

DER BEFUND (Boot fnFL2x120, WEG2-DC-BREAKDOWN stage=release): die schlafende
Phase laesst auf der 5090 1818 MiB (PP0) bzw. 2074 MiB (TP0) liegen, davon
1284 bzw. 1056 MiB `other (context+driver+communicator+non-torch)`. Kein Tag
und kein empty_cache() erreicht diesen Posten (allocator_cache_released_mib=0.0
auf jedem Sleep). Ein Teil davon ist die Local-Memory-Reservierung des
Treibers: Stack je Thread x SMs x Threads je SM, 1024 B x 170 x 1536 =
255 MiB auf der 5090, die der Treiber nie von selbst zurueckgibt.

Der Park senkt beim Sleep das Stack-Limit (cuCtxSetLimit), der Wake setzt den
gemerkten Wert zurueck, bevor der kv_cache-Fit die Karte liest. Hermetisch:
ein Treiber-Fake, der die Regel des echten Treibers spielt (die Reservierung
folgt dem Limit), und eine NVML-Uhr, die sie abliest.
"""

from __future__ import annotations

import types

import pytest

from sglang.srt.weg2 import sleep_lmem as sl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20
#: RTX 5090 (GB202, sm_120): 170 SMs x 1536 resident threads.
THREADS_5090 = 170 * 1536
#: RTX 3080 (GA102, sm_86): 68 SMs x 1536 resident threads.
THREADS_3080 = 68 * 1536
#: x120 PP0 nvml_proc_used at the fifth sleep, MiB.
PP0_SLEEP_NVML = 1818 * MIB


class _Driver:
    """Plays the driver rule: the reservation is stack x threads; rungs in
    ``refuse`` answer like CUDA_ERROR_INVALID_VALUE (1)."""

    def __init__(self, *, stack: int, threads: int, base: int, refuse=(), get_fails=False):
        self.stack = stack
        self.threads = threads
        self.base = base
        self.refuse = set(refuse)
        self.get_fails = get_fails
        self.sets: list = []

    def get_stack_bytes(self) -> int:
        if self.get_fails:
            raise RuntimeError("cuCtxGetLimit(STACK_SIZE) returned CUresult 201")
        return self.stack

    def set_stack_bytes(self, value: int) -> None:
        self.sets.append(value)
        if value in self.refuse:
            raise RuntimeError(f"cuCtxSetLimit(STACK_SIZE, {value}) returned CUresult 1")
        self.stack = value

    def nvml(self) -> int:
        return self.base + self.stack * self.threads


def _pp0_driver(**kw) -> _Driver:
    lmem = 1024 * THREADS_5090
    return _Driver(stack=1024, threads=THREADS_5090, base=PP0_SLEEP_NVML - lmem, **kw)


def test_the_5090_post_is_255_mib_and_the_park_hands_it_back():
    """Derived property: the size the report quotes (255 MiB on the 5090,
    102 MiB on a 3080) is the driver formula, and the park's NVML reading
    moves by exactly the reservation the limit carried."""
    assert round(sl.lmem_mib(stack_bytes=1024, threads=THREADS_5090)) == 255
    assert round(sl.lmem_mib(stack_bytes=1024, threads=THREADS_3080)) == 102
    drv = _pp0_driver()
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    assert park.refused == ""
    assert (park.saved_stack_bytes, park.parked_stack_bytes) == (1024, 0)
    assert round(park.released_nvml_mib()) == 255
    assert drv.nvml() == PP0_SLEEP_NVML - 1024 * THREADS_5090
    post = park.format_post()
    assert "lmem 255->0 MiB released" in post and "NVML -255 MiB measured" in post


def test_a_refused_rung_falls_through_to_the_next():
    drv = _pp0_driver(refuse={0})
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    assert drv.sets == [0, 16]
    assert park.refused == "" and park.parked_stack_bytes == 16


def test_no_rung_below_the_saved_limit_is_a_named_refusal_not_a_set():
    """A context already at the bottom (or a second park of a parked one)
    must not be 'parked' to a value that is then saved as the one to restore."""
    drv = _Driver(stack=0, threads=THREADS_5090, base=0)
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    assert drv.sets == []
    assert "no rung below the saved 0 B" in park.refused
    assert "KEPT" in park.format_post()


def test_an_unreadable_limit_refuses_without_touching_the_context():
    drv = _pp0_driver(get_fails=True)
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    assert drv.sets == [] and park.refused.startswith("get: RuntimeError")


def test_the_wake_restores_the_saved_limit_and_names_a_dormant_regrowth():
    """A dormant rank still launches barlink/lane kernels; the driver grows
    the reservation for them. The wake line says by how much, and restores
    the SAVED value, not the regrown one."""
    drv = _pp0_driver()
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    drv.stack = 64  # a dormant collective needed 64 B per thread
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml)
    assert rec.refused == "" and rec.restored_stack_bytes == 1024 and drv.stack == 1024
    line = rec.format_line(park=park)
    assert "regrown_while_dormant=64 B" in line and "lmem 16->255 MiB" in line
    assert drv.nvml() == PP0_SLEEP_NVML


def test_a_context_already_back_at_the_saved_limit_is_not_set_again():
    drv = _pp0_driver()
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    drv.stack = 1024
    n = len(drv.sets)
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml)
    assert len(drv.sets) == n and rec.restored_stack_bytes == 1024


def test_a_refused_restore_is_loud_and_leaves_the_driver_growth():
    drv = _pp0_driver()
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    drv.refuse = {1024}
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml)
    assert rec.refused.startswith("set: RuntimeError")
    assert "REFUSED" in rec.format_line(park=park)
    assert "grows it on demand" in rec.format_line(park=park)


def test_the_residue_posts_carry_before_and_after_per_post():
    """x120 PP0: world 24 + pp 96 MiB group windows, lanes p2/p4 128 MiB
    each (unborrowed) -- kept; lmem released."""
    drv = _pp0_driver()
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    posts = sl.format_residue_posts(
        park=park,
        group_windows=[("world:0", 24 * MIB), ("pp:0", 96 * MIB)],
        lane_windows=[("p2", 128 * MIB), ("p4", 128 * MIB)],
    )
    assert posts.startswith("posts=[lmem 255->0 MiB released")
    assert "barlink_group_windows 120->120 MiB KEPT (world:0:24, pp:0:96)" in posts
    assert "bar1_lane_windows 256->256 MiB KEPT (p2:128, p4:128)" in posts
    off = sl.format_residue_posts(park=None, group_windows=[], lane_windows=[])
    assert off.startswith("posts=[lmem n/a (park off)")


# -- the wiring on the manager (unbound methods on a fake self) ---------------


def _manager_fake(drv):
    return types.SimpleNamespace(
        _weg2_lmem_park=None,
        _weg2_lmem_base_stack=0,
        _weg2_sm_threads=lambda: THREADS_5090,
        _weg2_nvml_self_bytes=drv.nvml,
    )


@pytest.fixture
def mgr_cls(monkeypatch):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    return wu.SchedulerWeightUpdaterManager


def test_a_second_sleep_without_a_wake_keeps_the_first_park(mgr_cls, monkeypatch):
    """Bug class of the design, not of a boot: a chunked or repeated sleep
    parking again would save the PARKED 0 B as the limit to restore, and the
    woken rank would run with nothing restored."""
    drv = _pp0_driver()
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: drv)
    fake = _manager_fake(drv)
    mgr_cls._weg2_park_lmem_at_sleep(fake)
    first = fake._weg2_lmem_park
    assert first.saved_stack_bytes == 1024
    mgr_cls._weg2_park_lmem_at_sleep(fake)
    assert fake._weg2_lmem_park is first
    mgr_cls._weg2_restore_lmem_at_wake(fake)
    assert fake._weg2_lmem_park is None and drv.stack == 1024


def test_the_switch_off_leaves_the_context_alone(mgr_cls, monkeypatch):
    from sglang.srt.environ import envs

    drv = _pp0_driver()
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: drv)
    fake = _manager_fake(drv)
    with envs.SGLANG_WEG2_SLEEP_RELEASE_LMEM.override(False):
        mgr_cls._weg2_park_lmem_at_sleep(fake)
    assert fake._weg2_lmem_park is None and drv.sets == []


def test_park_runs_before_the_census_and_restore_before_the_kv_fit():
    """Order is the point: the census must read the card AFTER the park (or
    the residue line cannot show it), and the kv_cache fit must read it
    AFTER the restore (or the wake could book the same 255 MiB twice)."""
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    i_park = src.index("self._weg2_park_lmem_at_sleep()")
    i_census = src.index("self._weg2_log_sleep_acceptance(", i_park)
    i_cache = src.rindex("torch.get_device_module().empty_cache()", 0, i_park)
    assert i_cache < i_park < i_census
    i_restore = src.index("self._weg2_restore_lmem_at_wake()")
    i_pre = src.index('self._weg2_log_dc_breakdown("wake-pre-kv', i_restore)
    i_fit = src.index("kv_resume_fit_refusal(_kv_free", i_restore)
    assert i_restore < i_pre < i_fit
    assert "_weg2_lmem_park: Any = None" in src  # slots dataclass field
    assert "_weg2_lmem_base_stack: int = 0" in src  # H47, same rule


# -- H47: the wake restores a need, not a high-water ---------------------------
#: fnFL2x151 PP0: boot base 1248 B (first park), 7104 B after one launch of the
#: arena run-write kernel (229376-B element), restored at every wake (1769 MiB).
PP0_BASE = 1248
PP0_HIGH_WATER = 7104


def test_wake_target_rule():
    # known largest kernel stack of the phase: max(base, that)
    assert sl.wake_target_stack(saved=PP0_HIGH_WATER, base=PP0_BASE, phase_kernel_stack=1000) == (PP0_BASE, "")
    assert sl.wake_target_stack(saved=PP0_HIGH_WATER, base=PP0_BASE, phase_kernel_stack=3008) == (3008, "")
    # unknown: the saved value up to the cap, else the base and "oversized"
    assert sl.wake_target_stack(saved=1504, base=1504) == (1504, "")
    assert sl.wake_target_stack(saved=2048, base=PP0_BASE) == (2048, "")
    assert sl.wake_target_stack(saved=PP0_HIGH_WATER, base=PP0_BASE) == (PP0_BASE, "oversized")
    # the boot base: the first park's stack when sane, else the driver default
    assert sl.boot_base_stack(PP0_BASE) == PP0_BASE
    assert sl.boot_base_stack(PP0_HIGH_WATER) == sl.DRIVER_DEFAULT_STACK_BYTES == 1024
    assert sl.boot_base_stack(0) == 1024


def test_an_oversized_high_water_is_not_restored_and_says_so():
    drv = _Driver(stack=PP0_BASE, threads=THREADS_5090, base=0)
    first = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml)
    assert first.base_stack_bytes == PP0_BASE
    sl.restore_lmem(driver=drv, park=first, nvml_bytes=drv.nvml)
    assert drv.stack == PP0_BASE
    drv.stack = PP0_HIGH_WATER  # the run-write kernel grew it during the phase
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml,
                        base_stack_bytes=first.base_stack_bytes)
    assert (park.saved_stack_bytes, park.base_stack_bytes) == (PP0_HIGH_WATER, PP0_BASE)
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml)
    assert rec.refused == "" and drv.stack == PP0_BASE and rec.restored_stack_bytes == PP0_BASE
    assert round(sl.lmem_mib(stack_bytes=drv.stack, threads=THREADS_5090)) == 311
    line = rec.skip_line()
    assert line.startswith(f"WEG2-WAKE-LMEM restore SKIPPED saved={PP0_HIGH_WATER} reason=oversized")
    assert "1769 MiB not reserved" in line


def test_a_known_phase_kernel_stack_sets_the_target():
    drv = _Driver(stack=PP0_HIGH_WATER, threads=THREADS_5090, base=0)
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=drv.nvml, base_stack_bytes=PP0_BASE)
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml, phase_kernel_stack_bytes=3008)
    assert drv.stack == 3008 and rec.skip_line() == ""


def test_a_sane_saved_stack_is_restored_as_before():
    drv = _Driver(stack=1504, threads=THREADS_3080, base=0)
    park = sl.park_lmem(driver=drv, threads=THREADS_3080, nvml_bytes=drv.nvml)
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=drv.nvml)
    assert drv.stack == 1504 and rec.skip_line() == ""


def test_the_manager_keeps_the_first_parks_base_and_logs_the_skip(mgr_cls, monkeypatch):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    warned: list = []
    monkeypatch.setattr(wu.logger, "warning", lambda fmt, *a: warned.append(fmt % a))
    drv = _Driver(stack=PP0_BASE, threads=THREADS_5090, base=0)
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: drv)
    fake = _manager_fake(drv)
    mgr_cls._weg2_park_lmem_at_sleep(fake)
    assert fake._weg2_lmem_base_stack == PP0_BASE
    mgr_cls._weg2_restore_lmem_at_wake(fake)
    drv.stack = PP0_HIGH_WATER
    mgr_cls._weg2_park_lmem_at_sleep(fake)
    assert fake._weg2_lmem_base_stack == PP0_BASE  # a grown stack never becomes the base
    mgr_cls._weg2_restore_lmem_at_wake(fake)
    assert drv.stack == PP0_BASE
    assert any("restore SKIPPED saved=7104 reason=oversized" in w for w in warned)
