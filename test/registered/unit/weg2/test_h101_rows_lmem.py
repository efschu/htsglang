# SPDX-License-Identifier: Apache-2.0
"""H101 -- rc9p D-TP0: 'Triton Error [CUDA]: out of memory' at the first launch
of the QSA rows form (16, 1, 2).

DER BEFUND (Container-Abnahme NF rc9p, Baum d1c7094ba6, D-Log Z. 64368-64460):
das X-Direkt-Extend weg2-22-25 (uncached=3585 auf einem 23744-Praefix) war der
erste Request des Boots mit Praefix UND mehr als 512 neuen Zeilen; der L20-Tisch
gab ihm (16, 1, 2). Diese Variante ist im CUDA-13-Image (PTX 9.0) REG 128 /
STACK 2320 B je Thread (cuobjdump -res-usage, Container-Triton-Cache
SEVCQXIANJ...), der Kontext hielt 1024 B -- der Launch musste die
Local-Memory-Reservierung von 255 auf 578 MiB wachsen lassen (agent09252021:
'WEG2-SLEEP-LMEM lmem 578->0 MiB ... NVML -580 MiB measured') und starb daran.
Gruppe P fuhr die Form nie: ihr Arm setzt SGLANG_FORCE_QSA_ROWS_CONFIG=
inf=64/8/2, Gruppe D hatte die Zeile nicht.

Der Fix, hier CPU-hermetisch gepinnt:
(a) sm120-Default: nur der Eintrag >512 wird (64, 8, 2) -- REG 152 / STACK 0
    unter PTX 9.0 (UZIGNZUEW4..., genau die D-Variante);
(b) Boot-Prewarm aller Rows-Formen x USE_COUNTS vor dem ersten Sleep;
(c) LMEM-Zensus am #1056-Loader-Chokepoint: LOCAL_SIZE jedes geladenen Kerns,
    Vorab-Wachstum per cuCtxSetLimit vor dem ersten Launch (benannte
    Verweigerung), und der H15-Wake restauriert das Zensus-Maximum (gebucht).
"""

from __future__ import annotations

import inspect
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import rows_prewarm as rp
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.srt.utils import lmem_census as lc
from sglang.srt.weg2 import sleep_lmem as sl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20
THREADS_5090 = 170 * 1536
#: the CUDA-13 build of the D (16, 1, 2) form (cuobjdump, SEVCQXIANJ...).
STACK_16_1_2_CU13 = 2320
#: what D-TP0's context held all of rc9p ('WEG2-WAKE-LMEM stack 0->1024 B').
CTX_STACK_RC9P = 1024


@pytest.fixture(autouse=True)
def _clean_state():
    def _reset():
        lc._reset_for_test()
        sa._ROWS_CONFIG_CACHE.clear()
        sa._ROWS_PREWARM_SIG["sigs"].clear()
        sa._ROWS_PREWARM_SIG["done"] = False

    _reset()
    yield
    _reset()


def _on_device(capability, name="NVIDIA GeForce RTX 5090"):
    return mock.patch.multiple(
        sa.torch.cuda,
        get_device_capability=lambda *a: capability,
        get_device_name=lambda *a: name,
    )


# -- (a) the sm120 default -----------------------------------------------------


def test_rc9p_numbers_the_growth_is_323_mib_on_top_of_255():
    assert round(sl.lmem_mib(stack_bytes=CTX_STACK_RC9P, threads=THREADS_5090)) == 255
    assert round(sl.lmem_mib(stack_bytes=STACK_16_1_2_CU13, threads=THREADS_5090)) == 578
    v = lc.GrowVerdict(kernel="_sparse_attn_rows_fwd", local_bytes=STACK_16_1_2_CU13,
                       found_stack_bytes=CTX_STACK_RC9P, stack_bytes=CTX_STACK_RC9P,
                       threads=THREADS_5090, action="refused", free_mib=1706.8)
    assert round(v.grow_mib()) == 323


def test_sm120_default_replaces_only_the_spilling_band():
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((12, 0)):
        picks = {q: sa._get_rows_config(q) for q in (1, 4, 32, 33, 64, 65, 128, 129, 512, 513, 3585, 16384)}
    assert picks[1] == picks[4] == picks[32] == (32, 8, 2)  # decode / verify unchanged
    assert picks[33] == picks[64] == (64, 8, 2)
    assert picks[65] == picks[128] == (64, 4, 2)
    assert picks[129] == picks[512] == (32, 4, 2)
    # rc9p's X-direct extend (3585 rows) and every longer one: spill-free
    assert picks[513] == picks[3585] == picks[16384] == (64, 8, 2)
    # the bands up to 512 are the L20 table's own entries
    for q in (1, 33, 65, 129):
        assert picks[q] == next(c for lim, c in sa._L20_CONFIGS if q <= lim)


def test_sm86_keeps_the_table_and_an_env_table_still_wins():
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((8, 6), "NVIDIA GeForce RTX 3080"):
        assert sa._get_rows_config(3585) == (16, 1, 2)
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("sm120:inf=16/1/2"), _on_device((12, 0)):
        assert sa._get_rows_config(3585) == (16, 1, 2)  # an operator's A/B still reaches the metal
    sa._ROWS_CONFIG_CACHE.clear()
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("sm86:inf=32/8/2"), _on_device((12, 0)):
        assert sa._get_rows_config(3585) == (64, 8, 2)  # names sm86 only -> H101 default


def test_the_recommended_d_profile_line_equals_the_code_default():
    """The profile workaround sent to 27B for the image
    (SGLANG_FORCE_QSA_ROWS_CONFIG=sm120:32=32/8/2,64=64/8/2,128=64/4/2,512=32/4/2,inf=64/8/2)
    selects exactly what the H101 default selects."""
    raw = "sm120:32=32/8/2,64=64/8/2,128=64/4/2,512=32/4/2,inf=64/8/2"
    assert sa.parse_rows_config(raw, 120) == sa._SM120_ROWS_CONFIGS
    assert sa.parse_rows_config(raw, 86) is None


def test_launch_forms_walk_the_effective_table():
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((12, 0)):
        assert sa.rows_launch_forms(120) == [
            (1, (32, 8, 2)), (33, (64, 8, 2)), (65, (64, 4, 2)), (129, (32, 4, 2))
        ]
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("inf=64/8/2"), _on_device((12, 0)):
        assert sa.rows_launch_forms(120) == [(1, (64, 8, 2))]  # P's arm: one form
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((8, 6), "NVIDIA GeForce RTX 3080"):
        forms = sa.rows_launch_forms(86)
    assert forms[-1] == (513, (16, 1, 2)) and len(forms) == 5


# -- the launch records the specialization the prewarm needs -------------------


class _RecordedKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def test_launches_record_each_specialization_once_and_weakly():
    rec = _RecordedKernel()
    q = torch.zeros(3, 24, 8, dtype=torch.bfloat16)
    k = torch.zeros(32, 2, 8).to(torch.float8_e4m3fn)
    v = k.clone()
    rows = torch.zeros(3, 2051, dtype=torch.int32)
    q_draft = torch.zeros(1, 24, 8, dtype=torch.bfloat16)
    rows_draft = torch.zeros(1, 2048, dtype=torch.int32)  # another width = another build
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((12, 0)), \
            mock.patch.object(sa, "_sparse_attn_rows_fwd", rec):
        sa.sparse_attn_rows_triton(q, k, v, rows, 0.3)
        sa.sparse_attn_rows_triton(q[:1], k, v, rows[:1], 0.3)  # same key: no second entry
        sa.sparse_attn_rows_triton(q_draft, k, v, rows_draft, 0.3)
    sigs = sa.rows_prewarm_signatures()
    assert [(s["heads"], s["head_dim"], s["dtype"], s["k"]) for s in sigs] == [
        (24, 8, torch.bfloat16, 2051), (24, 8, torch.bfloat16, 2048)]
    assert sigs[0]["k_pool"] is k and sigs[0]["v_pool"] is v
    # closed after the prewarm: serving launches record nothing more
    sa.close_rows_prewarm_recording()
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((12, 0)), \
            mock.patch.object(sa, "_sparse_attn_rows_fwd", rec):
        sa.sparse_attn_rows_triton(q, k, v, torch.zeros(3, 7, dtype=torch.int32), 0.3)
    assert len(sa._ROWS_PREWARM_SIG["sigs"]) == 2
    rec.calls.clear()  # the recorded launches hold uint8 views whose _base is the pool
    del k, v, sigs
    assert sa.rows_prewarm_signatures() == []  # weak: never pins a pool


def test_the_launch_line_names_its_table():
    rec = _RecordedKernel()
    sa._ROWS_LAUNCH_SEEN.clear()
    q = torch.zeros(3585, 24, 8, dtype=torch.bfloat16)
    k = torch.zeros(32, 2, 8).to(torch.float8_e4m3fn)
    rows = torch.zeros(3585, 4, dtype=torch.int32)
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), _on_device((12, 0)), \
            mock.patch.object(sa, "_sparse_attn_rows_fwd", rec), \
            mock.patch.object(sa.logger, "info") as info:
        sa.sparse_attn_rows_triton(q, k, k.clone(), rows, 0.3)
    sa._ROWS_LAUNCH_SEEN.clear()
    line = info.call_args[0][0] % info.call_args[0][1:]
    assert "cfg=64/8/2 first_total_q=3585 table=h101-sm120" in line


# -- (b) the boot prewarm ------------------------------------------------------


def test_prewarm_launches_every_form_with_and_without_counts():
    launched = []
    stack = iter([1024, 1024])
    sig = {"heads": 24, "head_dim": 8, "dtype": torch.bfloat16, "k": 2051,
           "k_pool": object(), "v_pool": object()}

    def make(sig_, total_q, use_counts):
        return ("q", total_q), ("rows", total_q), ("c", total_q) if use_counts else None

    res = rp.prewarm_rows_forms(
        signatures=[sig],
        forms=[(1, (32, 8, 2)), (33, (64, 8, 2)), (65, (64, 4, 2)), (129, (32, 4, 2))],
        launch=lambda q, kp, vp, rows, scale, counts: launched.append((q[1], counts is not None)),
        stack_bytes=lambda: next(stack),
        census=lambda: (0, ""),
        make_inputs=make,
    )
    assert launched == [(t, c) for t in (1, 33, 65, 129) for c in (False, True)]
    assert res.status == "ok" and len(res.forms) == 8
    assert res.forms[1] == "32/8/2+counts@1"
    line = res.line(THREADS_5090)
    assert line.startswith("H101 QSA-ROWS-PREWARM ok forms=[32/8/2@1, 32/8/2+counts@1,")
    assert "stack 1024->1024 B lmem 255->255 MiB" in line


def test_prewarm_without_a_recorded_launch_is_a_named_skip():
    res = rp.prewarm_rows_forms(signatures=[], forms=[(1, (32, 8, 2))], launch=None,
                                stack_bytes=lambda: 0, census=lambda: (0, ""),
                                make_inputs=None)
    assert res.status.startswith("skipped: no rows launch recorded")


def test_a_failing_form_does_not_stop_the_others_and_is_named():
    def launch(q, kp, vp, rows, scale, counts):
        if q == 33:
            raise RuntimeError("Triton Error [CUDA]: out of memory")

    res = rp.prewarm_rows_forms(
        signatures=[{"k_pool": 1, "v_pool": 2}], forms=[(1, (32, 8, 2)), (33, (64, 8, 2))],
        launch=launch, stack_bytes=lambda: 1024, census=lambda: (2320, "_sparse_attn_rows_fwd"),
        make_inputs=lambda s, t, c: (t, None, None),
    )
    assert res.status == "partial" and len(res.forms) == 2 and len(res.errors) == 2
    assert "64/8/2@33: RuntimeError: Triton Error [CUDA]: out of memory" in res.errors[0]
    assert "census_max=2320(_sparse_attn_rows_fwd)" in res.line()


def test_the_scheduler_prewarms_before_the_sampling_barrier():
    from sglang.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler.init_model_worker)
    # after the capture (whose launches record the specializations), before
    # the #603b barrier (which pairs the ranks up after it)
    i_graphs = src.index("self.init_all_cuda_graphs()")
    i_rows = src.index("self.warm_qsa_rows_forms()")
    i_samp = src.index("self.warm_sampling_backend()", i_rows)
    assert i_graphs < i_rows < i_samp
    body = inspect.getsource(sch.Scheduler.warm_qsa_rows_forms)
    assert "run_boot_prewarm()" in body


def test_a_form_a_worker_does_not_prewarm(monkeypatch):
    from sglang.srt import rank_role

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    monkeypatch.setattr(rp, "prewarm_rows_forms", mock.Mock(side_effect=AssertionError("ran")))
    assert rp.run_boot_prewarm() is None


# -- (c) the census and the pre-grow at the loader chokepoint -------------------


class _Driver:
    def __init__(self, stack, refuse=0):
        self.stack = stack
        self.refuse = refuse
        self.sets = []

    def get_stack_bytes(self):
        return self.stack

    def set_stack_bytes(self, value):
        self.sets.append(value)
        if self.refuse:
            self.refuse -= 1
            raise RuntimeError(f"cuCtxSetLimit(STACK_SIZE, {value}) returned CUresult 2")
        self.stack = value


def _ensure(drv, local=STACK_16_1_2_CU13, capturing=False, emptied=None):
    return lc.ensure_stack(
        kernel="_sparse_attn_rows_fwd", local_bytes=local, driver=drv, threads=THREADS_5090,
        free_bytes=lambda: int(1706.8 * MIB),
        empty_cache=(lambda: emptied.append(1)) if emptied is not None else (lambda: None),
        capturing=lambda: capturing,
    )


def test_a_kernel_above_the_context_grows_it_before_its_first_launch():
    drv = _Driver(CTX_STACK_RC9P)
    v = _ensure(drv)
    assert v.action == "grown" and drv.stack == STACK_16_1_2_CU13 and drv.sets == [2320]
    line = v.line()
    assert line.startswith("H101 LMEM-GROW kernel=_sparse_attn_rows_fwd local=2320 B ctx_stack=1024->2320 B")
    assert "grow=+323 MiB total=578 MiB driver_free=1707 MiB" in line


def test_a_refusal_hands_back_the_torch_cache_and_retries_once():
    emptied = []
    drv = _Driver(CTX_STACK_RC9P, refuse=1)
    v = _ensure(drv, emptied=emptied)
    assert v.action == "grown-after-empty-cache" and emptied == [1] and drv.stack == 2320


def test_a_second_refusal_is_named_not_anonymous():
    emptied = []
    drv = _Driver(CTX_STACK_RC9P, refuse=2)
    v = _ensure(drv, emptied=emptied)
    assert v.action == "refused" and drv.stack == CTX_STACK_RC9P and emptied == [1]
    line = v.line()
    assert line.startswith("H101 LMEM-GROW REFUSED kernel=_sparse_attn_rows_fwd")
    assert "grow=+323 MiB" in line and "an OOM there is THIS post" in line


def test_covered_kernels_and_capture_touch_nothing():
    drv = _Driver(CTX_STACK_RC9P)
    assert _ensure(drv, local=992).action == "" and drv.sets == []
    v = _ensure(drv, capturing=True)
    assert v.action == "deferred-capturing" and drv.sets == []
    assert "deferred=capturing" in v.line()


def test_the_census_reads_local_size_from_the_loaded_kernel(monkeypatch):
    seen = {}
    monkeypatch.setattr(lc, "ensure_stack", lambda **kw: seen.update(kw) or lc.GrowVerdict(
        kernel=kw["kernel"], local_bytes=kw["local_bytes"], found_stack_bytes=1024,
        stack_bytes=2320, threads=kw["threads"], action="grown"))
    monkeypatch.setattr(lc, "_device_threads", lambda: THREADS_5090)
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: _Driver(CTX_STACK_RC9P))
    # Triton 3.x: n_spills = CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4
    kern = types.SimpleNamespace(name="_sparse_attn_rows_fwd", n_spills=STACK_16_1_2_CU13 // 4)
    v = lc.on_module_loaded(kern)
    assert v.action == "grown" and seen["local_bytes"] == 2320
    assert lc.census_max() == (2320, "_sparse_attn_rows_fwd")
    # no local memory, or a build without the attribute: no census, no set
    assert lc.on_module_loaded(types.SimpleNamespace(name="l2norm_fwd_kernel", n_spills=0)) is None
    assert lc.on_module_loaded(types.SimpleNamespace(name="x")) is None
    assert lc.census_max() == (2320, "_sparse_attn_rows_fwd")


def test_the_chokepoint_books_after_the_load_inside_the_window():
    from sglang.srt.utils import triton_loader_window as tlw

    src = inspect.getsource(tlw.install_triton_loader_window)
    i_win = src.index("with cold_build_window(")
    i_load = src.index("result = original(self)", i_win)
    i_book = src.index("on_module_loaded(self)", i_load)
    i_ret = src.index("return result", i_book)
    assert i_win < i_load < i_book < i_ret


# -- the wake restores the booked census ------------------------------------------


def test_the_wake_restores_the_census_even_above_the_h47_cap():
    drv = _Driver(STACK_16_1_2_CU13)
    drv.threads = THREADS_5090
    nv = lambda: drv.stack * THREADS_5090  # noqa: E731
    park = sl.park_lmem(driver=drv, threads=THREADS_5090, nvml_bytes=nv, base_stack_bytes=1024)
    assert park.saved_stack_bytes == 2320 and drv.stack == 0
    # without the census: H47 skips the oversized high-water, the launch regrows it
    rec_old = sl.restore_lmem(driver=_Driver(0), park=park, nvml_bytes=nv)
    assert rec_old.restored_stack_bytes == 1024 and rec_old.skip_reason == "oversized"
    # with the census: booked, restored, no skip line
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=nv,
                          booked_stack_bytes=2320, booked_kernel="_sparse_attn_rows_fwd")
    assert drv.stack == 2320 and rec.skip_line() == "" and rec.booked_stack_bytes == 2320
    line = rec.format_line(park=park)
    assert line.startswith("WEG2-WAKE-LMEM stack 0->2320 B")
    assert line.endswith("booked=2320(_sparse_attn_rows_fwd)")


def test_a_census_below_the_old_target_changes_nothing():
    assert sl.booked_wake_target(target=1504, skip="", saved=1504, booked=992) == (1504, "", 0)
    assert sl.booked_wake_target(target=1248, skip="oversized", saved=7104, booked=None) == (1248, "oversized", 0)
    # a booked stack below the saved high-water raises the target but keeps the skip named
    assert sl.booked_wake_target(target=1248, skip="oversized", saved=7104, booked=2320) == (2320, "oversized", 2320)


def test_the_manager_passes_the_census_to_the_wake(monkeypatch):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    drv = _Driver(STACK_16_1_2_CU13)
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: drv)
    fake = types.SimpleNamespace(
        _weg2_lmem_park=None, _weg2_lmem_base_stack=1024,
        _weg2_sm_threads=lambda: THREADS_5090,
        _weg2_nvml_self_bytes=lambda: drv.stack * THREADS_5090,
    )
    lc.record("_sparse_attn_rows_fwd", STACK_16_1_2_CU13)
    infos = []
    monkeypatch.setattr(wu.logger, "info", lambda fmt, *a: infos.append(fmt % a))
    wu.SchedulerWeightUpdaterManager._weg2_park_lmem_at_sleep(fake)
    assert any(i.startswith("WEG2-SLEEP-LMEM lmem 578->0 MiB released (stack 2320->0 B")
               and i.endswith("census=2320(_sparse_attn_rows_fwd)") for i in infos)
    wu.SchedulerWeightUpdaterManager._weg2_restore_lmem_at_wake(fake)
    assert drv.stack == 2320
    assert any("booked=2320(_sparse_attn_rows_fwd)" in i for i in infos)


def test_the_p_planner_still_reads_the_sleep_line():
    from sglang.srt.planner import p_card_chunk as pc

    line = ("[2026-09-26 10:23:19 PP0] WEG2-SLEEP-LMEM lmem 578->0 MiB released (stack 2320->0 B "
            "x 261120 threads, derived; NVML -580 MiB measured; 2.8 ms) census=2320(_sparse_attn_rows_fwd)")
    m = pc._RX_SLEEP_LMEM.search(line)
    assert m is not None and int(m.group(2)) == 2320


if __name__ == "__main__":
    unittest.main()
