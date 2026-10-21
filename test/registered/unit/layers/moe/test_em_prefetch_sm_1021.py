"""EM-PREFETCH-SM (21.10., NF D mixed prefill): the EM-PREFETCH side-stream copy must not take the forward stream's SMs.

rc31b D (``..._82784e1138_1021_202608``) against rc30 at equal T class and rows_on: the critical copy fell 100-150 ms
per mixed pass, the rest of the pass grew 170-230 ms (``/spinning/gpu-arb/deskq/done/nf-rc31-bs3-regression-1021.md``
§B4). Per rank, the MoE apply on the x4 3080 went 132 -> 262 ms at T 256-1023 (TP2 139 -> 210, TP0 50 -> 79) and did
not move at T < 256. The root, in code and in the CUDA limits: a prefill-sized Marlin block (dense ``gptq_marlin`` and
MoE ``moe_wna16_marlin``, ``blocks_per_sm == 1``) asks for the whole opt-in shared memory, 99 KB, plus the 1 KB the
driver reserves per block -- the SM's entire 100 KB on sm_86 and sm_120. The prefetch copy is a persistent Triton
kernel of 32 programs (4 warps, 72 registers, 0 B shared, its own 1 KB reserved): every SM holding one is closed to
Marlin, and Marlin's grid is exactly one wave of ``sms`` blocks, so the GEMM runs a second wave (~2x). Small-T Marlin
splits the shared memory over up to 4 blocks per SM, a copy block still fits -- the apply there did not move.

What must hold here (CPU harness, fake streams / kernels):

* the side-stream prefetch copy launches ``em_prefetch_copy_programs()`` programs (default 4), every synchronous copy
  keeps ``COPY_PROGRAMS``; the knob is clamped and the same on every rank; the grid block is thread-local;
* install builds the narrow copy variant as a ZERO-row copy (no JIT in a serving forward, no byte moved);
* a fork opens the Marlin SM window with the copy's programs, the target's join closes it -- exactly once, also when
  the join comes from another path; a body that raised opens nothing; the CPU inline path opens nothing;
* under graph capture and with the switch off the reserve is 0; a leaked window is closed at the next forward start;
* both JIT wrappers hand the reserve to the kernel as the LAST argument, and both C++ entries take it as their last
  parameter and subtract it from ``sms`` BEFORE the workspace bound (and, MoE, after the #49 override);
* the EM-FETCH line appends ``pf_programs`` / ``pf_sm_reserve`` / ``pf_gbps`` after ``pf_wait_ms``.
"""

import importlib.util
import logging
import os
import re
import threading
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.jit_kernel import marlin_sm_share as M  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.moe import em_fetch_probe as P  # noqa: E402
from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("_t_em_pf_sm_base", os.path.join(_HERE, "test_em_prefetch_1021.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
S51 = B.S51
_probe = B._probe  # noqa: F401 -- autouse: fake events, CPU allowed
_switch = B._switch  # noqa: F401 -- autouse: EM-PREFETCH knobs + Marlin window reset
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))


def _routes():
    return torch.tensor(S51.ROUTES, dtype=torch.int32)


# ---------------------------------------------------------------------------------------------- the copy grid

class _Kern:
    def __init__(self, log):
        self.log = log

    def __getitem__(self, grid):
        def _launch(*args, **kw):
            self.log.append((tuple(grid), kw["PROGRAMS"]))
        return _launch


def _cuda_copy(monkeypatch):
    log = []
    monkeypatch.setattr(ep, "_copy_kernel", lambda: _Kern(log))
    monkeypatch.setattr(ep, "_word_rows", lambda t: t)
    row = SimpleNamespace(shape=(4, 16), stride=lambda i: 16, device=SimpleNamespace(type="cuda"))
    return log, [row, row], [row, row]


def test_copy_rows_launches_the_grid_of_its_block_and_32_outside(monkeypatch):
    log, srcs, dsts = _cuda_copy(monkeypatch)
    ep.copy_rows(srcs, dsts, None, None, None)
    with ep.copy_grid(4):
        ep.copy_rows(srcs, dsts, None, None, None)
        with ep.copy_grid(99):                        # clamped to the synchronous grid
            ep.copy_rows(srcs, dsts, None, None, None)
        ep.copy_rows(srcs, dsts, None, None, None)    # the inner block restored the outer one
    with ep.copy_grid(0):                             # clamped to one program
        ep.copy_rows(srcs, dsts, None, None, None)
    ep.copy_rows(srcs, dsts, None, None, None)
    C = ep.COPY_PROGRAMS
    assert log == [((C,), C)] * 2 + [((4,), 4)] * 2 + [((C,), C)] * 2 + [((4,), 4)] * 2 + [((1,), 1)] * 2 \
        + [((C,), C)] * 2


def test_the_grid_block_is_thread_local(monkeypatch):
    seen = []
    with ep.copy_grid(4):
        th = threading.Thread(target=lambda: seen.append(ep.copy_programs()))
        th.start()
        th.join()
        seen.append(ep.copy_programs())
    assert seen == [ep.COPY_PROGRAMS, 4]


def test_the_programs_knob_defaults_to_4_clamps_and_is_read_once():
    assert eo.em_prefetch_copy_programs() == 4 == eo.EM_PREFETCH_PROGRAMS_DEFAULT
    for raw, want in ((8, 8), (64, ep.COPY_PROGRAMS), (1, 1)):
        with envs.FLLIPER_MOE_POOL_EM_PREFETCH_PROGRAMS.override(raw):
            eo.pool_em_prefetch_reset_for_tests()
            assert eo.em_prefetch_copy_programs() == want
    with envs.FLLIPER_MOE_POOL_EM_PREFETCH_PROGRAMS.override(8):
        assert eo.em_prefetch_copy_programs() == 1        # read once per process (the same value on every rank)


def test_the_prefetch_copy_runs_narrow_and_every_synchronous_copy_keeps_the_full_grid(monkeypatch):
    a, b = B._pair(monkeypatch)
    seen = []
    real = ep.copy_rows

    def _copy(srcs, dsts, src_rows, dst_rows, count):
        pf = b._pool_pf_buffers
        seen.append(("pf" if (pf is not None and src_rows is pf.gather_src) else "sync", ep.copy_programs()))
        return real(srcs, dsts, src_rows, dst_rows, count)

    monkeypatch.setattr(ep, "copy_rows", _copy)
    a.set_em_prefetch(b, _routes())
    S51._extend(a, seed=1)
    got = S51._extend(b, seed=2)
    assert torch.equal(got[:, 0], S51._truth(seed=2))           # the grid changes no byte
    assert ("pf", 4) in seen
    assert {p for kind, p in seen if kind == "pf"} == {4}
    assert {p for kind, p in seen if kind == "sync"} == {ep.COPY_PROGRAMS}


def test_install_builds_the_narrow_copy_variant_as_a_zero_row_copy(monkeypatch):
    b = S51._pool_cache(monkeypatch, warm=[8, 9])
    assert b.em_prefetch_prebuild() is True                      # allocates the prefetch buffers (install's call)
    pf = b._pool_pf_buffers
    pf.gather_count.fill_(3)                                     # whatever a step left: no row may move
    seen = []
    real = ep.copy_rows

    def _copy(srcs, dsts, src_rows, dst_rows, count):
        seen.append((ep.copy_programs(), int(count.reshape(-1)[0]), src_rows is pf.gather_src))
        return real(srcs, dsts, src_rows, dst_rows, count)

    monkeypatch.setattr(ep, "copy_rows", _copy)
    before = B._tables_snapshot(b._pool_tables)
    bank = {k: v.clone() for k, v in b._resident.items()}
    assert b.em_prefetch_prebuild() is True
    assert seen == [(4, 0, True)]
    for k, v in B._tables_snapshot(b._pool_tables).items():
        assert torch.equal(v, before[k]), k
    for k, v in b._resident.items():
        assert torch.equal(v, bank[k]), k


# ---------------------------------------------------------------------------------------------- the Marlin window

def _forked(monkeypatch, log=None, body=None):
    log = [] if log is None else log
    B._fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_begin, b._pool_pf_done = B._Event("begin", log), B._Event("done", log)
    monkeypatch.setattr(b, "_em_prefetch_streams", lambda: True)
    monkeypatch.setattr(b, "_em_prefetch_body", body or (lambda lanes, waves, probe=None: None))
    return b, log


def test_a_fork_opens_the_window_with_the_copy_programs_and_the_join_closes_it_once(monkeypatch):
    b, _ = _forked(monkeypatch)
    assert M.sm_reserve() == 0
    assert b.prefetch_pool_em(_routes()) is True
    assert M.sm_reserve() == 4 and b._em_pf_sm_open is True
    b._join_pool_prefetch()
    assert M.sm_reserve() == 0 and b._em_pf_sm_open is False
    b._join_pool_prefetch()
    assert M._STATE["open"] == 0                                  # never below zero, never twice


def test_a_chain_of_layers_holds_one_window_at_a_time(monkeypatch):
    b, log = _forked(monkeypatch)
    c = S51._pool_cache(monkeypatch)
    c._pool_pf_begin, c._pool_pf_done = B._Event("begin2", log), B._Event("done2", log)
    monkeypatch.setattr(c, "_em_prefetch_streams", lambda: True)
    monkeypatch.setattr(c, "_em_prefetch_body", lambda lanes, waves, probe=None: None)
    b.prefetch_pool_em(_routes())          # L -> L+1
    assert M.sm_reserve() == 4
    b._join_pool_prefetch()                # L+1 joins: its own waves run on the full grid
    assert M.sm_reserve() == 0
    c.prefetch_pool_em(_routes())          # L+1 -> L+2
    assert M.sm_reserve() == 4
    c._join_pool_prefetch()
    assert M.sm_reserve() == 0


def test_every_join_path_closes_the_window(monkeypatch):
    b, _ = _forked(monkeypatch)
    b.prefetch_pool_em(_routes())
    b._deferred_rows = SimpleNamespace(full=None)
    b._promote_deferred()                  # review A3's join, not run_eager_pool's
    assert M.sm_reserve() == 0 and b._em_pf_sm_open is False


def test_a_raising_body_opens_no_window_and_its_join_closes_nothing(monkeypatch):
    def _boom(lanes, waves, probe=None):
        raise RuntimeError("copy launch")

    b, _ = _forked(monkeypatch, body=_boom)
    with pytest.raises(RuntimeError):
        b.prefetch_pool_em(_routes())
    assert M.sm_reserve() == 0 and b._em_pf_sm_open is False
    b._join_pool_prefetch()
    assert M._STATE["open"] == 0


def test_the_inline_desk_path_opens_no_window(monkeypatch):
    b = S51._pool_cache(monkeypatch)
    assert b.prefetch_pool_em(_routes()) is True
    assert M.sm_reserve() == 0 and M._STATE["open"] == 0


def test_no_reserve_under_capture_or_with_the_switch_off(monkeypatch):
    b, _ = _forked(monkeypatch)
    b.prefetch_pool_em(_routes())
    monkeypatch.setattr(M, "_capturing", lambda: True)
    assert M.sm_reserve() == 0                                    # a captured grid would outlive the window
    monkeypatch.setattr(M, "_capturing", lambda: False)
    assert M.sm_reserve() == 4
    b._join_pool_prefetch()
    with envs.FLLIPER_MOE_POOL_EM_PREFETCH_SM_SHARE.override(False):
        eo.pool_em_prefetch_reset_for_tests()
        b2, _ = _forked(monkeypatch)
        b2.prefetch_pool_em(_routes())
        assert M.sm_reserve() == 0 and M._STATE["open"] == 0
        b2._join_pool_prefetch()
        assert M._STATE["open"] == 0


def test_a_leaked_window_is_closed_at_the_next_forward_start_and_named_once(monkeypatch, caplog):
    b, _ = _forked(monkeypatch)
    b.prefetch_pool_em(_routes())
    with caplog.at_level(logging.WARNING, logger=M.__name__):
        M.forward_start()
        M.open_window(4)
        M.forward_start()
    assert M.sm_reserve() == 0 and M._STATE["open"] == 0
    assert sum("still open at a forward start" in r.getMessage() for r in caplog.records) == 1
    M.forward_start()                                             # nothing open: nothing said, nothing changed
    assert M._STATE["leaks"] == 2


def test_the_scheduler_funnel_closes_leaks_at_every_forward():
    src = open(os.path.join(_REPO, "python/flliper/srt/managers/scheduler.py")).read()
    i = src.index("def _run_batch_forward(")
    body = src[i:i + 2500]
    assert "_marlin_sm_share.forward_start()" in body
    assert body.index("_em_fetch_probe.on_forward(self.forward_ct)") < body.index("_marlin_sm_share.forward_start()")


# ---------------------------------------------------------------------------------------------- the Marlin launch

class _Mod:
    def __init__(self, name):
        self.name, self.args = name, None

    def __getattr__(self, attr):
        if attr == self.__dict__.get("name"):
            def _call(*args):
                self.args = args
            return _call
        raise AttributeError(attr)


def test_the_dense_wrapper_hands_the_reserve_as_the_last_argument(monkeypatch):
    from flliper.jit_kernel import gptq_marlin as G

    mod = _Mod("gptq_marlin_gemm")
    monkeypatch.setattr(G, "_jit_gptq_marlin_module", lambda dtype: mod)
    a = torch.zeros(4, 16, dtype=torch.bfloat16)
    kw = dict(a=a, c=None, b_q_weight=torch.zeros(1, dtype=torch.int32), b_scales=torch.zeros(1, 8),
              global_scale=None, b_zeros=None, g_idx=None, perm=None, workspace=torch.zeros(1, dtype=torch.int32),
              b_q_type=SimpleNamespace(id=5), size_m=4, size_n=8, size_k=16)
    G.gptq_marlin_gemm(**kw)
    assert mod.args[-1] == 0
    M.open_window(4)
    G.gptq_marlin_gemm(**kw)
    assert mod.args[-1] == 4 and mod.args[-2] is False           # after is_zp_float
    M.close_window()


def test_the_moe_wrapper_hands_the_reserve_as_the_last_argument(monkeypatch):
    from flliper.jit_kernel import moe_wna16_marlin as W

    mod = _Mod("moe_wna16_marlin_gemm")
    monkeypatch.setattr(W, "_jit_moe_wna16_marlin_module", lambda dtype: mod)
    monkeypatch.setattr(W, "_log_marlin_switches", lambda device: None)
    a = torch.zeros(4, 16, dtype=torch.bfloat16)
    kw = dict(a=a, c_or_none=None, b_q_weight=torch.zeros(1, dtype=torch.int32), b_bias_or_none=None,
              b_scales=torch.zeros(2, 1, 8), global_scale_or_none=None, b_zeros_or_none=None, g_idx_or_none=None,
              perm_or_none=None, workspace=torch.zeros(1, dtype=torch.int32),
              sorted_token_ids=torch.zeros(16, dtype=torch.int32), expert_ids=torch.zeros(1, dtype=torch.int32),
              num_tokens_post_padded=torch.zeros(1, dtype=torch.int32), topk_weights=torch.zeros(4, 2),
              moe_block_size=16, top_k=2, mul_topk_weights=False, is_ep=False, b_q_type=SimpleNamespace(id=5),
              size_m=4, size_n=8, size_k=16)
    W.moe_wna16_marlin_gemm(**kw)
    assert mod.args[-1] == 0
    M.open_window(4)
    W.moe_wna16_marlin_gemm(**kw)
    assert mod.args[-1] == 4 and mod.args[-2] is False           # after is_zp_float
    M.close_window()


def _entry(path, name):
    src = open(os.path.join(_REPO, "python/flliper/jit_kernel/csrc/gemm", path)).read()
    i = src.index("void %s(" % name)
    return src[i:]


@pytest.mark.parametrize("path,name,after,before", [
    ("marlin/gptq_marlin.cuh", "gptq_marlin_gemm", "cudaDevAttrMultiProcessorCount",
     "workspace.size(0) >= sms"),
    ("marlin_moe/moe_wna16_marlin.cuh", "moe_wna16_marlin_gemm", "marlin_moe_sms_override()",
     "int64_t min_workspace_size ="),
])
def test_the_cuda_entries_take_the_reserve_last_and_narrow_sms_before_the_workspace_bound(path, name, after, before):
    body = _entry(path, name)
    sig = body[:body.index(") {")]
    assert re.search(r"bool is_zp_float,\s*int64_t sm_reserve$", sig), sig[-120:]
    clamp = body.index("if (sm_reserve > 0 && sm_reserve < sms)")
    assert "sms -= static_cast<int>(sm_reserve);" in body[clamp:clamp + 120]
    assert body.index(after) < clamp < body.index(before)


# ---------------------------------------------------------------------------------------------- the line

def test_the_em_fetch_line_appends_programs_reserve_and_link_rate_after_pf_wait(monkeypatch, caplog):
    P.on_forward(91)
    rec = P.begin_layer(torch.device("cpu"))
    e0 = rec.mark()
    e1 = rec.mark()
    rec.prefetch_copy(e0, e1, programs=4, row_bytes=1 << 20)
    rec.prefetch_sm_reserve(4)
    rec.counter[P.PF_FETCHED:P.PF_FETCHED + 1].add_(10)
    rec.pf_layers = 1
    rec.end_layer(6)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        P.on_forward(92)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)][-1]
    m = re.search(r"pf_layers=1 pf_predicted=\d+ pf_fetched=10 pf_hits=\d+ pf_copy_ms=([\d.]+) pf_wait_ms=[\d.]+ "
                  r"pf_programs=4 pf_sm_reserve=4 pf_gbps=([\d.]+)$", line)
    assert m, line
    ms, gbps = float(m[1]), float(m[2])
    assert ms > 0 and abs(gbps - 10 * (1 << 20) / (ms / 1000.0) / 1e9) < 0.01


def test_a_live_fork_puts_the_reserve_and_the_grid_on_the_record(monkeypatch):
    log = []
    B._fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_begin, b._pool_pf_done = B._Event("begin", log), B._Event("done", log)
    monkeypatch.setattr(b, "_em_prefetch_streams", lambda: True)
    P.on_forward(93)
    rec = P.begin_layer(torch.device("cpu"))
    b.prefetch_pool_em(_routes(), probe=rec)
    assert rec.pf_sm_reserve == 4 and rec.pf_programs == 4 and rec.pf_row_bytes == b._em_fetch_row_bytes()
    b._join_pool_prefetch()
