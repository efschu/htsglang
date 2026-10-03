"""P-HC-DYNROWS (30.09.): the GatedResidual torch.compile fallbacks compile once.

Metal (NF y4k 09301110 / y4l 09301150, P logs, PP2 = RTX 3080, sm86): the
model-level hyper-connection mixer of the last stage has no CuTe mix (sm_100
only) and no fused Triton mix above 16 rows (_FUSED_MIX_MAX_ROWS), so every
prefill runs ``GatedResidual._mix_compute`` = ``torch.compile(...)``.
FWD-TIMING-PREFILL other_ms (the tail after the last layer mark, i.e. this
mixer + final norm + logits): 1265.7 / 1228.1 on forward 1 (16384 rows, the
static compile), 328.9 / 332.2 on forward 2 (65 rows, the automatic-dynamic
recompile), 5-8 ms from then on.

Hermetic on CPU inductor: the stock wrapper builds two graphs over changing
row counts; with SGLANG_ENABLE_HC_COMPILE_DYNAMIC_ROWS one. The served bytes
are the compiled ones: equal, row count by row count (16, 17, 65, 16384), to
a static compile of the same function; the eager composition differs from
BOTH by bf16 rounding of the fused intermediates (asserted close, not equal
-- that is today's served path, unchanged).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch._dynamo as dyn
from torch._dynamo.utils import counters

from sglang.srt.environ import envs
from sglang.srt.layers import hyperconnection as hcm

HC, HS, LR = 4, 64, 16
ROWS = (16384, 65, 8258, 16, 17)


def _switch(on):
    return envs.SGLANG_ENABLE_HC_COMPILE_DYNAMIC_ROWS.override(on)


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    dyn.reset()
    counters.clear()
    yield
    dyn.reset()


def _mixer(use_combine=False, seed=0):
    torch.manual_seed(seed)
    cfg = hcm.HyperConnectionConfig(hc_count=HC, hidden_size=HS, params_dtype=torch.bfloat16,
                                    hc_lowrank=LR, hc_per_branch_norm=True)
    g = hcm.GatedResidual(cfg, use_mix=True, use_combine=use_combine)
    with torch.no_grad():
        for p in g.parameters():
            p.normal_(0.0, 0.05)
    return g


def _x(m, seed):
    torch.manual_seed(seed)
    return torch.randn(m, HC * HS, dtype=torch.bfloat16)


def _graphs():
    return counters["stats"]["unique_graphs"]


def test_base_behaviour_stock_wrapper_recompiles_on_a_new_row_count():
    """The defect as the metal shows it (switch off = today)."""
    g = _mixer()
    for i, m in enumerate(ROWS):
        g.mix(_x(m, i))
    assert _graphs() == 2


def test_switch_on_compiles_the_mixer_once_over_every_row_count():
    with _switch(True):
        g = _mixer()
    assert g._compile_dynamic_rows is True
    for i, m in enumerate(ROWS):
        g.mix(_x(m, i))
    assert _graphs() == 1


def test_every_layer_shares_the_one_graph():
    """PP0/PP1 and D carry the same class per layer (bf16 unless
    SGLANG_HC_MIXER_INT8): one code object, one graph for all instances."""
    with _switch(True):
        mixers = [_mixer(seed=s) for s in range(3)]
    for i, m in enumerate(ROWS):
        for g in mixers:
            g.mix(_x(m, i))
    assert _graphs() == 1


@pytest.mark.parametrize("m", [16, 17, 65, 16384])
def test_mix_bytes_equal_the_static_compile(m):
    with _switch(True):
        g = _mixer()
    for i, r in enumerate(ROWS):  # the serving order: the graph is the dynamic one
        g.mix(_x(r, i))
    x = _x(m, 99)
    got, (hin, normed) = g.mix(x)
    fn = g._mix_compute._torchdynamo_orig_callable
    args = (normed, g.input_mix_weight_down.weight, g.input_mix_weight_up.weight, HC, HS)
    dyn.reset()
    ref = torch.compile(fn, dynamic=False)(*args).to(torch.bfloat16)
    assert torch.equal(got, ref)
    eager = fn(*args).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), eager.float(), atol=1 / 64, rtol=1 / 64)


@pytest.mark.parametrize("m", [16, 17, 65, 16384])
def test_combine_compiles_once_and_bytes_equal_the_static_compile(m):
    with _switch(True):
        g = _mixer(use_combine=True)
    assert not g._jit_combine_ok  # 4 x 64 is no JIT combine shape: the compiled fallback
    for i, r in enumerate(ROWS):
        x = _x(r, i)
        _, res = g.mix(x)
        g.combine(torch.randn(r, HS, dtype=torch.bfloat16), res)
    assert _graphs() == 2  # one mix graph + one combine graph
    x = _x(m, 7)
    _, res = g.mix(x)
    blk = torch.randn(m, HS, dtype=torch.bfloat16)
    got = g.combine(blk, res)
    fn = g._combine_compute._torchdynamo_orig_callable
    dyn.reset()
    ref = torch.compile(fn, dynamic=False)(blk, res[0], res[1], g.block_inject_weight.weight, HC, HS)
    assert torch.equal(got, ref.to(torch.bfloat16))


def test_switch_default_off():
    assert envs.SGLANG_ENABLE_HC_COMPILE_DYNAMIC_ROWS.get() is False
    g = _mixer()
    assert g._compile_dynamic_rows is False
