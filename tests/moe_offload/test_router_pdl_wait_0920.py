"""#49 (20.09.): the Triton fused-gate router is launched with PDL on sm_90+;
a PDL dependent that never calls gdc_wait() reads its input (the gate logits)
before the primary finished writing them. b113aea441 removed the wait and kept
the launch; this pins the pair: wait BEFORE the scores load, launch_dependents
AFTER the last read of them."""

import inspect

from sglang.jit_kernel import moe_fused_gate as mfg


def test_pdl_wait_precedes_the_scores_load_and_launch_dependents_follows():
    k = mfg._router_triton_kernel
    src = getattr(k, "src", None) or inspect.getsource(getattr(k, "fn", k))
    i_wait = src.index("tl.extra.cuda.gdc_wait()")
    i_scores = src.index("scores = tl.load(row_ptr")
    i_launch = src.index("tl.extra.cuda.gdc_launch_dependents()")
    assert i_wait < i_scores < i_launch
    # both sit under the same compile-time switch as the launch attribute
    before_wait = src[:i_wait].rstrip().splitlines()[-1]
    assert "if USE_PDL:" in before_wait
    before_launch = src[:i_launch].rstrip().splitlines()[-1]
    assert "if USE_PDL:" in before_launch


def test_the_launch_attribute_and_the_kernel_switch_are_one_decision():
    src = inspect.getsource(mfg.moe_fused_gate)
    assert 'extra = {"launch_pdl": True} if use_pdl else {}' in src
    assert "USE_PDL=use_pdl" in src
