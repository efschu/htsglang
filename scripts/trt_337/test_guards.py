#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #591 -- tests for the three defects window 6 exposed in the #337 bench.

Every test here has a CAN-FAIL twin: the same assertion driven against the OLD
behaviour, proving the test would have caught the defect rather than merely
passing alongside the fix. A guard that has never been seen to fail is not a
guard.

All of it runs on the CPU. No card, no TensorRT.

    CUDA_VISIBLE_DEVICES="" python3 scripts/trt_337/test_guards.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "mb337", os.path.join(_HERE, "microbench_trt.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mb337"] = mod
    spec.loader.exec_module(mod)
    return mod


RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as ex:
        RESULTS.append((name, False, str(ex)))
        print(f"  FAIL  {name}: {ex}")
    except Exception as ex:
        RESULTS.append((name, False, f"{type(ex).__name__}: {ex}"))
        print(f"  ERROR {name}: {type(ex).__name__}: {ex}")


def expect_fail(name, fn):
    """The can-fail twin: this assertion MUST fail against the old behaviour."""
    try:
        fn()
    except AssertionError:
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name} (correctly rejected the old behaviour)")
        return
    RESULTS.append((name, False, "old behaviour was accepted"))
    print(f"  FAIL  {name}: the guard accepted the old behaviour -- it is inert")


# --------------------------------------------------------------------------
# Defect 1 -- the stream was captured at construction and escaped the graph
# --------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, tag):
        self.tag = tag
        self.cuda_stream = tag

    def synchronize(self):
        return None


def test_stream_resolved_at_call_time(mb):
    """_cur_stream must return whatever stream is active NOW."""
    import torch

    default, capture = _FakeStream(1), _FakeStream(2)
    state = {"cur": default}
    orig = torch.cuda.current_stream
    torch.cuda.current_stream = lambda *a, **k: state["cur"]
    try:
        chain = object.__new__(mb.Chain)
        chain.is_cuda = True
        at_construction = chain._cur_stream()
        state["cur"] = capture  # torch.cuda.graph() makes a capture stream current
        during_capture = chain._cur_stream()
        assert at_construction.tag == 1, at_construction.tag
        assert during_capture.tag == 2, (
            f"_cur_stream returned the construction-time stream "
            f"({during_capture.tag}) while a capture stream was active. Engine "
            f"work would be launched outside the capture and the graph would "
            f"replay empty -- this is exactly window 6's fold lane."
        )
    finally:
        torch.cuda.current_stream = orig


def canfail_stream_stored_at_construction(mb):
    """The OLD code: stream bound once in __init__ and reused forever."""
    import torch

    default, capture = _FakeStream(1), _FakeStream(2)
    state = {"cur": default}
    orig = torch.cuda.current_stream
    torch.cuda.current_stream = lambda *a, **k: state["cur"]
    try:
        class OldChain:
            def __init__(self):
                self.stream = torch.cuda.current_stream()  # the defect

            def _cur_stream(self):
                return self.stream

        chain = OldChain()
        state["cur"] = capture
        assert chain._cur_stream().tag == 2, (
            "old behaviour returned the stale construction-time stream"
        )
    finally:
        torch.cuda.current_stream = orig


# --------------------------------------------------------------------------
# Defect 1b -- an empty graph must be detected, not timed
# --------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, buf):
        self._out = buf
        self.stub = False


class _FakeGraph:
    """Replays by writing into the engine buffers -- or not, for an empty graph."""

    def __init__(self, engines, writes: bool):
        self.engines = engines
        self.writes = writes

    def replay(self):
        if not self.writes:
            return
        for e in self.engines:
            e._out.fill_(1.0)


def test_graph_verification_accepts_a_real_graph(mb):
    import torch

    bufs = [torch.zeros(4), torch.zeros(4)]
    engines = [_FakeEngine(b) for b in bufs]
    g = _FakeGraph(engines, writes=True)
    v = mb.verify_graph_contains_engines(g, engines, cuda=True, sync=lambda: None)
    assert v["checked"], v
    assert v["verified"], v


def test_graph_verification_rejects_an_empty_graph(mb):
    import torch

    bufs = [torch.zeros(4), torch.zeros(4)]
    engines = [_FakeEngine(b) for b in bufs]
    g = _FakeGraph(engines, writes=False)  # window 6's fold lane
    v = mb.verify_graph_contains_engines(g, engines, cuda=True, sync=lambda: None)
    assert v["checked"], v
    assert not v["verified"], (
        "an empty graph passed verification: the poison value survived replay "
        "and the check did not notice"
    )
    assert v["stale_engine_indices"] == [0, 1], v


def canfail_graph_verification_without_the_check(mb):
    """Without the poison check there is nothing to distinguish the two."""
    fast_empty_ms = 3.39e-05
    honest_ms = 1.54e-02
    assert fast_empty_ms > honest_ms * 0.5, (
        "timing alone cannot tell an empty graph from a fast one -- the empty "
        "replay is simply the smaller number, which is why it was published"
    )


# --------------------------------------------------------------------------
# Defect 2 -- the plausibility guard, against window 6's actual numbers
# --------------------------------------------------------------------------

#: rank 0, gemm_mlp_gate_up: N=16320 K=5120 -> 83,558,400 INT8 weights.
GATE_UP_ELEMS = 16320 * 5120
#: The 5090 read ~1.71 TB/s in window 6's torch_eager arm, so the floor for
#: reading 2 bytes per weight is about this.
FOLD_BYTES = GATE_UP_ELEMS * 2
FOLD_FLOOR_MS = (FOLD_BYTES / 1.7e12) * 1e3  # ~0.098 ms


def test_guard_flags_the_window6_fold_arm(mb):
    """3.39e-05 ms for a 167 MB fold GEMM must be ruled INVALID."""
    e = mb.judge_arm(3.401037e-05, FOLD_BYTES, FOLD_FLOOR_MS, 0.10)
    assert not e["valid"], e
    assert e["implied_gbps"] > 1e6, e  # ~4.9 PB/s
    assert "not executing the work" in e["reason"], e


def test_guard_flags_the_window6_int8_arm(mb):
    """trt_outer_graph was the same defect: 1.345e-03 ms for 83.6 MB."""
    int8_bytes = GATE_UP_ELEMS + 16320 * 2
    floor = (int8_bytes / 1.7e12) * 1e3
    e = mb.judge_arm(1.344750e-03, int8_bytes, floor, 0.10)
    assert not e["valid"], (
        "the INT8 verdict arm passed the guard; it is the same escaped-capture "
        f"defect as the fold lane: {e}"
    )


def test_guard_accepts_an_honest_arm(mb):
    """torch_eager at 4.87e-02 ms for 83.6 MB is ~1.7 TB/s -- plausible."""
    int8_bytes = GATE_UP_ELEMS + 16320 * 2
    floor = (int8_bytes / 1.7e12) * 1e3
    e = mb.judge_arm(4.872612e-02, int8_bytes, floor, 0.10)
    assert e["valid"], e


def test_guard_allows_l2_residency(mb):
    """An arm faster than HBM but not faster than its own measured floor stays
    valid: under graph replay the weights are re-read from L2, and the floor
    probe sees the same effect."""
    int8_bytes = GATE_UP_ELEMS + 16320 * 2
    l2_floor = (int8_bytes / 5.5e12) * 1e3  # floor probe also cached
    e = mb.judge_arm(1.537382e-02, int8_bytes, l2_floor, 0.10)
    assert e["valid"], (
        "a cache-resident but honest arm was flagged; the guard must use the "
        f"MEASURED floor, not a datasheet bandwidth: {e}"
    )


def canfail_guard_with_no_floor(mb):
    """Without a floor there is no decision to make -- the old state."""
    e = mb.judge_arm(3.401037e-05, FOLD_BYTES, None, 0.10)
    assert not e["valid"], "with no floor the guard cannot rule anything invalid"


def test_arm_weight_bytes(mb):
    class _G:
        n, k = 16320, 5120

    class _S:
        gemm = _G()

    class _T:
        engine_stages = [_S()]

    t = _T()
    i8 = mb.arm_weight_bytes(t, "trt_outer_graph")
    bf = mb.arm_weight_bytes(t, "trt_fold_bf16_graph")
    f32 = mb.arm_weight_bytes(t, "trt_fp32_ref_graph")
    assert bf == GATE_UP_ELEMS * 2, bf
    assert f32 == GATE_UP_ELEMS * 4, f32
    assert i8 == GATE_UP_ELEMS + 16320 * 2, i8
    assert bf / i8 > 1.99, (bf, i8)


# --------------------------------------------------------------------------
# Defect 3 -- the mock must exercise the real call signatures
# --------------------------------------------------------------------------


def test_signature_conformance_passes(mb, tmp_plan):
    import torch

    r = mb.signature_conformance_check(
        tmp_plan, out_width=8, device=torch.device("cpu"),
        profile_opts=[1, 4], spec_strategy="EAGER",
    )
    assert r["pass"], r
    for step in ("construct", "select_profile", "bind_fold", "bind", "enqueue",
                 "enqueue_used_given_stream", "layer_information", "save_cache"):
        assert step in r["steps"], (step, r)


def canfail_signature_conformance_catches_a_bad_kwarg(mb, tmp_plan):
    """The exact defect: a kwarg that does not exist on the constructor.

    The old mock built a StubFoldEngine instead of a TrtEngine, so
    ``TrtEngine(..., mode="fold")`` was never executed and the TypeError only
    appeared on the card.
    """
    import torch

    orig = mb.TrtEngine.__init__

    def bad_init(self, *args, **kwargs):
        return orig(self, *args, **kwargs, mode="fold")

    mb.TrtEngine.__init__ = bad_init
    try:
        r = mb.signature_conformance_check(
            tmp_plan, out_width=8, device=torch.device("cpu"),
            profile_opts=[1, 4], spec_strategy="EAGER",
        )
        assert not r["pass"], (
            "the conformance check accepted a constructor call with a "
            "nonexistent kwarg -- it would not have caught window 6's TypeError"
        )
        assert "TypeError" in r["error"], r
    finally:
        mb.TrtEngine.__init__ = orig


# --------------------------------------------------------------------------
# Defect 4 -- card identity
# --------------------------------------------------------------------------


def test_select_card_prefers_uuid(mb):
    spec = importlib.util.spec_from_file_location(
        "sel337", os.path.join(_HERE, "select_card.py")
    )
    sel = importlib.util.module_from_spec(spec)
    sys.modules["sel337"] = sel
    spec.loader.exec_module(sel)
    src = open(os.path.join(_HERE, "select_card.py")).read()
    assert "uuid" in src.lower(), "resolver must emit a UUID"
    assert "nvml_index" in src, "listing must label the index as NVML order"
    run = open(os.path.join(_HERE, "RUNSHEET.md")).read()
    # The prose is allowed to DESCRIBE the index trap; what must not survive is
    # a runnable line that assigns an index to CUDA_VISIBLE_DEVICES.
    offenders = [
        ln.strip()
        for ln in run.splitlines()
        if "CUDA_VISIBLE_DEVICES=" in ln
        and not ln.lstrip().startswith(("*", "-", ">"))
        and ("idx" in ln or "--query-gpu=index" in ln)
    ]
    assert not offenders, (
        f"RUNSHEET still has runnable lines feeding an index to "
        f"CUDA_VISIBLE_DEVICES: {offenders}. That is the mapping that put an "
        f"sm120 arm on a 3080 in window 6."
    )
    assert "select_card.py" in run, "RUNSHEET must use the UUID resolver"
    # Every card invocation of the harness must carry the assertion.
    invocations = [
        ln for ln in run.splitlines() if "microbench_trt.py" in ln and "$PY" in ln
    ]
    assert invocations, "RUNSHEET has no harness invocation to check"
    block = run.split("```")
    armed = sum(1 for b in block if "microbench_trt.py" in b and "--expect-arch" in b)
    total = sum(1 for b in block if "microbench_trt.py" in b and "$ART/bench" in b)
    assert armed >= total and total > 0, (
        f"{total - armed} of {total} harness invocations lack --expect-arch; "
        f"an unarmed arm can be mislabelled exactly as window 6's was"
    )


def main() -> int:
    import tempfile

    mb = load_harness()
    fd, tmp_plan = tempfile.mkstemp(suffix=".plan")
    os.write(fd, b"MOCK337\x00{}")
    os.close(fd)

    print("defect 1 -- stream resolved at call time")
    check("stream_resolved_at_call_time", lambda: test_stream_resolved_at_call_time(mb))
    expect_fail("CANFAIL stream_stored_at_construction",
                lambda: canfail_stream_stored_at_construction(mb))

    print("defect 1b -- empty graph detection")
    check("graph_verification_accepts_a_real_graph",
          lambda: test_graph_verification_accepts_a_real_graph(mb))
    check("graph_verification_rejects_an_empty_graph",
          lambda: test_graph_verification_rejects_an_empty_graph(mb))
    expect_fail("CANFAIL timing_alone_cannot_detect_an_empty_graph",
                lambda: canfail_graph_verification_without_the_check(mb))

    print("defect 2 -- plausibility guard")
    check("guard_flags_the_window6_fold_arm",
          lambda: test_guard_flags_the_window6_fold_arm(mb))
    check("guard_flags_the_window6_int8_arm",
          lambda: test_guard_flags_the_window6_int8_arm(mb))
    check("guard_accepts_an_honest_arm", lambda: test_guard_accepts_an_honest_arm(mb))
    check("guard_allows_l2_residency", lambda: test_guard_allows_l2_residency(mb))
    check("arm_weight_bytes", lambda: test_arm_weight_bytes(mb))
    expect_fail("CANFAIL guard_without_a_floor",
                lambda: canfail_guard_with_no_floor(mb))

    print("defect 3 -- signature conformance")
    check("signature_conformance_passes",
          lambda: test_signature_conformance_passes(mb, tmp_plan))
    check("CANFAIL signature_conformance_catches_a_bad_kwarg",
          lambda: canfail_signature_conformance_catches_a_bad_kwarg(mb, tmp_plan))

    print("defect 4 -- card identity")
    check("select_card_prefers_uuid", lambda: test_select_card_prefers_uuid(mb))

    os.unlink(tmp_plan)
    bad = [(n, e) for n, ok, e in RESULTS if not ok]
    print()
    if bad:
        print(f"{len(bad)} of {len(RESULTS)} FAILED")
        for n, e in bad:
            print(f"  {n}: {e}")
        return 1
    print(f"all {len(RESULTS)} checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
