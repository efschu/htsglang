# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#622 execution smoke: the replay tag on a REAL captured CUDA graph.

WHY THIS EXISTS SEPARATELY FROM THE HERMETIC TESTS
--------------------------------------------------
The 16 hermetic tests run on CPU. They drive the real ``check_aborted`` and
the real formatter, and they pin the call sites by reading the shipped
sources -- but they never execute ``FullCudaGraphBackend.replay`` or
``BreakableCUDAGraph.replay`` against an actual ``torch.cuda.CUDAGraph``.
Desk-written code that no run has executed is unvalidated, and "the tag is
written at replay time" is precisely the claim that a source-reading pin
cannot establish.

So this replays real captured graphs and asserts the tag advanced. It needs
ONE GPU, no model, no distributed bring-up and no BAR1 transport: with an
empty transport registry ``check_after_graph_replay`` returns on its first
truth test, which is exactly the default path this change must not disturb.

Run:
    PYTHONPATH=<tree>/python python scripts/probe/replay_tag_gpu_smoke_622.py
"""

from __future__ import annotations

import sys

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
)

FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + what)
    if not cond:
        FAILURES.append(what)


def _capture(fn):
    """Capture ``fn`` into a real CUDA graph, warmed on a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def smoke_full_backend() -> None:
    print("[1] FullCudaGraphBackend.replay on a real captured graph")
    x = torch.ones(1024, device="cuda")
    y = torch.zeros(1024, device="cuda")

    def fn():
        y.copy_(x * 2)
        return y

    graph, out = _capture(fn)

    # __new__: replay() needs only the graph and output maps. Constructing a
    # real backend would want a model runner; the method under test does not.
    b = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    key = "ShapeKey(size=7,probe=622)"
    b._graphs = {key: graph}
    b._outputs = {key: out}

    barlink_abort_gate.reset_replay_tag_for_test()
    before = barlink_abort_gate.current_replay()[3]
    ret = b.replay(key, None)
    torch.cuda.synchronize()
    kind, stored, index, seq = barlink_abort_gate.current_replay()

    check(seq == before + 1, f"replay advanced the counter ({before} -> {seq})")
    check(kind == "full", f"kind is 'full' (got {kind!r})")
    check(stored is key, "key stored by reference, not formatted on the path")
    check(index == -1, "no segment ordinal for a full graph")
    check(ret is out, "replay still returns the captured output")
    check(bool((ret == 2.0).all().item()), "the graph actually ran (y == 2)")
    check(key in barlink_abort_gate.format_current_replay(), "key reaches the line")
    print("      tag: " + barlink_abort_gate.format_current_replay())


def smoke_breakable_segments() -> None:
    print("[2] BreakableCUDAGraph.replay per segment, real captured graphs")
    a = torch.ones(512, device="cuda")
    outs = []
    segs = []
    for i in range(3):
        t = torch.zeros(512, device="cuda")

        def fn(t=t, i=i):
            t.copy_(a * (i + 1))
            return t

        g, o = _capture(fn)
        segs.append(g)
        outs.append(o)

    bg = BreakableCUDAGraph.__new__(BreakableCUDAGraph)
    bg._segments = segs
    bg._break_fns = [lambda: None, lambda: None]
    bg._deduped_cuda_graph = None

    barlink_abort_gate.reset_replay_tag_for_test()
    bg.replay()
    torch.cuda.synchronize()
    kind, _, index, seq = barlink_abort_gate.current_replay()

    check(seq == len(segs), f"one tag per segment ({seq} == {len(segs)})")
    check(kind == "breakable/seg", f"kind is 'breakable/seg' (got {kind!r})")
    check(index == len(segs) - 1, f"last segment ordinal is {len(segs) - 1}")
    check(
        all(bool((o == i + 1).all().item()) for i, o in enumerate(outs)),
        "every segment actually ran",
    )
    print("      tag: " + barlink_abort_gate.format_current_replay())


def smoke_default_path_is_untouched() -> None:
    print("[3] default path: no BAR1 transport registered -> gate returns early")
    check(
        barlink_abort_gate.registered() == [],
        "transport registry empty, so check_after_graph_replay short-circuits",
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible -- this smoke requires one GPU")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")
    smoke_full_backend()
    smoke_breakable_segments()
    smoke_default_path_is_untouched()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL GPU SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
