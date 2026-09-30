#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The proof: can an ``all_reduce`` over BAR1 be captured and replayed
multiple times -- and do the bytes check out after EVERY replay?

    python benchmark/bar1_graph_check.py 1,2,3 [port]

No server, no model, no planner. ``BarlinkBar1Transport`` is built directly,
as in ``bar1_diag.py``, so that an exception stays visible with its full
traceback instead of being translated into a ``logger.info``.

Why this program exists
------------------------
``bar1`` is not in ``CAPTURABLE_BARLINK_TRANSPORTS``. The bar is set out of
caution: the data path never touches the host, the round number lives in a
device word (``barlink_bar1_ext.py``, header comment), and the peer pointers
are fixed from bootstrap on -- capturable **by construction**. What was
missing was the proof. This program supplies it, and only after it passes
may ``FLLIPER_BARLINK_GRAPH_ENABLE=1`` be set.

What exactly is checked
-------------------------
1. **Both kernel variants individually.** ``1blk`` is an ordinary
   ``<<<1, threads>>>`` launch. ``grid`` is a
   ``cudaLaunchCooperativeKernel`` with ``grid.sync()``. The headers on
   this rig (CUDA 12.9) explicitly say of ``CU_LAUNCH_ATTRIBUTE_COOPERATIVE``
   "Valid for graph nodes, launches" (``cuda.h:2043``), so a cooperative
   launch can be represented as a graph node -- whether the driver also
   accepts it out of a stream capture is not stated there. Case ``grid``
   answers exactly that and reports the error code if it does not.

2. **Byte proof after EVERY replay, not just the first.** Every replay
   gets a DIFFERENT input. A capture that baked in a flag, a round number
   or a ring slot delivers correctly the first time and wrong afterward --
   that is the whole point.

3. **Multiple graphs.** flliper captures one graph per batch size. Case
   ``two-graphs`` captures two and replays them ALTERNATING. That exposes
   two captures sharing the same flag or result slot.

4. **The reservation itself.** Case ``reservation`` checks that ``_kernel``
   above the grid threshold, under capture, genuinely falls back to
   ``1blk`` instead of failing.

5. **Not just all_reduce.** The abort that made this path necessary came
   from a ``broadcast`` with 128 bytes in the draft graph -- not from an
   all_reduce. Cases ``broadcast`` and ``broadcast-two-graphs`` therefore
   run exactly that collective, at exactly that size, with EVERY rank once
   as the source. The check there is sharper than for all_reduce: a
   non-source rank starts with ITS OWN pattern in the buffer, and that
   pattern stays in place. A capture that moves nothing on replay
   therefore leaves its own pattern standing -- and that is
   distinguishable from the source's.

6. **Direct mode under capture.** Case ``pipe-direct`` captures three
   graphs, each of which gets a reserved ring slot in the BAR1 window, and
   replays them interleaved. Two additional proofs are needed here that no
   other case needs: the result tensor must REALLY sit in the window
   (otherwise the case measured the ``direct=0`` control path and proved
   nothing), and the read-back also goes over the host instead of over the
   receiving card's L2 -- a different path to the same bytes, because the
   same path would hide a broken path's own fault. Case
   ``pipe-direct-pool-empty`` is the negative control for this: without
   graph slots, every captured call must fall back to ``direct=0`` and
   still be correct.

Every case runs in FRESH processes. A failed capture leaves the stream in
capture state and makes everything after it unusable; a case that poisons
another would not be a proof.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Replays per graph. More than two, because a ring slot with L = 2 only
# shows a problem from the third round on.
REPLAYS = 5


# ===========================================================================
# The cases
# ===========================================================================
#
# `umgebung` ("environment") is set BEFORE building the transport -- the
# knob values are read once by `BarlinkBar1Transport.__init__`.
CASES = [
    {
        "name": "1blk-small",
        "zweck": "The ordinary launch, small payload. If this fails, "
                 "bar1 is fundamentally done with graphs.",
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40)},
        "groessen": [64 << 10],
        "gate": True,
    },
    {
        "name": "1blk-large",
        "zweck": "The same launch above the grid threshold, so size, "
                 "not variant, is the variable.",
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40)},
        "groessen": [8 << 20],
        "gate": True,
    },
    {
        "name": "grid",
        "zweck": "cudaLaunchCooperativeKernel UNDER capture. The one "
                 "question that could not be settled without free cards.",
        "umgebung": {
            "FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": "0",
            "FLLIPER_BARLINK_BAR1_GRAPH_GRID": "1",
        },
        "groessen": [64 << 10, 8 << 20],
        # NOT a gate: if this fails, that is not a reason against bar1 --
        # the reservation kicks in instead, and that is what the next case
        # checks.
        "gate": False,
    },
    {
        "name": "reservation",
        "zweck": "Above the threshold, but with "
                 "FLLIPER_BARLINK_BAR1_GRAPH_GRID=0: _kernel must fall back "
                 "to 1blk under capture instead of failing.",
        # The 0 is EXPLICITLY there, ever since the default started coming
        # from FLLIPER_BARLINK_GRAPH_ENABLE. Without it, this case would
        # depend on whether the release is set in the caller's environment
        # -- and a case that checks something different depending on the
        # environment checks nothing.
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": "0",
                     "FLLIPER_BARLINK_BAR1_GRAPH_GRID": "0"},
        "groessen": [64 << 10, 8 << 20],
        "gate": True,
    },
    {
        "name": "two-graphs",
        "zweck": "Two captures, replayed alternately. Exposes shared "
                 "flag or result slots.",
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40)},
        "groessen": [64 << 10, 256 << 10],
        "verschraenkt": True,
        "gate": True,
    },
    {
        "name": "pipe",
        "zweck": "mesh_pipe captured. Direct mode switches itself off "
                 "under capture; this checks that the direct=0 path "
                 "carries.",
        "umgebung": {
            "FLLIPER_BARLINK_BAR1_PIPE": "1",
            "FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40),
            "FLLIPER_BARLINK_BAR1_PIPE_GRID_THRESHOLD": str(1 << 40),
        },
        # Between pipe_from (256 KiB) and ring_from (1 MiB) -- only there does
        # `algorithm_for` pick mesh_pipe at all.
        "groessen": [512 << 10],
        "gate": True,
    },
    {
        "name": "pipe-direct",
        "zweck": "Direct mode CAPTURED: three graphs, three reserved "
                 "ring slots, replayed interleaved. Additionally reads "
                 "back over a DIFFERENT read path (host instead of "
                 "device) and proves the result tensor really sits in "
                 "the BAR1 window.",
        "umgebung": {
            "FLLIPER_BARLINK_BAR1_PIPE": "1",
            "FLLIPER_BARLINK_BAR1_PIPE_DIRECT": "1",
            "FLLIPER_BARLINK_BAR1_PIPE_DIRECT_GRAPH": "1",
            # 2 eager + 3 graph slots. Without the three above, the pool
            # would be empty and every captured call would fall back to
            # direct=0 -- the case would then pass without ever having
            # run direct mode. That is why `direkt` below verifies the
            # result tensor's location afterward.
            "FLLIPER_BARLINK_BAR1_PIPE_RESULT_RING": "5",
            "FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40),
            "FLLIPER_BARLINK_BAR1_PIPE_GRID_THRESHOLD": str(1 << 40),
        },
        "groessen": [512 << 10, 640 << 10, 768 << 10],
        "verschraenkt": True,
        "direkt": "all",
        "gate": True,
    },
    {
        "name": "pipe-direct-pool-empty",
        "zweck": "Negative control for the case before it: same setup, "
                 "but a ring with no graph slots (L=2). Every captured "
                 "call MUST fall back to direct=0 and still deliver the "
                 "right bytes -- a fallback that delivers wrong numbers "
                 "would be worse than none at all.",
        "umgebung": {
            "FLLIPER_BARLINK_BAR1_PIPE": "1",
            "FLLIPER_BARLINK_BAR1_PIPE_DIRECT": "1",
            "FLLIPER_BARLINK_BAR1_PIPE_DIRECT_GRAPH": "1",
            "FLLIPER_BARLINK_BAR1_PIPE_RESULT_RING": "2",
            "FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40),
            "FLLIPER_BARLINK_BAR1_PIPE_GRID_THRESHOLD": str(1 << 40),
        },
        "groessen": [512 << 10, 768 << 10],
        "verschraenkt": True,
        "direkt": "none",
        "gate": True,
    },
    {
        "name": "broadcast",
        "kollektiv": "broadcast",
        "zweck": "The cases from the standard run: 12- and 128-byte "
                 "broadcast in the draft graph. Every rank once as the "
                 "source, so no edge is left unchecked.",
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40)},
        # 12 comes first, and not out of tidiness: the first attempt
        # covered 128 and rejected 12 (floor of 16), and because the gate
        # only ran 128, it passed green while the standard run aborted. A
        # size UNDER one 16-byte packet has belonged in every broadcast
        # case ever since.
        "groessen": [12, 128, 64 << 10],
        "gate": True,
    },
    {
        "name": "broadcast-two-graphs",
        "kollektiv": "broadcast",
        "zweck": "Two broadcast captures, replayed alternately. A "
                 "shared slot or a baked-in half only shows up here.",
        "umgebung": {"FLLIPER_BARLINK_BAR1_GRID_THRESHOLD": str(1 << 40)},
        "groessen": [12, 128, 64 << 10],
        "verschraenkt": True,
        "gate": True,
    },
]


# ===========================================================================
# Input pattern and expected value -- computed independently of the transport
# ===========================================================================


def _pattern(n: int, rang: int, round: int, dev) -> torch.Tensor:
    """The input of rank ``rang`` in replay ``round``.

    float32 with small integers: the sum over up to eight ranks stays well
    under 2^24 and is therefore BIT-EXACT. A tolerance-based comparison
    would be the bug here -- it would smear over exactly the deviation a
    half-written buffer produces.

    The ``i % 97`` term depends on the index, not just the rank: a chunk
    decomposition that skips or shifts a range would otherwise not show
    up, because all elements would look the same.
    """
    i = torch.arange(n, dtype=torch.float32, device=dev)
    return (rang + 1) * 1000.0 + round * 7.0 + (i % 97)


def _expected_tensor(n: int, welt: int, round: int, dev) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float32, device=dev)
    header = sum((r + 1) * 1000.0 for r in range(welt))
    return header + welt * (round * 7.0 + (i % 97))


def _compare(actual: torch.Tensor, want: torch.Tensor) -> tuple[int, str]:
    wrong = torch.ne(actual, want)
    n = int(wrong.sum().item())
    if n == 0:
        return 0, ""
    first_bad = int(wrong.nonzero()[0].item())
    return n, (
        f"{n} of {actual.numel()} elements wrong, first at {first_bad}: "
        f"actual {float(actual[first_bad]):.1f}, expected {float(want[first_bad]):.1f}"
    )


# ===========================================================================
# One graph
# ===========================================================================


def _record(t, n: int, rang: int, dev):
    """Warm up, capture, return ``(graph, input, output)``.

    The warmup runs on a side stream -- the usual prescription for
    ``torch.cuda.graph``, and additionally necessary here because the
    kernel hits the peer's flag rows on the first run.

    The barrier before and after is a HOST BARRIER over the gloo group. It
    only stands around the warmup and around the capture; there is none
    inside the capture, otherwise the program would be checking something
    other than the hot path.
    """
    input = _pattern(n, rang, 0, dev)

    strom = torch.cuda.Stream(device=dev)
    strom.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(strom):
        for _ in range(3):
            t.barlink_all_reduce(None, input)
    torch.cuda.current_stream(dev).wait_stream(strom)
    torch.cuda.synchronize(dev)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = t.barlink_all_reduce(None, input)
    torch.cuda.synchronize(dev)
    dist.barrier()
    return graph, input, output


def _record_broadcast(t, n: int, rang: int, dev, src: int):
    """Same thing for ``broadcast`` -- and it is IN PLACE.

    Input and output are the same buffer, because the seam promises
    exactly that (``broadcast(tensor, src)`` returns the same tensor).
    That is exactly why the replay has to be checked more sharply here
    than for all_reduce: the buffer is filled with THIS rank's pattern
    before every replay, and if the replay moves nothing, that pattern is
    still there afterward.

    ``src`` is fixed at capture time and goes along as a kernel argument
    -- that is the condition under which a broadcast can be captured at
    all.
    """
    buffer = _pattern(n, rang, 0, dev)

    strom = torch.cuda.Stream(device=dev)
    strom.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(strom):
        for _ in range(3):
            t.barlink_broadcast(None, buffer, src)
    torch.cuda.current_stream(dev).wait_stream(strom)
    torch.cuda.synchronize(dev)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t.barlink_broadcast(None, buffer, src)
    torch.cuda.synchronize(dev)
    dist.barrier()
    return graph, buffer


def _in_window(t, tensor) -> bool:
    """Does ``tensor`` sit in this rank's exported result ring?

    That is the answer to "did direct mode really run". The transport
    name and the environment variable only say what was REQUESTED;
    whether the ring slot was actually assigned shows up solely at the
    result tensor's address. Without this check, a case could pass that
    in truth measured the ``direct=0`` control path throughout
    (measurement discipline, rule 5: never silently fall back to a
    different path).
    """
    window = t.result_window()
    if window is None:
        return False
    start, length = window
    p = int(tensor.data_ptr())
    return start <= p < start + length


def _read_back_via_host(output, want_cpu) -> tuple[int, str]:
    """Byte proof over a DIFFERENT read path than the one written.

    Measurement discipline, rule 3: "Write a pattern, read it back over a
    **different** path, compare every byte. If the read-back takes the
    same path as the write, a broken path hides its own fault."

    Here the write comes from the neighboring card's compute kernel via
    PCIe into the BAR1 aperture. The comparison above reads the same
    bytes with a device kernel -- i.e. over the receiving card's L2, and
    that exact L2 is NOT coherent with incoming PCIe writes
    (BEFUND_L2_NICHT_KOHAERENT.md). This second read-back instead goes
    over the host copy: a DMA read request from the host to the same
    region, a different path to the same bytes.

    The expected value comes in as a CPU tensor, so computing the
    expectation does not itself run over the card again.
    """
    actual = output.detach().to("cpu", copy=True)
    wrong = torch.ne(actual, want_cpu)
    n = int(wrong.sum().item())
    if n == 0:
        return 0, ""
    first_bad = int(wrong.nonzero()[0].item())
    return n, (
        f"host read-back: {n} of {actual.numel()} elements wrong, first "
        f"at {first_bad}: actual {float(actual[first_bad]):.1f}, expected "
        f"{float(want_cpu[first_bad]):.1f}"
    )


def _check_graphs(t, sizes, interleaved, rang, welt, dev, log_lines,
                    collective: str = "all_reduce", direct_expectation=None):
    """Capture, replay, prove after EVERY replay.

    ``kollektiv`` selects WHAT is captured. The replay-and-proof loop is
    the same for both -- what differs is the expected value and whether
    input and output are the same buffer.

    ``direkt_erwartung`` is ``None`` (don't care), ``"all"`` (every
    captured result tensor must sit in the BAR1 window) or ``"none"``
    (none may). It is checked at capture time, i.e. before a single
    replay has run -- a case that never actually achieved direct mode
    should not only show up at the end.
    """
    graphs = []
    for n_bytes in sizes:
        n = n_bytes // 4                       # float32
        if not t.handles(collective, n_bytes):
            log_lines.append(
                f"SKIPPED {n_bytes} bytes ({collective}): handles() -> "
                f"False (window {t.window_minimum()} bytes, max_bytes "
                f"{t.max_bytes} bytes, min_bytes {t.min_bytes}). Not a "
                f"finding, just a size this path does not carry."
            )
            continue
        if collective == "broadcast":
            # Every rank once as the source: otherwise exactly the edge
            # the draft pick uses in production would stay unchecked.
            for src in range(welt):
                graph, buffer = _record_broadcast(t, n, rang, dev, src)
                log_lines.append(
                    f"captured: broadcast, {n_bytes} bytes, src={src}, "
                    f"{t.bc_rounds(n_bytes)} rounds"
                )
                graphs.append((n_bytes, n, graph, buffer, buffer, src))
            continue
        algo = t.algorithm_for(n_bytes)
        graph, input, output = _record(t, n, rang, dev)
        in_window = _in_window(t, output)
        log_lines.append(
            f"captured: {n_bytes} bytes, algorithm {algo!r}, "
            f"result tensor {'IN' if in_window else 'NOT in'} the BAR1 window"
        )
        if direct_expectation == "all" and not in_window:
            raise AssertionError(
                f"{n_bytes} bytes: the result tensor is NOT in the BAR1 "
                f"window, so direct mode did not run. This case would "
                f"have measured and passed the direct=0 control path "
                f"without answering the question. Check the cause: the "
                f"result ring's graph pool "
                f"(FLLIPER_BARLINK_BAR1_PIPE_RESULT_RING), "
                f"FLLIPER_BARLINK_BAR1_PIPE_DIRECT_GRAPH."
            )
        if direct_expectation == "none" and in_window:
            raise AssertionError(
                f"{n_bytes} bytes: the result tensor sits in the BAR1 "
                f"window, even though the graph pool should be empty. "
                f"The result ring's split is wrong."
            )
        graphs.append((n_bytes, n, graph, input, output, None))

    if not graphs:
        raise RuntimeError(
            f"not a single graph captured ({collective}) -- every size in "
            f"this case was rejected by handles()"
        )

    # Replay. Interleaved means: round by round, ALL graphs, so that a
    # slot shared between two captures shows up. Otherwise each graph on
    # its own, so a finding can be attributed to one graph.
    sequences = ([[(round, g) for g in graphs] for round in range(1, REPLAYS + 1)]
              if interleaved
              else [[(round, g) for round in range(1, REPLAYS + 1)]
                    for g in graphs])

    for step_sequence in sequences:
        for round, (n_bytes, n, graph, input, output, src) in step_sequence:
            input.copy_(_pattern(n, rang, round, dev))
            if src is None:
                want = _expected_tensor(n, welt, round, dev)
                want_cpu = _expected_tensor(n, welt, round, "cpu")
                wer = ""
            else:
                want = _pattern(n, src, round, dev)
                want_cpu = _pattern(n, src, round, "cpu")
                wer = f", src={src}"
            torch.cuda.synchronize(dev)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(dev)
            bad, text = _compare(output, want)
            if bad:
                raise AssertionError(
                    f"replay {round} of {n_bytes} bytes{wer}: {text}"
                )
            path_used = "device"
            if direct_expectation is not None:
                bad, text = _read_back_via_host(output, want_cpu)
                if bad:
                    raise AssertionError(
                        f"replay {round} of {n_bytes} bytes{wer}: {text}"
                    )
                path_used = "device+host"
            log_lines.append(
                f"replay {round}, {n_bytes} bytes{wer}: 0 of {n} "
                f"elements wrong ({path_used})"
            )
            dist.barrier()

    for entry in graphs:
        del entry


# ===========================================================================
# Worker
# ===========================================================================


def worker(local_rank: int, devs: list, port: str, fall: dict, store: str) -> None:
    rang = local_rank
    welt = len(devs)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rang)
    os.environ["WORLD_SIZE"] = str(welt)
    for k, v in fall.get("umgebung", {}).items():
        os.environ[k] = v

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"[r{rang}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    dist.init_process_group("gloo", rank=rang, world_size=welt)
    torch.cuda.set_device(devs[rang])
    dev = torch.device("cuda", devs[rang])
    torch.cuda.init()
    torch.zeros(1, device=dev)

    from flliper.srt.distributed.device_communicators.barlink_bar1 import (
        BarlinkBar1Transport,
    )
    from flliper.srt.distributed.device_communicators.barlink_matrix_transport import (
        _window_bytes,
    )

    log_lines: list[str] = []
    result = {"fall": fall["name"], "rang": rang, "ok": False,
                "refuse_reason": "", "protokoll": log_lines}
    t = None
    try:
        t = BarlinkBar1Transport(dist.group.WORLD, dev, _window_bytes())
        evidence_items = t.byte_proof_all()
        if not all(evidence_items.values()):
            raise RuntimeError(
                f"transport byte proof failed: {evidence_items}. Without it, "
                f"handles() says False to everything, and a graph over a "
                f"path that loses bytes proves nothing."
            )
        if t.pipe_on and not t.byte_proof_pipe():
            raise RuntimeError(
                "mesh_pipe byte proof failed -- this case needs it, "
                "otherwise algorithm_for never picks mesh_pipe at all"
            )
        collective = fall.get("kollektiv", "all_reduce")
        if collective == "broadcast":
            # The transport is built directly here, not via `build_bar1`
            # -- so the proofs the factory otherwise runs are not in place
            # yet. broadcast depends on both: the a2a proof (same kernel,
            # same slots) and its own.
            if not t.byte_proof_a2a():
                raise RuntimeError(
                    "a2a byte proof failed -- broadcast runs on this "
                    "kernel, without it handles() says False"
                )
            if not t.byte_proof_broadcast():
                raise RuntimeError(
                    "broadcast byte proof failed -- a graph over a path "
                    "that loses bytes proves nothing"
                )
        _check_graphs(
            t, fall["groessen"], fall.get("verschraenkt", False),
            rang, welt, dev, log_lines, collective, fall.get("direkt"),
        )
        result["ok"] = True
    except BaseException as e:
        result["refuse_reason"] = f"{type(e).__name__}: {e}"
        sys.stderr.write(
            f"\n===== [r{rang}] CASE {fall['name']!r} FAILED =====\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        try:
            if t is not None:
                t.close()
        except Exception:
            pass
        pathlib.Path(store, f"r{rang}.json").write_text(
            json.dumps(result, indent=1)
        )
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    if not result["ok"]:
        os._exit(1)


# ===========================================================================
# Invocation
# ===========================================================================


def main() -> int:
    devs = [int(x) for x in
            (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(",")]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 29593
    nur = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    print(f"BAR1 graph proof: devices {devs}, {len(devs)} ranks, "
          f"{REPLAYS} replays per graph.\n")

    stand = []
    for i, fall in enumerate(CASES):
        if nur and fall["name"] not in nur:
            continue
        print(f"--- case {fall['name']!r} " + "-" * 40)
        print(f"    {fall['zweck']}")
        print(f"    environment: {fall.get('umgebung', {})}")
        with tempfile.TemporaryDirectory() as store:
            try:
                mp.spawn(
                    worker,
                    args=(devs, str(port + i), fall, store),
                    nprocs=len(devs), join=True,
                )
                ok, refuse_reason = True, ""
            except Exception as e:
                ok, refuse_reason = False, str(e)
            row_list = []
            for r in range(len(devs)):
                p = pathlib.Path(store, f"r{r}.json")
                if p.is_file():
                    d = json.loads(p.read_text())
                    row_list.append(d)
                    if not d["ok"]:
                        ok = False
                        refuse_reason = refuse_reason or d["refuse_reason"]
        for d in row_list:
            for z in d["protokoll"]:
                print(f"    [r{d['rang']}] {z}")
        # NOTE: "PASSED"/"FAILED" is a cross-file marker text, regex-parsed
        # by scripts/gpu_battery/s11_bar1_e2e.py and matched by the gpu_battery
        # checks and their fixtures. It was translated from the former German
        # "BESTANDEN"/"GEFALLEN" together with every one of those consumers in
        # a single commit -- renaming it again without doing the same would
        # silently break them.
        print(f"    => {'PASSED' if ok else 'FAILED'}"
              + (f": {refuse_reason}" if refuse_reason else ""))
        print()
        stand.append((fall["name"], fall.get("gate", True), ok, refuse_reason))

    print("=" * 62)
    print("Summary")
    print("=" * 62)
    for name, gate, ok, refuse_reason in stand:
        mark_label = "PASSED" if ok else "FAILED"
        print(f"  {mark_label}  {'[Gate]' if gate else '[Info]'}  {name}"
              + (f"  -- {refuse_reason[:80]}" if refuse_reason else ""))

    gates = [(n, ok) for n, gate, ok, _ in stand if gate]
    absent = [n for n, ok in gates if not ok]
    print()
    if not gates:
        print("No gate case ran -- that is NOT a release.")
        return 2
    if absent:
        print(f"Failed gate cases: {', '.join(absent)}.")
        print("FLLIPER_BARLINK_GRAPH_ENABLE stays OFF.")
        return 1
    print("All gate cases passed.")
    print("Only now may FLLIPER_BARLINK_GRAPH_ENABLE=1 be set;")
    print("bar1/matrix then count as capturable in parallel_state.")
    info = [(n, ok) for n, gate, ok, _ in stand if not gate]
    for n, ok in info:
        if n == "grid" and ok:
            print()
            print("Also: case 'grid' passed -- the cooperative launch can "
                  "be captured on this rig. That makes the reservation in "
                  "BarlinkBar1Transport._kernel moot on its own: its default "
                  "depends on FLLIPER_BARLINK_GRAPH_ENABLE. To bring it back "
                  "individually, set FLLIPER_BARLINK_BAR1_GRAPH_GRID=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
