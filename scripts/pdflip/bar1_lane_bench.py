#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bench of the PD-flip lane transport (BAR1 lanes) between two physical cards,
driving the PRODUCTION code path -- barlink-switch-1023.

WHY THIS EXISTS
===============
In the 23.10. 05:46Z flip boot the P->D flip was bound by lanes where the 5090
deposited into a 3080's BAR1 window: every such batch cost ~33 ms regardless of
the slot size (7 MiB slot -> 0.2 GB/s, 19 MiB -> 0.6 GB/s, max_sync 50.5 /
67.1 ms), while the 3080 depositing into the 5090 cost ~0.2 ms per batch. The
per-batch operation that does not scale with the bytes is the Fix-A 16-byte
READBACK (a non-posted P2P READ of the peer's BAR1) in
``bar1_lanes._run_bar1_tag_streamed``. This script makes that cost visible on
its own (the readback probe) and inside the production lane (with and without
write order), in both directions and with both lanes at once, as a flip runs.

WHAT IT DRIVES (no re-implementation)
=====================================
* ``bar1_lanes.Bar1Lanes.setup`` -> ``open_window`` (the receiver's VMM window,
  dma-buf export, AF_UNIX serve) and ``connect_peer`` (holder attach, the
  buffer-order ``bar1_contiguous_prefix`` fix, ``resource1_wc`` mmap,
  ``cudaHostRegister(IoMemory)``) -- unmodified.
* ``bar1_lanes.run_bar1_units`` with ``weight_exchange_transport.CudartDeviceOps``
  (a thin subclass that only TIMES ``synchronize``/``stream_done_held``) --
  unmodified. Synthetic descs (FLAT + STRIDED2D) are cut by ``tp.batch_descs``
  inside ``run_bar1_units`` exactly as production cuts a weights tag.
* The ONE bench-side seam, NO change to production: ``open_window`` borrows a
  group window on a BAR1 < 4 GiB (a 3080), and a bench has no group windows. In
  the worker the module constant ``bar1_lanes.BIG_BAR_MIN`` is set to 1 so the
  receiver takes the very branch the 5090 takes (``vmm_alloc(slot * ring)`` +
  ``dmabuf_fd`` + the AF_UNIX serve) with the slot from
  ``FLLIPER_PDFLIP_BAR1_BIG_SLOT_MIB`` -- ``prod`` = what ``slot_bytes_for``
  gives that receiver's BAR1 (32 MiB big BAR, 4 MiB small BAR).

STRUCTURE
=========
Lane cells: one worker process per card (multiprocessing spawn,
``CUDA_DEVICE_ORDER=PCI_BUS_ID``, ``CUDA_VISIBLE_DEVICES`` = that card's UUID as
NVML names it, own BDF checked against NVML's). Worker 0 = group "P" rank 0,
worker 1 = group "D" rank 1, ``cross_pairs = [(0, 1), (1, 0)]``: lane ``p0`` =
fwd (src-card deposits into dst-card's window), ``p1`` = rev. One worker pair
per (slot, write-order) config runs the directions fwd, rev, both (= both lanes
at once, a flip's shape) one tag after the other, with a barrier per tag.

Raw cells (``--raw``): the hardware ceiling of the SAME mapping (dmabuf_holder +
resource1_wc + cudaHostRegister(IoMemory)) without credits, slots, readbacks:
the writer streams a fixed total in chunks into the peer window, with the copy
engine (``ops.memcpy_async`` on 1..N streams) and with the SM copier
(``lane_sm_copy.copier_for`` -- reused, no new kernel). nvbandwidth /
p2pBandwidthLatencyTest cannot measure it: GeForce blocks CUDA P2P, they skip
every pair; only the holder path reaches a peer's BAR1.

RULES
=====
* GPU only inside a RUNNING gpuq window (``barlink_round_bench.
  require_gpuq_window`` / ``preflight``, imported, not copied).
* The parent process never touches CUDA (NVML only), the workers own the cards.
* A card showing foreign utilisation > 5 % while none of our workers is alive is
  tagged CONTENDED in the artifact and the summary.
* Every byte that arrives is compared with its expected pattern (lane cells:
  per tag; raw cells: once per cell). A wrong byte makes the cell WRONG, loudly.

ONE command for the flip-lane cells on cards 0 (5090) and 1 (3080)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 scripts/pdflip/bar1_lane_bench.py \\
        --gpuq-booking <id> --src-card 0 --dst-card 1 --directions fwd,rev,both \\
        --write-order on,off --readback-probe 64 --out /path/lane_bench.json

``--plan`` prints the cell matrix, the worker environment and the runtime/VRAM
estimate and exits without touching gpuq, NVML or CUDA.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_PYTHON = os.path.join(REPO, "python")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import barlink_round_bench as _rb  # noqa: E402  (stdlib only at import time)

SCHEMA = "bar1_lane_bench/2"
CONTENDED = "CONTENDED"
DIRECTIONS = ("fwd", "rev", "both")
LANE_OF_DIR = {"fwd": ("p0",), "rev": ("p1",), "both": ("p0", "p1")}
DIR_OF_LANE = {"p0": "fwd", "p1": "rev"}
CHUNK = 4 << 20                      # pattern generation / check granule (bytes)
TAG_NAME = "weights_bench"
BARRIER_S = 900.0
CONTEND_PCT = 5.0

# --------------------------------------------------------------------------
# pure helpers: statistics
# --------------------------------------------------------------------------


def percentile(xs, q):
    """Linear-interpolation percentile (q in 0..100); None for no samples."""
    v = sorted(float(x) for x in xs)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * float(q) / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def dist(xs, gt=None):
    """{n, min, p50, p90, max, mean}; ``gt`` adds ``n_gt`` = samples above it."""
    v = [float(x) for x in xs]
    out = {
        "n": len(v),
        "min": min(v) if v else None,
        "p50": percentile(v, 50),
        "p90": percentile(v, 90),
        "max": max(v) if v else None,
        "mean": (sum(v) / len(v)) if v else None,
    }
    if gt is not None:
        out["n_gt"] = sum(1 for x in v if x > gt)
        out["gt"] = gt
    return out


def periods_ms(t0, finish_times):
    """Per-batch period of one tag: successive finish timestamps, the first
    measured from the tag start (it carries the plan/credit set-up)."""
    out, prev = [], float(t0)
    for t in finish_times:
        out.append((float(t) - prev) * 1000.0)
        prev = float(t)
    return out


_KV = re.compile(r"(\w+)=(\S+)")


def parse_lane_time(line):
    """The ``PDFLIP-BAR1 lane-time ...`` line as a dict: numbers as floats,
    everything else as the raw string. {} if it is not a lane-time line."""
    if "lane-time" not in str(line):
        return {}
    out = {}
    for k, v in _KV.findall(str(line)):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def contention_verdict(samples_by_card, threshold=CONTEND_PCT):
    """{card: {max_util, n, contended}} from NVML utilisation samples taken
    while no worker of ours is alive. Pure."""
    out = {}
    for card, xs in samples_by_card.items():
        vals = [float(x) for x in xs if x is not None]
        mx = max(vals) if vals else None
        out[int(card)] = {
            "max_util": mx,
            "n": len(vals),
            "contended": bool(mx is not None and mx > threshold),
        }
    return out


def any_contended(verdict):
    return any(v.get("contended") for v in verdict.values())


# --------------------------------------------------------------------------
# pure helpers: arguments and the cell matrix
# --------------------------------------------------------------------------


def _csv(s, conv=str):
    return [conv(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def _csv_ints(s):
    return _csv(s, int)


def _uniq(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src-card", type=int, default=0,
                    help="NVML index of the card that deposits in 'fwd' "
                         "(0 = RTX 5090 on this rig; names are resolved via "
                         "NVML at run time and printed)")
    ap.add_argument("--dst-card", type=int, default=1,
                    help="NVML index of the card that receives in 'fwd' "
                         "(1 = a 3080 on this rig)")
    ap.add_argument("--directions", default="fwd,rev,both",
                    help="fwd = src-card deposits into dst-card's window, rev "
                         "= the reverse, both = the two lanes at the same time")
    ap.add_argument("--slot-mib", default="prod,7,19",
                    help="receiver slot size(s) in MiB; 'prod' = the production "
                         "slot_bytes_for value of each receiver's BAR1 (32 big "
                         "BAR / 4 small BAR); 7 and 19 are what a 3080 "
                         "receiver got in the 23.10. boot")
    ap.add_argument("--ring", type=int, default=None,
                    help="ring slots (default: production ring_slots())")
    ap.add_argument("--tag-mib", type=int, default=640,
                    help="payload bytes per tag in MiB (a weights_N tag = 640)")
    ap.add_argument("--tags", type=int, default=3,
                    help="measured repetitions (tags) per direction")
    ap.add_argument("--warmup-tags", type=int, default=1,
                    help="uncounted tags before the measured ones, per direction")
    ap.add_argument("--write-order", default="on,off",
                    help="arms: FLLIPER_PDFLIP_BAR1_WRITE_ORDER=1 (on) / 0 (off), "
                         "set in the workers before import")
    ap.add_argument("--readback-probe", type=int, default=64,
                    help="N individually synchronized 16-byte reads from the "
                         "peer window per probed direction (0 = skip), plus "
                         "the write-only and read-after-write references")
    ap.add_argument("--connect-probe", choices=("off", "prod"), default="off",
                    help="the connect-time readback probe of Bar1Lanes."
                         "connect_peer (FLLIPER_PDFLIP_BAR1_READBACK_PROBE_N): "
                         "'off' sets it to 0 so a slow lane stays measurable; "
                         "'prod' leaves production's value, a refused lane "
                         "shows as a REFUSED cell with the named reason")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment for the workers (production switches)")
    ap.add_argument("--out", default="", help="JSON artifact path")
    ap.add_argument("--gpuq-booking", default="")
    ap.add_argument("--max-used-mib", type=int, default=64)
    ap.add_argument("--flliper-python", default=DEFAULT_PYTHON)
    ap.add_argument("--plan", action="store_true",
                    help="print the matrix + estimates and exit (no gpuq/NVML/CUDA)")
    ap.add_argument("--contend-pct", type=float, default=CONTEND_PCT,
                    help="foreign utilisation above this, with no worker of "
                         "ours alive, tags a card CONTENDED")
    # raw ceiling cells ----------------------------------------------------
    ap.add_argument("--raw", action="store_true",
                    help="also run the RAW ceiling cells (no credits/slots/"
                         "readbacks): CE and SM writes into the peer window")
    ap.add_argument("--skip-lanes", action="store_true",
                    help="run no lane cells (with --raw: the raw run alone)")
    ap.add_argument("--cards", default="",
                    help="raw cells: NVML indices, 2 or 3 (every ordered pair, "
                         "both, fan, ring); default src-card,dst-card")
    ap.add_argument("--raw-patterns", default="auto",
                    help="'auto' = all for the card count (2 cards: --directions), "
                         "or names: fwd,rev,both | 0>1,1>0,.. | 0<>1,.. | fan,ring")
    ap.add_argument("--raw-engines", default="ce,sm")
    ap.add_argument("--raw-total-mib", type=int, default=2048,
                    help="bytes streamed per pass (a cell is capped by --raw-max-s)")
    ap.add_argument("--raw-chunk-mib", default="4,16,64")
    ap.add_argument("--raw-streams", default="1,2,4",
                    help="CE: streams per writer edge (SM uses 1 stream)")
    ap.add_argument("--raw-sm-blocks", default="32,64,128")
    ap.add_argument("--raw-window-mib", type=int, default=64,
                    help="receiver window per incoming edge (BAR1 must fit it)")
    ap.add_argument("--raw-src-mib", type=int, default=256,
                    help="source buffer per card (window + source <= 512 MiB)")
    ap.add_argument("--raw-reps", type=int, default=2)
    ap.add_argument("--raw-max-s", type=float, default=6.0,
                    help="cap per timed pass: a pilot pass sets the rate and the "
                         "total is cut so one pass takes at most this long")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    bad = [d for d in _csv(a.directions) if d not in DIRECTIONS]
    if bad:
        ap.error(f"--directions: unknown {bad}; choose from {list(DIRECTIONS)}")
    a.directions = [d for d in DIRECTIONS if d in _csv(a.directions)]
    if not a.directions:
        ap.error("--directions: empty")
    slots = []
    for tok in _csv(a.slot_mib):
        if tok == "prod":
            slots.append("prod")
            continue
        try:
            v = int(tok)
        except ValueError:
            ap.error(f"--slot-mib: {tok!r} is neither 'prod' nor an integer MiB")
        if v < 1:
            ap.error("--slot-mib: slots must be >= 1 MiB")
        slots.append(str(v))
    a.slots = _uniq(slots)
    if not a.slots:
        ap.error("--slot-mib: empty")
    wos = _csv(a.write_order)
    badw = [w for w in wos if w not in ("on", "off")]
    if badw or not wos:
        ap.error("--write-order: use on,off")
    a.write_orders = _uniq(wos)
    if a.ring is not None and not 3 <= a.ring <= 16:
        ap.error("--ring: production ring is 3..16")
    if a.tag_mib < 1:
        ap.error("--tag-mib must be >= 1")
    if a.tags < 1 or a.warmup_tags < 0:
        ap.error("--tags >= 1 and --warmup-tags >= 0")
    if a.readback_probe < 0:
        ap.error("--readback-probe >= 0")
    if a.src_card == a.dst_card:
        ap.error("--src-card and --dst-card must differ")
    envx = {}
    for kv in a.env:
        if "=" not in kv:
            ap.error(f"--env {kv!r}: expected KEY=VAL")
        k, v = kv.split("=", 1)
        envx[k] = v
    a.env_extra = envx
    # raw ----------------------------------------------------------------
    cards = _csv_ints(a.cards) if a.cards else [a.src_card, a.dst_card]
    if len(cards) not in (2, 3) or len(set(cards)) != len(cards):
        ap.error("--cards: two or three distinct NVML indices")
    a.cards_list = cards
    a.raw_chunks = _csv_ints(a.raw_chunk_mib)
    a.raw_stream_counts = _csv_ints(a.raw_streams)
    a.raw_blocks = _csv_ints(a.raw_sm_blocks)
    a.raw_engine_list = _csv(a.raw_engines)
    if a.raw:
        if any(e not in ("ce", "sm") for e in a.raw_engine_list) or not a.raw_engine_list:
            ap.error("--raw-engines: ce,sm")
        if not a.raw_chunks or min(a.raw_chunks) < 1:
            ap.error("--raw-chunk-mib: integers >= 1")
        if not a.raw_stream_counts or min(a.raw_stream_counts) < 1:
            ap.error("--raw-streams: integers >= 1")
        if not a.raw_blocks or min(a.raw_blocks) < 1:
            ap.error("--raw-sm-blocks: integers >= 1")
        if a.raw_window_mib < max(a.raw_chunks):
            ap.error("--raw-window-mib must hold the largest chunk")
        if a.raw_src_mib < max(a.raw_chunks):
            ap.error("--raw-src-mib must hold the largest chunk")
        if a.raw_src_mib < a.raw_window_mib:
            ap.error("--raw-src-mib must be >= --raw-window-mib (the verify "
                     "pass writes source offset = window offset)")
        if a.raw_window_mib + a.raw_src_mib > 512:
            ap.error("--raw-window-mib + --raw-src-mib must be <= 512 MiB per card")
        if a.raw_reps < 1 or a.raw_total_mib < 1 or a.raw_max_s <= 0:
            ap.error("--raw-reps/--raw-total-mib/--raw-max-s must be positive")
        a.raw_pattern_list = expand_raw_patterns(len(cards), a.raw_patterns,
                                                 a.directions)
        if not a.raw_pattern_list:
            ap.error("--raw-patterns: nothing selected for this card count")
    else:
        a.raw_pattern_list = []
    if a.skip_lanes and not a.raw:
        ap.error("--skip-lanes without --raw leaves nothing to run")
    return a


class Cell(tuple):
    """(direction, slot, write_order) -- a summary row's identity."""

    __slots__ = ()

    def __new__(cls, direction, slot, write_order):
        return tuple.__new__(cls, (direction, slot, write_order))

    direction = property(lambda s: s[0])
    slot = property(lambda s: s[1])
    write_order = property(lambda s: s[2])


def expand_cells(directions, slots, write_orders):
    """slot outermost, then write order (adjacent in time: drift hits both
    arms), then direction."""
    return [Cell(d, s, w) for s in slots for w in write_orders for d in directions]


def group_configs(cells):
    """One worker pair per (slot, write order); the directions it runs."""
    cfgs, seen = [], {}
    for c in cells:
        key = (c.slot, c.write_order)
        if key not in seen:
            seen[key] = {"slot": c.slot, "write_order": c.write_order,
                         "directions": []}
            cfgs.append(seen[key])
        if c.direction not in seen[key]["directions"]:
            seen[key]["directions"].append(c.direction)
    return cfgs


def receivers_for(a, directions):
    """NVML indices of the cards that serve a window in these directions."""
    out = []
    for d in directions:
        if d in ("fwd", "both") and a.dst_card not in out:
            out.append(a.dst_card)
        if d in ("rev", "both") and a.src_card not in out:
            out.append(a.src_card)
    return out


def lanes_for(directions):
    out = []
    for d in directions:
        for lk in LANE_OF_DIR[d]:
            if lk not in out:
                out.append(lk)
    return out


# --------------------------------------------------------------------------
# pure helpers: synthetic descs and the byte pattern
# --------------------------------------------------------------------------

#: A weights tag's shape: large FLAT tensors, odd sizes, and strided matrices
#: (src pitch 2x run, dst pitch 1.5x run: a different dpitch on purpose).
TEMPLATES = (
    ("FLAT", 96 << 20),
    ("FLAT", 24 << 20),
    ("2D", 8192, 2048),
    ("FLAT", (7 << 20) + 4096),
    ("FLAT", 1 << 20),
    ("2D", 16384, 1024),
    ("FLAT", 48 << 20),
    ("FLAT", (3 << 20) + 512),
)


def _align(x, a=4096):
    return (int(x) + a - 1) // a * a


def plan_descs(tag_bytes, templates=TEMPLATES):
    """Deterministic synthetic descriptor plan carrying exactly ``tag_bytes``
    of payload. Each entry: param_name, kind, nbytes, rows, run_bytes, spitch,
    dpitch, src_off, dst_off (arena offsets), logical_off (payload stream
    offset the pattern is defined on)."""
    specs, rem, logical, so, do, i = [], int(tag_bytes), 0, 0, 0, 0
    while rem > 0:
        t = templates[i % len(templates)]
        i += 1
        kind = t[0]
        if kind == "2D":
            run, rows = int(t[1]), int(t[2])
            rows = min(rows, rem // run)
            if rows <= 0:
                kind = "FLAT"
                t = ("FLAT", rem)
        if kind == "FLAT":
            n = min(int(t[1]), rem)
            specs.append({"param_name": f"w{len(specs)}", "kind": "FLAT",
                          "nbytes": n, "rows": 1, "run_bytes": n,
                          "spitch": n, "dpitch": n,
                          "src_off": so, "dst_off": do, "logical_off": logical})
            so, do = _align(so + n), _align(do + n)
        else:
            sp, dp = 2 * run, run + run // 2
            n = rows * run
            specs.append({"param_name": f"w{len(specs)}", "kind": "STRIDED2D",
                          "nbytes": n, "rows": rows, "run_bytes": run,
                          "spitch": sp, "dpitch": dp,
                          "src_off": so, "dst_off": do, "logical_off": logical})
            so, do = _align(so + rows * sp), _align(do + rows * dp)
        logical += n
        rem -= n
    return specs


def footprints(specs):
    """(src arena bytes, dst arena bytes, payload bytes)."""
    sf = df = pay = 0
    for s in specs:
        if s["kind"] == "FLAT":
            sf = max(sf, s["src_off"] + s["nbytes"])
            df = max(df, s["dst_off"] + s["nbytes"])
        else:
            sf = max(sf, s["src_off"] + s["rows"] * s["spitch"])
            df = max(df, s["dst_off"] + s["rows"] * s["dpitch"])
        pay += s["nbytes"]
    return sf, df, pay


def pattern(start, n, seed, device):
    """uint8 tensor: byte k of the payload stream = f(start + k, seed), never
    0 (an untouched destination is zero) and position-dependent (a mis-ordered
    window piece or a stale slot differs)."""
    import torch

    i = torch.arange(int(start), int(start) + int(n), device=device,
                     dtype=torch.int64)
    v = ((i * 40503 + int(seed) * 7919) ^ (i >> 7)) % 251 + 1
    return v.to(torch.uint8)


def fill_src(arena, specs, seed):
    """Source arena: payload bytes = pattern, padding = 0xA5."""
    arena.fill_(0xA5)
    dev = arena.device
    for s in specs:
        base, n = s["src_off"], s["nbytes"]
        if s["kind"] == "FLAT":
            for c in range(0, n, CHUNK):
                m = min(CHUNK, n - c)
                arena[base + c:base + c + m].copy_(
                    pattern(s["logical_off"] + c, m, seed, dev))
        else:
            run, sp = s["run_bytes"], s["spitch"]
            step = max(1, CHUNK // run)
            for r0 in range(0, s["rows"], step):
                rr = min(step, s["rows"] - r0)
                blk = arena[base + r0 * sp:base + (r0 + rr) * sp].view(rr, sp)
                blk[:, :run].copy_(
                    pattern(s["logical_off"] + r0 * run, rr * run, seed,
                            dev).view(rr, run))


def check_dst(arena, specs, seed):
    """{bad_payload, bad_padding, first_bad}: payload bytes vs the pattern,
    pitch padding must still be zero."""
    dev = arena.device
    bad_p = bad_pad = 0
    first = ""
    for s in specs:
        base, n = s["dst_off"], s["nbytes"]
        b = 0
        if s["kind"] == "FLAT":
            for c in range(0, n, CHUNK):
                m = min(CHUNK, n - c)
                exp = pattern(s["logical_off"] + c, m, seed, dev)
                b += int((arena[base + c:base + c + m] != exp).sum().item())
        else:
            run, dp = s["run_bytes"], s["dpitch"]
            step = max(1, CHUNK // run)
            for r0 in range(0, s["rows"], step):
                rr = min(step, s["rows"] - r0)
                blk = arena[base + r0 * dp:base + (r0 + rr) * dp].view(rr, dp)
                exp = pattern(s["logical_off"] + r0 * run, rr * run, seed,
                              dev).view(rr, run)
                b += int((blk[:, :run] != exp).sum().item())
                if dp > run:
                    bad_pad += int((blk[:, run:] != 0).sum().item())
        bad_p += b
        if b and not first:
            first = s["param_name"]
    return {"bad_payload": bad_p, "bad_padding": bad_pad, "first_bad": first}


# --------------------------------------------------------------------------
# pure helpers: per-batch timing recorder (fed by the TimedOps subclass)
# --------------------------------------------------------------------------


class Recorder:
    """Per-thread record of the lane's batch finishes. Production's
    ``timed_sync`` calls ``stream_done_held`` (True = the stream had already
    drained, no blocking call) and otherwise ``synchronize``: exactly one
    event per finished batch is recorded here, in ``end()``'s order."""

    def __init__(self):
        self._tls = threading.local()

    def begin(self, t0=None):
        self._tls.ev = []
        self._tls.t0 = time.perf_counter() if t0 is None else t0

    def sync_done(self, t_start, t_end):
        ev = getattr(self._tls, "ev", None)
        if ev is not None:
            ev.append((t_end, t_end - t_start))

    def skip(self, t):
        ev = getattr(self._tls, "ev", None)
        if ev is not None:
            ev.append((t, 0.0))

    def end(self):
        ev = getattr(self._tls, "ev", None) or []
        t0 = getattr(self._tls, "t0", 0.0)
        self._tls.ev = None
        fin = [e[0] for e in ev]
        return {"batch_ms": periods_ms(t0, fin),
                "sync_ms": [e[1] * 1000.0 for e in ev]}


# --------------------------------------------------------------------------
# pure helpers: aggregation of worker results into cells
# --------------------------------------------------------------------------


def lane_stats(deps, cols, tag_bytes):
    """Statistics of one lane over its measured tags. ``deps``/``cols`` are the
    depositor's and collector's run records (same tag order)."""
    walls = [d["wall_ms"] for d in deps if not d.get("err")]
    nbytes = [d["bytes"] for d in deps if not d.get("err")]
    rates = [b / (w / 1e3) / 1e9 for b, w in zip(nbytes, walls) if w > 0]
    batch = [x for d in deps for x in d.get("batch_ms", [])]
    fields = {}
    for key in ("total_ms", "wait_ms", "copy_sync_ms", "credit_ms", "issue_ms",
                "sync_ms", "sync_cpu_ms", "max_sync_ms", "send_ms",
                "overlap_ms", "rate_GBs"):
        vals = [d["lane_time"][key] for d in deps
                if isinstance(d.get("lane_time", {}).get(key), float)]
        if vals:
            fields[key] = dist(vals)
    last = (deps[-1].get("lane_time") if deps else None) or {}
    errs = [d["err"] for d in deps if d.get("err")] + \
           [c["err"] for c in cols if c.get("err")]
    bad = [c.get("verify") for c in cols
           if c.get("verify") and (c["verify"]["bad_payload"] or c["verify"]["bad_padding"])]
    verified = [c for c in cols if c.get("verify")]
    return {
        "tags": len(deps),
        "tag_bytes": tag_bytes,
        "GBs": (sum(nbytes) / (sum(walls) / 1e3) / 1e9) if walls and sum(walls) > 0 else None,
        "GBs_p50": percentile(rates, 50),
        "wall_ms": dist(walls),
        "batch_ms": dist(batch),
        "sync_dur_ms": dist([x for d in deps for x in d.get("sync_ms", [])]),
        "lane_time": fields,
        "mode": last.get("mode"),
        "write_order": last.get("write_order"),
        "errors": errs,
        "verified_tags": len(verified),
        "wrong": bad,
        "correct": bool(verified) and not bad and not errs,
    }


def assemble_cell(direction, slot, write_order, ring, tag_bytes, results):
    """One summary cell from the workers' result dicts ({rank: result})."""
    index = {}
    for rr in results.values():
        for r in rr.get("runs", []):
            index[(r["direction"], r["lane"], r["role"], r["tag"])] = r
    cell = {"direction": direction, "slot": slot, "write_order": write_order,
            "ring": ring, "status": "ok", "lanes": {}, "slot_mib": {},
            "readback": {}, "lane_lines": [], "wrong": [], "notes": []}
    agg_bytes, agg_wall = 0.0, 0.0
    for lk in LANE_OF_DIR[direction]:
        ldir = DIR_OF_LANE[lk]
        tags = sorted({k[3] for k in index
                       if k[0] == direction and k[1] == lk and k[2] == "src"
                       and not index[k]["warm"]})
        deps = [index[(direction, lk, "src", t)] for t in tags]
        cols = [index[(direction, lk, "dst", t)] for t in tags
                if (direction, lk, "dst", t) in index]
        st = lane_stats(deps, cols, tag_bytes)
        cell["lanes"][ldir] = st
        for d in deps:
            if d.get("lane_time_raw"):
                cell["lane_lines"].append(d["lane_time_raw"])
        w = [d["wall_ms"] for d in deps if not d.get("err")]
        if w:
            agg_bytes += sum(d["bytes"] for d in deps if not d.get("err"))
            agg_wall = max(agg_wall, sum(w))
        for rr in results.values():
            win = (rr.get("window") or {}).get(lk)
            if win:
                cell["slot_mib"][ldir] = win["slot_mib"]
            pr = (rr.get("probes") or {}).get(ldir)
            if pr:
                cell["readback"][ldir] = pr
        if st["errors"]:
            cell["status"] = "ERROR"
            cell["notes"].extend(st["errors"][:3])
        if st["wrong"]:
            cell["status"] = "WRONG"
            cell["wrong"].extend(st["wrong"])
        elif not st["verified_tags"] and cell["status"] == "ok":
            cell["status"] = "ERROR"
            cell["notes"].append(f"lane {lk}: no collector verification came back")
    cell["agg_GBs"] = (agg_bytes / (agg_wall / 1e3) / 1e9) if agg_wall > 0 else None
    cell["correct"] = cell["status"] == "ok"
    return cell


def refused_cell(direction, slot, write_order, ring, status, why):
    return {"direction": direction, "slot": slot, "write_order": write_order,
            "ring": ring, "status": status, "lanes": {}, "slot_mib": {},
            "readback": {}, "lane_lines": [], "wrong": [], "notes": [why],
            "agg_GBs": None, "correct": False}


def summary_rows(cells):
    """Flat table rows: direction x slot x write order -> GB/s, per-batch
    p50/max, readback p50/max, correct. 'both' gets one row per lane plus the
    aggregate."""
    rows = []

    def rb(cell, ldir):
        p = (cell.get("readback") or {}).get(ldir) or {}
        ri = p.get("read_idle") or {}
        return ri.get("p50"), ri.get("max"), ri.get("n_gt")

    for c in cells:
        base = {"slot": c["slot"], "write_order": c["write_order"],
                "status": c["status"]}
        if c["status"] not in ("ok", "WRONG") or not c["lanes"]:
            rows.append({**base, "direction": c["direction"], "slot_mib": None,
                         "gbs": None, "batch_p50": None, "batch_max": None,
                         "rb_p50": None, "rb_max": None, "rb_gt5": None,
                         "correct": False,
                         "note": (c["notes"] or [""])[0][:90]})
            continue
        for ldir, st in c["lanes"].items():
            label = ldir if c["direction"] != "both" else f"both:{ldir}"
            p50, mx, gt = rb(c, ldir)
            rows.append({**base, "direction": label,
                         "slot_mib": (c["slot_mib"] or {}).get(ldir),
                         "gbs": st["GBs"],
                         "batch_p50": st["batch_ms"]["p50"],
                         "batch_max": st["batch_ms"]["max"],
                         "rb_p50": p50, "rb_max": mx, "rb_gt5": gt,
                         "correct": st["correct"], "note": ""})
        if c["direction"] == "both":
            rows.append({**base, "direction": "both:sum",
                         "slot_mib": None, "gbs": c["agg_GBs"],
                         "batch_p50": None, "batch_max": None,
                         "rb_p50": None, "rb_max": None, "rb_gt5": None,
                         "correct": c["correct"], "note": ""})
    return rows


def _f(x, w, p=2):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def format_summary(rows):
    head = (f"{'direction':<10} {'slot':>5} {'wo':>3} {'GB/s':>7} "
            f"{'batch p50':>10} {'batch max':>10} {'rb p50':>8} {'rb max':>8} "
            f"{'rb>5ms':>6}  correct")
    out = [head, "-" * len(head)]
    for r in rows:
        slot = (str(r["slot_mib"]) if r.get("slot_mib") else str(r["slot"]))
        if r["status"] not in ("ok", "WRONG"):
            tail = f"{r['status']}: {r['note']}"
        else:
            tail = "YES" if r["correct"] else "WRONG  <<<<<<"
        out.append(
            f"{r['direction']:<10} {slot:>5} {r['write_order']:>3} "
            f"{_f(r['gbs'], 7)} {_f(r['batch_p50'], 10, 3)} "
            f"{_f(r['batch_max'], 10, 3)} {_f(r['rb_p50'], 8, 3)} "
            f"{_f(r['rb_max'], 8, 3)} "
            f"{(str(r['rb_gt5']) if r['rb_gt5'] is not None else '-'):>6}  {tail}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# pure helpers: raw ceiling cells
# --------------------------------------------------------------------------


def raw_edges(n):
    """All ordered pairs of card positions, in a fixed order; lane key of edge
    k is ``p<k>``."""
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def expand_raw_patterns(n, spec, directions=DIRECTIONS):
    """[(name, [(src, dst), ...])]: edges that run AT ONCE in a pattern.
    2 cards: fwd, rev, both (per --directions). 3 cards: every ordered pair
    ``i>j``, every bidirectional pair ``i<>j``, ``fan`` (card 0 into the two
    others at once), ``ring`` (0>1, 1>2, 2>0 at once)."""
    allp = []
    if n == 2:
        m = {"fwd": [(0, 1)], "rev": [(1, 0)], "both": [(0, 1), (1, 0)]}
        allp = [(d, m[d]) for d in DIRECTIONS if d in directions]
    else:
        for s, d in raw_edges(n):
            allp.append((f"{s}>{d}", [(s, d)]))
        for i in range(n):
            for j in range(i + 1, n):
                allp.append((f"{i}<>{j}", [(i, j), (j, i)]))
        allp.append(("fan", [(0, d) for d in range(1, n)]))
        allp.append(("ring", [(i, (i + 1) % n) for i in range(n)]))
    if str(spec).strip() in ("", "auto"):
        return allp
    want = _csv(spec)
    return [p for p in allp if p[0] in want]


def expand_raw_cells(patterns, engines, chunks, streams, blocks):
    """The raw cell list. CE: x streams; SM: x blocks (one stream)."""
    cells = []
    for name, edges in patterns:
        for ch in chunks:
            for eng in engines:
                if eng == "ce":
                    for n in streams:
                        cells.append({"pattern": name, "edges": list(edges),
                                      "engine": "ce", "streams": n,
                                      "blocks": 0, "chunk_mib": ch})
                else:
                    for b in blocks:
                        cells.append({"pattern": name, "edges": list(edges),
                                      "engine": "sm", "streams": 1,
                                      "blocks": b, "chunk_mib": ch})
    for i, c in enumerate(cells):
        c["id"] = i
    return cells


def raw_plan(total, chunk, window, src):
    """[(window_off, src_off)] of one streamed pass of ``total`` bytes in
    ``chunk`` pieces: window and source offsets wrap at the largest multiple of
    the chunk that fits; always at least one chunk."""
    wu = max(chunk, window // chunk * chunk)
    su = max(chunk, src // chunk * chunk)
    n = max(1, total // chunk)
    return [((k * chunk) % wu, (k * chunk) % su) for k in range(n)]


def raw_total_for(pilot_rate_gbs, total_bytes, pilot_bytes, max_s):
    """Bytes of a timed pass: the requested total, cut so it takes at most
    ``max_s`` at the pilot's rate (never below the pilot)."""
    if not pilot_rate_gbs or pilot_rate_gbs <= 0:
        return min(total_bytes, pilot_bytes)
    return int(min(total_bytes, max(pilot_bytes, pilot_rate_gbs * 1e9 * max_s)))


def raw_agg(edge_walls_ms, edge_bytes):
    """Aggregate GB/s of the edges that ran at once: per rep, sum of bytes over
    the slowest edge's wall (conservative when they do not overlap fully);
    median over reps."""
    reps = min(len(w) for w in edge_walls_ms) if edge_walls_ms else 0
    out = []
    for r in range(reps):
        wall = max(w[r] for w in edge_walls_ms)
        if wall > 0:
            out.append(sum(edge_bytes) / (wall / 1e3) / 1e9)
    return statistics.median(out) if out else None


def assemble_raw_cell(cell, writer_msgs, verify_msgs):
    """Merge the writers' timing and the receivers' verification of one raw
    cell (lists of per-edge dicts) into the artifact cell."""
    edges = []
    for e in cell["edges"]:
        w = next((x for x in writer_msgs if tuple(x["edge"]) == tuple(e)), None)
        v = next((x for x in verify_msgs if tuple(x["edge"]) == tuple(e)), None)
        row = {"edge": list(e), "GBs": None, "GBs_best": None, "reps_ms": [],
               "bytes": None, "pilot_GBs": None, "err": "", "bad": None,
               "checked": None, "correct": False}
        if w:
            row.update({k: w[k] for k in ("GBs", "GBs_best", "reps_ms", "bytes",
                                          "pilot_GBs", "err") if k in w})
        if v:
            row["bad"], row["checked"] = v["bad"], v["checked"]
            if v.get("error"):
                row["err"] = row["err"] or f"verify: {v['error']}"
                row["bad"] = None          # unknown, not 'wrong bytes'
            row["correct"] = (v["bad"] == 0 and v["checked"] > 0 and not row["err"])
        elif not row["err"]:
            row["err"] = "no verification came back"
        edges.append(row)
    ok = [e for e in edges if e["reps_ms"] and not e["err"]]
    agg = (raw_agg([e["reps_ms"] for e in ok], [e["bytes"] for e in ok])
           if ok else None)
    wrong = [e for e in edges if e["bad"]]
    status = "ok"
    if wrong:
        status = "WRONG"
    elif any(e["err"] for e in edges):
        status = "ERROR"
    return {**{k: cell[k] for k in ("id", "pattern", "engine", "streams",
                                    "blocks", "chunk_mib")},
            "edges": edges, "agg_GBs": agg, "status": status,
            "correct": status == "ok"}


def raw_summary_rows(cells, card_names=None):
    rows = []
    for c in cells:
        sb = (f"{c['streams']}s" if c["engine"] == "ce" else f"{c['blocks']}blk")
        per = ", ".join(
            f"{e['edge'][0]}>{e['edge'][1]} {e['GBs']:.2f}" if e.get("GBs") else
            f"{e['edge'][0]}>{e['edge'][1]} -" for e in c["edges"])
        rows.append({"pattern": c["pattern"], "engine": c["engine"],
                     "sb": sb, "chunk_mib": c["chunk_mib"],
                     "agg": c["agg_GBs"], "per_edge": per,
                     "status": c["status"], "correct": c["correct"],
                     "util_max": c.get("util_max")})
    return rows


def format_raw_summary(rows):
    head = (f"{'pattern':<8} {'eng':<3} {'s/blk':>6} {'chunk':>5} {'agg GB/s':>9}  "
            f"{'per edge (card position > position, GB/s)':<44} correct")
    out = [head, "-" * len(head)]
    for r in rows:
        tail = ("YES" if r["correct"] else
                ("WRONG  <<<<<<" if r["status"] == "WRONG" else r["status"]))
        out.append(f"{r['pattern']:<8} {r['engine']:<3} {r['sb']:>6} "
                   f"{r['chunk_mib']:>5} {_f(r['agg'], 9)}  "
                   f"{r['per_edge']:<44} {tail}")
    return "\n".join(out)


def bar1_fit_reason(card, free_mib, need_mib, margin_mib=8):
    """'' if ``need_mib`` of BAR1 windows fit in what NVML reports free, else
    the named refusal."""
    if free_mib is None:
        return ""
    if need_mib + margin_mib > free_mib:
        return (f"card {card}: BAR1 windows need {need_mib} MiB (+{margin_mib} MiB "
                f"margin) but NVML reports only {free_mib} MiB BAR1 free -- lower "
                f"--raw-window-mib / --slot-mib or free the card")
    return ""


def raw_bar1_need(n, patterns, window_mib):
    """{card position: MiB of windows it must serve} for the patterns' edges."""
    edges = {tuple(e) for _, es in patterns for e in es}
    need = {i: 0 for i in range(n)}
    for s, d in edges:
        need[d] += window_mib
    return need


# --------------------------------------------------------------------------
# pure helpers: estimates
# --------------------------------------------------------------------------


def prod_ring():
    """Production ring_slots() (imports flliper lazily; 4 if it cannot)."""
    try:
        if DEFAULT_PYTHON not in sys.path:
            sys.path.insert(0, DEFAULT_PYTHON)
        from flliper.srt.pdflip.bar1_lanes import ring_slots
        return int(ring_slots())
    except Exception:  # noqa: BLE001
        return 4


def slot_value_mib(slot, big_bar):
    return {"prod": 32 if big_bar else 4}.get(slot) or int(slot)


def lane_vram_mib(a, ring):
    """Per-card VRAM of the lane cells (excluding the CUDA context), MiB."""
    specs = plan_descs(a.tag_mib << 20)
    sf, df, _ = footprints(specs)
    both = ("both" in a.directions) or (("fwd" in a.directions) and ("rev" in a.directions))
    arenas = sf + df if both else max(sf, df)
    slot = max(slot_value_mib(s, big_bar=True) for s in a.slots)
    window = slot * ring
    temps = 192          # pattern generation temporaries + scratch
    probe = slot         # the write probe's raw source buffer
    mib = lambda b: int(math.ceil(b / (1 << 20)))  # noqa: E731
    total = mib(arenas) + window + temps + probe
    return {"arena_src": mib(sf), "arena_dst": mib(df), "window": window,
            "temps": temps, "probe_src": probe, "total": total}


def raw_vram_mib(a, n_in=2):
    temps = 192
    total = a.raw_src_mib + a.raw_window_mib * n_in + a.raw_window_mib + temps
    return {"src": a.raw_src_mib, "windows": a.raw_window_mib * n_in,
            "rbuf": a.raw_window_mib, "temps": temps, "total": total}


def estimate_seconds(a, ring):
    """(typical, upper bound) seconds of the lane cells. Upper bound assumes
    the 0.2 GB/s stalled direction at every deposit."""
    cfgs = group_configs(expand_cells(a.directions, a.slots, a.write_orders))
    tag_gb = a.tag_mib * (1 << 20) / 1e9
    ntag = a.tags + a.warmup_tags
    fill = 2.0 * ntag
    typ = up = 0.0
    for cfg in cfgs:
        base = 30.0
        t_typ = t_up = base
        for d in cfg["directions"]:
            rate_typ = 3.0 if d == "rev" else 1.0
            t_typ += ntag * (tag_gb / rate_typ + fill / ntag)
            t_up += ntag * (tag_gb / 0.2 + fill / ntag)
        probes = len([d for d in cfg["directions"] if d in ("fwd", "rev")]) or \
            (2 if "both" in cfg["directions"] else 0)
        if a.readback_probe:
            t_typ += probes * (a.readback_probe * 0.002 + 2)
            t_up += probes * (a.readback_probe * 0.12 + 4)
        typ += t_typ
        up += t_up
    return typ, up


def estimate_raw_seconds(a):
    cells = expand_raw_cells(a.raw_pattern_list, a.raw_engine_list, a.raw_chunks,
                             a.raw_stream_counts, a.raw_blocks)
    per_typ = 1.2 + a.raw_reps * (a.raw_total_mib / 1024.0 / 8.0)
    per_up = 2.0 + (1 + a.raw_reps) * a.raw_max_s
    return 60 + len(cells) * per_typ, 60 + len(cells) * per_up, len(cells)


# --------------------------------------------------------------------------
# NVML (parent only; never CUDA)
# --------------------------------------------------------------------------


def _s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None


def nvml_init():
    import pynvml

    pynvml.nvmlInit()
    return pynvml


def norm_bdf(bus_id):
    s = str(_s(bus_id)).lower()
    return s[-12:]


def nvml_static(nv, idx):
    h = nv.nvmlDeviceGetHandleByIndex(int(idx))
    pci = nv.nvmlDeviceGetPciInfo(h)
    bar1 = _try(nv.nvmlDeviceGetBAR1MemoryInfo, h)
    return {
        "nvml_index": int(idx),
        "name": _s(nv.nvmlDeviceGetName(h)),
        "uuid": _s(nv.nvmlDeviceGetUUID(h)),
        "bdf": norm_bdf(pci.busId),
        "mem_total_mib": int(nv.nvmlDeviceGetMemoryInfo(h).total >> 20),
        "bar1_total_mib": int(bar1.bar1Total >> 20) if bar1 else None,
        "pcie_gen": _try(nv.nvmlDeviceGetCurrPcieLinkGeneration, h),
        "pcie_width": _try(nv.nvmlDeviceGetCurrPcieLinkWidth, h),
    }


def nvml_snapshot(nv, idx):
    h = nv.nvmlDeviceGetHandleByIndex(int(idx))
    util = _try(nv.nvmlDeviceGetUtilizationRates, h)
    mem = _try(nv.nvmlDeviceGetMemoryInfo, h)
    bar1 = _try(nv.nvmlDeviceGetBAR1MemoryInfo, h)
    pw = _try(nv.nvmlDeviceGetPowerUsage, h)
    procs = []
    for fn in (nv.nvmlDeviceGetComputeRunningProcesses,
               nv.nvmlDeviceGetGraphicsRunningProcesses):
        for p in (_try(fn, h) or []):
            used = getattr(p, "usedGpuMemory", None)
            procs.append({"pid": int(p.pid),
                          "mib": (int(used >> 20) if isinstance(used, int)
                                  and used < (1 << 60) else None)})
    return {
        "t": time.time(),
        "util_gpu": int(util.gpu) if util else None,
        "util_mem": int(util.memory) if util else None,
        "power_w": (pw / 1000.0) if pw is not None else None,
        "mem_used_mib": int(mem.used >> 20) if mem else None,
        "bar1_used_mib": int(bar1.bar1Used >> 20) if bar1 else None,
        "bar1_free_mib": int(bar1.bar1Free >> 20) if bar1 else None,
        "procs": procs,
    }


class Sampler(threading.Thread):
    """Parent-side NVML sampler while workers run (util, power, memory)."""

    def __init__(self, nv, cards, dt=0.25):
        super().__init__(daemon=True)
        self.nv, self.cards, self.dt = nv, list(cards), dt
        self.samples = []
        self._stop_ev = threading.Event()

    def run(self):
        while not self._stop_ev.is_set():
            row = {"t": time.time()}
            for c in self.cards:
                try:
                    s = nvml_snapshot(self.nv, c)
                    row[c] = (s["util_gpu"], s["power_w"], s["mem_used_mib"])
                except Exception:  # noqa: BLE001
                    row[c] = (None, None, None)
            self.samples.append(row)
            self._stop_ev.wait(self.dt)

    def stop(self):
        self._stop_ev.set()
        self.join(timeout=5)
        return self.samples


def window_stats(samples, cards, t0=None, t1=None):
    """{card: {util_max, util_mean, power_max, mem_used_max, n}} for the
    samples inside [t0, t1]."""
    out = {}
    for c in cards:
        rows = [s[c] for s in samples
                if c in s and (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
        u = [r[0] for r in rows if r[0] is not None]
        p = [r[1] for r in rows if r[1] is not None]
        m = [r[2] for r in rows if r[2] is not None]
        out[c] = {"util_max": max(u) if u else None,
                  "util_mean": (sum(u) / len(u)) if u else None,
                  "power_max": max(p) if p else None,
                  "mem_used_max": max(m) if m else None, "n": len(rows)}
    return out


# --------------------------------------------------------------------------
# workers (run in spawned child processes; CUDA only here)
# --------------------------------------------------------------------------


def _worker_env(a, info, write_order, slot_mib, ring):
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": info["uuid"],
        "FLLIPER_PDFLIP_BAR1_LANES": "1",
        "FLLIPER_PDFLIP_BAR1_CONNECT_S": "60",
        "PYTHONUNBUFFERED": "1",
    }
    if write_order is not None:
        env["FLLIPER_PDFLIP_BAR1_WRITE_ORDER"] = "1" if write_order == "on" else "0"
    if ring:
        env["FLLIPER_PDFLIP_BAR1_RING_SLOTS"] = str(ring)
    if a.connect_probe == "off":
        env["FLLIPER_PDFLIP_BAR1_READBACK_PROBE_N"] = "0"
    env.update(a.env_extra)
    return env


def _make_ops(tp, rec):
    base = tp.CudartDeviceOps

    class TimedOps(base):
        """The production ops; ``synchronize``/``stream_done_held`` also stamp
        the lane's batch finishes into ``rec``. Nothing else differs."""

        def synchronize(self, stream):
            t = time.perf_counter()
            super().synchronize(stream)
            rec.sync_done(t, time.perf_counter())

        def stream_done_held(self, stream):
            r = super().stream_done_held(stream)
            if r is True:
                rec.skip(time.perf_counter())
            return r

    return TimedOps()


def _setup_logging(res):
    lock = threading.Lock()
    tls = threading.local()
    res.setdefault("log", [])

    def cap(msg):
        s = str(msg)
        if not s.startswith("PDFLIP"):
            return
        with lock:
            if len(res["log"]) < 3000:
                res["log"].append(s)
        lines = getattr(tls, "lines", None)
        if lines is not None:
            lines.append(s)

    return cap, tls


def worker_main(spec, q, barrier):
    kind = spec.get("kind", "lane")
    res = {"kind": "final", "rank": spec["rank"], "ok": False,
           "card": spec["card"]}
    try:
        if kind == "lane":
            _lane_worker(spec, res, barrier)
        else:
            _raw_worker(spec, res, barrier, q)
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 -- the parent needs the reason
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-3000:]
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass
    finally:
        q.put(res)


def _imports(spec):
    os.environ.update(spec["env"])
    sys.path.insert(0, spec["python"])
    import torch

    torch.cuda.set_device(0)
    from flliper.srt.distributed.device_communicators.barlink_bar1 import bar1_window
    from flliper.srt.pdflip import bar1_lanes as bl
    from flliper.srt.pdflip import weight_exchange_transport as tp
    return torch, bl, tp, bar1_window


def _lane_worker(spec, res, barrier):
    torch, bl, tp, bar1_window = _imports(spec)
    from flliper.srt.pdflip.weight_exchange import XchgDesc

    rank = spec["rank"]
    cap, tls = _setup_logging(res)
    rec = Recorder()
    ops = _make_ops(tp, rec)
    lanes = bl.Bar1Lanes(spec["nonce"], "P" if rank == 0 else "D", rank, 0,
                         [(0, 1), (1, 0)], log=cap)
    bdf = lanes.bdf()
    if bdf.lower() != spec["expect_bdf"].lower():
        raise RuntimeError(f"worker on card {spec['card']}: CUDA says BDF {bdf}, "
                           f"NVML says {spec['expect_bdf']} -- refusing to "
                           f"measure the wrong card")
    res["bdf"] = bdf
    prod_mib = bl.slot_bytes_for(int(bar1_window(bdf).size)) >> 20
    slot_mib = prod_mib if spec["slot"] == "prod" else int(spec["slot"])
    # THE bench-side seam (no production change): take the big-BAR branch of
    # open_window on whatever card this is, with this slot (docstring).
    os.environ[bl.ENV_BIG_SLOT_MIB] = str(slot_mib)
    bl.BIG_BAR_MIN = 1
    lane_keys = spec["lane_keys"]
    lanes.setup(lane_keys)
    res["window"], miss = {}, []
    for lk in lane_keys:
        role = lanes.role(lk)
        w = lanes.window_for(lk, role)
        if w is None:
            miss.append(lk)
        else:
            res["window"][lk] = {"role": role, "slot_mib": w[1] >> 20,
                                 "ring": w[2], "window_mib": (w[1] * w[2]) >> 20}
    if miss:
        res["status"] = "REFUSED"
        res["refusals"] = dict(lanes.refusals)
        raise RuntimeError("lane(s) %s not established: %s" % (
            miss, "; ".join(f"{k}: {v}" for k, v in lanes.refusals.items())
            or "no refusal text"))
    barrier.wait(BARRIER_S)

    tag_bytes = spec["tag_mib"] << 20
    specs = plan_descs(tag_bytes)
    sf, df, _pay = footprints(specs)
    arenas = {}
    for lk in lane_keys:
        role = lanes.role(lk)
        arenas[lk] = torch.empty(sf if role == "src" else df, dtype=torch.uint8,
                                 device="cuda:0")

    def mk_descs(lk, role):
        s_rank, d_rank = ((0, 1) if lk == "p0" else (1, 0))
        base = arenas[lk].data_ptr()
        return [XchgDesc(tag=TAG_NAME, src_rank=s_rank, dst_rank=d_rank,
                         param_name=d["param_name"], kind=d["kind"],
                         nbytes=d["nbytes"], rows=d["rows"],
                         run_bytes=d["run_bytes"], spitch=d["spitch"],
                         dpitch=d["dpitch"], src_off=d["src_off"],
                         dst_off=d["dst_off"], pieces=1,
                         src_ptr=base if role == "src" else None,
                         dst_ptr=base if role == "dst" else None)
                for d in specs]

    descs = {lk: mk_descs(lk, lanes.role(lk)) for lk in lane_keys}
    runs = res.setdefault("runs", [])
    warm_n = spec["warmup"]
    BUDGET = 60.0

    def lane_thread(lk, role, direction, ti, seq, out):
        tls.lines = []
        rec.begin()
        t0 = time.perf_counter()
        err = ""
        try:
            err = bl.run_bar1_units(descs[lk], ops, lanes=lanes, lane_key=lk,
                                    role=role, seq=seq, phase="bench",
                                    budget_s=BUDGET, device=0, log=cap) or ""
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        wall = (time.perf_counter() - t0) * 1000.0
        got = rec.end()
        lt_raw = next((ln for ln in tls.lines
                       if "lane-time" in ln and f"seq={seq} " in ln + " "), "")
        out.append({"direction": direction, "lane": lk, "role": role, "tag": ti,
                    "warm": ti < warm_n, "seq": seq, "wall_ms": wall,
                    "bytes": tag_bytes, "err": err,
                    "lane_time_raw": lt_raw, "lane_time": parse_lane_time(lt_raw),
                    "batch_ms": got["batch_ms"], "sync_ms": got["sync_ms"],
                    "verify": None})
        tls.lines = None

    for di, direction in enumerate(spec["directions"]):
        lks = LANE_OF_DIR[direction]
        for ti in range(warm_n + spec["tags"]):
            for li, lk in enumerate(lks):
                seed = 5 + ti * 13 + li * 101 + di * 1009 + spec["cfg"] * 7919
                if lanes.role(lk) == "src":
                    fill_src(arenas[lk], specs, seed)
                else:
                    arenas[lk].zero_()
            torch.cuda.synchronize()
            barrier.wait(BARRIER_S)
            out, ths = [], []
            for lk in lks:
                role = lanes.role(lk)
                seq = f"c{spec['cfg']}-{direction}-{lk}-t{ti}"
                th = threading.Thread(target=lane_thread,
                                      args=(lk, role, direction, ti, seq, out))
                th.start()
                ths.append(th)
            for th in ths:
                th.join()
            barrier.wait(BARRIER_S)
            for r in out:
                if r["role"] == "dst":
                    li = lks.index(r["lane"])
                    seed = 5 + ti * 13 + li * 101 + di * 1009 + spec["cfg"] * 7919
                    r["verify"] = check_dst(arenas[r["lane"]], specs, seed)
            runs.extend(out)

    # -- the readback probe cell -------------------------------------------
    res["probes"] = {}
    n = spec["probe_n"]
    for direction in ("fwd", "rev"):
        if n <= 0:
            break
        lk = LANE_OF_DIR[direction][0]
        if lk not in lane_keys:
            continue
        barrier.wait(BARRIER_S)
        if lanes.role(lk) == "src":
            res["probes"][direction] = _probe_lane(ops, bl, lanes, lk, n)
        barrier.wait(BARRIER_S)
    res["refusals"] = dict(lanes.refusals)


def _lat_stats(lat_ms):
    d = dist(lat_ms, gt=5.0)
    d["samples_ms"] = [round(x, 4) for x in lat_ms]
    return d


def _probe_lane(ops, bl, lanes, lk, n):
    """Probe cell of one depositor lane: n individually synchronized 16-byte
    reads of the PEER window (peer BAR1 -> own scratch, the production Fix-A
    read), plus references so the cost can be attributed: the same read on the
    own card, a slot-sized write alone, and a slot-sized write followed by the
    16-byte read on one stream (= one production batch)."""
    base, slot, ring = lanes.window_for(lk, "src")
    scratch = bl.readback_scratch(ops, 0)
    st = ops.create_stream(0)
    src = ops.raw_malloc(0, int(slot))
    clk = time.perf_counter
    pc = bl.READBACK_BYTES

    def run(fn):
        lat = []
        for i in range(n):
            t = clk()
            fn(i)
            ops.synchronize(st)
            lat.append((clk() - t) * 1000.0)
        return lat

    def peer_read(i):
        ops.memcpy_async(scratch, base + (i % ring) * slot, pc, st)

    def local_read(i):
        ops.memcpy_async(scratch, src, pc, st)

    def write_only(i):
        ops.memcpy_async(base + (i % ring) * slot, src, int(slot), st)

    def write_then_read(i):
        o = base + (i % ring) * slot
        ops.memcpy_async(o, src, int(slot), st)
        ops.memcpy_async(scratch, o, pc, st)

    try:
        for i in range(4):          # warm: first read maps pages / JIT
            peer_read(i)
        ops.synchronize(st)
        out = {"read_idle": _lat_stats(run(peer_read)),
               "local_read": _lat_stats(run(local_read)),
               "write_only": _lat_stats(run(write_only)),
               "write_then_read": _lat_stats(run(write_then_read)),
               "slot_mib": int(slot) >> 20, "n": n}
        wo, wr = out["write_only"]["p50"], out["write_then_read"]["p50"]
        out["readback_price_ms"] = (wr - wo) if (wo is not None and wr is not None) else None
    finally:
        try:
            ops.destroy_stream(st)
        except Exception:  # noqa: BLE001
            pass
        try:
            ops.raw_free(src)
        except Exception:  # noqa: BLE001
            pass
    return out


# -- raw worker -------------------------------------------------------------


def _raw_worker(spec, res, barrier, q):
    torch, bl, tp, bar1_window = _imports(spec)
    from flliper.srt.pdflip import lane_sm_copy

    rank, n = spec["rank"], spec["n"]
    cap, _tls = _setup_logging(res)
    ops = tp.CudartDeviceOps()
    edges = [tuple(e) for e in spec["edges"]]
    lane_of = {e: f"p{k}" for k, e in enumerate(edges)}
    recv = bl.Bar1Lanes(spec["nonce"], "D", rank, 0, edges, log=cap)
    send = bl.Bar1Lanes(spec["nonce"], "P", rank, 0, edges, log=cap)
    bdf = recv.bdf()
    if bdf.lower() != spec["expect_bdf"].lower():
        raise RuntimeError(f"worker on card {spec['card']}: CUDA says BDF {bdf}, "
                           f"NVML says {spec['expect_bdf']}")
    res["bdf"] = bdf
    ring = int(os.environ.get(bl.ENV_RING, "0") or 0) or bl.ring_slots()
    W = spec["window_mib"]
    if W % ring:
        raise RuntimeError(f"--raw-window-mib {W} is not a multiple of ring {ring}")
    os.environ[bl.ENV_BIG_SLOT_MIB] = str(W // ring)
    bl.BIG_BAR_MIN = 1                    # same bench-side seam as the lane cells
    incoming = [lane_of[e] for e in edges if e[1] == rank]
    outgoing = [lane_of[e] for e in edges if e[0] == rank]
    recv.setup(incoming)
    send.setup(outgoing)
    miss = [lk for lk in incoming if lk not in recv.recv] + \
           [lk for lk in outgoing if lk not in send.peers]
    if miss:
        res["status"] = "REFUSED"
        res["refusals"] = {**recv.refusals, **send.refusals}
        raise RuntimeError(f"raw lane(s) {miss} not established: "
                           f"{res['refusals']}")
    barrier.wait(BARRIER_S)

    want_sm = any(c["engine"] == "sm" for c in spec["cells"])
    copier, sm_why = (lane_sm_copy.copier_for(0) if want_sm else (None, ""))
    res["sm"] = {"available": copier is not None, "why": sm_why,
                 "build_ms": getattr(copier, "build_ms", None)}
    S = spec["src_mib"] << 20
    src_t = torch.empty(S, dtype=torch.uint8, device="cuda:0")
    rbuf = torch.empty(W << 20, dtype=torch.uint8, device="cuda:0")
    scratch = bl.readback_scratch(ops, 0)
    ce_copy, _ = bl.deposit_copiers(ops)[:2]
    streams = {}

    def get_streams(k):
        if k not in streams:
            streams[k] = [ops.create_stream(0) for _ in range(k)]
        return streams[k]

    def seed_of(cell_id, src_rank):
        return 3 + cell_id * 31 + src_rank * 7

    def issue(cell, lane, plan, chunk, sts, timed_out):
        base, slot, rg = send.window_for(lane, "src")
        sp = src_t.data_ptr()
        for k, (o, s) in enumerate(plan):
            st = sts[k % len(sts)]
            if cell["engine"] == "ce":
                ce_copy(base + o, sp + s, chunk, st)
            else:
                copier.copy_async(base + o, sp + s, chunk, st, blocks=cell["blocks"])

    def stream_pass(cell, e, total, ring_out):
        lane = lane_of[e]
        chunk = cell["chunk_mib"] << 20
        base, slot, rg = send.window_for(lane, "src")
        plan = raw_plan(total, chunk, int(slot) * rg, S)
        sts = get_streams(cell["streams"])
        t0 = time.perf_counter()
        try:
            issue(cell, lane, plan, chunk, sts, None)
            for st in sts:
                ops.synchronize(st)
            ring_out[e] = ((time.perf_counter() - t0) * 1000.0,
                           len(plan) * chunk, "")
        except BaseException as exc:  # noqa: BLE001
            ring_out[e] = (0.0, 0, f"{type(exc).__name__}: {exc}")

    def run_threads(cell, my_out, total_of):
        out, ths = {}, []
        for e in my_out:
            th = threading.Thread(target=stream_pass,
                                  args=(cell, e, total_of[e], out))
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        return out

    for cell in spec["cells"]:
        cid = cell["id"]
        cell_edges = [tuple(e) for e in cell["edges"]]
        my_out = [e for e in cell_edges if e[0] == rank]
        my_in = [e for e in cell_edges if e[1] == rank]
        skip = (cell["engine"] == "sm" and copier is None)
        if my_out and not skip:
            fill_src(src_t, [_flat_spec(S)], seed_of(cid, rank))
        torch.cuda.synchronize()
        barrier.wait(BARRIER_S)
        t_start = time.time()
        wmsg, werr = [], ""
        chunk = cell["chunk_mib"] << 20
        pilot_b = min(spec["total_mib"] << 20, 128 << 20)
        totals, pilot = {}, {}
        if skip:
            werr = f"SM-REFUSED: {sm_why}"
        if my_out and not skip:
            pilot = run_threads(cell, my_out, {e: pilot_b for e in my_out})
            for e in my_out:
                ms, nb, er = pilot[e]
                rate = (nb / (ms / 1e3) / 1e9) if ms > 0 else 0.0
                totals[e] = raw_total_for(rate, spec["total_mib"] << 20,
                                          pilot_b, spec["max_s"])
        reps = {e: [] for e in my_out}
        errs = {e: "" for e in my_out}
        for r in range(spec["reps"]):
            barrier.wait(BARRIER_S)
            if my_out and not skip:
                got = run_threads(cell, my_out, totals)
                for e in my_out:
                    ms, nb, er = got[e]
                    if er:
                        errs[e] = er
                    reps[e].append((ms, nb))
        barrier.wait(BARRIER_S)
        # untimed verify pass (same engine/chunk), arrival read, then the receivers check
        if my_out and not skip:
            for e in my_out:
                lane = lane_of[e]
                base, slot, rg = send.window_for(lane, "src")
                Wb = int(slot) * rg
                wu = max(chunk, Wb // chunk * chunk)
                sts = get_streams(cell["streams"])
                vplan = [(k * chunk, k * chunk) for k in range(wu // chunk)]
                try:
                    issue(cell, lane, vplan, chunk, sts, None)
                    for st in sts:
                        ops.synchronize(st)
                    ops.memcpy_async(scratch, base + wu - bl.READBACK_BYTES,
                                     bl.READBACK_BYTES, sts[0])
                    ops.synchronize(sts[0])
                except BaseException as exc:  # noqa: BLE001
                    errs[e] = errs[e] or f"verify write: {type(exc).__name__}: {exc}"
        barrier.wait(BARRIER_S)
        vmsg = []
        for e in my_in:
            lane = lane_of[e]
            w = recv.recv[lane]
            Wb = int(w.slot_bytes) * int(w.ring)
            wu = max(chunk, Wb // chunk * chunk)
            bad = checked = 0
            try:
                ops.rdma_flush(0)
                ops.memcpy_async(rbuf.data_ptr(), int(w.dptr), wu, 0)
                ops.synchronize(0)
                seed = seed_of(cid, e[0])
                for c in range(0, wu, CHUNK):
                    m = min(CHUNK, wu - c)
                    exp = pattern(c, m, seed, "cuda:0")
                    bad += int((rbuf[c:c + m] != exp).sum().item())
                    checked += m
            except BaseException as exc:  # noqa: BLE001
                bad, checked = -1, 0
                vmsg.append({"edge": list(e), "bad": 1, "checked": 0,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            vmsg.append({"edge": list(e), "bad": bad, "checked": checked})
        for e in my_out:
            rs = reps[e]
            gb = [nb / (ms / 1e3) / 1e9 for ms, nb in rs if ms > 0]
            pm, pb, _pe = pilot.get(e, (0.0, 0, ""))
            wmsg.append({"edge": list(e),
                         "GBs": statistics.median(gb) if gb else None,
                         "GBs_best": max(gb) if gb else None,
                         "reps_ms": [ms for ms, _ in rs],
                         "bytes": rs[0][1] if rs else 0,
                         "pilot_GBs": (pb / (pm / 1e3) / 1e9) if pm > 0 else None,
                         "err": werr or errs[e]})
        barrier.wait(BARRIER_S)
        q.put({"kind": "cell", "rank": rank, "cell": cid, "writers": wmsg,
               "verify": vmsg, "t0": t_start, "t1": time.time()})
    for lst in streams.values():
        for st in lst:
            try:
                ops.destroy_stream(st)
            except Exception:  # noqa: BLE001
                pass


def _flat_spec(n):
    return {"param_name": "raw", "kind": "FLAT", "nbytes": int(n), "rows": 1,
            "run_bytes": int(n), "spitch": int(n), "dpitch": int(n),
            "src_off": 0, "dst_off": 0, "logical_off": 0}


# --------------------------------------------------------------------------
# parent orchestration
# --------------------------------------------------------------------------


def _spawn(ctx, specs, barrier, q):
    procs = []
    for sp in specs:
        p = ctx.Process(target=worker_main, args=(sp, q, barrier),
                        name=f"bar1lane-{sp.get('kind', 'lane')}-{sp['rank']}")
        p.start()
        procs.append(p)
    return procs


def _collect(procs, q, n_final, deadline, on_cell=None):
    import queue as _queue

    finals = {}
    while len(finals) < n_final and time.time() < deadline:
        try:
            m = q.get(timeout=2.0)
        except _queue.Empty:
            if all(p.exitcode is not None for p in procs):
                try:
                    while True:           # drain what the dead workers left
                        m = q.get(timeout=0.5)
                        if m.get("kind") == "final":
                            finals[m["rank"]] = m
                        elif on_cell:
                            on_cell(m)
                except _queue.Empty:
                    pass
                break
            continue
        if m.get("kind") == "final":
            finals[m["rank"]] = m
        elif on_cell:
            on_cell(m)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
            p.join(timeout=10)
    return finals


def _cleanup_shm(nonce_prefix):
    root = "/dev/shm"
    try:
        for name in os.listdir(root):
            if name == f"pdflip-bar1-{nonce_prefix}" or \
                    name.startswith(f"pdflip-bar1-{nonce_prefix}"):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    except OSError:
        pass


def run_lane_configs(a, infos, nv, art, ring):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    cells_all = []
    cfgs = group_configs(expand_cells(a.directions, a.slots, a.write_orders))
    src, dst = infos[a.src_card], infos[a.dst_card]
    need_free = {c: (nvml_snapshot(nv, c)["bar1_free_mib"]) for c in (a.src_card, a.dst_card)}
    print(f"# BAR1 free before the lane cells: "
          f"card {a.src_card} {need_free[a.src_card]} MiB, "
          f"card {a.dst_card} {need_free[a.dst_card]} MiB", flush=True)
    for ci, cfg in enumerate(cfgs):
        wo, slot = cfg["write_order"], cfg["slot"]
        t0 = time.time()
        # BAR1 must hold the receiver windows (named refusal, not a late ENOMEM)
        why = ""
        for card in receivers_for(a, cfg["directions"]):
            big = (infos[card].get("bar1_total_mib") or 0) >= 4096
            need = slot_value_mib(slot, big) * ring
            why = why or bar1_fit_reason(card, need_free[card], need)
        if why:
            print(f"# config {ci} slot={slot} wo={wo}: REFUSED {why}", flush=True)
            for d in cfg["directions"]:
                cells_all.append(refused_cell(d, slot, wo, ring, "REFUSED", why))
            continue
        nonce = f"lb{os.getpid()}c{ci}"
        barrier, q = ctx.Barrier(2), ctx.Queue()
        specs = []
        for rank, card in enumerate((a.src_card, a.dst_card)):
            specs.append({"kind": "lane", "rank": rank, "card": card,
                          "expect_bdf": infos[card]["bdf"], "nonce": nonce,
                          "python": a.flliper_python, "cfg": ci,
                          "env": _worker_env(a, infos[card], wo, None, ring),
                          "slot": slot, "directions": cfg["directions"],
                          "lane_keys": lanes_for(cfg["directions"]),
                          "tag_mib": a.tag_mib, "tags": a.tags,
                          "warmup": a.warmup_tags, "probe_n": a.readback_probe})
        print(f"# config {ci + 1}/{len(cfgs)}: slot={slot} write-order={wo} "
              f"directions={cfg['directions']} ...", flush=True)
        snap_before = {c: nvml_snapshot(nv, c) for c in (a.src_card, a.dst_card)}
        sampler = Sampler(nv, (a.src_card, a.dst_card))
        sampler.start()
        procs = _spawn(ctx, specs, barrier, q)
        deadline = time.time() + 600 + 60 * (a.tags + a.warmup_tags) * len(cfg["directions"])
        finals = _collect(procs, q, 2, deadline)
        samples = sampler.stop()
        _cleanup_shm(nonce)
        snap_after = {c: nvml_snapshot(nv, c) for c in (a.src_card, a.dst_card)}
        nv_during = window_stats(samples, (a.src_card, a.dst_card), t0, time.time())
        art["nvml_during"].append({
            "config": ci, "slot": slot, "write_order": wo,
            "before": {str(k): v for k, v in snap_before.items()},
            "after": {str(k): v for k, v in snap_after.items()},
            "per_card": {str(k): v for k, v in nv_during.items()}})
        art["worker_logs"].append({"config": ci, "ranks": {
            str(r): {"log": f.get("log", []), "refusals": f.get("refusals", {}),
                     "error": f.get("error"), "trace": f.get("trace")}
            for r, f in finals.items()}})
        bad = [f for f in finals.values() if not f.get("ok")]
        if len(finals) < 2 or bad:
            why = "; ".join(f"rank {r}: {f.get('error')}" for r, f in finals.items()
                            if f.get("error")) or \
                f"worker(s) died without a result (exit codes " \
                f"{[p.exitcode for p in procs]})"
            status = "REFUSED" if any(f.get("status") == "REFUSED"
                                      for f in finals.values()) else "ERROR"
            print(f"# config {ci}: {status}: {why[:300]}", flush=True)
            for d in cfg["directions"]:
                cells_all.append(refused_cell(d, slot, wo, ring, status, why[:600]))
            continue
        for d in cfg["directions"]:
            cell = assemble_cell(d, slot, wo, ring, a.tag_mib << 20, finals)
            lt = [x for rr in finals.values() for x in rr.get("log", [])
                  if x.startswith("PDFLIP-BAR1 lane-time")]
            cell["worker_lane_time_lines"] = len(lt)
            cells_all.append(cell)
            print(f"#   {d}: {cell['status']} "
                  f"{cell['agg_GBs'] if cell['agg_GBs'] is None else round(cell['agg_GBs'], 2)} GB/s",
                  flush=True)
    return cells_all


def run_raw(a, infos, nv, art):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    cards = a.cards_list
    n = len(cards)
    pats = a.raw_pattern_list
    cells = expand_raw_cells(pats, a.raw_engine_list, a.raw_chunks,
                             a.raw_stream_counts, a.raw_blocks)
    edges = sorted({tuple(e) for _, es in pats for e in es})
    # BAR1 fit: print free first, refuse by name
    snaps = {c: nvml_snapshot(nv, c) for c in cards}
    print("# BAR1 free before the raw cells: " + ", ".join(
        f"card {c} {snaps[c]['bar1_free_mib']} MiB (total {infos[c].get('bar1_total_mib')})"
        for c in cards), flush=True)
    need = raw_bar1_need(n, pats, a.raw_window_mib)
    why = "; ".join(r for i, c in enumerate(cards) if (
        r := bar1_fit_reason(c, snaps[c]["bar1_free_mib"], need[i])))
    if why:
        print(f"# raw cells REFUSED: {why}", flush=True)
        return [], why
    nonce = f"lbraw{os.getpid()}"
    barrier, q = ctx.Barrier(n), ctx.Queue()
    ring = a.ring or prod_ring()
    specs = []
    for rank, card in enumerate(cards):
        env = _worker_env(a, infos[card], None, None, ring)
        env.pop("FLLIPER_PDFLIP_BAR1_WRITE_ORDER", None)
        env["FLLIPER_PDFLIP_BAR1_READBACK_PROBE_N"] = "0"
        specs.append({"kind": "raw", "rank": rank, "n": n, "card": card,
                      "expect_bdf": infos[card]["bdf"], "nonce": nonce,
                      "python": a.flliper_python, "env": env, "edges": edges,
                      "window_mib": a.raw_window_mib, "src_mib": a.raw_src_mib,
                      "total_mib": a.raw_total_mib, "reps": a.raw_reps,
                      "max_s": a.raw_max_s, "cells": cells})
    got = {}

    def on_cell(m):
        got.setdefault(m["cell"], []).append(m)

    sampler = Sampler(nv, cards)
    sampler.start()
    print(f"# raw: {len(cells)} cells on cards {cards} ...", flush=True)
    procs = _spawn(ctx, specs, barrier, q)
    _, up, _nc = estimate_raw_seconds(a)
    finals = _collect(procs, q, n, time.time() + up + 300, on_cell)
    samples = sampler.stop()
    _cleanup_shm(nonce)
    out = []
    for c in cells:
        msgs = got.get(c["id"], [])
        w = [x for m in msgs for x in m["writers"]]
        v = [x for m in msgs for x in m["verify"]]
        ac = assemble_raw_cell(c, w, v)
        if msgs:
            ac["util"] = window_stats(samples, cards, min(m["t0"] for m in msgs),
                                      max(m["t1"] for m in msgs))
            ac["util_max"] = {str(k): s["util_max"] for k, s in ac["util"].items()}
            ac["util"] = {str(k): s for k, s in ac["util"].items()}
        else:
            ac["status"] = "ERROR"
            ac["edges"] = [dict(e, err="cell never completed") for e in ac["edges"]]
        out.append(ac)
    art["worker_logs"].append({"raw": True, "ranks": {
        str(r): {"log": f.get("log", []), "refusals": f.get("refusals", {}),
                 "error": f.get("error"), "trace": f.get("trace"),
                 "sm": f.get("sm")} for r, f in finals.items()}})
    errs = [f"rank {r}: {f['error']}" for r, f in finals.items() if f.get("error")]
    return out, "; ".join(errs)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_artifact(a, argv):
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "argv": list(argv),
        "args": {k: v for k, v in vars(a).items() if not k.startswith("raw_pattern_list")},
        "cards": [], "gpuq": {}, "baseline": {}, "after": {},
        "contended": False, "contention": {},
        "nvml_during": [], "worker_logs": [],
        "cells": [], "raw_cells": [], "summary": [], "raw_summary": [],
        "summary_text": "", "raw_summary_text": "", "notes": [],
    }


def print_plan(a):
    ring = a.ring or prod_ring()
    print(f"# schema {SCHEMA}  ring={ring}  python={a.flliper_python}")
    if not a.skip_lanes:
        cells = expand_cells(a.directions, a.slots, a.write_orders)
        cfgs = group_configs(cells)
        print(f"# LANE cells: {len(cells)} in {len(cfgs)} worker-pair config(s); "
              f"src card {a.src_card}, dst card {a.dst_card}")
        for c in cells:
            print(f"#   {c.direction:<5} slot={c.slot:<5} write-order={c.write_order}")
        specs = plan_descs(a.tag_mib << 20)
        sf, df, pay = footprints(specs)
        k2 = sum(1 for s in specs if s["kind"] == "STRIDED2D")
        print(f"# tag = {pay >> 20} MiB payload in {len(specs)} descs "
              f"({len(specs) - k2} FLAT, {k2} STRIDED2D); arenas src "
              f"{sf >> 20} MiB / dst {df >> 20} MiB; tags {a.tags} "
              f"(+{a.warmup_tags} warm-up) per direction")
        v = lane_vram_mib(a, ring)
        typ, up = estimate_seconds(a, ring)
        print(f"# lane VRAM per card (book mib, context NOT included): "
              f"{v['total']} = arenas {v['arena_src']}+{v['arena_dst']} + window "
              f"{v['window']} + temps {v['temps']} + probe {v['probe_src']}")
        print(f"# lane runtime: typical {typ / 60:.1f} min, upper bound "
              f"{up / 60:.1f} min (0.2 GB/s at every deposit)")
        print(f"# worker env: {json.dumps(_worker_env(a, {'uuid': '<NVML UUID>'}, 'on', None, a.ring))}")
    if a.raw:
        cells = expand_raw_cells(a.raw_pattern_list, a.raw_engine_list,
                                 a.raw_chunks, a.raw_stream_counts, a.raw_blocks)
        typ, up, nc = estimate_raw_seconds(a)
        n = len(a.cards_list)
        print(f"# RAW cells: {nc} on cards {a.cards_list}; patterns "
              f"{[p[0] for p in a.raw_pattern_list]}; engines {a.raw_engine_list}; "
              f"chunks {a.raw_chunks} MiB; CE streams {a.raw_stream_counts}; "
              f"SM blocks {a.raw_blocks}")
        rv = raw_vram_mib(a, n_in=n - 1)
        print(f"# raw VRAM per card (book mib, context NOT included): "
              f"{rv['total']} = src {rv['src']} + windows {rv['windows']} + "
              f"verify buf {rv['rbuf']} + temps {rv['temps']}; BAR1 windows "
              f"{a.raw_window_mib * (n - 1)} MiB per card")
        print(f"# raw runtime: typical {typ / 60:.1f} min, upper bound {up / 60:.1f} min")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    if a.plan:
        print_plan(a)
        return 0
    ring = a.ring or prod_ring()
    lane_cards = [] if a.skip_lanes else [a.src_card, a.dst_card]
    raw_cards = a.cards_list if a.raw else []
    cards = sorted(set(lane_cards) | set(raw_cards))
    window = _rb.require_gpuq_window(cards, a.gpuq_booking)
    print(f"# gpuq window {window.get('id')} owner {window.get('owner')!r} cards "
          f"{window.get('cards')} -- running. The service plans, it does not "
          f"lock; the NVML/nvidia-smi checks below are the real ones.", flush=True)
    _rb.preflight(cards, a.max_used_mib)

    nv = nvml_init()
    art = new_artifact(a, argv)
    art["gpuq"] = {k: window.get(k) for k in ("id", "owner", "cards", "state", "purpose")}
    infos = {c: nvml_static(nv, c) for c in cards}
    art["cards"] = [infos[c] for c in cards]
    print("# NVML index -> name -> BDF (resolved now, never assumed):")
    for c in cards:
        i = infos[c]
        print(f"#   {c} -> {i['name']} -> {i['bdf']}  BAR1 {i['bar1_total_mib']} MiB  "
              f"PCIe gen{i['pcie_gen']} x{i['pcie_width']}  VRAM {i['mem_total_mib']} MiB",
              flush=True)

    # -- baseline: no worker of ours is alive -----------------------------
    before = {c: nvml_snapshot(nv, c) for c in cards}
    util_samples = {c: [before[c]["util_gpu"]] for c in cards}
    for _ in range(5):
        time.sleep(0.4)
        for c in cards:
            util_samples[c].append(nvml_snapshot(nv, c)["util_gpu"])
    verdict = contention_verdict(util_samples, a.contend_pct)
    art["baseline"] = {str(c): before[c] for c in cards}
    art["contention"] = {str(c): v for c, v in verdict.items()}
    art["contended"] = any_contended(verdict)
    for c, v in verdict.items():
        if v["contended"]:
            procs = before[c]["procs"]
            print(f"# {CONTENDED}: card {c} ({infos[c]['name']}) shows "
                  f"{v['max_util']}% utilisation with no worker of ours alive "
                  f"(foreign processes per NVML: {procs or 'none listed'}) -- "
                  f"every number measured on it is confounded", flush=True)

    rc = 0
    try:
        if not a.skip_lanes:
            art["cells"] = run_lane_configs(a, infos, nv, art, ring)
            rows = summary_rows(art["cells"])
            art["summary"] = rows
            art["summary_text"] = format_summary(rows)
        if a.raw:
            art["raw_cells"], raw_err = run_raw(a, infos, nv, art)
            if raw_err:
                art["notes"].append(f"raw: {raw_err}")
            rrows = raw_summary_rows(art["raw_cells"])
            art["raw_summary"] = rrows
            art["raw_summary_text"] = format_raw_summary(rrows)
    except Exception as exc:  # noqa: BLE001 -- the artifact is written anyway
        rc = 2
        art["notes"].append(f"ABORTED: {type(exc).__name__}: {exc}")
        art["notes"].append(traceback.format_exc()[-2000:])
        print(f"# ABORTED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        after = {c: nvml_snapshot(nv, c) for c in cards}
        art["after"] = {str(c): after[c] for c in cards}
        post = contention_verdict({c: [after[c]["util_gpu"]] for c in cards},
                                  a.contend_pct)
        art["contention_after"] = {str(c): v for c, v in post.items()}
        if any_contended(post):
            art["contended"] = True
        art["finished_utc"] = utc_now()

    wrong = [c for c in art["cells"] if c["status"] == "WRONG"] + \
            [c for c in art["raw_cells"] if c["status"] == "WRONG"]
    print()
    if art["contended"]:
        hot = sorted({c for key in ("contention", "contention_after")
                      for c, v in art.get(key, {}).items() if v["contended"]})
        print(f"*** {CONTENDED}: foreign load on card(s) {hot} -- see the "
              f"artifact; treat the numbers on those cards as suspect ***")
    if art["summary_text"]:
        print("=== LANE CELLS (production run_bar1_units; GB/s = payload / "
              "depositor wall incl. the collector's done) ===")
        print(art["summary_text"])
    if art["raw_summary_text"]:
        print("\n=== RAW CEILING CELLS (no credits; CE streams / SM blocks) ===")
        print(art["raw_summary_text"])
    if wrong:
        rc = 1
        print("\n*** WRONG BYTES ARRIVED in %d cell(s) -- those numbers are void ***"
              % len(wrong))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(art, f, indent=1, default=str)
        print(f"\nartifact: {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
