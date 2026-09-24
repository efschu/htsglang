"""The P cut's stage cost, FITTED from a boot's own #PGAP lines (27B line).

Default OFF: nothing here runs unless the launcher is given
``--pp-cut-stage-fit PATH|auto``. Unset, ``solve_p_cut`` prices exactly as
before (the bsscale 4096-chunk family split, ``family_costs_from_measurement``).

WHY. The standing cost model was calibrated on 4096-token chunks (bsscale,
2026-09-07) plus a card-rate library that puts the 5090 at 3.99x a 3080 per
layer. Group P now runs 512-token chunks, and the #PGAP instrument measures
every forward on the card. Joined to the chunk's prefix depth, one boot gives,
per rank, a straight line ``gpu_fwd = a + b * (prefix + C/2) / 1000`` with
1.2-3.6 ms residual over 0..255k prefix (xsn426 eager, xsn428 graph, 764-1667
chunks per rank). At 512 the measured per-layer ratio 5090 : 3080 is 3.08
(not 3.99), and the 5090's attention slope is 1.63x its 4096 value (192 CTAs
on 170 SMs, see layers/attention/fi_prefill_wave_split.py) -- both move the
optimal cut, and neither is visible to a 4096 calibration.

THE MODEL, per stage r of a candidate cut with n_r layers of which A_r are
full attention:

    T_r(prefix) = n_r * layer_ms[r] + stage_fixed_ms[r]
                  + A_r * attn_ms_per_1k[r] * (prefix + C/2) / 1000

from the fit ``a_r, b_r`` of the MEASURED cut (n_r, A_r):
    * ``attn_ms_per_1k[r] = b_r / A_r`` -- the slope is attention alone (a GDN
      layer's cost does not depend on the prefix);
    * ``layer_ms``: stages on the SAME card model share one per-layer rate,
      taken from a MIDDLE stage of that card (no embedding, no final norm /
      lm_head / draft producer); a first or last stage of that card keeps the
      remainder as ``stage_fixed_ms`` (per forward). Measured: PP2 (last)
      carries +4.1 ms per 512 chunk over PP1 at the same 11 layers, and +0.6 to
      1.0 ms per 512 tokens at 4096 -- the scaling of a per-FORWARD term, not a
      per-layer one. A card with no middle stage folds its intercept into its
      per-layer rate (the 5090 carries the embedding that way);
    * ASSUMPTION, stated: an attention layer's prefix-INDEPENDENT cost (its
      MLP and projections) equals a GDN layer's on the same card. By FLOPs
      they differ by ~3 % (380.8 vs 391.8 GOP per 512 tokens) plus the GDN
      recurrent kernels; moving one attention layer is mispriced by at most
      ~0.15 ms (5090) / ~0.45 ms (3080) per chunk by it.

DEVICE TIME ONLY. A forward the HOST paced (launch span >= 90 % of its gpu_fwd
AND the card idle > 1 ms before it -- eager PP0 below ~10k prefix, ~57 ms of
launch for 42 layers) is excluded from the fit and counted in the provenance: the model prices what the
cards do, which is what the prefill graph (shallow) and the eager deep split
(device-bound by construction) leave.

PIECEWISE when the fitted boot ran the deep split (its rank lines
``FI-WAVE-SPLIT ... from_prefix=N`` / ``PREFILL-GRAPH eager reason=deep_split``):
the shallow and the deep side are fitted separately at N, because the deep
side's attention slope is the split's, not the graph's.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: A rank needs this many device-bound full chunks for its line.
MIN_SAMPLES = 32
#: ... spread over at least this many 1k-tokens of prefix, or the slope is
#: not determined.
MIN_SPREAD_1K = 4.0
#: A forward was PACED BY THE HOST -- and is excluded -- when its host launch
#: span is at least this share of its gpu_fwd AND the card idled more than
#: HOST_PACED_GAP_MS before it. Both halves are needed: a host BLOCKED on a
#: full launch queue also shows launch ~ gpu_fwd (xsn426 PP0 at depth: launch
#: 147 vs gpu_fwd 158 ms), but then the card never idles (gap 0.2-0.3 ms) and
#: the forward is device time; a host-paced one leaves gaps (PP0 below 8k: gap
#: 5-9 ms at launch 57 >= gpu_fwd 55 ms).
HOST_BOUND_RATIO = 0.9
HOST_PACED_GAP_MS = 1.0
#: ``auto`` scans at most this many of the newest P logs.
AUTO_SCAN_LOGS = 8

_PGAP_RE = re.compile(
    r"\b(?P<rank>PP\d+)\] #PGAP pp_rank=\S+ fwd=(?P<fwd>-?\d+) tokens=(?P<tok>\S+) "
    r"gpu_gap_ms=(?P<gap>\S+) gpu_fwd_ms=(?P<gpu>[0-9.]+) host\[(?P<host>[^\]]*)\]"
)
_ADMIT_RE = re.compile(
    r"\b(?P<rank>PP\d+)\] #969N ADMIT slot=\S+ fwd_ct=(?P<fct>-?\d+) bs=(?P<bs>\d+) "
    r"extend=(?P<ext>\S+) input_ids=\S+ rids=\[(?P<rids>[^\]]*)\]"
)
_CACHED_RE = re.compile(r"\b(?P<rank>PP\d+)\] Prefill batch,.*?#cached-token: (?P<c>\d+)")
_CHUNK_RE = re.compile(r"server_args=ServerArgs\(.*?\bchunked_prefill_size=(-?\d+)")
_STAGE_RE = re.compile(r"server_args=ServerArgs\(.*?\bpp_stage_ratio=\[([^\]]*)\]")
_ATTN_RE = re.compile(r"server_args=ServerArgs\(.*?\bpp_attn_stage_ratio=\[([^\]]*)\]")
_SPLIT_FROM_RE = re.compile(r"FI-WAVE-SPLIT .*?\bfrom_prefix=(\d+)")


class StageFitRefused(ValueError):
    """The log cannot carry a stage fit; the text says which input is missing."""


@dataclasses.dataclass(frozen=True)
class PgapSample:
    prefix: int
    gpu_ms: float
    launch_ms: float
    gap_ms: float = 0.0

    @property
    def host_paced(self) -> bool:
        return (
            self.launch_ms >= HOST_BOUND_RATIO * self.gpu_ms
            and self.gap_ms > HOST_PACED_GAP_MS
        )


@dataclasses.dataclass(frozen=True)
class PgapLog:
    path: str
    chunk_tokens: int
    counts: Tuple[int, ...]
    attn: Tuple[int, ...]
    samples: Tuple[Tuple[PgapSample, ...], ...]
    joined: int
    unjoined: int
    cached_lines: Tuple[int, ...]
    split_from_prefix: int


def _ints(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[,\s]+", text.strip()) if x)


def read_pgap_log(path: str) -> PgapLog:
    """One P log's full-chunk (prefix, gpu_fwd, launch) samples per rank.

    THE PREFIX IS RECONSTRUCTED, as ``launcher.read_mean_prefill_prefix`` does:
    group P admits one chunked request per pass, so a request is a run of
    ``#969N ADMIT`` passes on one rank, and the prefix at pass j is the sum of
    the run's earlier ``extend``. A run ends at a chunk shorter than the full
    chunk (the prompt's last chunk, the 1-token end anchor) or at a change of
    the (8-character) rid. ``#PGAP fwd`` is the ADMIT ``fwd_ct`` + 1 (the
    scheduler counts the forward inside run_batch); a pair is used only when
    its token counts agree. A prefix-cache hit at a run's first chunk is not
    added (no line carries it per pass) -- the count of ``#cached-token > 0``
    prefill lines is returned so a reader can see whether that matters.
    """
    chunk = None
    counts: Tuple[int, ...] = ()
    attn: Tuple[int, ...] = ()
    split_from = 0
    state: Dict[str, Tuple[str, int, bool]] = {}
    admitted: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
    pgap: List[Tuple[str, int, str, float, float, float]] = []
    cached: Dict[str, int] = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if chunk is None and "server_args=ServerArgs(" in line:
                m = _CHUNK_RE.search(line)
                if m:
                    chunk = int(m.group(1))
                ms, ma = _STAGE_RE.search(line), _ATTN_RE.search(line)
                if ms:
                    counts = _ints(ms.group(1))
                if ma:
                    attn = _ints(ma.group(1))
                continue
            if "#969N ADMIT" in line:
                m = _ADMIT_RE.search(line)
                if m is None or not m.group("ext").isdigit():
                    continue
                rank, ext, rid = m.group("rank"), int(m.group("ext")), m.group("rids")
                prev_rid, acc, prev_short = state.get(rank, ("", 0, True))
                if rid != prev_rid or prev_short:
                    acc = 0
                full = chunk if chunk else ext
                admitted[(rank, int(m.group("fct")) + 1)] = (acc, ext, int(m.group("bs")))
                state[rank] = (rid, acc + ext, ext < full)
            elif "#PGAP pp_rank" in line:
                m = _PGAP_RE.search(line)
                if m is None:
                    continue
                host = dict(
                    kv.split("=", 1) for kv in m.group("host").split() if "=" in kv
                )
                try:
                    launch = float(host.get("launch", "0") or 0.0)
                except ValueError:
                    launch = 0.0
                try:
                    gap = float(m.group("gap"))
                except ValueError:
                    # "-": the rank's FIRST forward (no previous end) -- the
                    # boot's warm-up pass (JIT, first touch; xsn426: 1700 /
                    # 1005 / 893 ms). Treated as an idle card, so a host-paced
                    # first pass is excluded like any other.
                    gap = float("inf")
                pgap.append(
                    (m.group("rank"), int(m.group("fwd")), m.group("tok"),
                     float(m.group("gpu")), launch, gap)
                )
            elif "Prefill batch," in line and "#cached-token" in line:
                m = _CACHED_RE.search(line)
                if m is not None and int(m.group("c")) > 0:
                    cached[m.group("rank")] = cached.get(m.group("rank"), 0) + 1
            elif "FI-WAVE-SPLIT" in line and not split_from:
                m = _SPLIT_FROM_RE.search(line)
                if m is not None:
                    split_from = int(m.group(1))
            elif "reason=deep_split" in line and not split_from:
                split_from = -1  # seen, threshold unknown until a FI line names it
    if not chunk or chunk <= 0:
        raise StageFitRefused(f"{path}: no server_args chunked_prefill_size")
    if not counts or len(counts) != len(attn):
        raise StageFitRefused(
            f"{path}: server_args carries no pp_stage_ratio / pp_attn_stage_ratio pair"
        )
    ranks = ["PP%d" % r for r in range(len(counts))]
    per: Dict[str, List[PgapSample]] = {r: [] for r in ranks}
    joined = unjoined = 0
    for rank, fwd, tok, gpu, launch, gap in pgap:
        info = admitted.get((rank, fwd))
        if info is None or rank not in per:
            unjoined += 1
            continue
        prefix, ext, bs = info
        if not tok.isdigit() or int(tok) != ext:
            unjoined += 1
            continue
        joined += 1
        if bs == 1 and ext == chunk:
            per[rank].append(PgapSample(prefix, gpu, launch, gap))
    return PgapLog(
        path=path,
        chunk_tokens=int(chunk),
        counts=tuple(counts),
        attn=tuple(attn),
        samples=tuple(tuple(per[r]) for r in ranks),
        joined=joined,
        unjoined=unjoined,
        cached_lines=tuple(cached.get(r, 0) for r in ranks),
        split_from_prefix=max(0, split_from),
    )


#: Robust refit: a residual beyond max(TRIM_FLOOR_MS, TRIM_MADS x 1.4826 x
#: MAD) is a stall, not the line (a JIT compile inside a launch, a flip
#: seam), and is dropped before the refit -- counted, never silently.
TRIM_FLOOR_MS = 5.0
TRIM_MADS = 5.0


def _lsq(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sd = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, sd


def _robust_lsq(
    xs: Sequence[float], ys: Sequence[float], passes: int = 2
) -> Tuple[float, float, float, int]:
    """Least squares, then up to ``passes`` refits without the stalls."""
    keep = list(range(len(xs)))
    a, b, sd = _lsq(xs, ys)
    for _ in range(int(passes)):
        res = sorted(abs(ys[i] - a - b * xs[i]) for i in keep)
        mad = res[len(res) // 2]
        cut = max(TRIM_FLOOR_MS, TRIM_MADS * 1.4826 * mad)
        nxt = [i for i in keep if abs(ys[i] - a - b * xs[i]) <= cut]
        if len(nxt) == len(keep) or len(nxt) < 3:
            break
        keep = nxt
        a, b, sd = _lsq([xs[i] for i in keep], [ys[i] for i in keep])
    return a, b, sd, len(xs) - len(keep)


@dataclasses.dataclass(frozen=True)
class RankLine:
    """One rank's fitted line over one prefix segment (per chunk)."""

    a_ms: float
    b_ms_per_1k: float
    sd_ms: float
    n: int
    host_bound: int
    prefix_lo: int
    prefix_hi: int
    trimmed: int = 0


def fit_rank_lines(
    log: PgapLog,
    *,
    lo: int = 0,
    hi: Optional[int] = None,
    min_samples: int = MIN_SAMPLES,
) -> Tuple[RankLine, ...]:
    """``a + b * (prefix + C/2) / 1000`` per rank over prefixes in [lo, hi)."""
    half = 0.5 * float(log.chunk_tokens)
    out: List[RankLine] = []
    for r, samples in enumerate(log.samples):
        inside = [s for s in samples if s.prefix >= lo and (hi is None or s.prefix < hi)]
        dev = [s for s in inside if not s.host_paced]
        if len(dev) < int(min_samples):
            raise StageFitRefused(
                f"{log.path}: rank PP{r} has {len(dev)} device-bound full "
                f"{log.chunk_tokens}-token chunks with prefix in [{lo}, "
                f"{'inf' if hi is None else hi}) ({len(inside) - len(dev)} host-paced "
                f"excluded); the fit needs {int(min_samples)}"
            )
        xs = [(s.prefix + half) / 1000.0 for s in dev]
        if max(xs) - min(xs) < MIN_SPREAD_1K:
            raise StageFitRefused(
                f"{log.path}: rank PP{r}'s chunks span only "
                f"{(max(xs) - min(xs)) * 1000:.0f} prefix tokens; the attention "
                f"slope needs {MIN_SPREAD_1K * 1000:.0f}"
            )
        a, b, sd, trimmed = _robust_lsq(xs, [s.gpu_ms for s in dev])
        if a <= 0.0 or b < 0.0:
            raise StageFitRefused(
                f"{log.path}: rank PP{r} fits a={a:.2f} ms b={b:.4f} ms/1k -- a "
                "non-positive intercept or a negative slope is not a stage cost"
            )
        out.append(RankLine(a, b, sd, len(dev) - trimmed, len(inside) - len(dev), int(lo),
                            -1 if hi is None else int(hi), trimmed))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class DepthLinearStageCost:
    """Per-stage prefill cost of ONE chunk, linear in the prefix.

    Duck-compatible with ``pp_cut.FamilyDepthCost`` where the solver reads it
    (``stage_ms(counts, attn_counts, prefix_tokens)`` and
    ``ref_prefix_tokens``), so ``pp_cut_launch.solve_launch_cut`` prices every
    candidate with it unchanged.
    """

    layer_ms: Tuple[float, ...]
    attn_ms_per_1k: Tuple[float, ...]
    stage_fixed_ms: Tuple[float, ...]
    chunk_tokens: int
    ref_prefix_tokens: float
    deep_from_prefix: int = 0
    deep_layer_ms: Tuple[float, ...] = ()
    deep_attn_ms_per_1k: Tuple[float, ...] = ()
    deep_stage_fixed_ms: Tuple[float, ...] = ()

    def stage_ms(
        self,
        counts: Sequence[int],
        attn_counts: Sequence[int],
        prefix_tokens: float,
    ) -> Tuple[float, ...]:
        if not len(counts) == len(attn_counts) == len(self.layer_ms):
            raise ValueError(
                f"counts {tuple(counts)}, attn {tuple(attn_counts)} and the "
                f"{len(self.layer_ms)}-stage fitted cost disagree on the number "
                "of stages."
            )
        deep = (
            self.deep_from_prefix > 0
            and float(prefix_tokens) >= float(self.deep_from_prefix)
            and bool(self.deep_layer_ms)
        )
        lm = self.deep_layer_ms if deep else self.layer_ms
        am = self.deep_attn_ms_per_1k if deep else self.attn_ms_per_1k
        fx = self.deep_stage_fixed_ms if deep else self.stage_fixed_ms
        de = (float(prefix_tokens) + 0.5 * float(self.chunk_tokens)) / 1000.0
        out: List[float] = []
        for r, (n, a) in enumerate(zip(counts, attn_counts)):
            if int(a) > int(n):
                raise ValueError(
                    f"stage {r} holds {int(n)} layers of which {int(a)} are full "
                    "attention; a stage cannot hold more attention layers than layers."
                )
            out.append(int(n) * lm[r] + fx[r] + int(a) * am[r] * de)
        return tuple(out)


def _stage_rates(
    lines: Sequence[RankLine],
    counts: Sequence[int],
    attn: Sequence[int],
    card_names: Sequence[str],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], List[str]]:
    last = len(counts) - 1
    notes: List[str] = []
    layer = [0.0] * len(counts)
    fixed = [0.0] * len(counts)
    slope = [0.0] * len(counts)
    for r, (ln, n, a) in enumerate(zip(lines, counts, attn)):
        if int(a) <= 0:
            raise StageFitRefused(
                f"the fitted cut holds no attention layer on stage {r}, so its "
                "attention slope per layer is not measured"
            )
        slope[r] = ln.b_ms_per_1k / float(a)
    for name in dict.fromkeys(card_names):
        stages = [r for r, c in enumerate(card_names) if c == name]
        middle = [r for r in stages if 0 < r < last]
        if middle:
            m = middle[0]
            rate = lines[m].a_ms / float(counts[m])
            for r in stages:
                rem = lines[r].a_ms - rate * float(counts[r])
                if rem < 0.0:
                    layer[r] = lines[r].a_ms / float(counts[r])
                    notes.append(
                        f"stage {r} ({name}) is cheaper per layer than middle stage "
                        f"{m}; it keeps its own rate {layer[r]:.3f}"
                    )
                else:
                    layer[r], fixed[r] = rate, rem
        else:
            for r in stages:
                layer[r] = lines[r].a_ms / float(counts[r])
    return tuple(layer), tuple(slope), tuple(fixed), notes


def fit_stage_cost(
    log: PgapLog,
    card_names: Sequence[str],
    *,
    split_from_prefix: int = 0,
    min_samples: int = MIN_SAMPLES,
) -> Tuple[DepthLinearStageCost, str]:
    """The fitted :class:`DepthLinearStageCost` and its one provenance line.

    ``card_names`` are THIS boot's cards in stage order; the fitted boot's
    stage r is taken to have run on this boot's stage-r card (the launcher
    orders cards the same way every boot: the 5090 first, then the 3080s by
    NVML index). ``split_from_prefix`` > 0 fits two segments at that prefix
    (the deep split); it defaults to the threshold the log's own rank lines
    name.
    """
    if len(card_names) != len(log.counts):
        raise StageFitRefused(
            f"{log.path} ran {len(log.counts)} P stages, this boot has "
            f"{len(card_names)}"
        )
    thr = int(split_from_prefix or log.split_from_prefix or 0)
    half = 0.5 * float(log.chunk_tokens)
    if thr > 0:
        shallow = fit_rank_lines(log, lo=0, hi=thr, min_samples=min_samples)
        deep = fit_rank_lines(log, lo=thr, hi=None, min_samples=min_samples)
    else:
        shallow = fit_rank_lines(log, min_samples=min_samples)
        deep = ()
    lm, am, fx, notes = _stage_rates(shallow, log.counts, log.attn, card_names)
    dlm = dam = dfx = ()
    if deep:
        dlm, dam, dfx, dnotes = _stage_rates(deep, log.counts, log.attn, card_names)
        notes += ["deep: " + x for x in dnotes]
    all_prefix = [s.prefix for s in log.samples[0]]
    ref = (sum(all_prefix) / len(all_prefix)) if all_prefix else 0.0
    cost = DepthLinearStageCost(
        layer_ms=lm,
        attn_ms_per_1k=am,
        stage_fixed_ms=fx,
        chunk_tokens=int(log.chunk_tokens),
        ref_prefix_tokens=float(ref),
        deep_from_prefix=thr if deep else 0,
        deep_layer_ms=dlm,
        deep_attn_ms_per_1k=dam,
        deep_stage_fixed_ms=dfx,
    )

    def _seg(lines: Sequence[RankLine]) -> str:
        return "; ".join(
            "PP%d a=%.2f b=%.4f sd=%.2f n=%d host-paced=%d stalls=%d"
            % (r, x.a_ms, x.b_ms_per_1k, x.sd_ms, x.n, x.host_bound, x.trimmed)
            for r, x in enumerate(lines)
        )

    prov = (
        "STAGE FIT (--pp-cut-stage-fit) from %s: chunk %d, fitted cut %s / attn %s, "
        "%d #PGAP forwards joined to their prefix (%d unjoined), prefix-cache-hit "
        "lines %s; per rank gpu_fwd = a + b*(prefix+%d)/1000 over DEVICE-bound full "
        "chunks: %s%s -> per stage layer_ms %s, attn ms/1k/layer %s, fixed ms/forward "
        "%s%s; cards %s; ASSUMPTION an attention layer's prefix-independent cost = a "
        "GDN layer's on the same card%s"
        % (
            os.path.basename(log.path),
            log.chunk_tokens,
            ",".join(map(str, log.counts)),
            ",".join(map(str, log.attn)),
            log.joined,
            log.unjoined,
            ",".join(map(str, log.cached_lines)),
            int(half),
            _seg(shallow) if not deep else "prefix<%d: %s" % (thr, _seg(shallow)),
            "" if not deep else " | prefix>=%d: %s" % (thr, _seg(deep)),
            ",".join("%.3f" % x for x in lm),
            ",".join("%.4f" % x for x in am),
            ",".join("%.2f" % x for x in fx),
            ""
            if not deep
            else " | deep layer_ms %s attn %s fixed %s"
            % (
                ",".join("%.3f" % x for x in dlm),
                ",".join("%.4f" % x for x in dam),
                ",".join("%.2f" % x for x in dfx),
            ),
            ",".join(card_names),
            ("; " + "; ".join(notes)) if notes else "",
        )
    )
    return cost, prov


def newest_pgap_log(
    evidence_dir: str,
    chunk_tokens: int,
    stages: int,
    accept: Optional[Callable[[str], bool]] = None,
    min_samples: int = MIN_SAMPLES,
) -> Tuple[Optional[str], List[str]]:
    """Newest ``*.P.log`` whose #PGAP lines carry a fit at ``chunk_tokens``.

    Returns the path (None when none qualifies) and one line per log looked
    at, so a skipped log is named with its reason rather than passed silently.
    """
    seen: List[str] = []
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".P.log")]
    except OSError as exc:
        return None, ["%s: %s" % (evidence_dir, exc)]
    paths = sorted(
        (os.path.join(evidence_dir, n) for n in names),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for path in paths[: AUTO_SCAN_LOGS * 4]:
        if accept is not None and not accept(path):
            continue
        if len(seen) >= AUTO_SCAN_LOGS:
            break
        try:
            log = read_pgap_log(path)
            if log.chunk_tokens != int(chunk_tokens):
                raise StageFitRefused("chunk %d, this boot %d" % (log.chunk_tokens, int(chunk_tokens)))
            if len(log.counts) != int(stages):
                raise StageFitRefused("%d stages, this boot %d" % (len(log.counts), int(stages)))
            fit_rank_lines(log, min_samples=min_samples)
        except (StageFitRefused, OSError, ValueError, ZeroDivisionError) as exc:
            seen.append("%s: skipped (%s)" % (os.path.basename(path), exc))
            continue
        seen.append("%s: taken" % os.path.basename(path))
        return path, seen
    return None, seen


def depth_profile(
    spec: str,
    chunk_tokens: int,
    fitted_log: Optional[PgapLog] = None,
) -> Tuple[Tuple[Tuple[float, float], ...], str]:
    """``(prefix, weight)`` points the solver averages its makespan over.

    ``ladder:2048,8192,32768`` -- every chunk start of each prompt length,
    each RUNG weighted equally (a rung of 64 chunks does not outvote one of 4),
    i.e. the mean over rungs of the mean chunk time: the P-ladder the boots
    report. ``fit`` -- the fitted log's own full chunks, equally weighted.
    """
    text = str(spec or "").strip()
    c = int(chunk_tokens)
    if c <= 0:
        raise ValueError("chunk_tokens must be positive")
    if text.startswith("ladder:"):
        rungs = [int(x) for x in text[len("ladder:"):].split(",") if x.strip()]
        if not rungs or min(rungs) <= 0:
            raise ValueError(f"--pp-cut-depth-profile {text!r}: positive prompt lengths expected")
        pts: List[Tuple[float, float]] = []
        for p in rungs:
            starts = list(range(0, p, c))
            for d in starts:
                pts.append((float(d), 1.0 / (len(rungs) * len(starts))))
        return tuple(pts), "ladder %s at chunk %d (%d chunk starts, rungs weighted equally)" % (
            ",".join(map(str, rungs)), c, len(pts))
    if text == "fit":
        if fitted_log is None or not fitted_log.samples or not fitted_log.samples[0]:
            raise ValueError("--pp-cut-depth-profile fit needs --pp-cut-stage-fit")
        prefixes = [s.prefix for s in fitted_log.samples[0]]
        w = 1.0 / len(prefixes)
        return tuple((float(d), w) for d in prefixes), "the fitted log's %d PP0 full chunks (mean prefix %.0f)" % (
            len(prefixes), sum(prefixes) / len(prefixes))
    raise ValueError(
        f"--pp-cut-depth-profile {text!r}: expected 'ladder:N,N,...' or 'fit'"
    )
