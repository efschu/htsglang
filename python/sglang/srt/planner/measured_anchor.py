"""Anchor the parse-time capacity model to the last boot of the SAME FORM.

WHY THIS EXISTS (#48, the open half of #62)
-------------------------------------------
``PerfCostModel`` has two arithmetics.  Without a measurement it prices every
post from the model config; with one (``measured`` + ``measured_mlp_vector``)
it prices every NON-WEIGHT post from the previous boot of this configuration
and anchors the weight term additively (``measured_weight_bias``), so the
family model's DELTAS between candidate vectors stay model-driven while its
absolute error disappears.

The measured arithmetic was reachable only through
``uneven_perf.load_measured_registry``, which reads a JSON file that the
runtime writes ONLY under ``SGLANG_MEASURED_KV_BUDGET``.  On the weg2 flip
form that env is not set, so the file is never written -- measured on this
rig 2026-09-21: ``~/.cache/sglang`` holds 156 ``kv_budget-*-seam-rank*.json``
records and not one current ``kv_budget-<digest>.json``.  The weg2 launcher
therefore built the cost model with neither argument and planned on the
unanchored family model, which reads ~2x the measured census of the same form
(fnFL2v72 ``[vram-census]``: 11.78 / 11.06 / 10.49 GiB of model tensors per
rank against the model's 24583 / 22164 / 21852 MiB).

This module reads the posts out of the boot LOG instead, because that is the
measurement this form actually takes.  It produces components in exactly the
``load_measured_registry`` schema, so nothing downstream learns a new shape.

THE ONE IDENTITY THAT MAKES THIS EXACT
--------------------------------------
``predict_capacity``'s measured branch computes, per rank,

    free_r = device_total/ranks_on_gpu
           - residual_residency_r
           - (model_weights_r(vec) + bias_r)
           - mamba_aux_pool_r
           - required_free_r

and the boot log states ``free_r`` itself, exactly and per rank, as the
``KV pool sizing: available_bytes=`` integer.  So ``residual_residency`` is
not parsed -- it is SOLVED for, as the residue that makes the identity hold
at the vector the measurement was taken under:

    residual_r := total_share_r - available_bytes_r
                - weights_alloc_r - mamba_aux_r - required_free_r

Two consequences, both deliberate.  (1) The anchor reproduces the measured
boot's own KV budget to the byte at the measured vector -- the 2-decimal GiB
rounding of ``weights_alloc`` and ``mamba_aux`` cancels, because the same
rounded value is subtracted in the derivation and added back by the bias.
(2) Away from the measured vector only the WEIGHT DELTA moves, which is what
the family model is good at and the only thing it is trusted with here.

WHAT THIS MODULE REFUSES TO DO
------------------------------
Every post below is either read from the log, or handed in by the caller as
configuration the caller itself owns.  Nothing is defaulted.  In particular
``required_free_bytes`` is NOT parsed: no instrument in a weg2 boot log states
it.  The caller passes the reserve it is itself putting on the command line
(``--rank-user-reserve-mib``), and the provenance line says so.  The runtime's
own ``required_free`` is ``safety + max_paused_rung_tag``; the second half is
a serving-time quantity that is not visible in a boot log at all, and it is
named as open in the provenance rather than silently taken as zero.

A missing post is a NAMED refusal, never a zero that looks like a
measurement.  The caller falls back to the heuristic path, which stays
byte-identical.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

GIB = 1 << 30
MIB = 1 << 20

#: The posts ``uneven_perf.load_measured_registry`` demands of every
#: component.  Kept as a literal tuple rather than imported so that a change
#: on either side shows up as a test failure instead of a silent drift.
REQUIRED_POSTS: Tuple[str, ...] = (
    "device_total_bytes",
    "ranks_on_gpu",
    "residual_residency_bytes",
    "weights_alloc_bytes",
    "required_free_bytes",
    "mamba_aux_pool_bytes",
)


class MeasuredAnchorRefused(Exception):
    """A named refusal: which rank, which post, which log.

    Raised rather than returned so that no caller can mistake a partial
    anchor for a complete one.  The weg2 launcher catches it, prints it, and
    keeps the heuristic path.
    """


@dataclasses.dataclass(frozen=True)
class MeasuredAnchor:
    """Components in the ``load_measured_registry`` schema, plus provenance."""

    components: Tuple[dict, ...]
    mlp_vector: Tuple[int, ...]
    source_path: str
    boot_tag: str
    age_s: float
    #: One line for the boot log: which source won, which boot, how old, and
    #: which post came from configuration rather than from a measurement.
    provenance: str

    @property
    def uniform_kv_cell(self) -> bool:
        """Did every rank size its pool at the SAME KV cell?

        Only then does an anchored per-rank ``available_bytes`` mean the same
        thing on every rank, and only then may the capacity model's derived
        TOKEN VECTOR be read back and shipped.  On a form whose token axis is
        pinned to one rank (``--rank-tp-ratio 1,0,0``) the cells differ by
        20x and the answer is False -- see ``cell_size_bytes``.
        """
        return anchor_has_uniform_kv_cell(self.components)


# --- the line shapes, all of them already emitted by every weg2 boot -------
#
# The logger prefix carries the rank ordinal ("[... TP1] ..."); the census tag
# carries the full pp/tp position.  Which of the two is THE ordinal is a
# property of the group geometry (group D is pp=1,tp=3 so it is tp; group P is
# pp=3,tp=1 so it is pp), so the axis is handed in, never wired in here.
_RE_PREFIX = re.compile(r"\[\d[\d\-]* [\d:]+ (TP|PP)(\d+)[^\]]*\]")
_RE_CENSUS_LOAD = re.compile(
    r"\[vram-census\] pp(\d+)tp(\d+) after load: "
    r".*?torch allocated ([\d.]+) GiB"
)
_RE_IDLE = re.compile(
    r"\[vram-idle\] after pools: card free ([\d.]+) of ([\d.]+) GiB, "
    r"allocated ([\d.]+), reserved ([\d.]+)"
)
_RE_SIZING = re.compile(
    r"KV pool sizing: available_bytes=(\d+) .*?cell_size=(\d+), "
    r"page_size=(\d+) -> max_total_num_tokens=(\d+)"
)
#: The mamba/SSM pool, from the instrument that states it.  Derived instead
#: from the allocator difference ``allocated_after_pools - allocated_after
#: _load`` it is not merely imprecise but SIGN-WRONG on a rank that funds no
#: token share: measured on fnFL2v72 rank 1 the after-pools reading (11.39
#: GiB) is BELOW the after-load one (11.46 GiB), so the difference clamps to
#: zero and a clamp of that shape hides the defect instead of naming it.
#: ``slots x per_req`` is the pool's own arithmetic and reads 0.00 MiB per
#: request on exactly the ranks that hold no state.
_RE_MAMBA = re.compile(
    r"\[auto-mamba\].*?max_mamba_cache_size=(\d+) slots "
    r"\([\d.]+ GB @ per_req=([\d.]+) MiB"
)
_RE_BUDGETS = re.compile(r"--rank-gpu-memory-mib[= ]([\d,]+)")
_RE_MODEL = re.compile(r"model_path='([^']*)'")


def anchor_has_uniform_kv_cell(components: Sequence[dict]) -> bool:
    """True when every component names the same measured KV cell.

    Written as a free function so the weg2 launcher can ask it of a
    ``PerfCostModel.measured`` list directly, without having to keep the
    anchor object alive next to the model.  A component that predates the
    ``cell_size_bytes`` post answers False: an unknown cell is not a matching
    cell, and the conservative answer keeps the token vector where it is.
    """
    cells = {c.get("cell_size_bytes") for c in components}
    return len(cells) == 1 and None not in cells


def _slot(fields: Dict[int, Dict[str, float]], rank: int) -> Dict[str, float]:
    return fields.setdefault(int(rank), {})


def parse_boot_log_posts(
    text: str, *, rank_axis: str
) -> Dict[int, Dict[str, float]]:
    """The raw per-rank posts a weg2 boot log states, and nothing else.

    ``rank_axis`` is ``"tp"`` or ``"pp"`` -- which ordinal of the census tag
    ``pp<N>tp<M>`` is THE rank of this group.  Derived by the caller from the
    group geometry, never assumed here.

    ``-draft`` census lines are skipped: the draft runner's allocator reading
    is not this rank's weight checkpoint.  Where a rank emits several
    ``[vram-idle]`` samples (the target runner and then the draft runner both
    report after their pools exist), the one with the LOWEST card-free is
    kept: that is the most complete resident state the log witnesses, and
    taking the roomier sample would fund KV that the boot had already spent.
    """
    if rank_axis not in ("tp", "pp"):
        raise ValueError("rank_axis must be 'tp' or 'pp', got %r" % (rank_axis,))
    fields: Dict[int, Dict[str, float]] = {}
    for line in text.splitlines():
        m = _RE_CENSUS_LOAD.search(line)
        if m:
            if "-draft" in line:
                continue
            rank = int(m.group(1) if rank_axis == "pp" else m.group(2))
            _slot(fields, rank)["weights_alloc_gib"] = float(m.group(3))
            continue
        pref = _RE_PREFIX.search(line)
        rank = int(pref.group(2)) if pref else None
        if rank is None:
            continue
        m = _RE_IDLE.search(line)
        if m:
            s = _slot(fields, rank)
            free = float(m.group(1))
            if free < s.get("card_free_gib", float("inf")):
                s["card_free_gib"] = free
                s["pools_alloc_gib"] = float(m.group(3))
            s["card_total_gib"] = float(m.group(2))
            continue
        m = _RE_SIZING.search(line)
        if m:
            s = _slot(fields, rank)
            # A rank sizes several pools (main, draft/MTP, index). The MAIN
            # one is the one with the largest cell, never the largest token
            # count -- on this form the 768-byte cell belongs to a rank that
            # owns no token share while the 14143-byte cell is the real one.
            if float(m.group(2)) >= s.get("cell_size", -1.0):
                s["cell_size"] = float(m.group(2))
                s["available_bytes"] = float(m.group(1))
                s["max_total_num_tokens"] = float(m.group(4))
            continue
        m = _RE_MAMBA.search(line)
        if m:
            _slot(fields, rank)["mamba_pool_bytes"] = float(
                int(m.group(1)) * float(m.group(2)) * MIB
            )
            continue
    return fields


def build_components(
    posts: Dict[int, Dict[str, float]],
    *,
    tp_size: int,
    ranks_on_gpu: Sequence[int],
    required_free_bytes: Sequence[int],
    source: str = "<log>",
) -> Tuple[dict, ...]:
    """Turn the raw posts into ``load_measured_registry``-shaped components.

    ``ranks_on_gpu`` and ``required_free_bytes`` are the caller's own
    configuration (the rank->card map it is building, and the user reserve it
    is putting on the command line).  Both are per rank and neither has a
    default: this function will not invent either.
    """
    if len(ranks_on_gpu) != tp_size:
        raise MeasuredAnchorRefused(
            "ranks_on_gpu has %d entries for a %d-rank group -- the rank->card "
            "map is the caller's own and must cover every rank"
            % (len(ranks_on_gpu), tp_size)
        )
    if len(required_free_bytes) != tp_size:
        raise MeasuredAnchorRefused(
            "required_free_bytes has %d entries for a %d-rank group -- the "
            "configured reserve must be stated per rank"
            % (len(required_free_bytes), tp_size)
        )

    missing: List[str] = []
    for rank in range(tp_size):
        s = posts.get(rank, {})
        for key, instrument in (
            ("weights_alloc_gib", "[vram-census] ... after load: torch allocated"),
            ("card_total_gib", "[vram-idle] after pools: ... of <total> GiB"),
            ("card_free_gib", "[vram-idle] after pools: card free"),
            ("mamba_pool_bytes", "[auto-mamba] ... slots (... @ per_req=)"),
            ("available_bytes", "KV pool sizing: available_bytes="),
        ):
            if key not in s:
                missing.append("rank%d:%s (from `%s`)" % (rank, key, instrument))
    if missing:
        raise MeasuredAnchorRefused(
            "the boot log %s states %d of the posts this anchor needs and is "
            "missing: %s. A partial anchor is not an anchor -- the heuristic "
            "path stands rather than defaulting the gaps to zero."
            % (source, tp_size * 5 - len(missing), ", ".join(missing))
        )

    out: List[dict] = []
    for rank in range(tp_size):
        s = posts[rank]
        on_gpu = max(int(ranks_on_gpu[rank]), 1)
        device_total = int(round(s["card_total_gib"] * GIB))
        total_share = device_total // on_gpu
        weights_alloc = int(round(s["weights_alloc_gib"] * GIB))
        kv_pool = int(s["available_bytes"])
        # The mamba/SSM pool as its own instrument states it. This is the post
        # ``_measured_mamba_aux_bytes`` re-scales when a candidate moves the
        # GDN units, so it has to BE the mamba pool -- every other non-KV pool
        # is fixed across candidates and belongs in the residue below.
        mamba_aux = int(round(s["mamba_pool_bytes"]))
        req_free = int(required_free_bytes[rank])
        # SOLVED, not parsed -- see the module docstring. This is the residue
        # that makes predict_capacity reproduce this boot's own
        # ``available_bytes`` at the vector it was measured under.
        residual = total_share - kv_pool - weights_alloc - mamba_aux - req_free
        if residual < 0:
            raise MeasuredAnchorRefused(
                "rank%d: the posts do not balance -- card share %.2f GiB minus "
                "KV %.2f minus weights %.2f minus mamba/aux %.2f minus the "
                "configured reserve %.2f leaves %.2f GiB, i.e. the log "
                "describes a card holding more than it has. Refused rather "
                "than clamped: a negative residue means one of these posts "
                "belongs to a different rank or a different boot (%s)."
                % (
                    rank,
                    total_share / GIB,
                    kv_pool / GIB,
                    weights_alloc / GIB,
                    mamba_aux / GIB,
                    req_free / GIB,
                    residual / GIB,
                    source,
                )
            )
        comp = {
            "device_total_bytes": device_total,
            "ranks_on_gpu": on_gpu,
            "residual_residency_bytes": int(residual),
            "weights_alloc_bytes": weights_alloc,
            "required_free_bytes": req_free,
            "mamba_aux_pool_bytes": int(mamba_aux),
            "kv_pool_bytes": kv_pool,
            "kv_pool_tokens": int(s["max_total_num_tokens"]),
            "max_total_num_tokens": int(s["max_total_num_tokens"]),
            # THE CELL THIS RANK SIZED ITS OWN POOL AT, and the reason the
            # token vector must not be read back off this anchor on every
            # form. A rank that owns no attention share sizes its pool at a
            # placeholder cell (measured fnFL2v72: 14143 B on rank 0 against
            # 768 B on ranks 1 and 2, which own no attention under
            # ``--rank-tp-ratio 1,0,0``), so its ``available_bytes`` is NOT
            # "bytes fundable for group KV" -- re-priced at the group cell it
            # reads as several hundred thousand fundable tokens on a rank that
            # can hold none. Carried per rank so the caller can see the
            # disagreement instead of averaging it away.
            "cell_size_bytes": int(s["cell_size"]),
            # Provenance travels WITH the component: a later reader must be
            # able to tell a log-derived balance from one the runtime wrote
            # itself. PerfCostModel ignores keys it does not know.
            "anchor_source": source,
            "anchor_kind": "boot-log",
        }
        out.append(comp)
    return tuple(out)


def _boot_tag(path: str) -> str:
    base = os.path.basename(path)
    parts = base.split("_")
    return parts[2] if len(parts) > 2 else base


def find_boot_logs(evidence_dirs: Sequence[str], group: str) -> List[str]:
    """Every boot log of this group, newest first (by mtime)."""
    found: List[str] = []
    for d in evidence_dirs:
        found.extend(glob.glob(os.path.join(d, "boot_*.%s.log" % group)))
    return sorted(set(found), key=lambda p: os.path.getmtime(p), reverse=True)


def read_measured_anchor(
    *,
    group: str,
    tp_size: int,
    rank_axis: str,
    ranks_on_gpu: Sequence[int],
    required_free_bytes: Sequence[int],
    mlp_vector: Sequence[int],
    budgets_mib: Sequence[int],
    model_path: Optional[str] = None,
    evidence_dirs: Sequence[str] = (),
    now: Optional[float] = None,
) -> MeasuredAnchor:
    """The newest boot log of the same FORM, as measured components.

    Same form means, and each of the three is checked rather than assumed:

    * same GROUP -- the file suffix (``.D.log`` / ``.P.log``); P and D have
      different geometries and different weight footprints;
    * same MODEL -- ``model_path='...'`` in the log's own argv echo must name
      ``model_path``;
    * same BUDGETS -- ``--rank-gpu-memory-mib <budgets>`` must appear in that
      echo.  This is what pins the weight vector: the measurement's own
      ``mlp_vector`` is only the caller's current one if the measured boot
      ran the same budgets, so a budget change invalidates the anchor
      instead of silently re-using a bias measured under another split.

    Raises ``MeasuredAnchorRefused`` -- named -- when no log qualifies or when
    a qualifying log is missing a post.
    """
    logs = find_boot_logs(evidence_dirs, group)
    if not logs:
        raise MeasuredAnchorRefused(
            "no boot_*.%s.log under %s -- nothing of this group has been "
            "measured yet" % (group, ", ".join(evidence_dirs) or "<no dir>")
        )
    want_budgets = ",".join(str(int(b)) for b in budgets_mib)
    rejected: List[str] = []
    for path in logs:
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            rejected.append("%s: unreadable (%s)" % (os.path.basename(path), exc))
            continue
        if model_path is not None and not any(
            model_path in m for m in _RE_MODEL.findall(text)
        ):
            rejected.append(
                "%s: different model (%s)"
                % (os.path.basename(path), ", ".join(sorted(set(_RE_MODEL.findall(text)))) or "none")
            )
            continue
        echoed = set(_RE_BUDGETS.findall(text))
        if want_budgets not in echoed:
            rejected.append(
                "%s: measured under budgets %s, planning %s"
                % (os.path.basename(path), "|".join(sorted(echoed)) or "none", want_budgets)
            )
            continue
        posts = parse_boot_log_posts(text, rank_axis=rank_axis)
        try:
            components = build_components(
                posts,
                tp_size=tp_size,
                ranks_on_gpu=ranks_on_gpu,
                required_free_bytes=required_free_bytes,
                source=os.path.basename(path),
            )
        except MeasuredAnchorRefused as exc:
            # An INCOMPLETE newest log must not shadow an older complete one.
            # Measured 2026-09-21: the newest group-D log of this form,
            # fnFL2v92, states 0 of 15 posts -- it died before the weights
            # were loaded -- while v89 two boots earlier states all of them.
            # Refusing outright there would have thrown away a good
            # measurement because a later attempt crashed early. The form
            # checks above still gate every candidate, and the provenance
            # names which log won and how old it is, so falling through is
            # not "keep looking until something fits".
            rejected.append("%s: %s" % (os.path.basename(path), exc))
            continue
        age = (now if now is not None else time.time()) - os.path.getmtime(path)
        tag = _boot_tag(path)
        prov = (
            "measured anchor: boot-log %s (tag %s, group %s, %.1f h old) won; "
            "posts device_total/weights_alloc/mamba_aux/KV-available read per "
            "rank from [vram-idle], [vram-census] and `KV pool sizing`; "
            "residual_residency SOLVED as the residue that reproduces that "
            "boot's own available_bytes; required_free is the CONFIGURED "
            "reserve %s MiB this boot ships, not a measurement -- the "
            "max-paused-rung-tag half of the runtime's required_free is not "
            "stated by any boot-log instrument and is OPEN"
            % (
                os.path.basename(path),
                tag,
                group,
                age / 3600.0,
                ",".join(str(int(b) >> 20) for b in required_free_bytes),
            )
        )
        return MeasuredAnchor(
            components=components,
            mlp_vector=tuple(int(v) for v in mlp_vector),
            source_path=path,
            boot_tag=tag,
            age_s=age,
            provenance=prov,
        )
    raise MeasuredAnchorRefused(
        "no boot_*.%s.log of this form: %s" % (group, "; ".join(rejected))
    )
