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
runtime writes ONLY under ``FLLIPER_MEASURED_KV_BUDGET``.  On the pdflip flip
form that env is not set, so the file is never written -- measured on this
rig 2026-09-21: ``~/.cache/flliper`` holds 156 ``kv_budget-*-seam-rank*.json``
records and not one current ``kv_budget-<digest>.json``.  The pdflip launcher
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
``required_free_bytes`` is NOT parsed: no instrument in a pdflip boot log states
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
import hashlib
import inspect
import json
import os
import re
import tempfile
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
    anchor for a complete one.  The pdflip launcher catches it, prints it, and
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
    #: S68: how the scan got there (index hits, logs actually read, wall
    #: time).  Kept OUT of ``provenance`` so the provenance text of a found
    #: anchor stays what it was; the launcher appends it to the W65 note.
    scan_note: str = ""

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


# --- the line shapes, all of them already emitted by every pdflip boot -------
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

    Written as a free function so the pdflip launcher can ask it of a
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
    """The raw per-rank posts a pdflip boot log states, and nothing else.

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


# --- S68 (19.10.): the per-log index ----------------------------------------
#
# WHY.  The scan below rejects a log on two FORM facts -- the set of
# ``model_path='...'`` echoes and the set of ``--rank-gpu-memory-mib`` echoes
# -- and both are facts of the FILE, not of the boot asking.  Before S68 every
# boot re-derived them by reading every ``boot_*.D.log`` whole: measured
# 19.10. on the NF line, 490 of 490 logs rejected (250 different model, 240
# other budgets), 15.8 GiB read, ~70-93 s, and because the scan sits between
# ``group P pid`` and ``BZ3 D-EARLY-GATE LAUNCH`` it pushed D's start behind
# P READY (+16 s on the critical path, growing ~5.5 s per GiB of new D log).
#
# WHAT.  The two form sets, and the parsed posts of a log that PASSED the form
# check (per rank axis), are kept per file in one JSON next to the logs, keyed
# by basename + (size, mtime_ns, st_ino, st_ctime_ns).  A key that no longer
# matches is a miss and the file is read exactly as before.  The scan order, the form checks, the
# completeness check and the winner are unchanged: the index only replaces
# "read the file again" by "the same answer the last read gave".  A log is
# never moved, rewritten or deleted (other tools read them).
#
# The fingerprint covers every regex and the parser source, so a change to
# what a read would extract invalidates the whole index instead of serving a
# stale answer.  An unwritable directory degrades to the old cold scan and
# says so in the scan note -- the index can make the scan faster, never fail
# it.
INDEX_BASENAME = ".measured_anchor_index.json"
#: 2 (S68 review 19.10.): key = size + mtime_ns + st_ino + st_ctime_ns (a
#: same-size replacement with a restored mtime -- ``cp -p``, ``rsync -t``,
#: ``touch -r`` -- still changes inode/ctime, and no tool can set ctime), and a
#: per-entry checksum, so a damaged entry is discarded and recomputed (heals)
#: instead of being served on every boot.
INDEX_SCHEMA = 2

#: Who may WRITE the index.  ``rw`` (default for a real boot: the launcher's
#: ``main`` sets it when not ``--dry-run``), ``ro`` (use what is there, never
#: write: dry runs, the oracle, tests that read a shared evidence dir), ``off``
#: (the pre-S68 cold scan).  The env wins over every caller default, so a dry
#: run or a test can be ALLOWED to write only explicitly.
INDEX_MODE_ENV = "FLLIPER_MEASURED_ANCHOR_INDEX"
INDEX_MODES = ("rw", "ro", "off")


def resolve_index_mode(default: str = "rw") -> str:
    env = (os.environ.get(INDEX_MODE_ENV) or "").strip().lower()
    mode = env if env in INDEX_MODES else default
    if mode not in INDEX_MODES:
        raise ValueError("index mode must be one of %s, got %r" % (INDEX_MODES, mode))
    return mode


def _stat_key(st: os.stat_result) -> List[int]:
    return [int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_ctime_ns)]


def _entry_sum(e: dict) -> str:
    body = {k: v for k, v in e.items() if k != "sum"}
    return hashlib.sha1(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _is_str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _valid_posts(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    for axis, ranks in raw.items():
        if axis not in ("tp", "pp") or not isinstance(ranks, dict):
            return False
        for rk, fields in ranks.items():
            if not (isinstance(rk, str) and rk.isdigit() and isinstance(fields, dict)):
                return False
            for fk, fv in fields.items():
                if not isinstance(fk, str) or isinstance(fv, bool) or not isinstance(fv, (int, float)):
                    return False
    return True


def _index_fingerprint() -> str:
    h = hashlib.sha1()
    h.update(("schema=%d\n" % INDEX_SCHEMA).encode())
    for rx in (_RE_MODEL, _RE_BUDGETS, _RE_PREFIX, _RE_CENSUS_LOAD, _RE_IDLE,
               _RE_SIZING, _RE_MAMBA):
        h.update(rx.pattern.encode() + b"\n")
    try:
        h.update(inspect.getsource(parse_boot_log_posts).encode())
    except (OSError, TypeError):  # no source shipped: patterns + schema only
        h.update(b"<no-source>")
    return h.hexdigest()[:16]


class _LogIndex:
    """basename -> {key, models, budgets, posts{axis: {...}}, sum} per dir."""

    def __init__(self, dirs: Sequence[str], mode: str = "rw"):
        self.fp = _index_fingerprint()
        self.mode = mode
        self.writable = mode == "rw"
        self.discarded = 0
        self.by_dir: Dict[str, Dict[str, dict]] = {}
        self.dirty: Dict[str, bool] = {}
        self.hits = 0
        self.reads = 0
        self.read_bytes = 0
        self.notes: List[str] = []
        for d in dict.fromkeys(os.path.abspath(x) for x in dirs if x):
            self.by_dir[d] = self._load(d)
            self.dirty[d] = False

    def _load(self, d: str) -> Dict[str, dict]:
        path = os.path.join(d, INDEX_BASENAME)
        try:
            with open(path, "r") as f:
                doc = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self.notes.append("index %s unreadable (%s), rebuilt" % (path, type(exc).__name__))
            return {}
        if not isinstance(doc, dict) or doc.get("fingerprint") != self.fp:
            self.notes.append("index %s from another parser, rebuilt" % path)
            return {}
        logs = doc.get("logs")
        return logs if isinstance(logs, dict) else {}

    def get(self, path: str, st: os.stat_result) -> Optional[dict]:
        d, base = os.path.split(os.path.abspath(path))
        entries = self.by_dir.setdefault(d, {})
        e = entries.get(base)
        if e is None:
            return None
        if not (isinstance(e, dict) and e.get("key") == _stat_key(st)):
            return None  # a changed (or replaced) file: an ordinary miss
        # the entry claims THIS file: it is served only when it is intact.  A
        # damaged one (bad checksum, wrong element types -- e.g. models=[1]) is
        # dropped here, so the read below recomputes and put() replaces it: it
        # heals on the first scan instead of costing the anchor on every boot.
        if (e.get("sum") != _entry_sum(e) or not _is_str_list(e.get("models"))
                or not _is_str_list(e.get("budgets")) or not _valid_posts(e.get("posts"))):
            del entries[base]
            self.dirty[d] = True
            self.discarded += 1
            return None
        self.hits += 1
        return e

    def put(self, path: str, st: os.stat_result, text: str) -> dict:
        d, base = os.path.split(os.path.abspath(path))
        e = {
            "key": _stat_key(st),
            "models": sorted(set(_RE_MODEL.findall(text))),
            "budgets": sorted(set(_RE_BUDGETS.findall(text))),
            "posts": {},
        }
        self.by_dir.setdefault(d, {})[base] = e
        self.dirty[d] = True
        return e

    def put_posts(self, path: str, e: dict, axis: str, posts: Dict[int, Dict[str, float]]) -> None:
        e.setdefault("posts", {})[axis] = {str(k): dict(v) for k, v in posts.items()}
        self.dirty[os.path.dirname(os.path.abspath(path))] = True

    @staticmethod
    def posts_of(e: dict, axis: str) -> Optional[Dict[int, Dict[str, float]]]:
        raw = (e.get("posts") or {}).get(axis)
        if not isinstance(raw, dict):
            return None
        try:
            return {int(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in raw.items()}
        except (TypeError, ValueError, AttributeError):  # a damaged entry is a miss
            return None

    def read(self, path: str) -> str:
        """The whole log, exactly as the pre-S68 scan read it."""
        with open(path, "r", errors="replace") as f:
            text = f.read()
            try:  # a one-off read: do not leave 30 MiB per log in the page cache
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        self.reads += 1
        self.read_bytes += len(text)
        return text

    def save(self) -> None:
        if not self.writable:
            return
        for d, dirty in self.dirty.items():
            if not dirty:
                continue
            # entries of logs that are gone are dropped, so the index never grows
            # past the directory it describes
            logs = {}
            for b, e in self.by_dir[d].items():
                if os.path.exists(os.path.join(d, b)):
                    e["sum"] = _entry_sum(e)
                    logs[b] = e
            path = os.path.join(d, INDEX_BASENAME)
            # mkstemp IN the target dir: unique across PID namespaces (the
            # launcher in the container, a dry run in CT999 and a warm run on
            # the host can share a PID), same filesystem, so os.replace is
            # atomic.  0644: the CT999 side reads what the container wrote.
            tmp = None
            try:
                fd, tmp = tempfile.mkstemp(dir=d, prefix=INDEX_BASENAME + ".", suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump({"fingerprint": self.fp, "logs": logs}, f, separators=(",", ":"))
                os.chmod(tmp, 0o644)
                os.replace(tmp, path)
                tmp = None
            except OSError as exc:
                self.notes.append("index %s not written (%s: %s); next scan is cold again"
                                  % (path, type(exc).__name__, exc))
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    def note(self, n_logs: int, wall_s: float) -> str:
        s = (" [W65-SCAN logs=%d index_hits=%d read=%d read_mib=%.1f wall_s=%.2f index=%s%s%s]"
             % (n_logs, self.hits, self.reads, self.read_bytes / MIB, wall_s,
                self.mode, (" discarded=%d" % self.discarded) if self.discarded else "",
                ("; " + "; ".join(self.notes)) if self.notes else ""))
        return s


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
    use_index: bool = True,
    index_mode: Optional[str] = None,
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

    S68: the form facts and the posts of each log come from the per-directory
    index when its size+mtime still match (see ``_LogIndex``); the answer is
    the same one a full read gives.  ``use_index=False`` is the pre-S68 cold
    scan (kept for the equivalence test).  ``index_mode`` (``rw``/``ro``/
    ``off``) says whether the index may be written; ``None`` resolves the env
    ``FLLIPER_MEASURED_ANCHOR_INDEX`` with ``rw`` as default.  The launcher
    passes ``ro`` for a dry run (the oracle included), so a dry run never
    writes into a shared evidence dir unless the env allows it explicitly.
    """
    t0 = time.monotonic()
    logs = find_boot_logs(evidence_dirs, group)
    if not logs:
        raise MeasuredAnchorRefused(
            "no boot_*.%s.log under %s -- nothing of this group has been "
            "measured yet" % (group, ", ".join(evidence_dirs) or "<no dir>")
        )
    mode = "off" if not use_index else (index_mode or resolve_index_mode("rw"))
    if mode not in INDEX_MODES:
        raise ValueError("index_mode must be one of %s, got %r" % (INDEX_MODES, mode))
    use_index = mode != "off"
    idx = _LogIndex(evidence_dirs if use_index else (), mode)
    want_budgets = ",".join(str(int(b)) for b in budgets_mib)
    rejected: List[str] = []
    won: Optional[MeasuredAnchor] = None
    try:
        won = _scan(idx, logs, use_index, group, tp_size, rank_axis, ranks_on_gpu,
                    required_free_bytes, mlp_vector, model_path, want_budgets, now,
                    rejected)
    finally:
        # also on an unexpected error: what was learned so far is kept
        if use_index:
            idx.save()
    note = idx.note(len(logs), time.monotonic() - t0)
    if won is not None:
        return dataclasses.replace(won, scan_note=note)
    exc = MeasuredAnchorRefused(
        "no boot_*.%s.log of this form: %s%s" % (group, "; ".join(rejected), note)
    )
    exc.scan_note = note
    raise exc


def _scan(idx, logs, use_index, group, tp_size, rank_axis, ranks_on_gpu,
          required_free_bytes, mlp_vector, model_path, want_budgets, now,
          rejected) -> Optional[MeasuredAnchor]:
    """The pre-S68 loop, newest first, first complete log of the form wins."""
    for path in logs:
        base = os.path.basename(path)
        text: Optional[str] = None
        try:
            st = os.stat(path)
            entry = idx.get(path, st) if use_index else None
            if entry is None:
                text = idx.read(path)
                entry = idx.put(path, st, text) if use_index else {
                    "models": sorted(set(_RE_MODEL.findall(text))),
                    "budgets": sorted(set(_RE_BUDGETS.findall(text))),
                }
        except OSError as exc:
            rejected.append("%s: unreadable (%s)" % (base, exc))
            continue
        models = entry["models"]
        if model_path is not None and not any(model_path in m for m in models):
            rejected.append(
                "%s: different model (%s)" % (base, ", ".join(models) or "none")
            )
            continue
        echoed = entry["budgets"]
        if want_budgets not in echoed:
            rejected.append(
                "%s: measured under budgets %s, planning %s"
                % (base, "|".join(echoed) or "none", want_budgets)
            )
            continue
        posts = _LogIndex.posts_of(entry, rank_axis) if use_index else None
        if posts is None:
            if text is None:
                try:
                    text = idx.read(path)
                except OSError as exc:
                    rejected.append("%s: unreadable (%s)" % (base, exc))
                    continue
            posts = parse_boot_log_posts(text, rank_axis=rank_axis)
            if use_index:
                idx.put_posts(path, entry, rank_axis, posts)
        try:
            components = build_components(
                posts,
                tp_size=tp_size,
                ranks_on_gpu=ranks_on_gpu,
                required_free_bytes=required_free_bytes,
                source=base,
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
            rejected.append("%s: %s" % (base, exc))
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
                base,
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
    return None


def warm_index(evidence_dirs: Sequence[str], group: str = "D") -> str:
    """Fill the index for every log of ``group`` without planning anything.

    For a quiet moment on the host (``python -m flliper.srt.planner.measured_anchor
    <dir> [group]``), so the first boot of an image carrying S68 does not pay
    the cold scan.  Reads each un-indexed log once; never writes a log.
    """
    t0 = time.monotonic()
    logs = find_boot_logs(evidence_dirs, group)
    idx = _LogIndex(evidence_dirs, "rw")  # an operator call: writing is its purpose
    for path in logs:
        try:
            st = os.stat(path)
            if idx.get(path, st) is None:
                idx.put(path, st, idx.read(path))
        except OSError:
            continue
    idx.save()
    return idx.note(len(logs), time.monotonic() - t0).strip()


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import sys

    print(warm_index([sys.argv[1]], sys.argv[2] if len(sys.argv) > 2 else "D"))
