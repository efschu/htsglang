"""IPC Phase 1: the rank's own state as a versioned record, not a log line.

User order 28.09. ~20:45Z: "also diese kommunikation über logs? das ist nicht
dein ernst oder? das muss man professionell ordentlich machen - die ganze
inter prozesskommunikation. über logfiles?" -- control values travel as
records with a schema version; log lines stay for humans.

Measured cause (rc12z29d, 28.09. 20:29Z): under the uneven-DCP cut the two
Form A workers own token rows (#239 F14) and print '#239 F14 KV-WORKER-WINDOW'
instead of the worker line the launcher counted, so the launcher half of
W7/W10 counted 'kv x1 blob x1' and refused a D group that had booted clean.
The launcher asked the question "does every rank hold its canonical window?"
through a string that a different change was free to reword.

Transport (Phase 1): each rank writes ONE json file per attach into the
directory the launcher names in ``SGLANG_WEG2_RANK_STATE_DIR`` (next to the
group log, inside the bind mount), atomically (tmp + rename). The launcher
clears the directory before it spawns the group, so a record is always this
launch's, never a previous launch's that happened to share the log path. A
rank that never wrote is a named absence (``missing``), never a count that
came out short.

Phase 2 (IPC-STATE-PLAN-0928.md): the same record gathered once at init over
the group's existing CPU group, graded in-rank, and served by TP0/PP0 on
``/weg2/state`` through the existing Scheduler -> TokenizerManager internal
state path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

#: Bumped on every incompatible change of :class:`RankState`. A reader
#: refuses a record of another version by name instead of guessing fields.
#: 2 (VRAM-Vertrag M2, 29.09.): schema 1 plus the optional block ``vram``
#: (weg2/vram_actual.py, ``weg2.rank_vram/1``). A schema-1 record still reads
#: (it carries no ``vram``); any other version is refused by name. A ``vram``
#: block this reader refuses costs only the block (``vram_refused``), never
#: the record: it is display and record, not a control value.
RANK_STATE_SCHEMA = 2
#: Versions this reader accepts; a record's ``vram`` is only legal from 2 on.
RANK_STATE_SCHEMAS_READ = (1, 2)

RANK_STATE_ENV = "SGLANG_WEG2_RANK_STATE_DIR"

#: ``RankState.role``
ROLE_STAGE = "stage"  # P: a PP stage (every layer family on its slice)
ROLE_ATTN_HOST = "attn_host"  # D TP0 / a D rank that holds attention
ROLE_FORM_A_WORKER = "form_a_worker"  # D: expert worker, no attention layer


class RankStateSchemaError(ValueError):
    """A record of another schema version, or one that is not a record."""


@dataclass(frozen=True)
class RankState:
    """What one rank IS after its storage attach -- the fields a launcher,
    a watcher or a peer rank decides on. Every rank writes one, including a
    rank that owns zero KV rows under the cut (27B requirement 3: that is
    ``kv_rows_per_page=0``, never silence)."""

    group: str
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    role: str
    #: --hicache-canonical-kv-page armed for this rank's process.
    canonical_armed: bool
    #: 27B requirement 5: APPLICABLE and ACTIVE are separate facts. A rank
    #: the component does not apply to owes nothing (a Form A worker holds
    #: no GDN blob; a worker without the cut holds no KV page window); a rank
    #: it applies to must hold it. rc12z29d died on the conflation: 'no line'
    #: was read as 'not active' for ranks where the line had changed.
    kv_page_applicable: bool
    kv_page_active: bool
    gdn_blob_applicable: bool
    gdn_blob_active: bool
    page_size: int
    #: Token rows of every page this rank owns: ``hi - lo`` under the
    #: uneven-DCP cut (0 = owns none), ``page_size`` without a cut.
    kv_rows_per_page: int
    #: ``(S, lo, hi)`` of the weighted owner range, None without a cut.
    dcp_owner: Optional[Tuple[int, int, int]] = None
    pid: int = 0
    ts: float = 0.0
    #: Attach counter of this process (re-attach after a cutover rewrites).
    seq: int = 0
    schema: int = RANK_STATE_SCHEMA
    #: VRAM-Vertrag M2: the rank's VRAM actual per PID and category
    #: (``weg2.rank_vram/1``, weg2/vram_actual.py); None with
    #: SGLANG_WEG2_VRAM_ACTUAL off. Display and records only -- no gate reads it.
    vram: Optional[dict] = None
    #: READER-SIDE: why this reader dropped the record's ``vram`` block
    #: (VRAM-ACTUAL-BLOCK-REFUSED), None otherwise. A writer never sets it.
    vram_refused: Optional[str] = None

    @property
    def rank_key(self) -> str:
        return f"tp{self.tp_rank}pp{self.pp_rank}"

    def to_json(self) -> str:
        d = asdict(self)
        if d["dcp_owner"] is not None:
            d["dcp_owner"] = list(d["dcp_owner"])
        if d["vram_refused"] is None:
            del d["vram_refused"]  # reader-side finding; a writer's record never carries it
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RankState":
        try:
            d = json.loads(text)
        except ValueError as e:
            raise RankStateSchemaError(f"not a RankState record: {e}") from e
        if not isinstance(d, dict):
            raise RankStateSchemaError("not a RankState record: top level is not an object")
        schema = d.get("schema")
        if schema not in RANK_STATE_SCHEMAS_READ:
            raise RankStateSchemaError(
                f"RankState schema {schema!r}, this reader knows {list(RANK_STATE_SCHEMAS_READ)}"
            )
        names = set(cls.__dataclass_fields__)
        if schema == 1:
            names -= {"vram", "vram_refused"}
        unknown = sorted(set(d) - names)
        if unknown:
            raise RankStateSchemaError(f"RankState schema {schema} has no field(s) {unknown}")
        if d.get("vram") is not None:
            # The block is display and record, never a control value: a block
            # this reader refuses is dropped BY NAME and the record -- the
            # W7/W10 facts -- still reads (coordinator 29.09.).
            from sglang.srt.weg2.vram_actual import VramBlock, VramBlockSchemaError

            try:
                VramBlock.from_dict(d["vram"])
            except VramBlockSchemaError as e:
                d["vram"] = None
                d["vram_refused"] = str(e)
        if d.get("dcp_owner") is not None:
            d["dcp_owner"] = tuple(int(x) for x in d["dcp_owner"])
        try:
            return cls(**d)
        except TypeError as e:
            raise RankStateSchemaError(f"RankState record incomplete: {e}") from e


def rank_state_dir_for_log(group_log: str) -> str:
    """The launcher's choice of directory for one group: next to its log, so
    it lives wherever the log lives (the bind mount, 27B requirement 4)."""
    return f"{group_log}.rankstate"


def rank_state_path(state_dir: str, state: RankState) -> str:
    return os.path.join(state_dir, f"{state.group or 'G'}.{state.rank_key}.json")


def write_rank_state(state: RankState, state_dir: Optional[str]) -> Optional[str]:
    """Atomic write of this rank's record; None when no directory was named
    (a boot outside the weg2 launcher). Raises on an I/O failure: a record
    the launcher will decide on must not be silently absent."""
    if not state_dir:
        return None
    os.makedirs(state_dir, exist_ok=True)
    path = rank_state_path(state_dir, state)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(state.to_json())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def clear_rank_state_dir(state_dir: str) -> int:
    """Remove every record of a previous launch; the number removed."""
    n = 0
    try:
        names = os.listdir(state_dir)
    except FileNotFoundError:
        return 0
    for name in names:
        if name.endswith(".json") or ".json.tmp." in name:
            try:
                os.unlink(os.path.join(state_dir, name))
                n += 1
            except FileNotFoundError:
                pass
    return n


def read_group_states(state_dir: str) -> Tuple[List[RankState], List[str]]:
    """``(records, refusals)``: every readable record of the directory, and
    one line per file that is not a record of this schema."""
    states: List[RankState] = []
    bad: List[str] = []
    try:
        names = sorted(os.listdir(state_dir))
    except FileNotFoundError:
        return states, bad
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(state_dir, name)
        try:
            with open(path) as f:
                states.append(RankState.from_json(f.read()))
        except (OSError, RankStateSchemaError) as e:
            bad.append(f"{name}: {e}")
    return states, bad


@dataclass
class CanonicalVerdict:
    """W7/W10 graded on records. ``n_kv``/``n_blob`` count ranks exactly
    as the log count did (a Form A worker satisfies both), so the parallel
    log count can be compared number for number."""

    ok: bool
    n_kv: int
    n_blob: int
    n_worker: int
    expected: int
    missing: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    by_rank: Dict[str, str] = field(default_factory=dict)

    def line(self) -> str:
        ranks = " ".join(f"{k}={v}" for k, v in sorted(self.by_rank.items()))
        s = (
            f"kv x{self.n_kv} blob x{self.n_blob} of {self.expected} rank(s)"
            f" ({self.n_worker} Form A worker(s)); {ranks or 'no records'}"
        )
        if self.missing:
            s += f"; MISSING {','.join(self.missing)}"
        if self.reasons:
            s += "; " + "; ".join(self.reasons)
        return s


def _flag(applicable: bool, active: bool) -> str:
    return "active" if active else ("MISSING" if applicable else "n/a")


def grade_canonical(states: List[RankState], expected: int, group: str = "") -> CanonicalVerdict:
    """Every one of ``expected`` ranks reports, has the format armed, and
    holds every component that APPLIES to it (applicable => active). Counts
    match the old log count rank for rank (a Form A worker satisfies both),
    so the parallel count can be compared number for number. Duplicate
    records of one rank, a record of another group and a group-size
    disagreement are refusals, not tolerated noise."""
    reasons: List[str] = []
    by_key: Dict[str, RankState] = {}
    for s in states:
        if group and s.group != group:
            reasons.append(f"{s.rank_key} reports group {s.group!r}, graded {group!r}")
            continue
        if s.rank_key in by_key:
            reasons.append(f"{s.rank_key} reported twice")
            continue
        by_key[s.rank_key] = s
    sizes = {(s.tp_size, s.pp_size) for s in by_key.values()}
    if len(sizes) > 1:
        reasons.append(f"ranks disagree on (tp, pp): {sorted(sizes)}")
    if sizes:
        tp, pp = next(iter(sizes))
        want = [f"tp{t}pp{p}" for p in range(pp) for t in range(tp)]
        if tp * pp != expected:
            reasons.append(f"group reports {tp}x{pp} ranks, launcher expects {expected}")
    else:
        want = []
    missing = [k for k in want if k not in by_key]
    if not want:
        missing = [f"{expected} rank(s): no record at all"]
    n_kv = n_blob = n_worker = 0
    by_rank: Dict[str, str] = {}
    for k, s in by_key.items():
        worker = s.role == ROLE_FORM_A_WORKER
        n_worker += int(worker)
        kv_ok = s.canonical_armed and (s.kv_page_active or not s.kv_page_applicable)
        blob_ok = s.canonical_armed and (s.gdn_blob_active or not s.gdn_blob_applicable)
        n_kv += int(kv_ok)
        n_blob += int(blob_ok)
        by_rank[k] = (
            f"{s.role}/kv={_flag(s.kv_page_applicable, s.kv_page_active)}"
            f"/blob={_flag(s.gdn_blob_applicable, s.gdn_blob_active)}/rows{s.kv_rows_per_page}"
        )
        if not s.canonical_armed:
            reasons.append(f"{k} ({s.role}) canonical page format not armed")
            continue
        if not worker and not s.kv_page_applicable:
            reasons.append(f"{k} ({s.role}) holds attention but reports no KV page applicable")
        if not kv_ok:
            reasons.append(f"{k} ({s.role}) KV page applicable, not active")
        if not blob_ok:
            reasons.append(f"{k} ({s.role}) GDN blob applicable, not active")
    ok = not reasons and not missing and n_kv >= expected and n_blob >= expected
    return CanonicalVerdict(
        ok=ok, n_kv=n_kv, n_blob=n_blob, n_worker=n_worker, expected=expected,
        missing=missing, reasons=reasons, by_rank=by_rank,
    )


def build_rank_state(
    *,
    group: str,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    form_a_worker: bool,
    canonical_on: bool,
    canonical_kv_built: bool,
    canonical_blob_built: bool,
    has_mamba_pool: bool,
    page_size: int,
    owner_ctx: Optional[Tuple[int, int, int]],
    seq: int,
) -> RankState:
    """The cache controller's facts after ``_generate_storage_config``,
    mapped to the record. Pure, so the mapping is testable without a rank.

    Applicability, per component:
      * KV page: every rank that holds attention; a Form A worker only when
        it owns token rows under the cut (#239 F14, the window is built for
        it) and at least one of them. Without the cut the worker rides the
        null storage tier.
      * GDN blob: every non-worker rank whose device pool HAS a mamba pool
        (the bound pool is the witness, as in W7 ``check_mamba_blob_present``);
        never a Form A worker (mamba/QSA/draft stay the host's).
    """
    if form_a_worker:
        role = ROLE_FORM_A_WORKER
    elif pp_size > 1:
        role = ROLE_STAGE
    else:
        role = ROLE_ATTN_HOST
    if form_a_worker:
        kv_applicable = bool(canonical_on and canonical_kv_built)
        blob_applicable = False
    else:
        kv_applicable = bool(canonical_on)
        blob_applicable = bool(canonical_on and has_mamba_pool)
    if owner_ctx is None:
        rows = int(page_size)
        owner = None
    else:
        S, lo, hi = (int(x) for x in owner_ctx)
        owner = (S, lo, hi)
        rows = hi - lo if int(page_size) > 1 else int(page_size)
    if form_a_worker and rows == 0:
        # 27B B3: a rank owning no token rows owes no KV page -- it still
        # REPORTS (rows 0), it just is not graded on a window it cannot use.
        kv_applicable = False
    return RankState(
        group=group, tp_rank=int(tp_rank), tp_size=int(tp_size), pp_rank=int(pp_rank),
        pp_size=int(pp_size), role=role, canonical_armed=bool(canonical_on),
        kv_page_applicable=kv_applicable, kv_page_active=bool(canonical_on and canonical_kv_built),
        gdn_blob_applicable=blob_applicable, gdn_blob_active=bool(canonical_on and canonical_blob_built),
        page_size=int(page_size), kv_rows_per_page=int(rows), dcp_owner=owner,
        pid=os.getpid(), ts=time.time(), seq=int(seq),
    )
