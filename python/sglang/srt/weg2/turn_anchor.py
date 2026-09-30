# SPDX-License-Identifier: Apache-2.0
"""TURN ANCHOR (NF, 29.09.; y3m boot ...dauer09292136, P and front logs).

THE MEASURED GAP. A hybrid (GDN) reader resumes only at a recurrent anchor at
or below its fork. Group P anchors a prompt at its chunk ends and at its END
(``floor_page(N)``, H63 fold / #1233). A Claude-Code session forks at two
places, and only one of them had an anchor:

* a side request forks at the main turn's generation prompt (``N-4``) --
  the END anchor serves it (and CLAIM ANCHOR where ``N % page`` is 0/1);
* the NEXT turn forks at the start of the previous request's LAST message
  (front ``SESSION-PREFIX``: weg2-50-66 left weg2-46-61 at common=64352 of
  64515, N-163; the gaps of 125-171 tokens are that last message). There P
  had no anchor: 46-61's END anchor 64512 lies past the fork, the read fell
  back to the previous chunk anchor -- P 22:02:22 weg2-50-71 ``#1028B FETCH
  CAP kv=1005 claimed=989 lost=16 MAMBA (11, 988)``, and on a fresh session
  (turn 1 anchored only at its END ~14912) ``#1035c ZERO-ANSWER cause=CAPPED
  by=mamba``. The same boundary is where every NEW session forks from an old
  one: the end of the shared system prompt + tools is the start of the first
  user message.

WHY NOT A CHUNK CUT (P-FORK-CUT, weg2/p_fork_cut.py). A cut there is only
free when the extend spans two forwards anyway; a follow-up turn is ONE
forward (delta < 16384), and every extra forward costs P the ~1.7 s
expert-stream floor (H118). WHY NOT MOVING THE ONE TRACK: the last forward's
single extend track already carries the END anchor (floor_page(N)), which
D's leg 2 and every side request read.

THE REPAIR: A SECOND EXTEND TRACK in the same forward. The chunked GDN kernel
materialises the recurrent state at every FLA-chunk boundary of the forward
(``h``, the tensor the default track already gathers from), the conv window
is a gather of the forward's own inputs, and the PLE side states are a
gather of the same ``[state | chunk]`` layout. Snapshotting a second grid
point is therefore one more row in each of those gathers -- no forward, no
kernel, no chunk boundary moves. The prompt's LAST message start is found
in the token ids (the ``<|im_start|>`` before the generation prompt's), a
pure function of the ids and one env value: every PP rank derives the same
position, no transport, no collective, no log line in any control path.

  * ``SGLANG_WEG2_TURN_ANCHOR_TOKEN=<id>`` (launcher ``--turn-anchor-token``,
    group P). Armed only where the ordering is proven: group P, extra_buffer
    tracks, no checkpoint interval, overlap schedule OFF (the tree insert then
    runs after the forward that wrote the slot was launched on this thread).
  * prepare_for_extend: a step ``[prefix, end)`` whose interior holds the
    turn position ``t = floor_page(i)`` (``i`` = index of the last message's
    ``<|im_start|>``), strictly below the step's own track, on the step's
    FLA grid, draws ONE mamba slot (no eviction, no reserve; none free = no
    turn anchor, counted) as the second track destination.
  * the GDN backend / the PLE model code add the rows and MARK the batch's
    ``TurnTracks``; the kinds required are fixed at plan time from the pool
    (GDN always, the PLE side states when their pools are enabled), so a
    forward that did not run them in Python (graph replay, split batch,
    other backend) leaves a mark missing.
  * the tree's next insert of the request (cache_unfinished_req /
    cache_finished_req) first inserts ``key[:t]`` with that slot as its
    mamba value -- the chunk path's own insert/match/lock/publish sequence --
    or frees the slot when a mark is missing. The step's own insert follows
    unchanged (its protected prefix is now t).

FORK TRACK (NF y5k, 30.09.): the same second-track form at PP0's told fork
(``req._weg2_fork_told``, weg2/p_fork_cut.py) when P-FORK-CUT found the cut
``paid`` -- the store's KV of this prompt ends there, so the next request with
that prefix stops reading there; without an anchor at it, it fell back to the
previous chunk anchor (``_note_fork_step``).

COST. Compute: one gather row per GDN/PLE layer per request (~34 MiB of
state copied on PP0 per anchor, ~50 us at card bandwidth, inside a forward of
>= 1.7 s). Device: one mamba slot per request with a turn boundary in the
step, from the existing pool, held for one forward and then owned by the
tree like any chunk anchor (evictable once the request's lock moved on).
Host/L3: one more anchor per turn through the existing arena (share cap and
H19 displacement unchanged) -- the same size as a chunk anchor.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PENDING_ATTR = "_weg2_turn_pending"
POS_ATTR = "_weg2_turn_pos"
#: TWIN ANCHOR (30.09., weg2/twin_anchor.py): the step's twin-boundary tracks,
#: a list of plans of the same shape as PENDING_ATTR's, ascending by position
TWIN_PENDING_ATTR = "_weg2_twin_pending"
#: FORK TRACK (30.09., NF y5k weg2-0-4): the step's track at PP0's told fork
#: (``req._weg2_fork_told``, P-FORK-CUT) when the cut was not free -- one plan
#: of the same shape as PENDING_ATTR's
FORK_PENDING_ATTR = "_weg2_fork_pending"

#: the generation prompt of a thinking Qwen3.x turn is 5 tokens, the
#: non-thinking one 8; the same window the FORK ANCHOR uses.
GEN_TAIL = 16

_COUNTS: dict = {}


def _count(key: str) -> int:
    n = _COUNTS.get(key, 0) + 1
    _COUNTS[key] = n
    return n


def _log_due(n: int) -> bool:
    return n <= 16 or (n & (n - 1)) == 0


def counts() -> dict:
    return dict(_COUNTS)


# -- the switch --------------------------------------------------------------------
def token() -> Optional[int]:
    """The turn-start token id, or None = switch off."""
    try:
        from sglang.srt.environ import envs

        tok = envs.SGLANG_WEG2_TURN_ANCHOR_TOKEN.get()
    except Exception:  # noqa: BLE001 - unreadable = off
        return None
    try:
        tok = int(tok) if tok is not None else None
    except (TypeError, ValueError):
        return None
    return tok if tok is not None and tok > 0 else None


def armed(server_args: Any = None) -> Optional[int]:
    """The token id when the turn anchor runs on THIS process, else None.

    Group P only (the group that writes the END anchor the next turn cannot
    use), extra_buffer tracks without a checkpoint interval (the track form
    the second row joins), and no overlap schedule (the insert must follow
    the forward launch on the same thread, see the module doc)."""
    tok = token()
    if tok is None:
        return None
    if (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "P":
        return None
    if server_args is None:
        try:
            from sglang.srt.runtime_context import get_server_args

            server_args = get_server_args()
        except Exception:  # noqa: BLE001
            return None
    try:
        if not server_args.enable_mamba_extra_buffer():
            return None
    except Exception:  # noqa: BLE001
        return None
    if getattr(server_args, "mamba_checkpoint_interval", None) is not None:
        return None
    if not bool(getattr(server_args, "disable_overlap_schedule", False)):
        return None
    return tok


# -- geometry (pure) ---------------------------------------------------------------
def _rindex(ids: Sequence[int], tok: int, lo: int, hi: int, step: int = 4096) -> Optional[int]:
    """Index of the last ``tok`` in ``ids[lo:hi]``, or None. Scans backwards
    in slices, so the cost is the last message's length, not the prompt's
    (a 262k prompt with a 160-token last message reads one slice)."""
    end = int(hi)
    while end > lo:
        start = max(int(lo), end - step)
        seg = list(ids[start:end])
        seg.reverse()
        try:
            return end - 1 - seg.index(tok)
        except ValueError:
            end = start
    return None


def boundary(ids: Optional[Sequence[int]], tok: Optional[int],
             tail: int = GEN_TAIL) -> Optional[int]:
    """Index ``i`` of the ``<|im_start|>`` that opens the prompt's LAST message.

    The prompt must end with a generation prompt (a turn-start token among the
    final ``tail`` ids, excluding the very last); ``i`` is the turn-start token
    before THAT one. None when the switch is off, there is no generation
    prompt, or the prompt has a single message. ``ids[:i + 1]`` is what the
    next turn (and a side request of this one) shares with this prompt."""
    if tok is None or ids is None:
        return None
    n = len(ids)
    if n < 3:
        return None
    from sglang.srt.weg2 import fork_anchor as _fa

    g = _fa.fork_cut(ids, tok, tail)
    if g is None:
        return None
    return _rindex(ids, tok, 0, g)


def anchor_pos(ids: Optional[Sequence[int]], tok: Optional[int], grain: int) -> Optional[int]:
    """The turn anchor position ``floor_grain(i)``: a reader sharing
    ``ids[:i + 1]`` claims at most ``floor_page(i)`` (bigram keys take the
    next token too), so the anchor sits exactly where its claim ends."""
    i = boundary(ids, tok)
    if i is None:
        return None
    grain = max(1, int(grain))
    t = (int(i) // grain) * grain
    return t if t > 0 else None


def req_anchor_pos(req: Any, tok: int, grain: int) -> Optional[int]:
    """``anchor_pos`` of a request's PROMPT, cached on the request (the scan
    runs once per request, not per step)."""
    ids = getattr(req, "origin_input_ids", None)
    if ids is None:
        return None
    key = (len(ids), int(tok), int(grain))
    cached = getattr(req, POS_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    t = anchor_pos(ids, tok, grain)
    try:
        setattr(req, POS_ATTR, (key, t))
    except Exception:  # noqa: BLE001
        pass
    return t


def step_target(prefix: int, end: int, t: Optional[int], chunk: int,
                main_track: Optional[int]) -> Optional[int]:
    """The second track of the step ``[prefix, end)``, or None.

    Only a position strictly inside the step, strictly below the step's own
    (main) track -- the insert order is the turn node first, then the step's
    node below it on the same path -- and on the step's FLA-chunk grid (the
    kernel keeps the state at ``prefix + k * chunk`` only)."""
    if t is None or chunk <= 0:
        return None
    t, prefix, end = int(t), int(prefix), int(end)
    if not (prefix < t < end):
        return None
    if main_track is None or t >= int(main_track):
        return None
    if (t - prefix) % int(chunk) != 0:
        return None
    return t


# -- the batch descriptor ----------------------------------------------------------
class TurnTracks:
    """The second tracks of one extend batch: per row the target, the slot,
    and the geometry the scheduler computed them from. The forward adds the
    rows and marks what it wrote; the tree inserts only fully marked ones.

    ``need``/``done``: the kinds of state the forward is required to write
    (``gdn`` always; ``ple_conv`` / ``ple_ngram`` when the pool carries the
    Qwen4-Exp PLE side states -- fixed at plan time, and the model code
    registers them again when it runs) and the kinds it did write.
    """

    __slots__ = ("bs", "rows", "slots", "targets", "prefixes", "ext_lens",
                 "need", "done", "dst_phys", "rows_dev", "offsets_dev")

    def __init__(self, bs: int, need: Iterable[str] = ()):
        self.bs = int(bs)
        self.rows: List[int] = []
        self.slots: List[Any] = []   # one-element slot tensors (virtual ids)
        self.targets: List[int] = []
        self.prefixes: List[int] = []
        self.ext_lens: List[int] = []
        self.need = {"gdn", *need}
        self.done: set = set()
        self.dst_phys = None      # device tensor of translated slot ids (backend)
        self.rows_dev = None      # device tensor of row indices (backend)
        self.offsets_dev = None   # device tensor of t - prefix (backend)

    def add(self, row: int, slot: Any, t: int, prefix: int, ext_len: int) -> None:
        self.rows.append(int(row))
        self.slots.append(slot)
        self.targets.append(int(t))
        self.prefixes.append(int(prefix))
        self.ext_lens.append(int(ext_len))

    def __len__(self) -> int:
        return len(self.rows)

    def complete(self) -> bool:
        return bool(self.rows) and self.need <= self.done

    def geometry_matches(self, bs: int, prefix_lens_cpu: Optional[Sequence[int]],
                         ext_lens_cpu: Optional[Sequence[int]]) -> bool:
        """The forward runs the batch the scheduler described: same row count,
        and every turn row has the prefix and extend length it was planned
        with. A split or re-ordered batch answers False (no rows added)."""
        if int(bs) != self.bs or prefix_lens_cpu is None or ext_lens_cpu is None:
            return False
        try:
            for r, p, e in zip(self.rows, self.prefixes, self.ext_lens):
                if int(prefix_lens_cpu[r]) != p or int(ext_lens_cpu[r]) != e:
                    return False
        except (IndexError, TypeError, ValueError):
            return False
        return True


def h_rows(ext_lens_cpu: Sequence[int], rows: Sequence[int], targets: Sequence[int],
           prefixes: Sequence[int], chunk: int, mamba2: bool = False) -> List[int]:
    """Index into the forward's flattened ``h`` of the state at each target.

    Mirrors ``_init_track_ssm_indices``: row ``r`` owns ``(len - 1) // chunk + 1``
    states (GDN; ``len // chunk`` on Mamba2), laid out row after row, the
    ``k``-th being the state after ``k * chunk`` tokens of the step."""
    chunk = int(chunk)
    offs, acc = [], 0
    for L in ext_lens_cpu:
        offs.append(acc)
        L = int(L)
        acc += (L // chunk) if mamba2 else ((L - 1) // chunk + 1 if L > 0 else 0)
    return [offs[int(r)] + (int(t) - int(p)) // chunk for r, t, p in zip(rows, targets, prefixes)]


def conv_starts(ext_lens_cpu: Sequence[int], rows: Sequence[int], targets: Sequence[int],
                prefixes: Sequence[int], conv_len: int) -> List[int]:
    """First flattened input position of each target's conv window (the
    ``conv_len`` inputs before the target), as ``_init_track_conv_indices``."""
    qsl, acc = [], 0
    for L in ext_lens_cpu:
        qsl.append(acc)
        acc += int(L)
    return [qsl[int(r)] + (int(t) - int(p)) - int(conv_len) for r, t, p in zip(rows, targets, prefixes)]


# -- the scheduler side --------------------------------------------------------------
def note_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
              prefix: int, end: int, track_mask: bool, main_track: Optional[int],
              chunk: int, page: int, tok: Optional[int],
              twin_bounds: Sequence[int] = (),
              fork_told: int = 0) -> Optional[TurnTracks]:
    """prepare_for_extend, per request: plan the second track of this step,
    (TWIN ANCHOR, weg2/twin_anchor.py) one more track per twin boundary
    ``twin_bounds`` inside it, and (FORK TRACK) one at PP0's told fork
    ``fork_told`` when this step runs through it. Returns the (possibly new)
    batch descriptor."""
    stale = getattr(req, PENDING_ATTR, None)
    if stale is not None:
        # a plan the tree never consumed (a path that skipped the insert):
        # the slot goes back before a new one is drawn
        _free_slot(batch, stale[1])
        setattr(req, PENDING_ATTR, None)
        n = _count("stale")
        if _log_due(n):
            logger.warning("WEG2 TURN-ANCHOR STALE rid=%s: an unconsumed plan's slot "
                           "freed (n=%d)", str(getattr(req, "rid", "?")), n)
    for st_pend in getattr(req, TWIN_PENDING_ATTR, None) or ():
        _free_slot(batch, st_pend[1])
    if getattr(req, TWIN_PENDING_ATTR, None):
        setattr(req, TWIN_PENDING_ATTR, None)
    fk_stale = getattr(req, FORK_PENDING_ATTR, None)
    if fk_stale is not None:
        _free_slot(batch, fk_stale[1])
        setattr(req, FORK_PENDING_ATTR, None)
    if not track_mask or len(getattr(req, "output_ids", None) or ()) > 0:
        return desc
    desc = _note_turn_twin_steps(batch, desc, req, row, prefix, end, main_track,
                                 chunk, page, tok, twin_bounds)
    if fork_told:
        desc = _note_fork_step(batch, desc, req, row, prefix, end, main_track,
                               chunk, page, int(fork_told))
    return desc


def _note_turn_twin_steps(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                          prefix: int, end: int, main_track: Optional[int], chunk: int,
                          page: int, tok: Optional[int],
                          twin_bounds: Sequence[int]) -> Optional[TurnTracks]:
    """The turn track and the twin-boundary tracks of this step (the body of
    :func:`note_step` before the fork track joined it, unchanged)."""
    t_abs = req_anchor_pos(req, tok, page) if tok is not None else None
    t = step_target(prefix, end, t_abs, chunk, main_track)
    if twin_bounds:
        desc = _note_twin_steps(batch, desc, req, row, prefix, end, main_track, chunk,
                                page, twin_bounds, turn=t if t is not None and t % max(1, int(page)) == 0 else None)
    if t is None:
        return desc
    if t % max(1, int(page)) != 0:
        return desc
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    slot = _alloc_slot(batch)
    rid = str(getattr(req, "rid", "?"))
    if slot is None:
        n = _count("no_slot")
        if _log_due(n):
            logger.info("WEG2 TURN-ANCHOR SKIP rid=%s reason=no_free_slot turn=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
        return desc
    if desc is None:
        desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
    # the slot stays a tensor: no device read on the scheduler thread
    desc.add(row, slot, t, prefix, end - prefix)
    setattr(req, PENDING_ATTR, (t, slot, desc, int(prefix), int(end)))
    n = _count("planned")
    if _log_due(n):
        logger.info(
            "WEG2 TURN-ANCHOR TRACK n=%d rid=%s step=[%d,%d) turn=%d boundary=%s "
            "prompt=%d main=%s (second extend track: the state where the prompt's "
            "last message starts, for the next turn and side requests)",
            n, rid, prefix, end, t, _boundary_of(req, tok), len(req.origin_input_ids),
            main_track,
        )
    return desc


def _note_twin_steps(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                     prefix: int, end: int, main_track: Optional[int], chunk: int,
                     page: int, bounds: Sequence[int], turn: Optional[int]) -> Optional[TurnTracks]:
    """TWIN ANCHOR: one extra track per twin boundary ``B`` inside this step
    (same grid, below the main track, above the protected prefix, not the
    turn position, which the turn track writes anyway). Plans are kept on
    ``TWIN_PENDING_ATTR`` ascending; the tree inserts them with the turn plan
    in position order before the step's own insert."""
    from sglang.srt.weg2 import twin_anchor as _tw

    protected = int(getattr(req, "cache_protected_len", 0) or 0)
    plans = []
    rid = str(getattr(req, "rid", "?"))
    for b in sorted({int(x) for x in bounds}):
        if turn is not None and b == turn:
            _tw.note_planned(rid, b)  # the turn track writes this very position
            continue
        if b <= protected or b % max(1, int(page)) != 0:
            continue
        if step_target(prefix, end, b, chunk, main_track) is None:
            continue
        slot = _alloc_slot(batch)
        if slot is None:
            n = _count("twin_no_slot")
            if _log_due(n):
                logger.info("WEG2 TWIN-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
                            "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, b, prefix, end, n)
            _tw.note_declined(rid, b)
            continue
        if desc is None:
            desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
        desc.add(row, slot, b, prefix, end - prefix)
        plans.append((b, slot, desc, int(prefix), int(end)))
        _tw.note_planned(rid, b)
        n = _count("twin_planned")
        if _log_due(n):
            logger.info(
                "WEG2 TWIN-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d main=%s (extra "
                "extend track at a queued twin's shared boundary: the twin resumes here "
                "instead of the previous chunk anchor)", n, rid, prefix, end, b, main_track)
    if plans:
        setattr(req, TWIN_PENDING_ATTR, plans)
    return desc


def fork_target(prefix: int, end: int, fork: int, prompt_len: int, chunk: int,
                page: int, main_track: Optional[int]) -> Optional[int]:
    """FORK TRACK position of the step ``[prefix, end)`` for the told fork
    ``fork``, or None. The deepest position ``<= min(fork, prompt_len - 1)``
    that is page-aligned (a reader's claim ends on a page, the same floor as
    P-FORK-CUT's ``fork_cut``) and on the step's FLA grid, strictly inside the
    step and below its own track (:func:`step_target`). Pure."""
    page, chunk = max(1, int(page)), int(chunk)
    if chunk <= 0 or int(fork) <= int(prefix):
        return None
    t = (min(int(fork), int(prompt_len) - 1) // page) * page
    # page and grid both: at most chunk / gcd(page, chunk) page steps down
    for _ in range(max(1, chunk)):
        if t <= int(prefix):
            return None
        if (t - int(prefix)) % chunk == 0:
            return step_target(prefix, end, t, chunk, main_track)
        t -= page
    return None


def _note_fork_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                    prefix: int, end: int, main_track: Optional[int], chunk: int,
                    page: int, fork: int) -> Optional[TurnTracks]:
    """FORK TRACK (NF y5k, 30.09.; P PP0 22:12:15, weg2-0-4).

    PP0's told fork (``req._weg2_fork_told``, weg2/p_fork_cut.py) is the depth
    to which the store holds this prompt's KV -- where the NEXT request with
    this prefix will stop reading. P-FORK-CUT ends a chunk there only when the
    cut is free; a follow-up turn is ONE forward, so the cut is ``paid`` and
    nothing ever anchored the fork: 'P-FORK-CUT TOLD fork=12544 src=store',
    the step ran [2560, 14096) through it, '#1028B FETCH CAP kv=196 claimed=40
    lost=156 ... MAMBA=(1, 39)' -- 9984 readable KV tokens recomputed, and
    again by every later request with the same fork.

    One more track in the same forward (the turn anchor's machinery: one
    gather row per GDN/PLE layer, one mamba slot from the pool, no eviction,
    no reserve), inserted as its own node before the step's insert and
    published with the chain. Rank-uniform: the told fork is PP0's verdict,
    set alike on every stage at admission. Skipped where the turn or a twin
    track already writes the same position."""
    t = fork_target(prefix, end, fork, len(getattr(req, "origin_input_ids", None) or ()),
                    chunk, page, main_track)
    if t is None:
        return desc
    rid = str(getattr(req, "rid", "?"))
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    turn = getattr(req, PENDING_ATTR, None)
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    if turn is not None:
        taken.add(int(turn[0]))
    if t in taken:
        n = _count("fork_shared")
        if _log_due(n):
            logger.info("WEG2 FORK-ANCHOR SHARED rid=%s at=%d fork_told=%d step=[%d,%d) "
                        "(n=%d; the turn/twin track writes this very position)",
                        rid, t, fork, prefix, end, n)
        return desc
    slot = _alloc_slot(batch)
    if slot is None:
        n = _count("fork_no_slot")
        if _log_due(n):
            logger.info("WEG2 FORK-ANCHOR SKIP rid=%s reason=no_free_slot at=%d fork_told=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)",
                        rid, t, fork, prefix, end, n)
        return desc
    if desc is None:
        desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
    desc.add(row, slot, t, prefix, end - prefix)
    setattr(req, FORK_PENDING_ATTR, (t, slot, desc, int(prefix), int(end)))
    n = _count("fork_planned")
    if _log_due(n):
        logger.info(
            "WEG2 FORK-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d fork_told=%d main=%s "
            "(extra extend track at the store fork the step runs through: the next "
            "request with this prefix reads its KV up to here instead of recomputing "
            "from the previous anchor)", n, rid, prefix, end, t, fork, main_track)
    return desc


def has_pending(req: Any) -> bool:
    """A turn, twin or fork plan waits for the tree's next insert of ``req``."""
    return (getattr(req, PENDING_ATTR, None) is not None
            or bool(getattr(req, TWIN_PENDING_ATTR, None))
            or getattr(req, FORK_PENDING_ATTR, None) is not None)


def pop_all_pending(req: Any) -> List[Tuple[str, Any]]:
    """Every pending plan of ``req`` as ``(kind, plan)``, ascending by position
    (the tree inserts the lower node first); both attributes cleared."""
    out: List[Tuple[str, Any]] = []
    turn = pop_pending(req)
    if turn is not None:
        out.append(("turn", turn))
    twins = getattr(req, TWIN_PENDING_ATTR, None) or ()
    if twins:
        setattr(req, TWIN_PENDING_ATTR, None)
        out.extend(("twin", p) for p in twins)
    fork = getattr(req, FORK_PENDING_ATTR, None)
    if fork is not None:
        setattr(req, FORK_PENDING_ATTR, None)
        out.append(("fork", fork))
    out.sort(key=lambda kp: int(kp[1][0]))
    return out


def _boundary_of(req: Any, tok: int) -> Optional[int]:
    try:
        return boundary(req.origin_input_ids, tok)
    except Exception:  # noqa: BLE001 - an instrument
        return None


def slot_state_kinds(pool: Any) -> Tuple[str, ...]:
    """The per-slot states besides GDN conv/ssm the second track must write:
    the Qwen4-Exp PLE side states when their pools are enabled (the same two
    ``MambaPool.register_slot_state`` carries). Fixed at PLAN time, so a
    forward whose model code did not run in Python (a captured graph) leaves
    them unmarked and the plan is refused -- never inserted half-written.

    P-TURN-REUSE (NF y3u, 30.09.; P log ...0930_002717): the n-gram history
    is required only on the stage that KEEPS it. The pool carries it on every
    PP stage (``ngram_context_len`` is not stage-filtered in
    model_runner_kv_cache_mixin, the short-conv layer ids are), but only the
    stage that owns a PLE layer builds and commits the PLE batch
    (``Qwen4ExpModel._stage_has_ple``) -- on NF's P (PP3) that is PP0 alone.
    The PLE layers ARE the short-conv layers (config ``short_conv_layer_ids``
    = ``ple_layer_ids - 1``), so a stage whose short-conv pool is off owns no
    PLE layer, and no forward there writes an n-gram row for ANY track (the
    default track neither). Requiring it anyway refused every turn anchor on
    PP1/PP2 ('TURN-ANCHOR SKIP reason=unmarked:ple_ngram', 100 % in y3r/y3t/
    y3u), PP0's anchor stayed a partial arena slot, and the next turn read
    only to the older anchor: weg2-2-8 '#1028B FETCH CAP kv=1089 claimed=941
    ... mamba (11, 940)', 9528 tokens re-prefilled on P."""
    kinds = []
    scp = getattr(pool, "short_conv_pool", None)
    stage_keeps_ple = scp is None or bool(getattr(scp, "enabled", False))
    if scp is not None and bool(getattr(scp, "enabled", False)):
        kinds.append("ple_conv")
    ngp = getattr(pool, "ngram_pool", None)
    if ngp is not None and bool(getattr(ngp, "enabled", False)):
        if stage_keeps_ple:
            kinds.append("ple_ngram")
        else:
            n = _count("ngram_not_on_stage")
            if _log_due(n):
                logger.info("WEG2 TURN-ANCHOR NGRAM-OFF-STAGE n=%d: this PP stage owns no PLE "
                            "layer (short-conv pool off); its n-gram rows are written by no "
                            "track, so the turn plan does not require them", n)
    return tuple(kinds)


def _alloc_slot(batch: Any):
    try:
        alloc = batch.req_to_token_pool.mamba_allocator
    except AttributeError:
        return None
    try:
        slot = alloc.alloc(1)
    except Exception:  # noqa: BLE001 - an accelerator, never a wall
        return None
    return slot


def _free_slot(batch_or_pool: Any, slot) -> None:
    if slot is None:
        return
    pool = getattr(batch_or_pool, "req_to_token_pool", batch_or_pool)
    try:
        pool.mamba_allocator.free(slot)
    except Exception:  # noqa: BLE001
        logger.warning("WEG2 TURN-ANCHOR slot free raised", exc_info=True)


def pop_pending(req: Any):
    p = getattr(req, PENDING_ATTR, None)
    if p is not None:
        setattr(req, PENDING_ATTR, None)
    return p


# -- the forward side ----------------------------------------------------------------
def forward_rows(forward_batch: Any) -> Optional[TurnTracks]:
    """The batch's descriptor when the forward may add its rows: an extend
    (not a verify) whose geometry is the planned one."""
    tt = getattr(forward_batch, "weg2_turn_tracks", None)
    if tt is None or not len(tt):
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    try:
        if mode is None or not mode.is_extend() or mode.is_target_verify():
            return None
    except Exception:  # noqa: BLE001
        return None
    if not tt.geometry_matches(
        getattr(forward_batch, "batch_size", -1),
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        getattr(forward_batch, "extend_seq_lens_cpu", None),
    ):
        n = _count("geometry")
        if _log_due(n):
            logger.warning("WEG2 TURN-ANCHOR SKIP reason=geometry: the forward's batch is "
                           "not the planned one (bs=%s planned=%d); no rows added (n=%d)",
                           getattr(forward_batch, "batch_size", "?"), tt.bs, n)
        return None
    return tt


def insert_verdict(pend) -> Tuple[bool, str]:
    """(insert?, reason) for a consumed plan."""
    t, slot, desc, prefix, end = pend
    if desc is None or slot is None:
        return False, "no_plan"
    if not desc.complete():
        missing = sorted(desc.need - desc.done)
        return False, "unmarked:" + ",".join(missing)
    return True, "ok"


def _label(kind: str) -> str:
    return {"twin": "TWIN-ANCHOR", "fork": "FORK-ANCHOR"}.get(kind, "TURN-ANCHOR")


def _kind_prefix(kind: str) -> str:
    return kind + "_" if kind in ("twin", "fork") else ""


def note_insert(req: Any, t: int, prefix_len: int, taken: bool, prompt: int,
                step: Tuple[int, int], kind: str = "turn") -> None:
    label = _label(kind)
    n = _count(_kind_prefix(kind) + ("inserted" if taken else "exists"))
    from sglang.srt.weg2 import twin_anchor as _tw

    if kind == "twin" or _tw.status(str(getattr(req, "rid", "?")), int(t)) is not None:
        _tw.note_written(str(getattr(req, "rid", "?")), int(t))
    if _log_due(n):
        logger.info(
            "WEG2 %s INSERT n=%d rid=%s at=%d step=[%d,%d) prompt=%d "
            "prefix_len=%d value=%s counts=%s",
            label, n, str(getattr(req, "rid", "?")), t, step[0], step[1], prompt, prefix_len,
            "taken" if taken else "exists", counts(),
        )


def note_skip(req: Any, reason: str, t: int, kind: str = "turn") -> None:
    label = _label(kind)
    n = _count(_kind_prefix(kind) + "skip:" + reason.split(":")[0])
    from sglang.srt.weg2 import twin_anchor as _tw

    if kind == "twin" or _tw.status(str(getattr(req, "rid", "?")), int(t)) is not None:
        _tw.note_declined(str(getattr(req, "rid", "?")), int(t))
    if _log_due(n):
        logger.info("WEG2 %s SKIP rid=%s reason=%s turn=%d (n=%d) -- slot freed",
                    label, str(getattr(req, "rid", "?")), reason, t, n)
