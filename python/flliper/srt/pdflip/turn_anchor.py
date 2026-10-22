# SPDX-License-Identifier: Apache-2.0
"""TURN ANCHOR (NF, 29.09.; y3m boot ...dauer09292136, P and front logs).

THE MEASURED GAP. A hybrid (GDN) reader resumes only at a recurrent anchor at
or below its fork. Group P anchors a prompt at its chunk ends and at its END
(``floor_page(N)``, H63 fold / #1233). A Claude-Code session forks at two
places, and only one of them had an anchor:

* a side request forks at the main turn's generation prompt (``N-4``) --
  the END anchor serves it (and CLAIM ANCHOR where ``N % page`` is 0/1);
* the NEXT turn forks at the start of the previous request's LAST message
  (front ``SESSION-PREFIX``: pdflip-50-66 left pdflip-46-61 at common=64352 of
  64515, N-163; the gaps of 125-171 tokens are that last message). There P
  had no anchor: 46-61's END anchor 64512 lies past the fork, the read fell
  back to the previous chunk anchor -- P 22:02:22 pdflip-50-71 ``#1028B FETCH
  CAP kv=1005 claimed=989 lost=16 MAMBA (11, 988)``, and on a fresh session
  (turn 1 anchored only at its END ~14912) ``#1035c ZERO-ANSWER cause=CAPPED
  by=mamba``. The same boundary is where every NEW session forks from an old
  one: the end of the shared system prompt + tools is the start of the first
  user message.

WHY NOT A CHUNK CUT (P-FORK-CUT, pdflip/p_fork_cut.py). A cut there is only
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

  * ``FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN=<id>`` (launcher ``--turn-anchor-token``,
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
(``req._pdflip_fork_told``, pdflip/p_fork_cut.py) when P-FORK-CUT found the cut
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

PENDING_ATTR = "_pdflip_turn_pending"
POS_ATTR = "_pdflip_turn_pos"
#: TWIN ANCHOR (30.09., pdflip/twin_anchor.py): the step's twin-boundary tracks,
#: a list of plans of the same shape as PENDING_ATTR's, ascending by position
TWIN_PENDING_ATTR = "_pdflip_twin_pending"
#: FORK TRACK (30.09., NF y5k pdflip-0-4): the step's track at PP0's told fork
#: (``req._pdflip_fork_told``, P-FORK-CUT) when the cut was not free -- one plan
#: of the same shape as PENDING_ATTR's
FORK_PENDING_ATTR = "_pdflip_fork_pending"
#: GEN-FORK TRACK (S57, 19.10., NF rc15 pdflip-42-262): the step's track at the
#: generation prompt's fork ``floor_page(g)`` when the step's own (END) track lies
#: past it -- one plan of the same shape as PENDING_ATTR's
GEN_PENDING_ATTR = "_pdflip_gen_pending"
#: SYS-GRID TRACK (S62 4b, 19.10., NF rc15 pdflip-2-8): the step's tracks on the
#: first message's grid (system prompt + tool list), a list of plans of the same
#: shape as PENDING_ATTR's, ascending by position
SYSGRID_PENDING_ATTR = "_pdflip_sysgrid_pending"
#: CLOSE-FORK TRACK (20.10., NF rc27 pdflip-50-825 / 51-826): the step's track at the
#: close of the message BEFORE the last one, ``floor_page(e - 1)``, when the turn
#: track ``floor_page(i)`` lies past it -- one plan of the same shape as PENDING_ATTR's
CLOSE_PENDING_ATTR = "_pdflip_close_pending"
#: IMAGE-END TRACK (21.10., NF rc30 pdflip-127-1253 / 130-1258): the step's track at the first
#: page boundary at or past the end of the step's last image, ``ceil_page(e)``, when the step's
#: own track lies past it -- one plan of the same shape as PENDING_ATTR's
IMAGE_PENDING_ATTR = "_pdflip_image_pending"
#: BRANCH TRACK (NF rc32 22.10., D-FORK-ANCHOR review A2): group D's branching point (the match's
#: ``mamba_branching_seqlen`` on the state-holding rank) as a SECOND track of the step instead of the
#: step's own track -- one plan of the same shape as PENDING_ATTR's
BRANCH_PENDING_ATTR = "_pdflip_branch_pending"
#: the branching point handed from the extend-track setup (``ScheduleBatch._mamba_radix_cache_v2_req_prepare_
#: for_extend``) to :func:`note_step` of the same step; consumed (cleared) by note_step
BRANCH_POS_ATTR = "_pdflip_d_branch_at"
#: HEAD TRACK (22.10., NF rc32 pdflip-14-469, Kopf-Anker): the step's tracks at the '</system-reminder' closes
#: of the prompt's FIRST user message (the Claude-Code head), a list of plans of the same shape as PENDING_ATTR's,
#: ascending by position
HEAD_PENDING_ATTR = "_pdflip_head_pending"

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
        from flliper.srt.environ import envs

        tok = envs.FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN.get()
    except Exception:  # noqa: BLE001 - unreadable = off
        return None
    try:
        tok = int(tok) if tok is not None else None
    except (TypeError, ValueError):
        return None
    return tok if tok is not None and tok > 0 else None


def _group() -> str:
    return (os.environ.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper()


def d_fork_anchor_on() -> bool:
    """``FLLIPER_PDFLIP_D_FORK_ANCHOR`` (default ON): the fork tracks on group D.

    Review F6 (H99, no fallback): an unreadable value is NOT turned into a default here -- one rank answering True
    from an exception while its peers read the env would split ``d_mode``; the error propagates and stops by name."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_D_FORK_ANCHOR.get())


def d_mode() -> bool:
    """D-FORK-ANCHOR: True on a group-D process with the token and the switch -- the plans are POSITION plans
    (every rank of the group plans the same positions from the same ids) and a plan without a written state is
    inserted SPLIT-ONLY instead of dropped (``_pdflip_turn_insert_one``), so every rank's tree has the same
    nodes. Env reads only (the same answer on every rank of the group)."""
    return _group() == "D" and token() is not None and d_fork_anchor_on()


def d_split_only_rank() -> bool:
    """D-FORK-ANCHOR: a rank that holds no recurrent state (a Form A expert worker, a KV-only rank) plans
    every fork track SPLIT-ONLY -- no slot, no gather row, the node ``key[:t]`` without a mamba value; its
    anchors are bookkeeping and it follows the host's depth with the mamba rule suspended (H98).

    Review A4 (21.10.), DELIBERATE ASYMMETRY: on a D fork node the worker's mamba bookkeeping does NOT mirror TP0's
    (TP0: the state, workers: none). Consequences, all accepted by design: (a) the #639b premise "same count,
    replicated LRU, same tombstone set" (mem_cache/common.py peer_needs_mamba_evict) does not hold for fork nodes --
    harmless on Form A, a worker in the follow walk reads no mamba validity; (b) an R12 STATE of TP0 that names an
    anchor host copy on a fork node reconciles to ``short`` on the worker (form_a_host_shadow._reconcile) -- EXPECTED
    for fork anchors, no action; (c) TP0 holds one more slot per taken fork: the review A1 guard
    (UnifiedRadixCache._d_fork_donation_room) keeps it from ever costing the step's donation that slot, so the step
    node is inserted on every rank alike. E1 (workers draw a byteless mirror slot) was weighed and not built: a worker
    runs no GDN forward, so its "marks" would be a second, unverified verdict, and TP0-only slots exist anyway
    (S109R device reads) -- the guard covers the one place the asymmetry could split the tree."""
    # No bare except (H99 audit, the bare-except trap): an undecidable rank form is a named stop in
    # ``form_a_follow_active`` -- swallowing it into "holds state" would let this rank plan a slot its peers do not.
    from flliper.srt.managers.tp_match_floor import this_rank_follows
    from flliper.srt.rank_role import kv_only_rank

    return bool(this_rank_follows()) or bool(kv_only_rank())


def d_branch_rides(server_args: Any, page: Any, chunk: Any, has_output: bool) -> bool:
    """BRANCH TRACK (NF rc32 22.10., review A2 of nf-rc31-forkanchor): True when the step's branching point goes
    to a SECOND track and the step keeps its own track at the step-end grid point.

    THE MEASURED GAP (rc32 ded00bd5a7, boot ..._1022_030836, D): upstream's extend-track setup lets the match's
    branching point REPLACE the step's own track (``mamba_track_seqlen_aligned = mamba_branching_seqlen``). The
    branching point exists only on the state-holding rank (TP0); a Form A worker's follow walk drops it
    (``follow_rematch``, H98) -- ``#59 FINISH-TRACK ... branching=57088`` on TP0, ``branching=-`` on TP1/TP2
    for pdflip-4-121. So TP0's main track was rank-local, :func:`step_target` refused every fork position at or
    above it on TP0 alone, and the D fork plan set split: 7 of 7 branching readers of that boot carried a
    worker turn plan TP0 never made (pdflip-0-13 / 2-61 / 4-90 / 4-121 / 4-143 / 4-149 / 6-170), and the step-end
    anchor itself was lost on TP0 (pdflip-4-143 resumed on 4-141's branching anchor 101760 instead of its
    step-end anchor 102784: 1024 tokens recomputed).

    THE REPAIR: on group D with the fork tracks armed, the step's own track stays where every rank puts it (a pure
    function of the step's geometry), and the branching point rides the fork-track machinery as one more plan
    (:func:`_note_branch_step`): a slot when one is free, the node at the branching point in every case. The node
    needs no worker plan: the branching point is the page-aligned end of the KV match, so every rank's tree has a
    node boundary there once the step inserts (the divergence split, or the leaf end it extends) -- exact when the
    page is a multiple of the FLA chunk (``branching = floor_chunk(reached) == reached``); any other geometry keeps
    upstream's replacement. A request with output (a retracted re-prefill: :func:`note_step` plans nothing for
    it) keeps upstream's replacement too. Env reads and geometry only: the same answer on every rank."""
    if has_output:
        return False
    if _group() != "D" or armed(server_args) is None:
        return False
    try:
        page, chunk = int(page), int(chunk)
    except (TypeError, ValueError):
        return False
    return page > 0 and chunk > 0 and page % chunk == 0


def armed(server_args: Any = None) -> Optional[int]:
    """The token id when the turn anchor runs on THIS process, else None.

    Group P (the group that writes the END anchor the next turn cannot use) with extra_buffer tracks without a
    checkpoint interval (the track form the second row joins) and no overlap schedule.

    D-FORK-ANCHOR (21.10.): group D too (``d_mode``), WITH the overlap schedule. THE ORDERING, proven from the
    code: (a) the second row is appended to the SAME gather tensors as the step's own track
    (``GDNAttnBackend._pdflip_turn_rows`` extends ``track_ssm_h_src/dst`` and ``track_conv_indices`` in
    ``init_forward_metadata``), so ONE kernel launch writes the step's slot and the fork slot; (b) the marks
    are set in that same Python call, i.e. inside ``run_batch`` of the step's batch, and the plan is consumed
    only by the request's next tree insert (``cache_unfinished_req`` at the next batch's scheduling or in
    ``process_batch_result``, ``cache_finished_req``) -- both after ``run_batch`` returned on this thread;
    (c) the fork node is inserted in the SAME insert call, immediately before the step's own node, with no GPU
    read in between (insert / match / lock are bookkeeping). So every reader ordering that is correct for the
    step's own track slot under overlap is correct for the fork slot. A MIXED-D batch (a short D prefill that
    takes the running decodes along, ``ScheduleBatch.mix_with_running``) keeps its plan: the tails join behind
    the planned rows (``TurnTracks.append_tail_rows``). A plan consumed without its marks (a
    batch that never ran, a geometry change, a captured graph) is inserted split-only and its slot freed: no
    anchor, no error."""
    tok = token()
    if tok is None:
        return None
    grp = _group()
    if grp == "D":
        if not d_fork_anchor_on():
            return None
        if server_args is None:
            try:
                from flliper.srt.runtime_context import get_server_args

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
        return tok
    if grp != "P":
        return None
    if server_args is None:
        try:
            from flliper.srt.runtime_context import get_server_args

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
    from flliper.srt.pdflip import fork_anchor as _fa

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

    def append_tail_rows(self, n: int) -> None:
        """MIXED-D (``ScheduleBatch.mix_with_running``): ``n`` decode tails join the batch BEHIND every planned
        row. The planned rows keep their index, prefix and extend length, and the forward's track math
        (``h_rows`` / ``conv_starts``, upstream ``_init_track_ssm_indices``) offsets a row only by the rows
        BEFORE it, so the plan stays exact; only the row count the forward checks grows. NF rc31b D TP0: 62 of
        62 ``split_only why=unmarked:gdn,ple_conv,ple_ngram`` were plans of a mixed forward, 64 of 64 ``taken``
        plans of an unmixed one -- the merge had dropped the descriptor."""
        n = int(n)
        if n < 0:
            raise ValueError(f"TurnTracks.append_tail_rows: negative tail count {n}")
        self.bs += n

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
    (TWIN ANCHOR, pdflip/twin_anchor.py) one more track per twin boundary
    ``twin_bounds`` inside it, and (FORK TRACK) one at PP0's told fork
    ``fork_told`` when this step runs through it. Returns the (possibly new)
    batch descriptor."""
    # BRANCH TRACK: the branching point this step's track setup handed over -- consumed here, never left standing
    branch_at = getattr(req, BRANCH_POS_ATTR, None)
    if branch_at is not None:
        setattr(req, BRANCH_POS_ATTR, None)
    stale = getattr(req, PENDING_ATTR, None)
    if stale is not None:
        # a plan the tree never consumed (a path that skipped the insert):
        # the slot goes back before a new one is drawn
        _free_slot(batch, stale[1])
        _decline_unconsumed(req, stale[0])
        setattr(req, PENDING_ATTR, None)
        n = _count("stale")
        if _log_due(n):
            logger.warning("PDFLIP TURN-ANCHOR STALE rid=%s: an unconsumed plan's slot "
                           "freed (n=%d)", str(getattr(req, "rid", "?")), n)
    for st_pend in getattr(req, TWIN_PENDING_ATTR, None) or ():
        _free_slot(batch, st_pend[1])
        _decline_unconsumed(req, st_pend[0])
    if getattr(req, TWIN_PENDING_ATTR, None):
        setattr(req, TWIN_PENDING_ATTR, None)
    fk_stale = getattr(req, FORK_PENDING_ATTR, None)
    if fk_stale is not None:
        _free_slot(batch, fk_stale[1])
        _decline_unconsumed(req, fk_stale[0])
        setattr(req, FORK_PENDING_ATTR, None)
    gn_stale = getattr(req, GEN_PENDING_ATTR, None)
    if gn_stale is not None:
        _free_slot(batch, gn_stale[1])
        _decline_unconsumed(req, gn_stale[0])
        setattr(req, GEN_PENDING_ATTR, None)
    for sg_pend in getattr(req, SYSGRID_PENDING_ATTR, None) or ():
        _free_slot(batch, sg_pend[1])
        _decline_unconsumed(req, sg_pend[0])
    if getattr(req, SYSGRID_PENDING_ATTR, None):
        setattr(req, SYSGRID_PENDING_ATTR, None)
    cl_stale = getattr(req, CLOSE_PENDING_ATTR, None)
    if cl_stale is not None:
        _free_slot(batch, cl_stale[1])
        _decline_unconsumed(req, cl_stale[0])
        setattr(req, CLOSE_PENDING_ATTR, None)
    im_stale = getattr(req, IMAGE_PENDING_ATTR, None)
    if im_stale is not None:
        _free_slot(batch, im_stale[1])
        _decline_unconsumed(req, im_stale[0])
        setattr(req, IMAGE_PENDING_ATTR, None)
    br_stale = getattr(req, BRANCH_PENDING_ATTR, None)
    if br_stale is not None:
        _free_slot(batch, br_stale[1])
        setattr(req, BRANCH_PENDING_ATTR, None)
    for hd_pend in getattr(req, HEAD_PENDING_ATTR, None) or ():
        _free_slot(batch, hd_pend[1])
        _decline_unconsumed(req, hd_pend[0])
    if getattr(req, HEAD_PENDING_ATTR, None):
        setattr(req, HEAD_PENDING_ATTR, None)
    if not track_mask or len(getattr(req, "output_ids", None) or ()) > 0:
        return desc
    if d_mode():
        # D-FORK-ANCHOR: the TURN, GEN-FORK, CLOSE-FORK and IMAGE-END tracks only (the twin / told-fork / sys-grid
        # tracks are P's -- their positions come from PP0's rows, not from the ids alone)
        twin_bounds, fork_told = (), 0
    desc = _note_turn_twin_steps(batch, desc, req, row, prefix, end, main_track,
                                 chunk, page, tok, twin_bounds)
    if fork_told:
        desc = _note_fork_step(batch, desc, req, row, prefix, end, main_track,
                               chunk, page, int(fork_told))
    if tok is not None and gen_fork_armed():
        desc = _note_gen_step(batch, desc, req, row, prefix, end, main_track,
                              chunk, page, tok)
    if tok is not None:
        # CLOSE-FORK TRACK: armed wherever the turn anchor is (``tok``), no own switch
        desc = _note_close_step(batch, desc, req, row, prefix, end, main_track,
                                chunk, page, tok)
    if tok is not None and sys_grid_armed() and not d_mode():
        desc = _note_sysgrid_steps(batch, desc, req, row, prefix, end, main_track,
                                   chunk, page, tok)
    if tok is not None:
        # IMAGE-END TRACK: armed wherever the turn anchor is (``tok``), no own switch
        desc = _note_image_step(batch, desc, req, row, prefix, end, main_track, chunk, page)
    if tok is not None and not d_mode():
        # HEAD TRACK (22.10.): armed wherever the turn anchor is on group P, no own switch; last, so a position
        # another track writes is shared, not doubled
        desc = _note_head_steps(batch, desc, req, row, prefix, end, main_track, chunk, page, tok)
    if branch_at is not None and d_mode():
        # BRANCH TRACK (review A2): the branching point the step's own track no longer takes, last so that a
        # fork position another track writes is shared, not doubled
        desc = _note_branch_step(batch, desc, req, row, prefix, end, main_track, chunk, page, int(branch_at))
    return desc


def _note_branch_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                      prefix: int, end: int, main_track: Optional[int], chunk: int,
                      page: int, branch_at: int) -> Optional[TurnTracks]:
    """BRANCH TRACK (NF rc32 22.10., review A2): the match's branching point as a second track of the step whose
    own track stays at the rank-agreed step end (:func:`d_branch_rides`). The same machinery and verdicts as the
    turn track: strictly inside the step, below the step's own track, on its FLA grid (``step_target``), page
    aligned, above the protected prefix, not a position another track already writes; a slot when one is free,
    split-only otherwise (no eviction, no reserve); inserted as its own node before the step's insert. Only the
    state-holding rank ever has a branching point, so only it plans one -- the node itself is a boundary on every
    rank's tree anyway (the KV match ends there), and its state is TP0's like every D fork state (review A4)."""
    rid = str(getattr(req, "rid", "?"))
    t = step_target(prefix, end, branch_at, chunk, main_track)
    if t is None or t % max(1, int(page)) != 0:
        n = _count("branch_off_step")
        if _log_due(n):
            logger.info("PDFLIP BRANCH-ANCHOR SKIP rid=%s reason=off_step at=%d step=[%d,%d) main=%s (n=%d; the "
                        "step's own track writes it or it is not on the step's grid)", rid, int(branch_at),
                        prefix, end, main_track, n)
        return desc
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    taken.update(int(p[0]) for p in (getattr(req, SYSGRID_PENDING_ATTR, None) or ()))
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR, GEN_PENDING_ATTR, CLOSE_PENDING_ATTR, IMAGE_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    if t in taken:
        n = _count("branch_shared")
        if _log_due(n):
            logger.info("PDFLIP BRANCH-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; another fork track writes "
                        "this very position)", rid, t, prefix, end, n)
        return desc
    slot, split_ok = _draw_slot(batch)
    if slot is None and not split_ok:
        return desc
    desc = _put_plan(batch, desc, req, row, BRANCH_PENDING_ATTR, t, slot, prefix, end)
    note_d_planned(req, "branch", t, prefix, end, slot)
    n = _count("branch_planned")
    if _log_due(n):
        logger.info(
            "PDFLIP BRANCH-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d main=%s state=%s (the match's branching "
            "point as a second extend track: the step's own track stays at the step end on every rank, review A2)",
            n, rid, prefix, end, t, main_track, "slot" if slot is not None else "split_only")
    return desc


def _decline_unconsumed(req: Any, pos: Any) -> None:
    """S52r (review 19.10.): a plan dropped unconsumed (its slot freed above) is never
    inserted -- a promise booked on it (twin boundary, or the S52 turn / fork track) is
    declined, so no late twin keeps waiting on an anchor that will not be written
    (p_twin_defer rule 5: declined = no promise, the twin registers at once). Ledger
    only; nothing is booked unless the twin anchor is armed, so a no-op otherwise."""
    from flliper.srt.pdflip import twin_anchor as _tw

    try:
        rid, p = str(getattr(req, "rid", "?")), int(pos)
    except (TypeError, ValueError):
        return
    if _tw.status(rid, p) == "planned":
        _tw.note_declined(rid, p)


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
    slot, split_ok = _draw_slot(batch)
    rid = str(getattr(req, "rid", "?"))
    if slot is None and not split_ok:
        n = _count("no_slot")
        if _log_due(n):
            logger.info("PDFLIP TURN-ANCHOR SKIP rid=%s reason=no_free_slot turn=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
        # S52: a twin boundary booked at this very position (``_note_twin_steps``: "the
        # turn track writes it") is NOT written now -- no promise is left standing on it
        from flliper.srt.pdflip import twin_anchor as _tw

        if _tw.status(rid, t) == "planned":
            _tw.note_declined(rid, t)
        return desc
    # the slot stays a tensor: no device read on the scheduler thread
    desc = _put_plan(batch, desc, req, row, PENDING_ATTR, t, slot, prefix, end)
    if not d_mode():
        # S52 (pdflip-1-30): the turn anchor is a promise to every twin with shared > t, also
        # one that reaches PP0 only after this step was planned (twin_anchor.deepest_track)
        from flliper.srt.pdflip import twin_anchor as _tw

        _tw.note_track_planned(rid, t)
    else:
        note_d_planned(req, "turn", t, prefix, end, slot)
    n = _count("planned")
    if _log_due(n):
        logger.info(
            "PDFLIP TURN-ANCHOR TRACK n=%d rid=%s step=[%d,%d) turn=%d boundary=%s "
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
    from flliper.srt.pdflip import twin_anchor as _tw

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
                logger.info("PDFLIP TWIN-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
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
                "PDFLIP TWIN-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d main=%s (extra "
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
    """FORK TRACK (NF y5k, 30.09.; P PP0 22:12:15, pdflip-0-4).

    PP0's told fork (``req._pdflip_fork_told``, pdflip/p_fork_cut.py) is the depth
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
            logger.info("PDFLIP FORK-ANCHOR SHARED rid=%s at=%d fork_told=%d step=[%d,%d) "
                        "(n=%d; the turn/twin track writes this very position)",
                        rid, t, fork, prefix, end, n)
        return desc
    slot = _alloc_slot(batch)
    if slot is None:
        n = _count("fork_no_slot")
        if _log_due(n):
            logger.info("PDFLIP FORK-ANCHOR SKIP rid=%s reason=no_free_slot at=%d fork_told=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)",
                        rid, t, fork, prefix, end, n)
        return desc
    if desc is None:
        desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
    desc.add(row, slot, t, prefix, end - prefix)
    setattr(req, FORK_PENDING_ATTR, (t, slot, desc, int(prefix), int(end)))
    # S52: the fork anchor is a promise to a later twin as well (twin_anchor.deepest_track)
    from flliper.srt.pdflip import twin_anchor as _tw

    _tw.note_track_planned(rid, t)
    n = _count("fork_planned")
    if _log_due(n):
        logger.info(
            "PDFLIP FORK-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d fork_told=%d main=%s "
            "(extra extend track at the store fork the step runs through: the next "
            "request with this prefix reads its KV up to here instead of recomputing "
            "from the previous anchor)", n, rid, prefix, end, t, fork, main_track)
    return desc


# -- GEN-FORK TRACK (S57) ----------------------------------------------------------
GEN_POS_ATTR = "_pdflip_gen_pos"


def gen_fork_armed() -> bool:
    """``FLLIPER_PDFLIP_GEN_FORK_ANCHOR`` (default ON, new features on): the
    generation-prompt fork track. Off = every path byte-identical to rc15.
    Read per call (an env read, no state): the same answer on every rank of
    one process group, which all boot with one env."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_GEN_FORK_ANCHOR.get())
    except Exception:  # noqa: BLE001 - unreadable = off
        return False


def gen_fork_pos(ids: Optional[Sequence[int]], tok: Optional[int], grain: int,
                 tail: int = GEN_TAIL) -> Optional[int]:
    """The generation-prompt fork position ``floor_grain(g)``.

    ``g`` is the index of the turn-start token that opens the prompt's
    generation prompt (``fork_anchor.fork_cut``: the last ``<|im_start|>``
    among the final ``tail`` ids, never the very last id). A SIBLING of this
    request -- the same chat up to and including that ``<|im_start|>``, then a
    different message instead of ``assistant`` -- shares ``ids[:g + 1]`` and
    claims at most ``floor_page(g)`` (bigram keys take the next token too, the
    bound turn/twin anchors sit on). Pure function of the ids and the env
    token: every PP rank derives the same position, nothing is transported.

    rc15 pdflip-40-260: N=49667, g=49662 -> 49600; its END anchor sat at
    floor_page(N-1)=49664, past the sibling pdflip-42-262's common=49663."""
    if tok is None or ids is None:
        return None
    from flliper.srt.pdflip import fork_anchor as _fa

    g = _fa.fork_cut(ids, tok, tail)
    if g is None:
        return None
    grain = max(1, int(grain))
    t = (int(g) // grain) * grain
    return t if t > 0 else None


def req_gen_pos(req: Any, tok: int, grain: int) -> Optional[int]:
    """``gen_fork_pos`` of a request's PROMPT, cached on the request."""
    ids = getattr(req, "origin_input_ids", None)
    if ids is None:
        return None
    key = (len(ids), int(tok), int(grain))
    cached = getattr(req, GEN_POS_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    t = gen_fork_pos(ids, tok, grain)
    try:
        setattr(req, GEN_POS_ATTR, (key, t))
    except Exception:  # noqa: BLE001
        pass
    return t


def _note_gen_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                   prefix: int, end: int, main_track: Optional[int], chunk: int,
                   page: int, tok: int) -> Optional[TurnTracks]:
    """GEN-FORK TRACK (S57, 19.10.; NF rc15 ..._1019_071429, pdflip-40-260 / 42-262).

    THE MEASURED GAP. 40-260 (N=49667) ran ONE step [39424, 49667). Its tracks:
    the END anchor at floor_page(N-1) = 49664 ('PDFLIP END-ANCHOR ... anchor=49664
    target=49666') and no turn track -- its last message started inside the
    step's first page, so the turn position floor_page(i) = 39424 was the step
    start itself (``step_target``: strictly inside). The sibling 42-262 shared
    common=49663 = N-4 (through the generation prompt's ``<|im_start|>`` at
    g = N-5) and claims floor_page(g) = 49600: the END anchor is ONE token past
    it, the store had the KV to 49600 ('PDFLIP-CLAIM-SHORT ... kv=775 claimed=616
    lost=159 by=mamba'), the Mamba anchor only at 39424 -> 10239 tokens
    recomputed on P ('RECOMPUTE-ECHT-P rid=pdflip-42-262').

    The END anchor serves a generation-prompt sibling only when
    floor_page(N-1) <= g, i.e. ``N % page`` not in {1, ..., N-1-g}: with NF's
    5-token generation prompt and page 64 it misses 4 of 64 prompt lengths.

    THE REPAIR: in exactly that geometry -- ``floor_page(g)`` strictly inside the
    step, below the step's own track, on its FLA grid (``step_target``) -- one
    more track in the same forward, the turn anchor's machinery (one gather row
    per GDN/PLE layer, one mamba slot from the pool, no eviction, no reserve;
    none free = no anchor, counted), inserted as its own node before the step's
    insert and published with the chain. Skipped where the turn, a twin or the
    fork track already writes the same position. A promise to a later twin
    like the turn/fork track (S52, ``twin_anchor.note_track_planned``)."""
    t = step_target(prefix, end, req_gen_pos(req, tok, page), chunk, main_track)
    if t is None or t % max(1, int(page)) != 0:
        return desc
    rid = str(getattr(req, "rid", "?"))
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    if t in taken:
        n = _count("gen_shared")
        if _log_due(n):
            logger.info("PDFLIP GEN-FORK-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; the "
                        "turn/twin/fork track writes this very position)", rid, t, prefix, end, n)
        return desc
    slot, split_ok = _draw_slot(batch)
    if slot is None and not split_ok:
        n = _count("gen_no_slot")
        if _log_due(n):
            logger.info("PDFLIP GEN-FORK-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
        return desc
    desc = _put_plan(batch, desc, req, row, GEN_PENDING_ATTR, t, slot, prefix, end)
    if not d_mode():
        from flliper.srt.pdflip import twin_anchor as _tw

        _tw.note_track_planned(rid, t)
    else:
        note_d_planned(req, "gen", t, prefix, end, slot)
    n = _count("gen_planned")
    if _log_due(n):
        logger.info(
            "PDFLIP GEN-FORK-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d prompt=%d main=%s "
            "(extra extend track at the generation prompt's fork: the step's own track lies "
            "past it, a sibling sharing the chat through <|im_start|> resumes here instead "
            "of the previous anchor)", n, rid, prefix, end, t,
            len(getattr(req, "origin_input_ids", None) or ()), main_track)
    return desc


# -- CLOSE-FORK TRACK (20.10.) -----------------------------------------------------
CLOSE_POS_ATTR = "_pdflip_close_pos"
#: ids between a message's close token and the next turn-start token: ``<|im_end|>``
#: ``\n`` on the Qwen chat template. Not trusted blindly: :func:`close_fork_pos` checks
#: that the last message and the generation prompt are opened behind the SAME two ids.
CLOSE_SEP = 2


def close_fork_pos(ids: Optional[Sequence[int]], tok: Optional[int], grain: int,
                   tail: int = GEN_TAIL) -> Optional[int]:
    """The close-fork position ``floor_grain(e - 1)``, ``e = i - CLOSE_SEP``.

    ``i`` is the index of the ``<|im_start|>`` that opens the prompt's LAST message
    (:func:`boundary`), ``e`` the index of the close token of the message before it.
    A SIBLING that continues that earlier message (a Claude-Code side request appends
    its instruction as one more content block to it, rc27 pdflip-51-826: PROMPT-DIVERGE
    at=29055, prev ``</system-reminder><|im_end|>\\n<|im_start|>user``, new
    ``</system-reminder>\\nDescribe ...``) shares ``ids[:e]`` and claims at most
    ``floor_page(e - 1)`` (bigram keys take the next token too -- the bound turn/twin/gen
    anchors sit on, ``twin_anchor.boundary(e)``). The turn anchor ``floor_page(i)``
    serves it only when ``floor_page(i) <= e - 1``; on page 64 it misses
    ``i % 64 in {0, 1, 2}``.

    The separator is verified, not assumed: ``ids[i - 2:i]`` must equal the two ids in
    front of the generation prompt's ``<|im_start|>`` (``fork_anchor.fork_cut``), which
    the template closes the same way, and neither may be the turn-start token. Pure
    function of the ids, the env token and the page: every PP rank derives the same
    position, nothing is transported."""
    i = boundary(ids, tok, tail)
    if i is None:
        return None
    from flliper.srt.pdflip import fork_anchor as _fa

    g = _fa.fork_cut(ids, tok, tail)
    e = int(i) - CLOSE_SEP
    if g is None or e < 1 or int(g) - CLOSE_SEP <= int(i):
        return None
    sep_last = tuple(int(x) for x in ids[e:int(i)])
    sep_gen = tuple(int(x) for x in ids[int(g) - CLOSE_SEP:int(g)])
    if sep_last != sep_gen or int(tok) in sep_last:
        return None
    grain = max(1, int(grain))
    t = ((e - 1) // grain) * grain
    return t if t > 0 else None


def req_close_pos(req: Any, tok: int, grain: int) -> Optional[int]:
    """``close_fork_pos`` of a request's PROMPT, cached on the request."""
    ids = getattr(req, "origin_input_ids", None)
    if ids is None:
        return None
    key = (len(ids), int(tok), int(grain))
    cached = getattr(req, CLOSE_POS_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    t = close_fork_pos(ids, tok, grain)
    try:
        setattr(req, CLOSE_POS_ATTR, (key, t))
    except Exception:  # noqa: BLE001
        pass
    return t


def _note_close_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                     prefix: int, end: int, main_track: Optional[int], chunk: int,
                     page: int, tok: int) -> Optional[TurnTracks]:
    """CLOSE-FORK TRACK (20.10.; NF rc27 ..._1020_145331, pdflip-50-825 / 51-826).

    THE MEASURED GAP. 50-825 (N=36064) ran [18240, 34624) + [34624, 36064) on P; its
    tracks: 'TURN-ANCHOR TRACK ... step=[18240,34624) turn=29056 boundary=29057', the
    chunk end 34624 and 'END-ANCHOR ... anchor=36032'. The side request 51-826 of the
    same session ('SESSION-PREFIX ... common=29055') shares ids[:29055] = up to the
    ``<|im_end|>`` at e = i - 2 and claims floor_page(29054) = 28992. P's own tree held
    the KV there ('MATCH-CENSUS-DEEP rid=pdflip-51-826 reached=28992 accepted=18240
    branching=28992') -- 50-825 had FINISHED before 51-826 was admitted, the store was
    not consulted for that span -- but the deepest recurrent anchor at or below 28992
    was the step start 18240: the turn anchor 29056 lay ONE token past the fork
    (i % 64 = 1). '#969 EXTENT ... ('pdflip-51-826', 18240, 28992, 18240, 10752)':
    10752 tokens recomputed that P's device tree held.

    THE REPAIR: in exactly that geometry -- floor_page(e - 1) strictly inside the step,
    below the step's own track, on its FLA grid (``step_target``), and not the position
    another track already writes -- one more track in the same forward (the turn
    anchor's machinery: one gather row per GDN/PLE layer, one mamba slot from the free
    list; none free = no anchor, counted; no eviction, no reserve), inserted as its own
    node before the step's insert and published with the chain. A promise to a later
    twin like the turn/gen tracks (S52 ledger). Armed wherever the turn anchor is."""
    t = step_target(prefix, end, req_close_pos(req, tok, page), chunk, main_track)
    if t is None or t % max(1, int(page)) != 0:
        return desc
    rid = str(getattr(req, "rid", "?"))
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR, GEN_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    if t in taken:
        n = _count("close_shared")
        if _log_due(n):
            logger.info("PDFLIP CLOSE-FORK-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; the "
                        "turn/twin/fork/gen track writes this very position)", rid, t, prefix,
                        end, n)
        return desc
    slot, split_ok = _draw_slot(batch)
    if slot is None and not split_ok:
        n = _count("close_no_slot")
        if _log_due(n):
            logger.info("PDFLIP CLOSE-FORK-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
        return desc
    desc = _put_plan(batch, desc, req, row, CLOSE_PENDING_ATTR, t, slot, prefix, end)
    if not d_mode():
        from flliper.srt.pdflip import twin_anchor as _tw

        _tw.note_track_planned(rid, t)
    else:
        note_d_planned(req, "close", t, prefix, end, slot)
    n = _count("close_planned")
    if _log_due(n):
        logger.info(
            "PDFLIP CLOSE-FORK-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d turn=%s prompt=%d "
            "main=%s (extra extend track at the close of the message before the last one: "
            "the turn anchor lies past it, a side request continuing that message resumes "
            "here instead of the previous anchor)", n, rid, prefix, end, t,
            req_anchor_pos(req, tok, page), len(getattr(req, "origin_input_ids", None) or ()),
            main_track)
    return desc


# -- IMAGE-END TRACK (21.10.) -------------------------------------------------------
def image_end_in(ids: Optional[Sequence[int]], lo: int, hi: int) -> Optional[int]:
    """One past the last image placeholder in ``ids[lo:hi]`` (a pad id >= ``MM_PAD_SHIFT_VALUE``:
    the scheduler's ids carry every image position as its item's pad value, the radix key), or
    None. A pure function of the ids: every PP rank derives the same position."""
    if ids is None:
        return None
    lo, hi = max(0, int(lo)), min(len(ids), int(hi))
    if hi <= lo:
        return None
    import numpy as np

    from flliper.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    try:
        seg = np.frombuffer(ids, dtype=np.int64)[lo:hi] if getattr(ids, "typecode", "") == "q" \
            else np.asarray(ids[lo:hi], dtype=np.int64)
    except (TypeError, ValueError):
        seg = np.asarray(list(ids[lo:hi]), dtype=np.int64)
    hit = np.flatnonzero(seg >= MM_PAD_SHIFT_VALUE)
    return None if hit.size == 0 else lo + int(hit[-1]) + 1


def image_end_pos(ids: Optional[Sequence[int]], prefix: int, end: int, grain: int) -> Optional[int]:
    """The image-end position ``ceil_grain(e)`` of the step ``[prefix, end)``: ``e`` = one past its
    last image placeholder. A reader sharing the prompt through ``ceil_grain(e)`` (bigram keys: one
    token more) claims it, and every image of the step then lies inside a recurrent anchor."""
    e = image_end_in(ids, prefix, end)
    if e is None:
        return None
    grain = max(1, int(grain))
    t = ((int(e) + grain - 1) // grain) * grain
    return t if t > 0 else None


def _note_image_step(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                     prefix: int, end: int, main_track: Optional[int], chunk: int,
                     page: int) -> Optional[TurnTracks]:
    """IMAGE-END TRACK (21.10.; NF rc30 ..._1d822bd150_1021_063252, session a4c8b097ac).

    THE MEASURED GAP. pdflip-127-1253 (N=63879, images=2, image_end=63714) ran ONE step
    [63680, 63879) on P ('#969 EXTENT ... (pdflip-127-1253, 63680, 63879, 63680, 199)'); its tracks:
    'END-ANCHOR ... anchor=63872' and nothing else -- the last message opens inside the step's
    first page, so turn and close positions round down onto the step start 63680 (``step_target``:
    strictly inside). P published 998 KV pages and the END anchor ('STORE-PRESENCE src=p_flush
    anchors=[pdflip-127-1253:63872]'). Its sibling pdflip-130-1258 (N=63880) shares the KV through
    page 996 and diverges inside page 997: 'PDFLIP-CLAIM-SHORT rid=pdflip-130-1258 asked=998 kv=996
    claimed=995 lost=1 short=3 by=mamba tokens=63680' -- the recurrent state exists only at 63680
    and 63872, so the claim falls back to 63680 < image_end 63714: the image lies in the uncached
    tail again (front 'D-PROBE ... covered=63680 ... image_in_tail=1'), P-only (SOLL R9) -> a second
    D->P flip for 200 tokens, 64 of them (63680..63744) recomputed although their KV was stored.

    THE REPAIR: one more track in the same forward at ``ceil_page(e)`` -- the first page boundary at
    or past the step's last image -- in exactly the geometry ``step_target`` admits (strictly inside
    the step, below the step's own track, on its FLA grid) and where no other track writes it (the
    turn anchor's machinery: one gather row per GDN/PLE layer, one mamba slot from the free list;
    none free = no anchor, counted; no eviction, no reserve), inserted as its own node before the
    step's insert and published with the chain. Every later request that shares the prompt through
    that page boundary -- the next turn, a sibling, a retry -- holds the image inside its covered
    prefix: it is priced as text and served by D, the image is never P-only again. A promise to a
    later twin like the turn/gen/close tracks (S52 ledger). Armed wherever the turn anchor is."""
    ids = getattr(req, "origin_input_ids", None)
    t = step_target(prefix, end, image_end_pos(ids, prefix, end, page), chunk, main_track)
    if t is None or t % max(1, int(page)) != 0:
        return desc
    rid = str(getattr(req, "rid", "?"))
    if int(getattr(req, "cache_protected_len", 0) or 0) >= t:
        return desc
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    taken |= {int(p[0]) for p in (getattr(req, SYSGRID_PENDING_ATTR, None) or ())}
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR, GEN_PENDING_ATTR, CLOSE_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    if t in taken:
        n = _count("image_shared")
        if _log_due(n):
            logger.info("PDFLIP IMAGE-END-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; another "
                        "track writes this very position)", rid, t, prefix, end, n)
        return desc
    # V1 (review of the rc31 merge with D-FORK-ANCHOR): on group D the image track is a POSITION plan like the
    # turn / close / gen tracks -- split-only on a Form A worker, split-only on a state rank without a free slot, its
    # slot in the G1 registry, no S52 ledger -- so whether the node at t exists never hangs on a rank-local free list
    slot, split_ok = _draw_slot(batch)
    if slot is None and not split_ok:
        n = _count("image_no_slot")
        if _log_due(n):
            logger.info("PDFLIP IMAGE-END-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
                        "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
        return desc
    desc = _put_plan(batch, desc, req, row, IMAGE_PENDING_ATTR, t, slot, prefix, end)
    if not d_mode():
        from flliper.srt.pdflip import twin_anchor as _tw

        _tw.note_track_planned(rid, t)
    else:
        note_d_planned(req, "image", t, prefix, end, slot)
    n = _count("image_planned")
    if _log_due(n):
        logger.info(
            "PDFLIP IMAGE-END-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d image_end=%s prompt=%d "
            "main=%s (extra extend track at the first page boundary past the step's last image: a "
            "later request sharing the prompt through it holds the image inside its covered prefix, "
            "D serves it -- no second flip)", n, rid, prefix, end, t, image_end_in(ids, prefix, end),
            len(ids or ()), main_track)
    return desc


# -- SYS-GRID TRACK (S62 4b) ------------------------------------------------------
SYSGRID_POS_ATTR = "_pdflip_sysgrid_pos"


def sys_grid_armed() -> bool:
    """``FLLIPER_PDFLIP_SYS_GRID_ANCHOR`` (default ON) and a positive grid. Off =
    every path byte-identical. An env read: one answer on every rank."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_SYS_GRID_ANCHOR.get()) and int(
            envs.FLLIPER_PDFLIP_SYS_GRID_TOKENS.get() or 0) > 0
    except Exception:  # noqa: BLE001 - unreadable = off
        return False


def _ids_env(name: str) -> Tuple[int, ...]:
    try:
        from flliper.srt.environ import envs

        raw = getattr(envs, name).get() or ""
        return tuple(int(x) for x in str(raw).replace(" ", "").split(",") if x)
    except Exception:  # noqa: BLE001 - unreadable = no tool snapping
        return ()


def sys_grid_config() -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """(grid, max points, tool-line head ids, name-field ids, MCP name prefix ids) from the env."""
    from flliper.srt.environ import envs

    return (int(envs.FLLIPER_PDFLIP_SYS_GRID_TOKENS.get() or 0),
            int(envs.FLLIPER_PDFLIP_SYS_GRID_MAX.get() or 0),
            _ids_env("FLLIPER_PDFLIP_SYS_GRID_TOOL_HEAD"),
            _ids_env("FLLIPER_PDFLIP_SYS_GRID_TOOL_NAME"),
            _ids_env("FLLIPER_PDFLIP_SYS_GRID_MCP_PREFIX"))


def _find(ids: Sequence[int], pat: Sequence[int], lo: int, hi: int) -> Optional[int]:
    """First index ``i`` in [lo, hi) with ``ids[i:i+len(pat)] == pat``, or None."""
    if not pat:
        return None
    first, n = int(pat[0]), len(pat)
    i = int(lo)
    while i < hi:
        try:
            i = ids.index(first, i, hi)
        except ValueError:
            return None
        if tuple(int(x) for x in ids[i:i + n]) == tuple(pat):
            return i
        i += 1
    return None


def segment_end(ids: Optional[Sequence[int]], tok: Optional[int]) -> Optional[int]:
    """Index ``u`` of the turn-start token that opens the prompt's SECOND message:
    ``ids[:u]`` is the first message (system prompt + tool list on a Claude-Code
    prompt). None when the prompt does not start with a turn-start token or has a
    single message."""
    if tok is None or ids is None or len(ids) < 2 or int(ids[0]) != int(tok):
        return None
    try:
        return int(ids.index(int(tok), 1))
    except ValueError:
        return None


def tool_group_bounds(ids: Sequence[int], lo: int, hi: int, head: Sequence[int],
                      name: Sequence[int], mcp: Sequence[int]) -> List[int]:
    """Indices of the tool lines in ``ids[lo:hi]`` whose GROUP differs from the line
    before (the first tool line included): built-in tools form one group, MCP tools
    one group per server (the name tokens after ``mcp`` up to its separator, the last
    id of ``mcp``). Where two Claude-Code tool lists differ -- another MCP server set,
    a server added or gone -- they fork at such a line (rc15 2-8: list86 -> list99 at
    the built-in -> MCP line, 'mcp__debugtools' vs 'mcp__claude_ai_Claude_Docs').
    Pure function of the ids; no tokenizer at run time."""
    if not head:
        return []
    heads: List[int] = []
    i = lo
    while True:
        j = _find(ids, head, i, hi)
        if j is None:
            break
        heads.append(j)
        i = j + len(head)
    out: List[int] = []
    prev = None
    sep = int(mcp[-1]) if mcp else None
    for k, b in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else hi
        group: Tuple[Any, ...] = ("?",)
        at = _find(ids, name, b + len(head), stop) if name else None
        if at is not None:
            p = at + len(name)
            if mcp and tuple(int(x) for x in ids[p:p + len(mcp)]) == tuple(mcp):
                q = p + len(mcp)
                srv = []
                while q < stop and len(srv) < 32 and int(ids[q]) != sep:
                    srv.append(int(ids[q]))
                    q += 1
                group = ("mcp", tuple(srv))
            else:
                group = ("builtin",)
        if group != prev:
            out.append(int(b))
        prev = group
    return out


def sys_grid_positions(ids: Optional[Sequence[int]], tok: Optional[int], page: int, grid: int,
                       max_points: int = 16, head: Sequence[int] = (), name: Sequence[int] = (),
                       mcp: Sequence[int] = ()) -> List[int]:
    """The SYS-GRID positions of a prompt, ascending, every one a multiple of ``page``
    and in (0, floor_page(u)] (u = :func:`segment_end`):

    * one point per grid window ``(G - grid, G]`` (G = grid, 2*grid, ... <= floor_page(u)):
      the LOWEST tool-group boundary (page-floored, :func:`tool_group_bounds`) inside the
      window, else G itself -- the count stays one per window, a tool-list fork at a group
      boundary is anchored exactly, any other fork loses at most two windows;
    * the lowest group boundary of the tail window ``(G_last, floor_page(u))``, if any;
    * ``floor_page(u)``: where a new first message forks off the same system prompt.

    More than ``max_points`` windows widen the grid (a power-of-two multiple of ``grid``,
    so the widened points land on the chunk ends) instead of adding points. Pure: ids, the env token, page and the config only."""
    u = segment_end(ids, tok)
    page = max(1, int(page))
    grid = (max(int(grid), page) // page) * page
    if u is None or grid <= 0:
        return []
    end_page = (int(u) // page) * page
    if end_page <= 0:
        return []
    if max_points > 0 and end_page // grid > max_points:
        k = -(-(end_page // grid) // int(max_points))
        # Review S62 (A4): a POWER-OF-TWO multiple, so a widened grid point falls on the
        # prefill chunk ends (16384 = 4 x 4096 on NF), which carry their chunk anchor
        # anyway. x3 (12288) missed them: a 150k-192k first message then held 13 grid
        # tracks PLUS 9-11 chunk anchors -- 22-27 arena anchors against the per-request
        # share of 21 (``MAMBA-ARENA ... cap=21``), i.e. SHARE displaced the request's
        # own shallowest anchors.
        grid *= 1 << (int(k) - 1).bit_length()
    bounds = sorted({(b // page) * page for b in tool_group_bounds(ids, 0, int(u), head, name, mcp)}
                    if head else set())
    bounds = [b for b in bounds if 0 < b < end_page]
    pts = set()
    prev = 0
    g = grid
    while g <= end_page:
        win = [b for b in bounds if prev < b <= g]
        pts.add(min(win) if win else g)
        prev = g
        g += grid
    tail = [b for b in bounds if prev < b < end_page]
    if tail:
        pts.add(min(tail))
    pts.add(end_page)
    return sorted(p for p in pts if p > 0)


def req_sysgrid_pos(req: Any, tok: int, grain: int) -> List[int]:
    """:func:`sys_grid_positions` of a request's PROMPT with the env config, cached on
    the request (one scan per request, not per step)."""
    ids = getattr(req, "origin_input_ids", None)
    if ids is None:
        return []
    try:
        grid, mx, head, name, mcp = sys_grid_config()
    except Exception:  # noqa: BLE001 - unreadable = no grid
        return []
    key = (len(ids), int(tok), int(grain), grid, mx, head, name, mcp)
    cached = getattr(req, SYSGRID_POS_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    pts = sys_grid_positions(ids, tok, grain, grid, mx, head, name, mcp)
    try:
        setattr(req, SYSGRID_POS_ATTR, (key, pts))
    except Exception:  # noqa: BLE001
        pass
    return pts


def _note_sysgrid_steps(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                        prefix: int, end: int, main_track: Optional[int], chunk: int,
                        page: int, tok: int) -> Optional[TurnTracks]:
    """SYS-GRID TRACK (S62 4b, 19.10.; NF rc15 ..._1019_071429, pdflip-0-2 / pdflip-2-8).

    THE MEASURED GAP. 2-8 forked from 0-2 inside the tool list ('PREFIX-DIVERGE-TEXT
    seg_role=system ... tools:list86 -> tools:list99', common=37859: the built-in tool
    'Write' ends, then 'mcp__debugtools__...' vs 'mcp__claude_ai_Claude_Docs__...'). 0-2's
    anchors: 64 / 16384 / 32768 / 49152 / turn 55872 / END 61312 -- the store had the KV to
    37824, the Mamba anchor at 32768: 5056 tokens recomputed by its first reader (P-FORK-CUT
    anchors the fork for the SECOND reader only). Nobody knows the fork before it happens.

    THE REPAIR: while the step computes the first message (system + tools), one more track
    per SYS-GRID position (:func:`req_sysgrid_pos`) strictly inside the step, below its own
    track, on its FLA grid -- the turn anchor's machinery (one gather row per GDN/PLE layer,
    one mamba slot from the free list; none free = that point skipped and counted, no
    eviction, no reserve), each inserted as its own node before the step's insert and
    published with the chain. A later prompt with the same system prefix hits the segment in
    the cache and runs no grid step: the cost is paid once per system-prefix variant. Skipped
    where a turn/twin/fork/gen track writes the same position. A promise to a later twin
    like the other tracks (S52)."""
    pts = req_sysgrid_pos(req, tok, page)
    if not pts:
        return desc
    rid = str(getattr(req, "rid", "?"))
    protected = int(getattr(req, "cache_protected_len", 0) or 0)
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR, GEN_PENDING_ATTR, CLOSE_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    from flliper.srt.pdflip import twin_anchor as _tw

    plans = []
    for t in pts:
        if step_target(prefix, end, t, chunk, main_track) is None:
            continue
        if t % max(1, int(page)) != 0 or t <= protected:
            continue
        if t in taken:
            n = _count("grid_shared")
            if _log_due(n):
                logger.info("PDFLIP SYS-GRID-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; another "
                            "track writes this very position)", rid, t, prefix, end, n)
            continue
        slot = _alloc_slot(batch)
        if slot is None:
            n = _count("grid_no_slot")
            if _log_due(n):
                logger.info("PDFLIP SYS-GRID-ANCHOR SKIP rid=%s reason=no_free_slot at=%d "
                            "step=[%d,%d) (n=%d; no eviction, no reserve)", rid, t, prefix, end, n)
            continue
        if desc is None:
            desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
        desc.add(row, slot, t, prefix, end - prefix)
        plans.append((t, slot, desc, int(prefix), int(end)))
        taken.add(t)
        _tw.note_track_planned(rid, t)
        n = _count("grid_planned")
        if _log_due(n):
            logger.info(
                "PDFLIP SYS-GRID-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d segment_end=%s main=%s "
                "(extra extend track on the first message's grid: a tool-list variant forking here "
                "reads its KV up to this anchor instead of the previous chunk anchor)",
                n, rid, prefix, end, t, segment_end(getattr(req, "origin_input_ids", None), tok),
                main_track)
    if plans:
        setattr(req, SYSGRID_PENDING_ATTR, plans)
    return desc


# -- HEAD TRACK (NF rc32 22.10., Kopf-Anker) -----------------------------------------
HEAD_POS_ATTR = "_pdflip_head_pos"
#: '</' 'system' '-' 'reminder' -- the close of a Claude-Code system-reminder block up to (not including) its '>',
#: which the tokenizer merges with whatever follows ('><' before the next block, '>' + '\n' before text). Verified
#: 22.10. against the NF (Qwen3.8-Flash-Next) and the Qwen3.8-27B tokenizer: the same four ids. A constant, not a
#: switch (order 22.10.: no off switch); another tokenizer simply never matches.
HEAD_CLOSE_IDS: Tuple[int, ...] = (510, 8678, 12, 75240)
#: at most this many head tracks per prompt, the lowest. A Claude-Code first user message carries 4 closes (rc32
#: pdflip-8-255, rebuilt from its transcript: instructions 17503, session context 18040, attribution 18175, the
#: SubagentHandback note 19468 -- the last one shares the turn track's page); 4 keeps a prompt whose SYS-GRID part
#: uses its slack (2 slots at a 64k / 131k / 250k system part against the arena share cap of 21) at most 2 over it
HEAD_MAX = 4


def head_positions(ids: Optional[Sequence[int]], tok: Optional[int], page: int,
                   max_points: int = HEAD_MAX) -> List[int]:
    """The HEAD positions of a prompt, ascending, distinct, at most ``max_points``: ``floor_page(b - 1)`` for every
    close ``ids[b-4:b] == HEAD_CLOSE_IDS`` inside the FIRST user message ``ids[u:v]`` (``u`` = :func:`segment_end`,
    the second turn-start; ``v`` = the next turn-start). ``b`` is the token that holds the close's ``>`` -- the first
    token a branching prompt may tokenize differently ('><system-reminder>' vs '>\nThis session ...'), so a reader
    that shares the head through the close shares ``ids[:b]`` and claims at most ``floor_page(b - 1)`` (the turn
    anchor's rule: a reader sharing ``ids[:i + 1]`` claims ``floor_page(i)``). The anchor sits exactly where that
    claim ends. Pure: the ids, the env token, the page and the constant close ids."""
    u = segment_end(ids, tok)
    if u is None:
        return []
    page = max(1, int(page))
    n = len(ids)
    try:
        v = int(ids.index(int(tok), int(u) + 1))
    except ValueError:
        v = n
    close = HEAD_CLOSE_IDS
    pts: List[int] = []
    i = int(u) + 1
    while len(pts) < int(max_points):
        j = _find(ids, close, i, v)
        if j is None:
            break
        b = j + len(close)
        if b > v:
            break
        t = ((b - 1) // page) * page
        if t > 0 and (not pts or t != pts[-1]):
            pts.append(t)
        i = b
    return pts


def req_head_pos(req: Any, tok: int, grain: int) -> List[int]:
    """:func:`head_positions` of a request's PROMPT, cached on the request (one scan per request, not per step)."""
    ids = getattr(req, "origin_input_ids", None)
    if ids is None:
        return []
    key = (len(ids), int(tok), int(grain), HEAD_CLOSE_IDS, HEAD_MAX)
    cached = getattr(req, HEAD_POS_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    pts = head_positions(ids, tok, grain)
    try:
        setattr(req, HEAD_POS_ATTR, (key, pts))
    except Exception:  # noqa: BLE001
        pass
    return pts


def _note_head_steps(batch: Any, desc: Optional[TurnTracks], req: Any, row: int,
                     prefix: int, end: int, main_track: Optional[int], chunk: int,
                     page: int, tok: int) -> Optional[TurnTracks]:
    """HEAD TRACK (NF rc32 22.10., boot ..._ded00bd5a7_1022_030836, pdflip-14-469: 6976 tokens recomputed on P).

    THE MEASURED GAP. A Claude-Code prompt's first user message is a run of system-reminder blocks (CLAUDE.md +
    memory, session context, attribution) and then the task text. Prompts of one head family branch at the END of
    such a block: a compaction keeps the instructions and the session context and replaces the rest with its summary
    ('PREFIX-DIVERGE-TEXT rid=pdflip-14-469 ... seg=1 seg_role=user ... before="...reported back.\n</system-reminder"
    prev_after="><system-reminder>\nAttribution ..." cur_after=">\nThis session is being continued ..."', common=18040),
    sibling agents share the head through the attribution block and the first words of the task. The front's
    PROMPT-DIVERGE saw 30 such 'moved' prompts in that boot: 9 left exactly at a close (17936 x8, 18040), 13 inside
    the text right after one, on the close's page (17977, 18098, 18102). The store holds their KV to that page; the
    recurrent state only where a P step ended or another track sat. pdflip-8-255 had written the head of the 18040
    family in ONE step [10816, 19792) with tracks at 11008 (fork), 19456 (turn) and 19776 (END) only, so 14-469
    claimed 172 of 281 pages ('PDFLIP-CLAIM-SHORT ... lost=109 by=mamba tokens=11008', 'S80 STORE-HOLE have=11008
    kv_end=17984 class=anchor'): 6976 tokens computed again. The FIRST branch at a close of a new head family pays;
    P-FORK-CUT / FORK TRACK anchor the fork only for the second reader.

    THE REPAIR: while a P step computes the first user message, one more track per head close
    (:func:`req_head_pos`) strictly inside the step, below its own track, on its FLA grid -- the turn anchor's
    machinery, exactly as the SYS-GRID track does for the system part (one gather row per GDN/PLE layer, one mamba
    slot from the free list; none free = that point skipped and counted, no eviction, no reserve, no timer), each
    inserted as its own node before the step's insert and published with the chain. A later prompt of the same head
    hits the head in the cache and runs no head step: the cost is paid once per head family (pdflip-8-255: three
    anchors 17472 / 17984 / 18112, 3 x 58834944 B). Skipped where another track writes the same position. A promise
    to a later twin like the other tracks (S52).

    GROUP P ONLY (R11): the positions are a pure function of the ids, so every PP stage plans the same ones; a stage
    without a free slot plans a subset, never another position. Group D is left out deliberately: at this base a D
    step's own track can be rank-local (TP0's branching point replaces it, review A2 of nf-rc31-forkanchor, fixed on
    desk/nf-rc32-forkanchor2-1022 as the 'branch' second track) and a head plan bounded by it would split the group's
    tree; the rc32 boot had 0 head branches recomputed on D (5 of 5 requests that prefilled through the head ran on
    P)."""
    pts = req_head_pos(req, tok, page)
    if not pts:
        return desc
    rid = str(getattr(req, "rid", "?"))
    protected = int(getattr(req, "cache_protected_len", 0) or 0)
    taken = {int(p[0]) for p in (getattr(req, TWIN_PENDING_ATTR, None) or ())}
    taken |= {int(p[0]) for p in (getattr(req, SYSGRID_PENDING_ATTR, None) or ())}
    for attr in (PENDING_ATTR, FORK_PENDING_ATTR, GEN_PENDING_ATTR, CLOSE_PENDING_ATTR, IMAGE_PENDING_ATTR):
        p = getattr(req, attr, None)
        if p is not None:
            taken.add(int(p[0]))
    from flliper.srt.pdflip import twin_anchor as _tw

    plans = []
    for t in pts:
        if step_target(prefix, end, t, chunk, main_track) is None:
            continue
        if t % max(1, int(page)) != 0 or t <= protected:
            continue
        if t in taken:
            n = _count("head_shared")
            if _log_due(n):
                logger.info("PDFLIP HEAD-ANCHOR SHARED rid=%s at=%d step=[%d,%d) (n=%d; another track writes "
                            "this very position)", rid, t, prefix, end, n)
            continue
        slot = _alloc_slot(batch)
        if slot is None:
            n = _count("head_no_slot")
            if _log_due(n):
                logger.info("PDFLIP HEAD-ANCHOR SKIP rid=%s reason=no_free_slot at=%d step=[%d,%d) (n=%d; no "
                            "eviction, no reserve)", rid, t, prefix, end, n)
            continue
        if desc is None:
            desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
        desc.add(row, slot, t, prefix, end - prefix)
        plans.append((t, slot, desc, int(prefix), int(end)))
        taken.add(t)
        _tw.note_track_planned(rid, t)
        n = _count("head_planned")
        if _log_due(n):
            logger.info(
                "PDFLIP HEAD-ANCHOR TRACK n=%d rid=%s step=[%d,%d) at=%d heads=%s main=%s (extra extend track at "
                "a system-reminder close of the first user message: a prompt of this head family branching there "
                "-- a compaction, a sibling agent -- reads its KV up to here instead of the previous chunk anchor)",
                n, rid, prefix, end, t, pts, main_track)
    if plans:
        setattr(req, HEAD_PENDING_ATTR, plans)
    return desc


def has_pending(req: Any) -> bool:
    """A turn, twin or fork plan waits for the tree's next insert of ``req``."""
    return (getattr(req, PENDING_ATTR, None) is not None
            or bool(getattr(req, TWIN_PENDING_ATTR, None))
            or getattr(req, FORK_PENDING_ATTR, None) is not None
            or getattr(req, GEN_PENDING_ATTR, None) is not None
            or bool(getattr(req, SYSGRID_PENDING_ATTR, None))
            or getattr(req, CLOSE_PENDING_ATTR, None) is not None
            or getattr(req, IMAGE_PENDING_ATTR, None) is not None
            or getattr(req, BRANCH_PENDING_ATTR, None) is not None
            or bool(getattr(req, HEAD_PENDING_ATTR, None)))


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
    gen = getattr(req, GEN_PENDING_ATTR, None)
    if gen is not None:
        setattr(req, GEN_PENDING_ATTR, None)
        out.append(("gen", gen))
    grid = getattr(req, SYSGRID_PENDING_ATTR, None) or ()
    if grid:
        setattr(req, SYSGRID_PENDING_ATTR, None)
        out.extend(("grid", p) for p in grid)
    close = getattr(req, CLOSE_PENDING_ATTR, None)
    if close is not None:
        setattr(req, CLOSE_PENDING_ATTR, None)
        out.append(("close", close))
    image = getattr(req, IMAGE_PENDING_ATTR, None)
    if image is not None:
        setattr(req, IMAGE_PENDING_ATTR, None)
        out.append(("image", image))
    branch = getattr(req, BRANCH_PENDING_ATTR, None)
    if branch is not None:
        setattr(req, BRANCH_PENDING_ATTR, None)
        out.append(("branch", branch))
    heads = getattr(req, HEAD_PENDING_ATTR, None) or ()
    if heads:
        setattr(req, HEAD_PENDING_ATTR, None)
        out.extend(("head", p) for p in heads)
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
    only to the older anchor: pdflip-2-8 '#1028B FETCH CAP kv=1089 claimed=941
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
                logger.info("PDFLIP TURN-ANCHOR NGRAM-OFF-STAGE n=%d: this PP stage owns no PLE "
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


def _draw_slot(batch: Any) -> Tuple[Any, bool]:
    """(slot, plan_without_slot). Group P: a slot or no plan (unchanged). Group D (``d_mode``): the plan is a
    POSITION plan on every rank -- a split-only rank draws nothing, a state-holding rank draws a slot when one
    is free and plans split-only when none is (no eviction, no reserve) -- so the plan/no-plan decision never
    depends on a rank-local quantity."""
    if d_mode():
        if d_split_only_rank():
            return None, True
        return _alloc_slot(batch), True
    return _alloc_slot(batch), False


def _put_plan(batch: Any, desc: Optional["TurnTracks"], req: Any, row: int, attr: str, t: int,
              slot: Any, prefix: int, end: int) -> Optional["TurnTracks"]:
    """Record one plan; only a plan with a slot gets a gather row (``desc``)."""
    if slot is not None:
        if desc is None:
            desc = TurnTracks(len(batch.reqs), need=slot_state_kinds(batch.req_to_token_pool))
        desc.add(row, slot, t, prefix, end - prefix)
        plan = (t, slot, desc, int(prefix), int(end))
        setattr(req, attr, plan)
        if d_mode():
            _register_inflight(req, attr, plan,
                               getattr(getattr(batch, "req_to_token_pool", None), "mamba_allocator", None))
    else:
        setattr(req, attr, (t, None, None, int(prefix), int(end)))   # D split-only
    return desc


# -- D-FORK-ANCHOR review G1 (21.10.): the in-flight fork slots of THIS rank -----------------------------------------
#: (req, attr, plan, allocator, clear epoch) of every D fork plan that holds a slot and was not consumed yet. Validated
#: lazily: an entry whose request no longer carries that very plan tuple (consumed by the insert, freed as stale), or
#: whose allocator was cleared since (review N1), is dropped on the next walk.
_INFLIGHT: List[Tuple[Any, str, Any, Any, int]] = []


def _epoch(alloc: Any) -> int:
    return int(getattr(alloc, "_pdflip_clear_epoch", 0) or 0)


def _register_inflight(req: Any, attr: str, plan: Any, alloc: Any = None) -> None:
    _prune_inflight()
    _INFLIGHT.append((req, attr, plan, alloc, _epoch(alloc)))


def _live(e) -> bool:
    req, attr, plan, alloc, ep = e
    return getattr(req, attr, None) is plan and _epoch(alloc) == ep


def _prune_inflight() -> None:
    _INFLIGHT[:] = [e for e in _INFLIGHT if _live(e)]


def on_pool_clear(alloc: Any) -> int:
    """Review N1 (21.10.): ``clear()`` of a mamba allocator (flush / sleep-leg reset, wake restore, vram_dial shrink)
    puts every slot back on the free list of a fresh generation. A request object that survives it still carries its
    unconsumed fork plan -- whose slot a new owner may hold from now on. Every such plan drawn from ``alloc`` turns
    split-only here (desc field ``"pool_reset"``): no later donation revokes that slot, no stale free and no insert
    release it again. Returns how many plans were neutralised."""
    n = 0
    keep = []
    for e in _INFLIGHT:
        req, attr, plan, a, _ep = e
        if a is not alloc:
            keep.append(e)
            continue
        if getattr(req, attr, None) is plan:
            t, _slot, _desc, prefix, end = plan
            setattr(req, attr, (t, None, "pool_reset", int(prefix), int(end)))
            n += 1
    _INFLIGHT[:] = keep
    if n:
        k = _count("d_pool_reset")
        if _log_due(k):
            logger.info("PDFLIP D-FORK-ANCHOR POOL-RESET n=%d plans=%d (the mamba allocator was cleared: the "
                        "unconsumed fork plans lost their slots and turn split-only)", k, n)
    return n


def inflight_count() -> int:
    _prune_inflight()
    return len(_INFLIGHT)


def revoke_for_donation() -> Any:
    """Review G1: a fork slot is drawn at ``prepare_for_extend`` (batch N) but a last-step plan is consumed only by
    ``process_batch_result(N)`` -- after the next scheduling. In that window ANOTHER request's step donation
    (``cache_unfinished_req`` of the chunked request, an earlier entry of the same result loop) draws from a pool that
    is short by exactly that slot on TP0 alone (a split-only worker drew none): TP0 alone would skip that request's
    node. So the state-holding D rank's donation, when the free list is empty, takes back the slot of the OLDEST
    unconsumed fork plan BEFORE it evicts anything: the plan turns split-only (``why=revoked_for_donation``), its node
    at t is still inserted on every rank. TP0's donation sources are then a superset of every worker's (the workers
    hold exactly these slots free). No reserve, no eviction.

    STREAM ORDER (checked at the code): the three callers are the step donations of
    ``MambaComponent.prepare_for_caching_req``. The two extra_buffer branches hand the slot to the donating request as
    its PING-PONG track buffer (``donate_mamba_ping_pong_slot``): its first write is a later forward's track gather on
    the forward stream, i.e. after batch N's late fork row into the same slot on that stream -- the old row is dead
    bytes, overwritten. The copy branch (no extra_buffer -- unreachable with D fork plans, ``armed`` requires
    extra_buffer) fences the device before its current-stream copy. No other allocation (S90b/S109R device reads,
    the admission COW) may take a fork slot. A plan whose request left without consuming it (the F5 leak) is
    revoked like any other -- its slot comes back. Returns the slot (a one-element tensor) or None."""
    if not d_mode() or d_split_only_rank():
        return None
    _prune_inflight()
    while _INFLIGHT:
        e = _INFLIGHT.pop(0)
        if not _live(e):
            continue
        req, attr, plan, _alloc, _ep = e
        t, slot, _desc, prefix, end = plan
        setattr(req, attr, (t, None, "revoked_for_donation", int(prefix), int(end)))
        n = _count("d_revoked")
        if n <= 64 or (n & (n - 1)) == 0:
            logger.info("PDFLIP D-FORK-ANCHOR REVOKED n=%d rid=%s kind=%s at=%d (an unconsumed fork plan gave its "
                        "slot to another request's step donation: the free list was empty on this rank and the "
                        "workers hold this slot free -- the fork goes split-only)", n,
                        str(getattr(req, "rid", "?")), _ATTR_KIND.get(attr, "?"), int(t))
        return slot
    return None


def _free_slot(batch_or_pool: Any, slot) -> None:
    if slot is None:
        return
    pool = getattr(batch_or_pool, "req_to_token_pool", batch_or_pool)
    try:
        pool.mamba_allocator.free(slot)
    except Exception:  # noqa: BLE001
        logger.warning("PDFLIP TURN-ANCHOR slot free raised", exc_info=True)


def pop_pending(req: Any):
    p = getattr(req, PENDING_ATTR, None)
    if p is not None:
        setattr(req, PENDING_ATTR, None)
    return p


# -- the forward side ----------------------------------------------------------------
def forward_rows(forward_batch: Any) -> Optional[TurnTracks]:
    """The batch's descriptor when the forward may add its rows: an extend
    (not a verify) whose geometry is the planned one."""
    tt = getattr(forward_batch, "pdflip_turn_tracks", None)
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
            logger.warning("PDFLIP TURN-ANCHOR SKIP reason=geometry: the forward's batch is "
                           "not the planned one (bs=%s planned=%d); no rows added (n=%d)",
                           getattr(forward_batch, "batch_size", "?"), tt.bs, n)
        return None
    return tt


def insert_verdict(pend) -> Tuple[bool, str]:
    """(insert?, reason) for a consumed plan. A revoked D plan (review G1) carries its reason in the desc field."""
    t, slot, desc, prefix, end = pend
    if isinstance(desc, str):
        return False, desc
    if desc is None or slot is None:
        return False, "no_plan"
    if not desc.complete():
        missing = sorted(desc.need - desc.done)
        return False, "unmarked:" + ",".join(missing)
    return True, "ok"


def _label(kind: str) -> str:
    return {"twin": "TWIN-ANCHOR", "fork": "FORK-ANCHOR",
            "gen": "GEN-FORK-ANCHOR", "grid": "SYS-GRID-ANCHOR",
            "close": "CLOSE-FORK-ANCHOR", "image": "IMAGE-END-ANCHOR",
            "branch": "BRANCH-ANCHOR", "head": "HEAD-ANCHOR"}.get(kind, "TURN-ANCHOR")


def _kind_prefix(kind: str) -> str:
    return kind + "_" if kind in ("twin", "fork", "gen", "grid", "close", "image", "branch", "head") else ""


def note_insert(req: Any, t: int, prefix_len: int, taken: bool, prompt: int,
                step: Tuple[int, int], kind: str = "turn") -> None:
    label = _label(kind)
    n = _count(_kind_prefix(kind) + ("inserted" if taken else "exists"))
    from flliper.srt.pdflip import twin_anchor as _tw

    if kind == "twin" or _tw.status(str(getattr(req, "rid", "?")), int(t)) is not None:
        _tw.note_written(str(getattr(req, "rid", "?")), int(t))
    if _log_due(n):
        logger.info(
            "PDFLIP %s INSERT n=%d rid=%s at=%d step=[%d,%d) prompt=%d "
            "prefix_len=%d value=%s counts=%s",
            label, n, str(getattr(req, "rid", "?")), t, step[0], step[1], prompt, prefix_len,
            "taken" if taken else "exists", counts(),
        )


def note_skip(req: Any, reason: str, t: int, kind: str = "turn") -> None:
    label = _label(kind)
    n = _count(_kind_prefix(kind) + "skip:" + reason.split(":")[0])
    from flliper.srt.pdflip import twin_anchor as _tw

    if kind == "twin" or _tw.status(str(getattr(req, "rid", "?")), int(t)) is not None:
        _tw.note_declined(str(getattr(req, "rid", "?")), int(t))
    if _log_due(n):
        logger.info("PDFLIP %s SKIP rid=%s reason=%s turn=%d (n=%d) -- slot freed",
                    label, str(getattr(req, "rid", "?")), reason, t, n)


# -- D-FORK-ANCHOR witnesses (21.10.) ------------------------------------------------
D_FORK_KIND_ATTR = "pdflip_d_fork_kind"
_ATTR_KIND = {PENDING_ATTR: "turn", CLOSE_PENDING_ATTR: "close", GEN_PENDING_ATTR: "gen", IMAGE_PENDING_ATTR: "image",
              BRANCH_PENDING_ATTR: "branch"}


def _d_role() -> str:
    return "worker" if d_split_only_rank() else "host"


def note_d_planned(req: Any, kind: str, t: int, prefix: int, end: int, slot: Any) -> None:
    n = _count("d_planned")
    if _log_due(n):
        logger.info("PDFLIP D-FORK-ANCHOR PLAN n=%d rid=%s kind=%s at=%d step=[%d,%d) role=%s state=%s",
                    n, str(getattr(req, "rid", "?")), kind, int(t), int(prefix), int(end), _d_role(),
                    "slot" if slot is not None else "split_only")


def note_d_set(req: Any, kind: str, t: int, state: str, why: str, depth: int) -> None:
    """Live witness: one line per fork node group D put into its tree (``state=taken`` = the recurrent state
    at t, ``split_only`` = the node without a state, ``exists`` = the node was already there)."""
    n = _count("d_set:" + state)
    if n <= 64 or (n & (n - 1)) == 0:
        logger.info("PDFLIP D-FORK-ANCHOR SET n=%d rid=%s kind=%s at=%d state=%s why=%s role=%s prefix_len=%d "
                    "(group D's fork node: the next turn / a side request forking there resumes at t)",
                    n, str(getattr(req, "rid", "?")), kind, int(t), state, why, _d_role(), int(depth))


def note_d_read(req: Any, kind: str, depth: int) -> None:
    """Live witness: a match resumed on a state group D's fork track wrote."""
    n = _count("d_read")
    if n <= 64 or (n & (n - 1)) == 0:
        logger.info("PDFLIP D-FORK-ANCHOR READ n=%d rid=%s kind=%s depth=%d (a match resumed on a group-D fork "
                    "anchor instead of the last state below it)", n, str(getattr(req, "rid", "?")), kind,
                    int(depth))
