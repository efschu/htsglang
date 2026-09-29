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
              chunk: int, page: int, tok: int) -> Optional[TurnTracks]:
    """prepare_for_extend, per request: plan the second track of this step.
    Returns the (possibly new) batch descriptor."""
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
    if not track_mask or len(getattr(req, "output_ids", None) or ()) > 0:
        return desc
    t_abs = req_anchor_pos(req, tok, page)
    t = step_target(prefix, end, t_abs, chunk, main_track)
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
    them unmarked and the plan is refused -- never inserted half-written."""
    kinds = []
    scp = getattr(pool, "short_conv_pool", None)
    if scp is not None and bool(getattr(scp, "enabled", False)):
        kinds.append("ple_conv")
    ngp = getattr(pool, "ngram_pool", None)
    if ngp is not None and bool(getattr(ngp, "enabled", False)):
        kinds.append("ple_ngram")
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


def note_insert(req: Any, t: int, prefix_len: int, taken: bool, prompt: int,
                step: Tuple[int, int]) -> None:
    n = _count("inserted" if taken else "exists")
    if _log_due(n):
        logger.info(
            "WEG2 TURN-ANCHOR INSERT n=%d rid=%s at=%d step=[%d,%d) prompt=%d "
            "prefix_len=%d value=%s counts=%s",
            n, str(getattr(req, "rid", "?")), t, step[0], step[1], prompt, prefix_len,
            "taken" if taken else "exists", counts(),
        )


def note_skip(req: Any, reason: str, t: int) -> None:
    n = _count("skip:" + reason.split(":")[0])
    if _log_due(n):
        logger.info("WEG2 TURN-ANCHOR SKIP rid=%s reason=%s turn=%d (n=%d) -- slot freed",
                    str(getattr(req, "rid", "?")), reason, t, n)
