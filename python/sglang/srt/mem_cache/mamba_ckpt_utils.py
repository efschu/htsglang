# Copyright 2025 SGLang Team
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
"""Boundary arithmetic for --mamba-checkpoint-interval.

With a fixed checkpoint interval G, every radix-cached mamba/GDN state must
sit at an absolute token position that is a multiple of G, so that the
checkpoint grid — and with it the prefix-resume point of identical requests —
is a pure function of the token history instead of the traffic-dependent
prefill split. These helpers are pure integer arithmetic so the invariants
can be unit-tested on CPU.
"""

from typing import Optional


def floor_to_interval(pos: int, interval: Optional[int]) -> int:
    """Round ``pos`` down to the checkpoint grid (identity when off)."""
    if interval is None:
        return pos
    return pos // interval * interval


def is_on_interval(pos: int, interval: Optional[int]) -> bool:
    """True when ``pos`` is a valid checkpoint position (always, when off)."""
    if interval is None:
        return True
    return pos % interval == 0


def is_resume_candidate(
    depth: int,
    interval: Optional[int],
    has_device_value: bool,
    has_host_value: bool = False,
    device_only: bool = True,
) -> bool:
    """Is a node at absolute token position ``depth`` a valid mamba resume
    anchor for the prefix match?

    The compound rule both match walks apply (``mamba_radix_cache.py``'s
    ``_match_prefix_helper`` and the unified component's match validators):
    the node must CARRY a state, and with ``--mamba-checkpoint-interval`` set
    it must SIT on the grid -- off-grid checkpoints (legacy entries, storage
    written by a non-interval run, unaligned edge paths) would re-introduce
    traffic-dependent resume points.

    ``device_only=False`` additionally accepts a host-backed state
    (``has_host_value``): under HiCache an evicted-but-backuped anchor is a
    valid match that triggers ``load_back``. The grid applies to those
    equally -- surviving on the host tier does not legalise an off-grid
    position.

    ``interval=None`` degenerates to the pure presence test, byte-identical
    to the pre-#747 behaviour of both walks.
    """
    return (
        resume_refusal_reason(
            depth,
            interval,
            has_device_value,
            has_host_value=has_host_value,
            device_only=device_only,
        )
        is None
    )


#: The two ways a node can fail to be a resume anchor. They are NOT variants of
#: one defect and their fixes point in opposite directions, which is the whole
#: reason they are named apart -- see ``resume_refusal_reason``.
RESUME_REFUSAL_ABSENT = "absent"
RESUME_REFUSAL_OFF_GRID = "off_grid"


def resume_refusal_reason(
    depth: int,
    interval: Optional[int],
    has_device_value: bool,
    has_host_value: bool = False,
    device_only: bool = True,
) -> Optional[str]:
    """Why this node is not a resume anchor, or ``None`` when it is one.

    THE COMPOUND RULE HAD ONE BIT OF OUTPUT AND TWO CAUSES, and #913/W42
    measured what that costs. The #904 match census named ``MambaComponent``
    as the refuser of 671 of 675 walks and could say nothing further, so the
    two readings below were indistinguishable in the only instrument that
    watches this path:

      ABSENT   the node carries no recurrent state at all. On the commit side
               that is the TOMBSTONE: ``commit_insert_component_data``
               declined the value (off-grid finish, or an exhausted int8
               checkpoint pool) and the node kept its KV without its state.
               The fix is on the WRITE side -- make the state exist.
      OFF_GRID the node HAS a state and sits at a position that is not a
               multiple of ``--mamba-checkpoint-interval``. Nothing is
               missing; the grid rule is declining a usable anchor to keep
               resume points from moving with traffic (#747 determinism).
               The fix, if any, is a POLICY change on the read side, and it
               is the one that must never be made blindly: accepting an
               off-grid anchor is exactly the #767 corruption direction when
               the state is not in fact the state at that depth.

    A single boolean forces whoever reads the census to guess which of those
    two it is looking at, and the guesses point at opposite files. This
    function is the one place the distinction is drawn; ``is_resume_candidate``
    is defined in terms of it so the predicate and the explanation cannot
    drift the way #747 records the two match lineages doing.

    Presence is checked FIRST and reported alone. An absent state that also
    sits off-grid is an ABSENT node: the grid is a property of a checkpoint,
    and there is no checkpoint to be on or off it. Reporting the grid there
    would send a reader to the read-side policy for a write-side hole.
    """
    present = has_device_value or (not device_only and has_host_value)
    if not present:
        return RESUME_REFUSAL_ABSENT
    if not is_on_interval(depth, interval):
        return RESUME_REFUSAL_OFF_GRID
    return None


def protect_deepest_anchors(
    interval: Optional[int], host_tier_present: bool = False
) -> bool:
    """Should eviction spare the deepest checkpoint anchors of every path?

    ``MambaRadixCache`` protects them (``mamba_radix_cache.py:1092-1105``)
    because "losing the deepest one silently moves the resume point of
    identical requests and re-introduces run-to-run drift". The protection is
    not about capacity, it is about DETERMINISM.

    That premise holds only while an evicted anchor is a LOST anchor, which is
    true of a device-only pool. With a host tier the node stays a valid match
    after its device value is dropped and is loaded back on the next hit
    (``unified_cache_components/mamba_component.py:71-74`` and ``:139-144``),
    so the resume point does not move and there is nothing to protect against.

    Hence the branch, stated once here rather than diverging per lineage:

    * ``host_tier_present=False`` -> protect, exactly as today;
    * ``host_tier_present=True``  -> do not protect; spilling an anchor is
      allowed because it can be matched and reloaded.

    #767: THE INTERVAL IS NOT A PRECONDITION, and treating it as one is what
    made identical requests drift. This read "no interval means no grid, so
    there are no anchors either way" and returned False. There ARE anchors: the
    ``no_buffer`` path donates a checkpoint for every finished request at
    ``cache_len = len(token_ids)``, grid or no grid, and the deepest one is the
    resume point whether or not an operator configured an interval. The
    interval only adds MORE anchors on top.

    Measured on one commit with the short-prompt determinism gate (identical
    temp-0 requests, which must return identical bytes): 48 slots idle over 10
    probes = 1 distinct, over 20 probes = 2, under 4-way load = 7 distinct with
    a degenerate one; 12 slots idle = 3. Divergence tracks slot pressure, which
    is exactly what eviction responds to -- so the protection has to hold
    whenever an evicted anchor is a LOST anchor, which is the device-only case
    and has nothing to do with the interval.

    THE CAPACITY TRADE IS REAL AND IS NOT THIS FUNCTION'S TO MAKE. Sparing
    anchors reduces the evictable set, so a pool that was only just large
    enough gets tighter. That is a SIZING answer (the slot count is a planner
    post), not a reason to hand back determinism.
    """
    return not host_tier_present


def checkpoint_truncation_align(
    existing_align: Optional[int],
    interval: Optional[int],
    chunked_prefill_size: Optional[int],
) -> tuple:
    """``(truncation_align_size, folded)`` after the checkpoint grid weighs in.

    #750: the interval folds into the prefill truncation alignment ONLY
    while ``interval <= chunked_prefill_size`` (or chunked prefill is off).
    There the scheduler clips every step end onto the grid, which is what
    makes every cached snapshot position an absolute interval multiple.

    A SPARSE grid -- ``interval > chunked_prefill_size``, validated to be an
    exact multiple -- needs no clipping at all: full chunk ends land at
    ``n * chunked_prefill_size``, and every ``(interval // chunk)``-th one
    is ON the grid by divisibility, while the retention rule
    (``mamba_radix_cache.py``: an off-grid finish is not cached) drops the
    ends between. Folding 8192 into the alignment would inflate the
    truncation unit 16x past a 512 chunk budget, which is exactly the C30
    refusal that made the old coupling a hard limit.

    ``folded`` tells the caller whether the interval became part of the
    alignment (and so belongs in the C30 sources list).
    """
    if interval is None:
        return existing_align, False
    if (
        chunked_prefill_size is not None
        and chunked_prefill_size > 0
        and interval > chunked_prefill_size
    ):
        return existing_align, False
    if existing_align is None:
        return interval, True
    import math

    return math.lcm(existing_align, interval), True


def mamba_checkpoint_track_target(
    prefix_len: int,
    extend_len: int,
    interval: int,
    chunk_size: int,
) -> Optional[int]:
    """Deepest absolute multiple of ``interval`` whose GDN state can be
    snapshotted within this extend step, or None if no boundary is reachable.

    The prefill kernels expose intermediate states only at
    ``prefix_len + k * chunk_size`` (FLA chunk grid) plus the final position,
    so a target is reachable iff it lies inside ``(prefix_len, end]`` and its
    offset from ``prefix_len`` is chunk-aligned. Callers validate
    ``interval % chunk_size == 0``; prefill steps end on the interval grid
    while ``interval <= chunked_prefill_size`` (the fold arm of
    ``checkpoint_truncation_align``), and on a #750 sparse grid
    (``interval`` a multiple of the chunk budget) a boundary inside a full
    step can only be the step's own end, so the final-position routing
    serves every anchor.
    """
    end = prefix_len + extend_len
    target = end // interval * interval
    if target <= prefix_len:
        return None
    if (target - prefix_len) % chunk_size != 0:
        return None
    return target


def retention_shrinks_protected(
    cache_len: Optional[int], protected_len: Optional[int]
) -> bool:
    """True when a proposed retention cap would truncate the protected prefix.

    #824. ``cache_unfinished_req`` publishes ``req.cache_protected_len =
    len(new_indices)`` at its end (unified_radix_cache.py:1234,
    mamba_radix_cache.py:1013), so that length is COMMITTED: the tree owns the
    KV below it and the caller writes only ``[protected, len(new_indices))``
    back into the token pool. A later step that retains LESS leaves the range
    between the two claimed by nobody. Both lineages assert on exactly that,
    and the assert killed rank PP0 -- and with it the instance, PP1 and PP2
    following seconds later on gloo peer-close -- on 2026-08-23 06:07:06::

        req.cache_protected_len=16384, len(new_indices)=8192,
        page_aligned_len=8192

    Reachable rather than theoretical, because the caps are NOT monotone:
    ``mamba_last_track_seqlen`` is set from ``mamba_branching_seqlen``
    (schedule_batch.py:2660), which can sit below an earlier track, and it is
    reset to ``None`` outright in several places.

    A PREDICATE, NOT A CORRECTED LENGTH, and that is the whole design decision.
    Raising the cap back up to ``protected_len`` is not available: the donated
    mamba state sits at exactly the tracked position, and pairing a state with
    a key at a different position is the silent corruption
    mamba_component.py:684-687 already refuses in the other direction ("Never
    floor"). Handing back a number would invite precisely that. Callers
    instead route a True into the decline machinery they already have --
    ``_decline_retention``, whose answer correctly differs by caller (0 for an
    unfinished step, ``None`` for a finished one, #783) -- so this helper never
    has to know which of those two is right.

    ``None`` cache_len is False: a component that already declined has nothing
    to shrink and must not be re-judged here.

    SHRINKAGE ONLY. Whether a position sits on the checkpoint grid is
    ``is_on_interval``'s decision and stays there; re-deciding it here would
    restart the two-lineage drift #747 exists to end.
    """
    if cache_len is None or protected_len is None:
        return False
    return int(cache_len) < int(protected_len)


# ---------------------------------------------------------------------------
# 27B line, 2026-09-24: WHERE P'S MAMBA ANCHORS GO, AND HOW MANY A PATH KEEPS.
#
# The user's decision of 24.09. (~15:20Z, "1 ja 2 ja 3 ja 4 spaeter") on the
# anchor question, points 2 and 3 (1 is Agent B's c255e10ddb, 4 -- the int8
# checkpoint pool -- is deferred):
#   2. inner anchors are deletable once the chain moved on; LRU like upstream;
#      an upper bound per path (~4); forks stay;
#   3. "the anchors are simply spread wider, not after every chunk": with
#      512-token chunks an anchor every ~4096 tokens (every 8th chunk) + at
#      forks + at the request end -- NO rigid 8192 grid.
# Point 3 is NOT the #747 `--mamba-checkpoint-interval` grid: that grid is also
# a READ rule (`is_resume_candidate` refuses every off-grid anchor, the end
# anchor N-1 included), which is exactly what made every short prefix hit
# unservable (#873, the per-node law of 27./28.08.). The spacing below is a
# WRITE-side thinning only: which chunk boundaries of an UNFINISHED request
# donate a state. Every anchor that exists stays matchable wherever it sits.
#
# Both switches are group-P-only (SGLANG_WEG2_GROUP=P) and default OFF; the arm
# switches them on with the approved values (4096, 4). Unset / empty / <= 0 /
# unparsable = off, byte-identical to the per-node default.
# ---------------------------------------------------------------------------

import functools as _functools
import os as _os
from typing import Mapping as _Mapping

#: Token spacing of the anchors a group-P chunked prefill donates.
ANCHOR_INTERVAL_ENV = "SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL"
#: Upstream --mamba-max-states-per-path (1417345f5f, `_evict_excess_path_states`).
MAX_STATES_PER_PATH_ENV = "SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH"

#: Why a chunk boundary carries an anchor (`weg2_anchor_step`).
ANCHOR_STEP_END = "end"
ANCHOR_STEP_INTERVAL = "interval"

#: `resume_refusal_reason`'s third answer, given by the component for a node the
#: per-path cap took (see UnifiedRadixCache._weg2_cap_path_states).
RESUME_REFUSAL_PATH_CAP = "path_cap"


def _weg2_group_p(env: _Mapping[str, str]) -> bool:
    return str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() == "P"


def _positive_int(env: _Mapping[str, str], name: str) -> int:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0


@_functools.lru_cache(maxsize=16)
def _profile_int_default(form_env: str, name: str) -> int:
    """UNIFY S7/S8: the published weg2 form's MODEL PROFILE default of ``name``
    (weg2/form.py MAMBA_ANCHOR_SWITCHES: grid4096 = 4096 / 4 per path, the 27B
    line's arm values; deepest / none = 0 = off). Cached per form string: the
    readers sit on the scheduler's per-node path."""
    if not form_env:
        return 0
    from sglang.srt.weg2.form import FORM_ENV, profile_switch_default

    return int(profile_switch_default(name, 0, {FORM_ENV: form_env}) or 0)


def _positive_int_or_profile(env: _Mapping[str, str], name: str, is_process_env: bool) -> int:
    """``name`` from ``env``; when unset in the PROCESS env, the profile default
    (an explicit mapping -- a desk caller -- keeps the plain read)."""
    if is_process_env and not str(env.get(name, "") or "").strip():
        # weg2/form.py FORM_ENV, spelled here: no import on the per-node path
        v = _profile_int_default(str(env.get("SGLANG_WEG2_FORM", "") or ""), name)
        return v if v > 0 else 0
    return _positive_int(env, name)


def weg2_anchor_interval(env: Optional[_Mapping[str, str]] = None) -> int:
    """Token spacing of group P's inner mamba anchors; 0 = off (an anchor at
    every chunk boundary, the per-node law's default)."""
    env = _os.environ if env is None else env
    if not _weg2_group_p(env):
        return 0
    return _positive_int_or_profile(env, ANCHOR_INTERVAL_ENV, env is _os.environ)


def weg2_max_states_per_path(env: Optional[_Mapping[str, str]] = None) -> int:
    """Upper bound of mamba anchors one root-to-tail path of group P's tree
    keeps; -1 = off (unbounded, upstream's default)."""
    env = _os.environ if env is None else env
    if not _weg2_group_p(env):
        return -1
    value = _positive_int_or_profile(env, MAX_STATES_PER_PATH_ENV, env is _os.environ)
    return value if value > 0 else -1


def weg2_anchor_step(
    pos: int, prompt_len: int, last_anchor: int, interval: int
) -> Optional[str]:
    """Why the chunk boundary at RAW token position ``pos`` of an UNFINISHED
    request donates a mamba anchor -- or ``None``: this boundary stays
    anchorless and the request keeps its KV until the next anchor step.

    * ``end`` -- ``pos >= prompt_len - 1``: the #1481 hand-back anchor N-1 (the
      end-anchor split puts a chunk boundary exactly there) and everything
      after it. Never thinned: it is the one anchor D resumes from, and the
      one a repeat of the prompt, the next turn or a regeneration resumes
      from -- the forks real traffic makes.
    * ``interval`` -- ``pos - last_anchor >= interval``: at least ``interval``
      tokens since this request's previous anchor (its resume anchor or the
      last one it donated). A DISTANCE, not an absolute grid: after a prefix
      hit at 1234 the next anchor sits ~4096 further on, whatever the chunk
      size and the alignment.

    Every input is RANK-UNIFORM on a PP group: ``pos`` is the forwarded
    extent, ``last_anchor`` the tree-owned prefix after PP0's geometry
    (`cap_req_geometry` / `truncate_prefix_to`), ``prompt_len`` the request.
    That is why there is no third, "fork" reason fed by the admission's
    ``key_match_depth``: on group P only PP0 reads the store (one
    `#1423 INSERT-PLACED`, PP0 only, in xsn422 and xsn423), so a rank's own
    key match depth can differ from its peers', and an anchor decided on it
    would be inserted on one rank and not on the others (raenge-nie-uneins).
    Forks keep their anchors under the per-path cap instead (a fork marked by
    the device insert that made it); an anchor exactly AT an arbitrary fork
    would need a chunk boundary there that PP0 decides and forwards.

    Everything else declines. ``interval <= 0`` means the switch is off; the
    caller does not ask then.
    """
    if pos >= prompt_len - 1:
        return ANCHOR_STEP_END
    if interval > 0 and pos - last_anchor >= interval:
        return ANCHOR_STEP_INTERVAL
    return None
