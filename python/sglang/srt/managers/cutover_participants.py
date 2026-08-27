# SPDX-License-Identifier: Apache-2.0
"""#859 SEED: the cutover's participants, declared instead of discovered.

WHY THIS FILE EXISTS, and it is not a taxonomy exercise.

Seven blockers in one night, each investigated as its own bug, share ONE
shape: **the cutover is not a first-class operation.** It is a sequence of
steps that each remember to move the piece they know about, and every piece
nobody remembered is found by a boot. The list, in the order the metal
produced them:

  * #822/H1   an id lived in two free lists          -> allocator lists
  * #861      the draft KV pool was never registered  -> a whole pool, missed
  * #861b     `prefix_indices` truthiness             -> a carried tensor field
  * #861c/F1  the draft host pool's size check        -> a binding, mis-guarded
  * #861c/F2  `uncached_prompt_tokens` at an          -> a counter, wrong
              existence question                        semantics for a caller
  * W37-C/R3  `checked=0` on 18 flips                 -> a gate, unreachable
  * W37-C     `batch_is_full` latched at running=0    -> a scheduler flag
              across the #856 retract                    nobody clears

Not one of those is exotic. Every one is "a thing that must be moved, reset or
re-derived at the seam, and was not". The cost of finding them one boot at a
time is one window each; the cost of declaring them is this file.

WHAT THIS IS AND IS NOT.

It is a REGISTRY plus two obligations per participant:

  * a CUTOVER HOOK -- the named function that moves/resets/re-derives it, so
    "who handles this at the seam" is answerable by reading rather than by
    grepping the runtime and hoping;
  * a REACHABILITY PROBE -- a named observable proving the hook actually RAN.
    This is the #719 STALE-GATE lesson generalised: W36 built a heartbeat so
    "clean" and "blind" could not be byte-identical, W37-C then logged
    `checked=0` eighteen times and the mechanism was still blind. A hook
    without a probe is a hook you cannot prove ran.

It is NOT a framework and it does NOT execute the cutover. Nothing here is on
a hot path. `test_cutover_participants_859.py` asserts that every registered
participant names both obligations and that the named symbols EXIST, which
converts "we forgot a participant" from a boot-time discovery into a desk-time
check for everything already on the list.

FIRST CUT, HONESTLY BOUNDED. It covers the participants THIS NIGHT surfaced,
plus the ones already known to be handled (they are the useful control: a
registry containing only defects would not show what a healthy entry looks
like). Entries whose obligation is not yet built carry ``hook=None`` and a
``gap`` naming what is missing -- those are the filed, unbuilt items, and they
are visible as a list rather than as institutional memory.

ADDING A PARTICIPANT IS THE POINT. When the next cutover-shaped defect is
found, it gets a row here BEFORE its fix lands, so the fix's own test can
assert the row is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Participant:
    """One thing the cutover must move, reset or re-derive.

    ``hook`` and ``probe`` are dotted symbol paths, checked for existence by
    the registry test. ``None`` means the obligation is NOT built; ``gap`` then
    says what is missing, and the test asserts a gap is always explained.
    """

    name: str
    what: str
    #: dotted path to the function that handles it at the cutover
    hook: Optional[str]
    #: dotted path (or log-line marker) proving the hook ran
    probe: Optional[str]
    #: the ticket that surfaced it
    ticket: str
    #: what is still missing, when hook or probe is None
    gap: Optional[str] = None
    #: how it was found: "boot" (a window paid for it) or "desk"
    found_by: str = "boot"


#: Marker probes are log substrings rather than symbols: some participants can
#: only be proven to have run by what they emit. Prefixed so the test can tell
#: the two kinds apart without a second field.
LOG = "log:"


REGISTRY: Tuple[Participant, ...] = (
    # ---------------------------------------------------------- pools
    Participant(
        name="target_kv_pool_binding",
        what="HiCacheController's device pool, host pool and allocator: all "
        "three move together or none does",
        hook="sglang.srt.mem_cache.hicache_phase_binding.rebind_for_cutover",
        probe="sglang.srt.mem_cache.hicache_phase_binding.coherence_check",
        ticket="#718/#719",
        found_by="boot",
    ),
    Participant(
        name="draft_kv_pool_registration",
        what="the drafter's own KV pool and its host twin -- registered at the "
        "cutover because its host indices are the target's, and the target "
        "host pool moves per phase",
        hook="sglang.srt.mem_cache.kv_cache_builder.rebind_hicache_draft_for_phase",
        probe=LOG + "HiCache draft KV registered",
        ticket="#861",
        found_by="boot",
    ),
    Participant(
        name="draft_pool_phase_ownership",
        what="the draft half must be DISARMED in the phase that owns no "
        "drafter, or a PP backup persists rows no drafter wrote",
        hook="sglang.srt.managers.cache_controller.HiCacheController.disarm_draft_kv_pool",
        probe=LOG + "#861 draft-half HiCache DISARMED",
        ticket="#861",
        found_by="desk",
    ),
    # ---------------------------------------------------------- allocator
    Participant(
        name="allocator_free_lists",
        what="an id may live in at most one free list; free_pages and "
        "release_pages overlapped and available_size double-counted",
        hook="sglang.srt.managers.kv_backing_relief.KvRowCap._settle_free_list_overlap",
        probe=LOG + "ONE-OWNER (free lists)",
        ticket="#822",
        found_by="boot",
    ),
    # ---------------------------------------------------------- requests
    Participant(
        name="resident_requests",
        what="every resident is retracted and re-admitted; retracted must "
        "equal re-admitted or requests are owned by nobody",
        hook="sglang.srt.managers.phase_flip_runtime.build_cutover_release",
        probe=LOG + "RESIDENTS RELEASED",
        ticket="#856",
        found_by="boot",
    ),
    Participant(
        name="draft_state_of_admitted_requests",
        what="a request admitted on a cached prefix whose draft rows nothing "
        "wrote must be scrubbed and must not speculate",
        hook="sglang.srt.managers.phase_flip_draft_bootstrap.arm_draft_cold_for_admission",
        probe=LOG + "ADMISSION draft-cold",
        ticket="#861/#861b",
        found_by="boot",
    ),
    Participant(
        name="carried_batch_spec_algorithm",
        what="a batch's own spec_algorithm field names the phase that BUILT "
        "it, and prepare_for_decode branches on it",
        hook="sglang.srt.managers.phase_flip_draft_bootstrap.retune_carried_batches_for_phase",
        probe=None,
        ticket="#631",
        gap="no probe: the retune returns a count but nothing logs it per "
        "cutover, so 'retuned 0 because none needed it' and 'retuned 0 "
        "because the reach missed them' are byte-identical -- the exact "
        "shape the #719 heartbeat exists to forbid",
        found_by="boot",
    ),
    Participant(
        name="carried_batch_spec_info",
        what="a TP batch's spec_info is dereferenced by merge_batch in a "
        "phase with no drafter (corpse I)",
        hook="sglang.srt.managers.phase_flip_draft_bootstrap.clear_spec_info_for_unspeculated_phase",
        probe=None,
        ticket="#631",
        gap="same missing probe as carried_batch_spec_algorithm",
        found_by="boot",
    ),
    # ---------------------------------------------------------- flags
    Participant(
        name="latched_batch_flags",
        what="THE CLASS behind three instances: a persistent flag on a "
        "ScheduleBatch whose only clear sites are FINISH paths, surviving "
        "the #856 retract. Members: spec_algorithm, spec_info, "
        "batch_is_full (the last latched True at running=0 and refused "
        "admission for ever)",
        hook="sglang.srt.managers.phase_flip_draft_bootstrap.reset_stale_batch_flags",
        probe=LOG + "#861c cleared latched batch flag(s)",
        ticket="W37-C/#861c",
        found_by="boot",
    ),
    # ---------------------------------------------------------- counters
    Participant(
        name="prefill_backlog_counters",
        what="ONE number was asked TWO questions: economics (would PP be "
        "cheaper) and existence (is a pass owed at all)",
        hook="sglang.srt.managers.scheduler.Scheduler._admissible_prefill_tokens",
        probe="sglang.srt.managers.phase_policy.PhasePolicyInputs.work_exists",
        ticket="#861c",
        found_by="boot",
    ),
    # ---------------------------------------------------------- gates
    Participant(
        name="stale_generation_gate",
        what="operations queued under one binding and consumed under the next",
        hook="sglang.srt.managers.cache_controller.operation_is_stale",
        probe=LOG + "STALE-GATE HEARTBEAT",
        ticket="#719/#760",
        found_by="boot",
    ),
    Participant(
        name="stale_generation_gate_reachability",
        what="the gate's own liveness: checked=0 across a full flip cycle "
        "means it was never REACHED, which protects nothing",
        hook="sglang.srt.managers.cache_controller.gate_heartbeat",
        probe=LOG + "STALE-GATE BLIND",
        ticket="#861c",
        found_by="boot",
    ),
    # ---------------------------------------------------------- caches
    Participant(
        name="prefix_tree",
        what="the tree is dropped at the seam and its rows returned to the "
        "allocator; a bare reset leaked 152 rows per cycle",
        hook="sglang.srt.managers.phase_flip_runtime.drop_prefix_tree_returning_rows",
        probe=LOG + "tree dropped returning",
        ticket="#856/W27",
        found_by="boot",
    ),
    Participant(
        name="hicache_writeback_fence",
        what="warm prefixes must reach the canonical store BEFORE the seam, "
        "or the next phase cannot read them back",
        hook="sglang.srt.mem_cache.hicache_flip_writeback.maybe_flip_writeback",
        probe=None,
        ticket="#703",
        gap="probe not confirmed: the fence logs, but no single greppable "
        "marker was verified against a boot in this pass",
        found_by="desk",
    ),
)


def participants_with_gaps() -> Tuple[Participant, ...]:
    """Registered participants whose hook or probe is not built.

    The list this returns IS the #859 backlog. It is deliberately reachable
    from code rather than living in a document: a backlog nobody can enumerate
    is a backlog that gets rediscovered by a boot.
    """
    return tuple(p for p in REGISTRY if p.hook is None or p.probe is None)


def participants_found_by_boot() -> Tuple[Participant, ...]:
    """The ones a GPU window paid for. The number this returns is the argument
    for the registry: every entry here cost a boot to discover."""
    return tuple(p for p in REGISTRY if p.found_by == "boot")


# ---------------------------------------------------------------------------
# CUT 2 (#859): DISCOVERY-DIFF. Default-inverted.
# ---------------------------------------------------------------------------
#
# Cut 1 above is ENUMERATIVE: it validates the declarations somebody remembered
# to write. It cannot find a participant nobody thought of, and the root
# property -- a component can participate in the flip implicitly and
# undeclared -- survived it untouched. Every blocker of W37-B/C/D was such a
# component.
#
# Cut 2 inverts the default. Instead of asking "are the declared entries
# well-formed", it DISCOVERS what the cutover actually touches and fails on
# anything undeclared. Undeclared is the failure; silence is not consent.
#
# THE READING-MOMENT AXIS IS PART OF THE DECLARATION (#861e). W37-D's last
# defect was not an undeclared participant -- `running_bs` is declared and
# handled -- but a term that read it INSIDE the transition that manufactures
# it. So a participant now also declares WHEN its state is validly readable,
# and a reader that consults it in the wrong window is as much a defect as a
# mover that forgets it.


class ReadWindow:
    """When a participant's state is a fact rather than an artefact."""

    #: Readable at any time; the cutover does not disturb it.
    ALWAYS = "always"
    #: Valid only OUTSIDE the retract/re-admit window. `running_bs` is the
    #: type case: the cutover empties the running batch, so a read inside the
    #: window reports a state the seam produced (W37-D: "nothing decoding" one
    #: line after "7 request(s) retracted").
    OUTSIDE_CUTOVER = "outside_cutover"
    #: Only meaningful DURING the cutover -- seam bookkeeping.
    DURING_CUTOVER = "during_cutover"


#: State the cutover mutates, with the window in which reading it is honest.
#: A term that reads any of these must either be evaluated outside the window
#: or go through a transition-coherent accessor. Named rather than discovered,
#: for the same reason as REGISTRY: what this list forgets, a boot finds.
MUTATED_STATE = {
    "running_batch.reqs": ReadWindow.OUTSIDE_CUTOVER,
    "running_bs": ReadWindow.OUTSIDE_CUTOVER,
    "waiting_queue": ReadWindow.OUTSIDE_CUTOVER,
    "tree_cache": ReadWindow.OUTSIDE_CUTOVER,
    "token_to_kv_pool_allocator": ReadWindow.OUTSIDE_CUTOVER,
    "batch_is_full": ReadWindow.OUTSIDE_CUTOVER,
    "spec_algorithm": ReadWindow.OUTSIDE_CUTOVER,
    "spec_info": ReadWindow.OUTSIDE_CUTOVER,
    "mem_pool_host": ReadWindow.OUTSIDE_CUTOVER,
    "seam_readmit_epoch": ReadWindow.DURING_CUTOVER,
}

#: Accessors that make an OUTSIDE_CUTOVER quantity honest inside the window.
#: A reader using one of these is coherent by construction.
COHERENT_ACCESSORS = frozenset(
    {
        "decode_work_bs",
        "demand_prefill_tokens",
        "work_exists",
        "_retracted_unfinished_bs",
        "_admissible_prefill_tokens",
        # #861f: asks whether a GENUINELY RESIDENT bundle is owed decode steps.
        # It reads `running_bs` raw ON PURPOSE -- residency is exactly what it
        # must measure, and it fires on >0 (the fail-safe direction) rather
        # than on the manufactured 0.
        "bundle_is_mid_flight",
    }
)


def discover_cutover_writes(source: str) -> set:
    """Attributes the cutover ASSIGNS, read out of its own source.

    Deliberately syntactic. A semantic analysis would be better and would also
    be a second thing to maintain; the point of a discovery-diff is that it
    costs nothing to keep true.
    """
    import ast

    found = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Attribute):
                found.add(t.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            found.add(node.args[1].value)
    return found


#: Attributes the cutover writes that are NOT participants in the #859 sense:
#: seam-local bookkeeping, counters, logging state. Declared so the diff has a
#: small, reviewable allow-list rather than a threshold.
NOT_PARTICIPANTS = frozenset(
    {
        "_stale_gate_zero_streak",
        "_seam_readmitted",
        "_seam_drain_ms",
        "_retracted_refs_retired",
        "residents_released",
        "tree_rows_returned",
        "hicache_seam_active",
        "_census_scheduler",
        "_seam_premise_refused_announced",
        "_seam_transport_announced",
        "phase_flip_active_stack",
        "_phase",
        "epoch",
    }
)


# ---------------------------------------------------------------------------
# #902: THE READ WINDOW'S TWIN -- where a resource's RELEASE site lives.
#
# NOTE_888b §5 wrote this extension out before it existed, and named the class
# it closes: "a resource held by a resident that the current layout forbids to
# make progress, with NO RELEASE PATH REACHABLE FROM THAT LAYOUT." The seat
# fix proved the shape is real; the note's own sweep found the siblings and
# left them as prose. Prose is what a boot finds one at a time.
#
# A participant already declares WHO moves it (`hook`), PROOF that the mover
# ran (`probe`), and WHEN its state is honest (`ReadWindow`). What it could
# not say is: this thing is HELD, here is the door out, and here are the paths
# that door lives on. Without that, "held with no reachable exit" is a
# property nobody can evaluate -- so it is found by a window, once per
# resource, for ever.
#
# NOTHING HERE CHANGES A RELEASE. This is declaration plus a check, and that
# boundary is deliberate: a conformance test that also rewired the releases it
# judges would be two changes wearing one ticket, and the failure mode of the
# second one is a wedged cutover mid-window. The check WARNS by name and hands
# its findings back; it never blocks a flip. Refusing here would be a
# behaviour change of exactly the kind that ended two of this fork's windows.
# ---------------------------------------------------------------------------


class ReleasePath:
    """The path a release site lives on -- i.e. what must be RUNNING for the
    door to be reachable."""

    #: The cutover itself releases it. Reachable from either layout by
    #: construction, because the seam runs in both.
    SEAM = "seam"
    #: Released only while decode steps run.
    DECODE = "decode"
    #: Released only while prefill batches are built.
    PREFILL = "prefill"
    #: Released only when the holder FINISHES or is ABORTED. Not reachable at
    #: a cutover: the seam RETRACTS residents, it does not finish them, so a
    #: parked holder keeps the resource for its whole life. This is not a
    #: pessimistic reading -- it is NOTE_888b's measured verdict for the kvso
    #: host region, in its own words: "both are finish/abort paths only. A
    #: parked session holds its host region for its whole life."
    FINISH_OR_ABORT = "finish_or_abort"
    #: Reachable from anywhere; no layout or phase gates it.
    ANY = "any"


@dataclass(frozen=True)
class HeldResource:
    """Something a resident HOLDS, and the door out of it.

    ``paths`` is the set of ReleasePath values its release site lives on. An
    EMPTY tuple means undeclared -- the state this file exists to make
    visible, not a shorthand for "none".
    """

    name: str
    what: str
    #: dotted path to the call that releases it; None when nothing does yet
    released_by: Optional[str]
    #: the paths that call lives on
    paths: Tuple[str, ...]
    ticket: str
    #: what is still missing, when the release is absent or unreachable
    gap: Optional[str] = None
    found_by: str = "boot"
    #: branch the named symbol lands with, when it is not on this base yet.
    #: Declared rather than omitted: a row left out until its branch merges is
    #: a row nobody adds afterwards, which is how the population thinned in
    #: the first place. The existence check skips these BY NAME.
    pending_branch: Optional[str] = None


#: The population from NOTE_888b §5's sweep, plus the two families that
#: post-date it (#890/#906 grants, #773 §8's pin). Rows whose door is absent
#: or unreachable carry a `gap` and are the filed, unbuilt items -- visible as
#: a list rather than as institutional memory, exactly as REGISTRY's are.
HELD_RESOURCES: Tuple[HeldResource, ...] = (
    HeldResource(
        name="request_seat",
        what="the req_to_token_pool seat a parked/spilled request occupies",
        released_by="sglang.srt.managers.schedule_batch.release_req",
        paths=(ReleasePath.SEAM, ReleasePath.DECODE),
        ticket="#888b",
        found_by="boot",
    ),
    HeldResource(
        name="device_kv_rows",
        what="the device KV rows that seat owns; freed by the same call",
        released_by="sglang.srt.managers.schedule_batch.release_req",
        paths=(ReleasePath.SEAM, ReleasePath.DECODE),
        ticket="#888b",
        found_by="boot",
    ),
    HeldResource(
        name="mamba_slot",
        what="the mamba/GDN state slot a resident holds",
        released_by="sglang.srt.managers.schedule_batch.release_req",
        paths=(ReleasePath.SEAM, ReleasePath.DECODE),
        ticket="#888b",
        found_by="boot",
    ),
    HeldResource(
        name="mamba_anchor_pin",
        what="#773 §8's pin on a mamba anchor, held until the write-through "
        "is acked",
        released_by=None,
        paths=(),
        ticket="#773 §8",
        gap="the pin release is ABSENT -- NOTE_773 defers it to its own "
        "ticket. Declared here so 'held with no door at all' is a row rather "
        "than a sentence in another document.",
        found_by="desk",
    ),
    HeldResource(
        name="kvso_host_region",
        what="the host region a spilled session occupies in the kv-session "
        "offload tier",
        released_by="sglang.srt.managers.kv_session_offload."
        "KVSessionOffloadManager.release_finished_spilled_req",
        paths=(ReleasePath.FINISH_OR_ABORT,),
        ticket="#888b",
        gap="finish/abort ONLY, and the seam RETRACTS rather than finishes, "
        "so a parked session holds its host region for its whole life. Same "
        "shape as the seat, host tier, still unfixed -- NOTE_888b named it "
        "and this row is where it stops being prose.",
        found_by="desk",
    ),
    HeldResource(
        name="draft_weights",
        what="the drafter's weights, spilled and restored across the seam",
        released_by="sglang.srt.managers.phase_flip_spill."
        "PhaseFlipSpillLadder.on_enter_pp",
        paths=(ReleasePath.SEAM,),
        ticket="#888b",
        found_by="desk",
    ),
    HeldResource(
        name="seam_transport_grant",
        what="the one-chunk seam-transport credit a retracted request holds "
        "(#906 slice 1); spent at the exempt admission, re-issued by the next "
        "cutover stamp",
        released_by="sglang.srt.managers.phase_purity.consume_seam_grant",
        paths=(ReleasePath.SEAM, ReleasePath.PREFILL),
        ticket="#890/#906",
        found_by="desk",
        pending_branch="fix/906-one-chunk-consumption",
    ),
    HeldResource(
        name="batch_is_full_latch",
        what="the scheduler flag latched at running=0 across the #856 retract "
        "-- a latch is a held resource whose door is the clear",
        released_by="sglang.srt.managers.phase_flip_draft_bootstrap."
        "reset_stale_batch_flags",
        paths=(ReleasePath.SEAM,),
        ticket="W37-C",
        gap="its OTHER clear sites are all finish paths, which #856 does not "
        "take -- the seam handler is the only reachable door, and its own "
        "call site says so: 'Every clear site for these is a FINISH path and "
        "#856 RETRACTS'. Declared so the dependency on that one handler is a "
        "row rather than a comment.",
        found_by="boot",
    ),
)


#: Which release paths a target layout CANNOT reach, under strict phase
#: purity. This is the whole reachability rule and it is two lines because the
#: purity law is two lines: the PP window forbids decode, and the TP window
#: forbids prefill (the seam-transport exemption is not a general prefill
#: path, which is exactly why #906 had to debit it per chunk).
LAYOUT_CANNOT_REACH = {
    "pp": (ReleasePath.DECODE,),
    "tp": (ReleasePath.PREFILL,),
}

#: Paths NO layout reaches at a cutover, whatever the purity setting. The seam
#: RETRACTS its residents -- #856's own words, "nothing is carried across" --
#: and a retracted request is not a finished one, so a door that only opens on
#: finish or abort stays shut for the whole of a parked holder's life. This is
#: NOTE_888b's verdict for the kvso host region, encoded rather than restated:
#: the first draft of this rule left it out, and the conformance test caught
#: the omission by asking for the finding the note had already established.
NEVER_REACHABLE_AT_CUTOVER = (ReleasePath.FINISH_OR_ABORT,)


def release_path_conformance(target_layout: str, *, strict: bool = True) -> list:
    """Resources with no release path reachable from ``target_layout``. #902.

    Returns a list of human-readable findings, empty when every held resource
    has a door the target layout can open. PURE: no imports, no runtime, no
    side effects -- the caller logs, and a test can assert on the list without
    a scheduler.

    Two kinds of finding, kept apart because their remedies differ:

      * UNDECLARED -- the row names no release site, or names one with no
        paths. Nothing can be said about reachability, which is worse than an
        unreachable door and is the state #902 exists to surface.
      * UNREACHABLE -- every declared path is one this layout forbids. That is
        NOTE_888b's class, evaluated instead of discovered.

    ``strict`` False lifts the prefill bar only: a non-strict boot may prefill
    in TP, so a PREFILL-only door is reachable there. Decode is never
    permitted in the PP window under either setting, so the pp row does not
    move.
    """
    forbidden = set(LAYOUT_CANNOT_REACH.get(target_layout, ()))
    forbidden.update(NEVER_REACHABLE_AT_CUTOVER)
    if not strict:
        forbidden.discard(ReleasePath.PREFILL)
    findings = []
    for res in HELD_RESOURCES:
        if res.released_by is None or not res.paths:
            findings.append(
                f"UNDECLARED {res.name} ({res.ticket}): {res.what} -- no "
                f"release path is declared, so whether any layout can free it "
                f"cannot be evaluated"
                + (f". {res.gap}" if res.gap else "")
            )
            continue
        reachable = [p for p in res.paths if p not in forbidden]
        if not reachable:
            findings.append(
                f"UNREACHABLE {res.name} ({res.ticket}): {res.what} -- its "
                f"release lives on {'/'.join(res.paths)}, and the "
                f"{target_layout} layout reaches none of those"
                + (f". {res.gap}" if res.gap else "")
            )
    return findings
