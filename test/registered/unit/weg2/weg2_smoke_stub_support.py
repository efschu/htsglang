# SPDX-License-Identifier: Apache-2.0
"""#1363 round 2: anti-drift support for `xchg_provider_smoke.py`'s BORROWED
production stubs (`_Stub`, `_LegStub`).

THE DRIFT (why this module exists, precedent: #624's BAR1 transport stub
audit, `27c1791941`, `test/registered/unit/distributed/barlink_stub_support.
py`). `xchg_provider_smoke.py`'s own docstring states its rule: "the stub
BORROWS the real unbound methods rather than re-implementing them, so what
runs here is byte-for-byte the code the flip leg runs." That rule protects
the METHOD BODIES from drift -- a borrowed method can never diverge from the
real one, because it IS the real one. It does NOT protect the STUB's OWN
SURFACE: when a borrowed method starts calling a NEW `self.<name>` the stub
does not have, Python raises `AttributeError` -- and because every one of
`_weg2_shadow_plan`'s callers wraps the whole derivation in `except
BaseException`, that `AttributeError` is caught, formatted into a string, and
returned as an ordinary "derivation failed" reason. The smoke's own `check()`
calls still catch this (a `plan is None` check fails), so THIS specific
class of drift is not silent -- but nothing FINDS it before the day it
happens, and nothing ROOT-CAUSES it once it does (a bare "AttributeError:
'_Stub' object has no attribute 'X'" names the symptom, not "a new production
method was added to this path").

#1394 ("draft weights join the real exchange") added exactly this: `_weg2_
shadow_plan` now calls `self._weg2_xchg_draft_plan_or_none(...)` directly,
`_Stub` never borrowed it, and section [1] of the smoke started failing --
found by a human/agent reading the output, not by any check built for the
purpose. THIS is the #624/#978 stub-drift class, and the fix shape is the
SAME one #624 used: not a bigger stub, an AUDIT -- every `self.<name>` the
REAL production path (transitively) reads must be either provided by the
stub or listed here with a reason. A new call turns the audit RED naming
it; a call the production code drops turns its stale exclusion RED the same
way.

WHY THIS IS A DIFFERENT SHAPE FROM #624'S OWN, NOT A COPY. #624 audits
`__init__`-ASSIGNED ATTRIBUTES (state a constructor sets) via a single
non-transitive AST walk of one method. This file audits `self.<name>` READS
(state and methods a call GRAPH touches) via a walk that is TRANSITIVE: the
production entry point calls other production methods on the SAME class,
and those may call further ones, so `self_reads_reachable` follows any
discovered name that is itself a method DEFINED ON THE REAL CLASS, and
stops at anything that is not (a plain instance attribute, or a name the
real code never actually holds as a callable).

A GETATTR-DEFAULT READ IS INVISIBLE TO THIS AUDIT, ON PURPOSE. `getattr(
self, "draft_worker", None)` and `_weg2_identity(self, "_weg2_group_name",
"")`-style calls pass the attribute NAME as a plain string argument, not as
`self.draft_worker`/`self._weg2_group_name` AST attribute syntax -- so they
never match the `ast.Attribute(value=Name("self"))` pattern this walk
looks for. That is exactly the right scope: a `getattr`-guarded read
already tolerates absence by construction (the real code supplies its own
default), so a stub missing that name cannot raise the way a direct
`self.X` access can. Auditing THOSE too would produce an ever-growing
exclusion table for names that were never actually a risk.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Sequence, Set


def self_reads_reachable(cls: type, entry_names: Sequence[str]) -> Set[str]:
    """Every `self.<name>` DIRECT READ (AST `Load`, never a `getattr(self,
    "name", ...)` call -- see the module docstring) reachable from
    `entry_names`' methods on `cls`, expanded TRANSITIVELY through any such
    name that is itself a method DEFINED ON `cls`.

    Walks `cls`'s OWN methods -- the REAL class, never the stub -- so a
    name the stub happens to lack does not prematurely end the walk: the
    whole point is to discover what the REAL call graph touches regardless
    of what any particular stub currently provides.
    """
    seen_methods: Set[str] = set()
    required: Set[str] = set()
    queue = list(entry_names)
    while queue:
        name = queue.pop()
        if name in seen_methods:
            continue
        seen_methods.add(name)
        member = cls.__dict__.get(name)
        if member is None:
            member = getattr(cls, name, None)
        if not (inspect.isfunction(member) or inspect.ismethod(member)):
            continue
        try:
            src = textwrap.dedent(inspect.getsource(member))
            tree = ast.parse(src)
        except (OSError, TypeError, SyntaxError):
            # A member this file cannot read the source of (e.g. a C
            # extension) cannot be walked further; it is not silently
            # dropped from `required` -- the CALLER that reached it already
            # added its name -- only its OWN body is unexamined.
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                continue
            attr = node.attr
            required.add(attr)
            if attr in seen_methods:
                continue
            sub = cls.__dict__.get(attr)
            if sub is None:
                sub = getattr(cls, attr, None)
            if inspect.isfunction(sub) or inspect.ismethod(sub):
                queue.append(attr)
    return required


#: `_LegStub` (`_weg2_xchg_bounce_leg`) exclusions. Reviewed 2026-09-14
#: against `weight_updater.py` @ 35e4924369. Both entries are guarded by a
#: runtime condition this smoke's OWN call (`xchg_provider_smoke.py`
#: section [4]) never satisfies -- not "harmless to skip" in general, only
#: for THIS caller's specific argument choice. A caller that ever starts
#: passing `tag=`/`rank=` to exercise the per-tag lockstep or the
#: undrained-lane check would need to revisit this table, which is exactly
#: what `test_no_stale_exclusions` cannot catch by itself (it only proves
#: the NAME still exists in the real class, not that the guard's shape is
#: unchanged) -- named here so a future reader knows to re-check the guard
#: condition, not just the attribute's continued existence.
LEGSTUB_EXCLUSIONS = {
    "_weg2_xchg_tag_seen": (
        "dataclass field (weight_updater.py:451, default None), read only "
        "inside `if tag is not None and phase == bx.PHASE_DEPOSIT:` -- this "
        "smoke calls _weg2_xchg_bounce_leg with no `tag=` (a single-shot "
        "execution smoke, not the per-tag lockstep #1374 added), so `tag` "
        "is always None and the branch never runs"),
    "_weg2_xchg_undrained_lanes": (
        "method (weight_updater.py:3906), read only inside `if phase == "
        "bx.PHASE_COLLECT and rank is not None and ...` -- this smoke "
        "never passes `rank=`, so it defaults to None and the branch never "
        "runs"),
}

#: `_Stub` (`_weg2_shadow_plan`) exclusions. Starts EMPTY, and empty is the
#: correct state, not a placeholder: every `self.<name>` this entry point
#: transitively reads is either borrowed onto `_Stub` as a class attribute
#: or set in `_Stub.__init__` (`tp_worker`). If this ever needs an entry,
#: it must carry a reason exactly like `LEGSTUB_EXCLUSIONS` above -- an
#: unreasoned name here would be the same silent drift this file exists to
#: name.
STUB_EXCLUSIONS: dict = {}

#: `scripts/weg2/xchg_leg_replay.py`'s `_Stub` exclusions (the SIX-PROCESS
#: desk replay of a whole exchange leg, #1378 Posten 1). This stub's borrow
#: list lived in `_Stub.__init__` (a `setattr` loop, not class attributes)
#: and was NOT audited by the original ratchet -- which is how it lost
#: `_weg2_shadow_plan`'s new `self._weg2_xchg_draft_plan_or_none` call
#: (#1394, weight_updater.py:3014 @ 18bb175bc6) and every rank of the
#: 2026-09-14 desk run died with
#: `AttributeError: '_Stub' object has no attribute '_weg2_xchg_draft_plan_
#: or_none'` BEFORE emitting a single leg. Same two names as
#: `LEGSTUB_EXCLUSIONS`, and the reasons were RE-VERIFIED FOR THIS CALLER,
#: not copied: `xchg_leg_replay.py`'s `rank_proc` calls
#: `stub._weg2_xchg_bounce_leg(descs=..., ops=..., boot_nonce=NONCE,
#: slot_bytes=..., depth=DEPTH, mode=..., shm_root=..., device=0, hook=...,
#: region=None, sems=sems)` -- neither `tag=` nor `rank=` -- so both guards
#: below are dead branches for exactly this caller too.
LEG_REPLAY_STUB_EXCLUSIONS = {
    "_weg2_seq_units_from_join": (
        "method (weight_updater.py, added by #1378 xsn52): the sequential "
        "transport's unit derivation from the join's tensors. METAL-ONLY: "
        "it needs real models and address books (the deposit and collect "
        "address books resolve per-side VRAM pointers). The leg replay's "
        "stub exercises the transport through its own plan derivation, "
        "not through this method."),
    "_weg2_model_for_group": (
        "method (weight_updater.py, added by #1378 xsn52): returns the "
        "model runner for a given group. METAL-ONLY: needs real workers. "
        "The leg replay's stub resolves addresses through its own "
        "borrowed _weg2_join_src_addr/_weg2_join_dst_addr."),
    "_weg2_card_uuid": (
        "method (weight_updater.py): resolves the rank's NVML uuid for "
        "the per-copy card lock. METAL-ONLY: needs a CUDA device. The "
        "sequential form's leg replay runs on the desk without CUDA."),
    "_weg2_rank_param_table": (
        "borrowed by the leg replay's _Stub.__init__ (a setattr loop, not "
        "a class attribute) -- the ratchet's dir() check finds it because "
        "the loop ran during setUp, but the source-level walk doesn't see "
        "it. Excluded because the stub provides it by construction."),
    "tp_worker": (
        "set in _Stub.__init__ as self.tp_worker = _FakeWorker(...) -- "
        "the production code reads it for the address books. The stub "
        "provides it by construction."),
    "_weg2_xchg_tag_seen": (
        "dataclass field (weight_updater.py:451, default None), read only "
        "inside `if tag is not None and phase == bx.PHASE_DEPOSIT:` -- "
        "xchg_leg_replay.py's rank_proc calls _weg2_xchg_bounce_leg with no "
        "`tag=` (it replays whole legs from manifests, not the per-tag "
        "lockstep #1374 added), so `tag` is always None and the branch "
        "never runs"),
    "_weg2_xchg_undrained_lanes": (
        "method (weight_updater.py:3906), read only inside `if phase == "
        "bx.PHASE_COLLECT and rank is not None and ...` -- xchg_leg_replay."
        "py never passes `rank=` (the replay's own cross-process teardown "
        "and digest collect replace the in-process lane audit), so it "
        "defaults to None and the branch never runs"),
}
