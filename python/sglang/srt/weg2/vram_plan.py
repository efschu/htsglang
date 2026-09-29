# SPDX-License-Identifier: Apache-2.0
"""VRAM-VERTRAG M1 (29.09.): ``vram_plan.json`` as a VIEW on today's solvers.

The launcher's budget passes (P budgets, map, dry, early D, D-only, real D)
each write one plan: per card and group the posts the solvers booked, each
with its provenance, the other group's sleep residue, and a CLOSURE per card
x phase x state (sum / total / rest / rest_to / bound_by). A rest of at least
one expert row that no user intent binds is IDLE -- VRAM the planner left
unused (/spinning/gpu-arb/docs/VRAM-VERTRAG-0929.md §3.5). M1 changes no
boot, argv or bolt: the plan only makes visible what the solvers decided.

Stdlib only: the rank side (M6) and the host side read the same schema.
The reader refuses a foreign schema version or an unknown field by name.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "weg2.vram_plan/1"
#: the passes that write a plan, in boot order
PASSES = ("p_budget", "map", "dry", "d_early", "d_only", "d")
PROVENANCE_KINDS = ("RECORD", "BUILTIN", "MODEL", "BORROWED", "UNMEASURED", "OVERRIDE")
#: a user intent that binds a rest (§3.5): not a planner error
BOUND_BY = ("ctx_max", "seats", "objective")
FIXED_CATEGORIES = ("weights", "draft", "experts_resident", "state_pools", "graphs", "cuda_ctx")
REST_TO = ("experts_resident", "experts_lru", "kv", "none")

TOP_KEYS = ("schema", "plan_id", "boot_id", "rev", "pass", "identity", "cards", "groups",
            "asleep", "budget_terms", "flip_legs", "closure", "open", "overrides")
CARD_KEYS = ("uuid", "nvml", "class", "total_mib", "driver_reserved_mib", "user_reserve_mib")
GROUP_KEYS = ("ranks",)
RANK_KEYS = ("card", "fixed", "elastic", "transient_by_state", "budget_mib",
             "torch_cache_cap_mib", "provenance", "cells")
CLOSURE_KEYS = ("card", "phase", "state", "sum_mib", "total_mib", "rest_mib", "row_mib",
                "rest_to", "bound_by", "idle", "terms")
OVERRIDE_KEYS = ("group", "key", "value", "source")


class VramPlanRefused(ValueError):
    """A plan this reader cannot trust, named by code (VRAM_PLAN_SCHEMA,
    VRAM_PLAN_FIELD, VRAM_PLAN_ID)."""

    def __init__(self, code: str, detail: str):
        super().__init__("%s: %s" % (code, detail))
        self.code = code


@dataclass(frozen=True)
class Closure:
    """One card x phase x state: what the solvers booked against the card."""

    card: str
    phase: str
    state: str
    total_mib: int
    terms: Tuple[Tuple[str, int], ...]
    row_mib: int
    rest_to: str = "experts_resident"
    bound_by: str = ""

    @property
    def sum_mib(self) -> int:
        return int(sum(int(v) for _k, v in self.terms))

    @property
    def rest_mib(self) -> int:
        return int(self.total_mib) - self.sum_mib

    @property
    def idle(self) -> bool:
        """At least one expert row left over that no user intent binds."""
        return self.rest_mib >= max(1, int(self.row_mib)) and not self.bound_by

    def to_dict(self) -> dict:
        return {"card": self.card, "phase": self.phase, "state": self.state,
                "sum_mib": self.sum_mib, "total_mib": int(self.total_mib),
                "rest_mib": self.rest_mib, "row_mib": int(self.row_mib),
                "rest_to": self.rest_to, "bound_by": self.bound_by, "idle": self.idle,
                "terms": {k: int(v) for k, v in self.terms}}


def bound_by_of(*, objective: str = "", ctx_at_max: bool = False, experts_full: bool = False,
                seats_at_cap: bool = False, rest_to: str = "experts_resident") -> str:
    """The user intent that binds a rest (§3.5), '' if none.

    - ``objective``: the profile asks for performance, not capacity (maxperf).
    - ``ctx_max``: every expert is resident already (the rest cannot become
      expert rows) and the KV holds the full context -- more KV serves nobody.
    - ``seats``: the rest could only buy seats and the seat count is reached.
    A rest that could still become expert rows is never bound by ctx or seats."""
    if str(objective).strip().lower() == "maxperf":
        return "objective"
    if experts_full and ctx_at_max:
        return "ctx_max"
    if rest_to == "none" and seats_at_cap:
        return "seats"
    return ""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_plan_id(plan: Mapping) -> str:
    """sha256 over the plan without its own id (same inputs -> same id)."""
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    return "sha256:" + hashlib.sha256(canonical(body).encode()).hexdigest()


def provenance_ok(text: str) -> bool:
    return any(str(text).startswith(k) for k in PROVENANCE_KINDS)


def make_plan(*, boot_id: str, rev: str, pass_name: str, identity: Mapping,
              cards: Sequence[Mapping], groups: Mapping, asleep: Mapping,
              budget_terms: Mapping, closure: Sequence[Closure], open_items: Sequence[str],
              overrides: Sequence[Mapping], flip_legs: Optional[Mapping] = None) -> dict:
    """The plan as a dict with its id; every nested key checked by the reader."""
    if pass_name not in PASSES:
        raise VramPlanRefused("VRAM_PLAN_FIELD", "pass %r not in %s" % (pass_name, PASSES))
    plan = {
        "schema": SCHEMA, "plan_id": "", "boot_id": str(boot_id or ""), "rev": str(rev or ""),
        "pass": pass_name, "identity": dict(identity), "cards": [dict(c) for c in cards],
        "groups": json.loads(canonical(groups)), "asleep": json.loads(canonical(asleep)),
        "budget_terms": json.loads(canonical(budget_terms)),
        "flip_legs": dict(flip_legs or {}), "closure": [c.to_dict() for c in closure],
        "open": sorted(set(str(x) for x in open_items)),
        "overrides": [dict(o) for o in overrides],
    }
    plan["plan_id"] = compute_plan_id(plan)
    read_plan(plan)
    return plan


def _unknown(where: str, got: Mapping, allowed: Sequence[str]) -> None:
    extra = sorted(set(got) - set(allowed))
    if extra:
        raise VramPlanRefused("VRAM_PLAN_FIELD", "%s: unknown field(s) %s" % (where, extra))


def read_plan(plan: Mapping) -> Mapping:
    """Check a plan against the schema; refuse by name. Returns it unchanged."""
    if not isinstance(plan, Mapping):
        raise VramPlanRefused("VRAM_PLAN_SCHEMA", "not an object")
    if plan.get("schema") != SCHEMA:
        raise VramPlanRefused("VRAM_PLAN_SCHEMA", "schema %r, this reader knows %r"
                              % (plan.get("schema"), SCHEMA))
    _unknown("plan", plan, TOP_KEYS)
    missing = [k for k in TOP_KEYS if k not in plan]
    if missing:
        raise VramPlanRefused("VRAM_PLAN_FIELD", "plan: missing field(s) %s" % missing)
    for i, c in enumerate(plan["cards"]):
        _unknown("cards[%d]" % i, c, CARD_KEYS)
    for g, body in plan["groups"].items():
        _unknown("groups.%s" % g, body, GROUP_KEYS)
        for r, rank in body.get("ranks", {}).items():
            _unknown("groups.%s.ranks.%s" % (g, r), rank, RANK_KEYS)
            for post, prov in (rank.get("provenance") or {}).items():
                if not provenance_ok(prov):
                    raise VramPlanRefused(
                        "VRAM_PLAN_FIELD", "groups.%s.ranks.%s.provenance.%s = %r is none of %s"
                        % (g, r, post, prov, PROVENANCE_KINDS))
    for i, c in enumerate(plan["closure"]):
        _unknown("closure[%d]" % i, c, CLOSURE_KEYS)
        if c.get("bound_by") and c["bound_by"] not in BOUND_BY:
            raise VramPlanRefused("VRAM_PLAN_FIELD", "closure[%d].bound_by %r not in %s"
                                  % (i, c["bound_by"], BOUND_BY))
    for i, o in enumerate(plan["overrides"]):
        _unknown("overrides[%d]" % i, o, OVERRIDE_KEYS)
    if plan["plan_id"] != compute_plan_id(plan):
        raise VramPlanRefused("VRAM_PLAN_ID", "plan_id %r does not hash its content"
                              % (plan["plan_id"],))
    return plan


def load_plan(path: str) -> Mapping:
    with open(path) as fh:
        return read_plan(json.load(fh))


def plan_file_name(pass_name: str) -> str:
    return "vram_plan-%s.json" % pass_name


def write_plan(state_dir: str, plan: Mapping, write_json_atomic) -> List[str]:
    """``vram_plan-<pass>.json`` and ``vram_plan.json`` (the latest pass, the
    one the boot runs with) into the boot's state directory."""
    os.makedirs(state_dir, exist_ok=True)
    paths = [os.path.join(state_dir, plan_file_name(plan["pass"])),
             os.path.join(state_dir, "vram_plan.json")]
    for p in paths:
        write_json_atomic(p, dict(plan))
    return paths


def idle_items(plan: Mapping, nvml_of: Optional[Mapping[str, int]] = None) -> List[str]:
    """``nvml<k>:<rest>(<bound_by or ->)`` for every closure row with a rest of
    at least one expert row -- bound or not (bound rows name their intent)."""
    nvml_of = nvml_of or {c["uuid"]: c["nvml"] for c in plan.get("cards", ())}
    out = []
    for c in plan.get("closure", ()):
        if int(c["rest_mib"]) >= max(1, int(c.get("row_mib") or 0)):
            out.append("%s/nvml%s:%d(%s)" % (c["phase"], nvml_of.get(c["card"], "?"),
                                             int(c["rest_mib"]), c.get("bound_by") or "-"))
    return out


def plan_line(plan: Mapping) -> str:
    """The one human line per pass (logs are for people, the file is the record)."""
    ov = ["%s:%s=%s" % (o["group"], o["key"], o["value"]) for o in plan.get("overrides", ())]
    return ("VRAM-PLAN pass=%s plan_id=%s open=[%s] idle=[%s] overrides=[%s]"
            % (plan["pass"], plan["plan_id"][:19], ", ".join(plan.get("open", ())),
               ", ".join(idle_items(plan)), ", ".join(ov)))


def _flat(obj, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(_flat(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flat(v, "%s[%d]" % (prefix, i)))
    else:
        out[prefix] = obj
    return out


def diff_plans(a: Mapping, b: Mapping, *, sections: Sequence[str] = (
        "groups", "asleep", "closure"), only_common: bool = True
               ) -> List[Tuple[str, object, object]]:
    """Every term that MOVED between two passes' plans (path, a, b): a post
    both passes price with a different value. A post only one pass prices (the
    P pass has no D group) is not a move; ``only_common=False`` lists those too."""
    out = []
    for s in sections:
        fa, fb = _flat(a.get(s, {}), s), _flat(b.get(s, {}), s)
        keys = (set(fa) & set(fb)) if only_common else (set(fa) | set(fb))
        for k in sorted(keys):
            if fa.get(k) != fb.get(k):
                out.append((k, fa.get(k), fb.get(k)))
    return out


def argv_budget_mismatches(plan: Mapping, group: str, argv: Sequence[str]) -> List[str]:
    """M1 acceptance (a): the plan's budget_mib per rank equals the group's
    ``--rank-gpu-memory-mib`` in the argv, rank by rank (card order)."""
    argv = list(argv)
    try:
        vals = [int(x) for x in argv[argv.index("--rank-gpu-memory-mib") + 1].split(",")]
    except (ValueError, IndexError):
        return ["argv of group %s names no --rank-gpu-memory-mib" % group]
    ranks = plan["groups"].get(group, {}).get("ranks", {})
    order = [c["uuid"] for c in plan["cards"]]
    by_card = {r["card"]: int(r["budget_mib"]) for r in ranks.values()}
    out = []
    for i, uuid in enumerate(order):
        if i < len(vals) and uuid in by_card and by_card[uuid] != vals[i]:
            out.append("%s ordinal %d: plan %d != argv %d" % (group, i, by_card[uuid], vals[i]))
    return out
