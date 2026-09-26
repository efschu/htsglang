# SPDX-License-Identifier: Apache-2.0
"""One launcher line, two groups -- the P/D env and argv split for Next Flash.

THE PROBLEM THIS SOLVES, in one sentence: the Next-Flash flip starts TWO
server groups from ONE launcher (P = PP3 prefill, D = Form A decode), and the
two need OPPOSITE values for the same environment variables -- so an
environment that is merely INHERITED is a boot that measures two different
layouts as if they were one.

The 27B flip already has this shape: ``weg2/launcher.py`` builds ``argv_p()``
(``--tp-size 1 --pp-size 3``) and ``argv_d()`` (``--tp-size 3 --pp-size 1``)
separately and diverges the environment in ``build_env(..., group="P"|"D")``.
This module is the Next-Flash content of that divergence, as pure data, so it
can be tested without a launcher, a GPU or a model.

THE TWO AXES THAT DIVERGE, both measured, neither optional:

  (1) DCP -- seam D of DESIGN_FLIP_NEXTFLASH_0920.md.
      P runs DCP ON: ``arm_fnw.sh:85-89`` sets ``SGLANG_UNEVEN_DCP=1`` and
      ``SGLANG_UNEVEN_DCP_WEIGHTED=1`` since WP3b (16.09.), with the comment
      "DCP=0 nur noch als Rueckfall-A/B".
      D runs DCP OFF: under Form A one rank holds every head and every token,
      so ``resolve_dcp_under_host_kv`` (``rank_role.py:906-947``) collapses
      dcp_size to 1 and REFUSES (``RankRoleError``, ``:925-938``) an explicit
      non-1 value rather than ignoring it.
      A flip that lets the D group inherit P's ``SGLANG_UNEVEN_DCP=1`` lands
      in that refusal -- loudly, which is better than silently, but inside
      the GPU window instead of at the desk. :func:`build_group_env` moves it
      to the desk.

  (2) The context ceiling -- seam A / slice 1.
      Both groups must run the mandatory 262144 (Memory
      ``KONTEXT-262K-PFLICHT``). The trap is NOT ``--context-length``: it is
      ``--max-total-tokens``. ``_apply_token_constraints``
      (``model_runner_kv_cache_mixin.py:5110-5134``) takes
      ``min(profiled_capacity, max_total_tokens)``, and the hybrid
      mamba/attention ceiling above it (``:5074-5108``) is
      ``max_running_requests x (context_len + req_to_token_extra)`` -- 262151
      for one request at 262144, which is what fnFA19 line 1027 prints.
      The PP3 launcher template ``run_fn7s2.sh`` carries
      ``EXTRA_FLAGS="${EXTRA_FLAGS:---max-total-tokens 40000}"`` as its
      DEFAULT. At CTX=262144 that default clamps the pool to 40000 tokens --
      15 % of the demanded context -- and nothing refuses, because a user
      limit BELOW the profiled capacity is the documented, intended use of
      the flag. The boot would come up healthy and fail the needle.
      :func:`solve_group_context` is that missing refusal, W113, at the desk.

What this module deliberately does NOT do (same discipline as
``flip_nextflash_plan`` and ``form_a_plan``): it reads no NVML, no torch, no
device and no live environment. Everything is injected; every default carries
the file and line it was read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from sglang.srt.flip_nextflash_plan import (
    FlipInfeasible,
    Weg2FlipKvRelayInfeasible,
)
from sglang.srt.name_compat import canonical_env
from sglang.srt.weg2.form import PROFILE_NEXTFLASH as _PROFILE_NEXTFLASH
from sglang.srt.weg2.form import PROFILES as _PROFILES

__all__ = [
    "FLIP_CONTEXT_TOKENS",
    "Weg2FlipGroupEnvInherited",
    "GroupContextPlan",
    "solve_group_context",
    "GroupEnvPlan",
    "build_group_env",
    "FlipGroupPlan",
    "build_flip_groups",
    "P_GROUP",
    "D_GROUP",
    "REQ_TO_TOKEN_EXTRA_MEASURED",
]

#: Memory ``KONTEXT-262K-PFLICHT`` (user, 19.09.): Next Flash is ALWAYS
#: computed and booted at 262144. Not a default -- a law. Both groups.
FLIP_CONTEXT_TOKENS = 262144

P_GROUP = "P"
D_GROUP = "D"

#: ``get_req_to_token_extra_context_len(server_args)`` as it resolved on
#: fnFA19: the hybrid cap printed ``262144 -> 262151``, so the per-request
#: headroom beyond context_len was 7 tokens at ``max_running_requests=1``
#: (``model_runner_kv_cache_mixin.py:5052-5057``). Injected, not computed:
#: this module does not own the spec-decode arithmetic that produces it.
REQ_TO_TOKEN_EXTRA_MEASURED = 7


class Weg2FlipGroupEnvInherited(FlipInfeasible):
    """W116 -- a flip group would inherit an environment value it must own.

    Raised at the desk for the value that would otherwise be refused inside
    the GPU window (``RankRoleError``, ``rank_role.py:925-938``) or, worse,
    accepted and silently measured as a different layout.

    The rule this enforces is Memory ``Uneven nie ab``: the fix is NEVER an
    empty override that unsets the axis. The D group sets its value
    EXPLICITLY (``DCP=0``, exactly as ``launch_fnFA19.sh`` does), and the P
    group keeps its own.
    """


# ==========================================================================
# Slice 1 (a): the context ceiling, and the --max-total-tokens trap
# ==========================================================================
@dataclass(frozen=True)
class GroupContextPlan:
    """What one group's KV pool is actually allowed to reach."""

    group: str
    context_tokens: int
    max_running_requests: int
    req_to_token_extra: int
    hybrid_cap_tokens: int
    user_limit_tokens: Optional[int]
    reachable_tokens: int

    def report(self) -> str:
        limit = (
            "none" if self.user_limit_tokens is None else str(self.user_limit_tokens)
        )
        return (
            f"CONTEXT CEILING group {self.group}: context {self.context_tokens}, "
            f"hybrid cap {self.hybrid_cap_tokens} "
            f"({self.max_running_requests} x ({self.context_tokens} + "
            f"{self.req_to_token_extra})), --max-total-tokens {limit} "
            f"-> reachable {self.reachable_tokens}"
        )


def solve_group_context(
    group: str,
    context_tokens: int = FLIP_CONTEXT_TOKENS,
    max_running_requests: int = 1,
    req_to_token_extra: int = REQ_TO_TOKEN_EXTRA_MEASURED,
    max_total_tokens: Optional[int] = None,
) -> GroupContextPlan:
    """Does this group's pool reach the context it is configured to accept?

    Three numbers, in the order the runtime applies them:

      1. ``context_tokens`` -- what ``--context-length`` admits per request.
      2. ``hybrid_cap`` -- ``max_running_requests x (context + extra)``, the
         PHYSICAL ceiling the hybrid mamba/attention sizing imposes
         (``_hybrid_kv_token_cap``). This is a ceiling on the pool, never a
         floor: it only ever shrinks a profiled capacity.
      3. ``max_total_tokens`` -- the operator's cap, applied as a plain
         ``min`` (``_apply_token_constraints``, :5127-5134). A value BELOW
         the context is legal, silent, and fatal for a 262k needle.

    The refusal is W113 because this is seam A seen from the other end: the
    KV relay asks "does the destination hold 262144 tokens"; this asks "is
    the destination even ALLOWED to hold them".
    """
    if context_tokens <= 0:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- group {group!r} has "
            f"context_tokens={context_tokens}, which is not a context"
        )
    if max_running_requests <= 0:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- group {group!r} has "
            f"max_running_requests={max_running_requests}; the hybrid cap "
            f"({max_running_requests} x context) would be non-positive and "
            f"the pool ceiling unreadable"
        )
    if req_to_token_extra < 0:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- group {group!r} has a negative "
            f"per-request headroom {req_to_token_extra}"
        )

    hybrid_cap = max_running_requests * (context_tokens + req_to_token_extra)
    reachable = hybrid_cap
    if max_total_tokens is not None:
        if max_total_tokens <= 0:
            raise Weg2FlipKvRelayInfeasible(
                f"W113 Weg2FlipKvRelayInfeasible -- group {group!r} has "
                f"--max-total-tokens {max_total_tokens}, which is not a pool"
            )
        reachable = min(reachable, max_total_tokens)

    if reachable < context_tokens:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- group {group!r} admits requests "
            f"of {context_tokens} tokens (--context-length) but its KV pool is "
            f"capped at {reachable} tokens "
            + (
                f"by --max-total-tokens {max_total_tokens} "
                if max_total_tokens is not None and max_total_tokens <= hybrid_cap
                else f"by the hybrid mamba/attention ceiling "
                f"{max_running_requests} x ({context_tokens} + "
                f"{req_to_token_extra}) = {hybrid_cap} "
            )
            + f"-- short by {context_tokens - reachable} tokens. This does NOT "
            f"refuse at boot: --max-total-tokens below the profiled capacity is "
            f"the documented use of the flag (model_runner_kv_cache_mixin.py:"
            f"5127-5134 logs a warning only when it is LARGER). The boot would "
            f"come up healthy and fail the needle. Raise --max-total-tokens to "
            f"at least {hybrid_cap}, or lower --context-length."
        )

    return GroupContextPlan(
        group=group,
        context_tokens=context_tokens,
        max_running_requests=max_running_requests,
        req_to_token_extra=req_to_token_extra,
        hybrid_cap_tokens=hybrid_cap,
        user_limit_tokens=max_total_tokens,
        reachable_tokens=reachable,
    )


# ==========================================================================
# Slice 1 (b) / seam D: the per-group environment
# ==========================================================================
#: Variables whose value must be decided PER GROUP and therefore must never
#: reach a group by inheritance. The value is the reason, printed in the
#: refusal so nobody has to look it up.
GROUP_OWNED_ENV: Dict[str, str] = {
    "SGLANG_UNEVEN_DCP": (
        "P runs DCP-aware QSA sparse attention (arm_fnw.sh:85-89, WP3b); "
        "Form A collapses dcp_size to 1 and refuses an explicit non-1 value "
        "(rank_role.py:925-938)"
    ),
    "SGLANG_UNEVEN_DCP_WEIGHTED": (
        "the weighted uneven-DCP geometry follows the same axis; Form A has "
        "no token axis to weight (rank_role.py:939-947)"
    ),
}

#: The per-group values. NOT an empty override on either side -- Memory
#: ``Uneven nie ab`` forbids unsetting an uneven axis; both sides state a
#: value. ``launch_fnFA19.sh`` already passes ``DCP=0`` explicitly and this
#: is that choice, moved from a shell line into testable data.
#: UNIFY S3: the rows live in the model-profile registry (weg2/form.py,
#: ``PROFILES["nextflash"].group_env``); this name is their view, so the
#: launcher's build_env and this module read ONE table.

GROUP_ENV_VALUES: Dict[str, Dict[str, str]] = {
    g: dict(_PROFILES[_PROFILE_NEXTFLASH].group_env[g]) for g in (P_GROUP, D_GROUP)
}


@dataclass(frozen=True)
class GroupEnvPlan:
    group: str
    env: Dict[str, str]
    owned: Tuple[str, ...]

    def report(self) -> str:
        owned = ", ".join(f"{k}={self.env[k]}" for k in self.owned)
        return f"GROUP ENV {self.group}: owns {owned} ({len(self.env)} vars total)"


def build_group_env(
    group: str,
    base_env: Mapping[str, str],
    extra: Optional[Mapping[str, str]] = None,
) -> GroupEnvPlan:
    """The environment for ONE flip group, with the owned axes set explicitly.

    ``base_env`` is what the launcher shares between the groups. If it
    already carries a group-owned variable, that is the inheritance bug this
    function exists to catch: refused as W116 rather than overwritten, because
    an override that silently wins hides which side actually decided.

    The caller therefore keeps group-owned variables OUT of the common
    prefix -- exactly the structure ``weg2/launcher.py:4615``
    (``build_env(..., group="P"|"D", ...)``) already has for
    ``CUDA_VISIBLE_DEVICES``.
    """
    if group not in GROUP_ENV_VALUES:
        raise Weg2FlipGroupEnvInherited(
            f"W116 Weg2FlipGroupEnvInherited -- {group!r} is not a flip group; "
            f"expected one of {sorted(GROUP_ENV_VALUES)}"
        )

    # Rename 1b: W116 judges the CANONICAL name -- a legacy/renamed spelling
    # of a group-owned variable is the same inheritance and must not slip past.
    base_env = canonical_env(dict(base_env))
    leaked = sorted(k for k in GROUP_OWNED_ENV if k in base_env)
    if leaked:
        reasons = "; ".join(f"{k}: {GROUP_OWNED_ENV[k]}" for k in leaked)
        raise Weg2FlipGroupEnvInherited(
            f"W116 Weg2FlipGroupEnvInherited -- group {group!r} would INHERIT "
            f"{', '.join(f'{k}={base_env[k]!r}' for k in leaked)} from the "
            f"common launcher environment. These are per-group axes and must "
            f"be set by the group, never shared: {reasons}. Move them out of "
            f"the common prefix into the per-group env (weg2/launcher.py:4615 "
            f"build_env(..., group=...) is where the two already diverge). Do "
            f"NOT fix this with an empty override -- Memory 'Uneven nie ab' "
            f"forbids unsetting an uneven axis; both sides state a value."
        )

    env = dict(base_env)
    env.update(GROUP_ENV_VALUES[group])
    if extra:
        extra = canonical_env(dict(extra))
        collide = sorted(k for k in extra if k in GROUP_OWNED_ENV)
        if collide:
            raise Weg2FlipGroupEnvInherited(
                f"W116 Weg2FlipGroupEnvInherited -- the caller's `extra` for "
                f"group {group!r} sets group-owned {collide}. That is the same "
                f"bug one layer up: the value belongs in GROUP_ENV_VALUES, "
                f"where both groups' choices are visible side by side."
            )
        env.update(extra)

    return GroupEnvPlan(group=group, env=env, owned=tuple(sorted(GROUP_OWNED_ENV)))


# ==========================================================================
# Slice 1 (c): both groups from one line
# ==========================================================================
@dataclass(frozen=True)
class FlipGroupPlan:
    """The whole flip start, as the launcher needs it."""

    context_tokens: int
    p_env: GroupEnvPlan
    d_env: GroupEnvPlan
    p_context: GroupContextPlan
    d_context: GroupContextPlan
    p_argv: Tuple[str, ...] = field(default=())
    d_argv: Tuple[str, ...] = field(default=())

    def report(self) -> str:
        return "\n".join(
            [
                f"FLIP GROUPS @ {self.context_tokens} tokens",
                "  " + self.p_env.report(),
                "  " + self.p_context.report(),
                "  " + self.d_env.report(),
                "  " + self.d_context.report(),
            ]
        )


def build_flip_groups(
    base_env: Mapping[str, str],
    context_tokens: int = FLIP_CONTEXT_TOKENS,
    max_running_requests: int = 1,
    req_to_token_extra: int = REQ_TO_TOKEN_EXTRA_MEASURED,
    p_max_total_tokens: Optional[int] = None,
    d_max_total_tokens: Optional[int] = None,
    p_extra_env: Optional[Mapping[str, str]] = None,
    d_extra_env: Optional[Mapping[str, str]] = None,
    p_argv: Sequence[str] = (),
    d_argv: Sequence[str] = (),
) -> FlipGroupPlan:
    """Both groups, from one common environment, with both axes decided.

    Every refusal fires HERE, at the desk, before a card is touched:
    W116 for an inherited group-owned axis, W113 for a pool that cannot
    reach the mandatory context.
    """
    p_env = build_group_env(P_GROUP, base_env, p_extra_env)
    d_env = build_group_env(D_GROUP, base_env, d_extra_env)
    p_ctx = solve_group_context(
        P_GROUP,
        context_tokens,
        max_running_requests,
        req_to_token_extra,
        p_max_total_tokens,
    )
    d_ctx = solve_group_context(
        D_GROUP,
        context_tokens,
        max_running_requests,
        req_to_token_extra,
        d_max_total_tokens,
    )
    return FlipGroupPlan(
        context_tokens=context_tokens,
        p_env=p_env,
        d_env=d_env,
        p_context=p_ctx,
        d_context=d_ctx,
        p_argv=tuple(p_argv),
        d_argv=tuple(d_argv),
    )
