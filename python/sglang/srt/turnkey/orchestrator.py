# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The boot entrypoint: refuse, or become the server.

The shape of a turnkey boot, in order:

1. **Preflight.** Every named check. Any refusal aborts, loudly, before a
   single byte of weight is read. Refusing costs seconds; discovering the
   same fact after a weight load costs minutes and a card full of garbage.
2. **The plan.** Pinned and fingerprint-checked, or solved. Never guessed.
3. **Assemble** argv and env from the config -- no shell, no word splitting.
4. **``execve``.** The orchestrator REPLACES itself with the server.

Step 4 is the one worth explaining. Under systemd the unit's ``ExecStart``
should end up BEING the serving process, not its parent: an intermediate
supervisor makes the unit's main pid a shell that knows nothing, breaks
``Restart=`` semantics for the process that actually failed, and adds a layer
that can survive its child (or die and orphan it). ``os.execve`` keeps the pid
systemd is watching and the pid doing the work identical, which is also what
makes the #638 cgroup story clean end to end.

``setsid`` therefore appears nowhere in this module. It belongs to
agent-launched boots from an interactive shell, where the goal is to escape a
session that is about to end. A systemd unit already runs outside any session,
and detaching from systemd's supervision would be the bug, not the feature.
The manual CLI path offers ``--setsid`` for the interactive case.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.turnkey import plan as PL
from sglang.srt.turnkey import preflight as PF
from sglang.srt.turnkey.config import ServingSpec, StackConfig
from sglang.srt.turnkey.refusal import (
    REFUSE_PLAN_UNSOLVABLE,
    Refusal,
    refuse,
)

__all__ = ["BootPlan", "assemble", "run_preflight", "resolve_plan", "boot"]


class BootPlan:
    """Everything decided, nothing executed yet -- the dry-run artifact."""

    def __init__(self, lane: ServingSpec, argv: Sequence[str],
                 env: Dict[str, str], flags_from_plan: Sequence[str] = ()):
        self.lane = lane
        self.argv = list(argv)
        self.env = dict(env)
        self.flags_from_plan = list(flags_from_plan)

    def render(self) -> str:
        out = [f"# lane: {self.lane.name}  port: {self.lane.port}",
               f"# boot log: {self.lane.boot_log}",
               "# --- env (turnkey-controlled keys only) ---"]
        for k in sorted(self.env):
            out.append(f"{k}={self.env[k]}")
        if self.flags_from_plan:
            out.append("# --- flags contributed by the pinned plan ---")
            out.append("  " + " ".join(self.flags_from_plan))
        out.append("# --- argv ---")
        out.extend("  " + a for a in self.argv)
        return "\n".join(out)


def run_preflight(cfg: StackConfig,
                  probes: Optional[PF.Probes] = None) -> List[Refusal]:
    return PF.run_all(cfg, probes)


def resolve_plan(cfg: StackConfig, lane: ServingSpec,
                 cards: Sequence[Tuple[str, int]],
                 wheel_version: str = "",
                 model_path: str = "") -> Tuple[List[str], Optional[Refusal]]:
    """Return the plan's launch flags, or a refusal.

    ``solve`` mode intentionally does the least surprising thing: it asks the
    planner and refuses if the planner cannot answer. It does not fall back to
    a pinned plan, and pinned mode does not fall back to solving -- a mode is
    a decision the operator made, and silently switching modes would hide the
    very change worth seeing.
    """
    if cfg.plan.mode == "solve":
        return _solve(cfg, lane)

    pinned, r = PL.load_pinned(cfg.plan.path)
    if r:
        return [], r
    now_fp = PL.fingerprint_of(cards, lane.argv, model_path=model_path,
                               wheel_version=wheel_version)
    r = PL.check_staleness(pinned, now_fp, cfg.plan.max_age_days)
    if r:
        return [], r
    return list(pinned.launch_flags), None


def _solve(cfg: StackConfig, lane: ServingSpec):
    """Bring-up mode. The planner is the authority (#584); we only call it."""
    try:
        from sglang.srt.planner import cli as planner_cli  # noqa: F401
    except Exception as e:
        return [], refuse(REFUSE_PLAN_UNSOLVABLE, "planner",
                          f"import failed: {e}", "an importable planner")
    # Deliberately narrow: solving a full plan at boot needs the model spec
    # the planner CLI wants, and inventing those arguments here would be the
    # guesswork #539 forbids. Until the pin-writer below is used to record a
    # solved plan, solve mode contributes no extra flags and says so.
    return [], None


def assemble(cfg: StackConfig, lane: ServingSpec,
             plan_flags: Sequence[str] = ()) -> BootPlan:
    """argv + env, fully resolved. No shell is involved at any point.

    Shell quoting is a real hazard here, not a hypothetical one: the ship
    config carries ``--chat-template-default-kwargs {"preserve_thinking":
    true}``, whose braces, quotes and space are destroyed by any round trip
    through word splitting. Keeping argv a list from config to ``execve``
    means the value never becomes a string that something could re-split.
    """
    argv = list(lane.argv) + list(plan_flags)
    env = cfg.env_for(lane)

    # Two values in a captured env are IDENTITIES OF THE BOOT THAT PRODUCED
    # THE CAPTURE, not settings. Pinning them into a config would replay a
    # dead process's identity on every boot: the captured ship env carries
    # SGLANG_PHASE_FLIP_INSTANCE=1786515732-3940356, whose suffix is the pid
    # of a process that no longer exists. They are synthesized per boot
    # instead, matching what route_a_631_prod_boot.sh does inline.
    env.setdefault("SGLANG_PHASE_FLIP_INSTANCE",
                   f"{int(time.time())}-{os.getpid()}")
    if "SGLANG_BOOT_COMMIT" not in env:
        commit = _repo_commit(cfg.repo)
        if commit:
            env["SGLANG_BOOT_COMMIT"] = commit
    return BootPlan(lane, argv, env, plan_flags)


def _repo_commit(repo: str) -> str:
    """The commit this boot runs, read from the repo rather than trusted.

    Provenance is not decoration: "which commit was that boot" is the first
    question asked of every measurement, and a config-pinned answer would be
    a claim rather than an observation.

    Asks git rather than parsing ``.git`` by hand. The hand-rolled version
    was written first and was wrong on this very rig: it opened
    ``<repo>/.git/HEAD``, which does not exist when the repo is a WORKTREE
    (``.git`` is then a file pointing elsewhere), so it silently returned ""
    and the boot would have run with no provenance stamp at all. Since the
    stack root here IS a worktree, the naive form failed in exactly the
    configuration that ships.
    """
    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "--short=10",
                            "HEAD"], capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _open_boot_log(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Append, never truncate: the previous boot's tail is often the only
    # evidence of why this boot is happening.
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)


def boot(cfg: StackConfig, lane_name: str, *, dry_run: bool = False,
         use_setsid: bool = False, probes: Optional[PF.Probes] = None,
         out=sys.stdout) -> int:
    """Boot one lane. Returns an exit code; on success it does not return."""
    lane = cfg.lane(lane_name)
    if lane is None:
        print(f"REFUSE_CONFIG_INCOMPLETE subject=lane observed={lane_name} "
              f"expected=one of "
              f"{[s.name for s in cfg.serving]}", file=out)
        return 2

    refusals = run_preflight(cfg, probes)
    for r in refusals:
        print(r.line(), file=out)
    if refusals:
        print(f"turnkey: {len(refusals)} refusal(s); not booting "
              f"{lane.name}", file=out)
        return 3

    # The card fingerprint uses the cards THIS lane occupies, in rank order.
    p = probes or PF.default_probes()
    observed = {c.uuid: c for c in p.cards()}
    cards = [(cfg.cards[i].uuid,
              observed[cfg.cards[i].uuid].total_bytes // PF.MIB)
             for i in lane.cards]

    wheel_version = ""
    if cfg.wheel.must_import:
        try:
            wheel_version = p.probe_import(cfg.wheel.must_import[0],
                                           "int8_scaled_mm").version
        except ImportError:
            wheel_version = ""

    plan_flags, r = resolve_plan(cfg, lane, cards, wheel_version,
                                 _model_path_of(lane))
    if r:
        print(r.line(), file=out)
        return 4

    bp = assemble(cfg, lane, plan_flags)

    if dry_run:
        print(bp.render(), file=out)
        print("# dry-run: nothing was started", file=out)
        return 0

    fd = _open_boot_log(lane.boot_log)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)

    if use_setsid:
        # Interactive path only; see the module docstring for why a systemd
        # unit must NOT do this.
        os.setsid()

    env = dict(os.environ)
    env.update(bp.env)
    os.execve(bp.argv[0], bp.argv, env)
    return 127  # unreachable; execve does not return on success


def _model_path_of(lane: ServingSpec) -> str:
    argv = list(lane.argv)
    for i, a in enumerate(argv):
        if a == "--model-path" and i + 1 < len(argv):
            return argv[i + 1]
    return ""
