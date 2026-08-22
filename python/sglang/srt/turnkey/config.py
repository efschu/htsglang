# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The stack description a turnkey boot reads -- one file, no guesses.

#539's acceptance is "after a container/host boot the ENTIRE stack stands
fully automatically, with NO LLM guesswork and no manual reconstruction".
That makes the config file the contract: everything a boot needs must be
stated here, and anything not stated is a refusal rather than a default.

Two design rules are load-bearing and both exist because of specific defects
observed on this rig:

**Cards are named by UUID, never by index.** NVML's index order and torch's
enumeration order are different orderings of the same hardware (the
device-order trap, AUDIT_331), and either can move across a reboot or a driver
reload. A config that says ``index = 1`` is a config that silently addresses a
different card after a boot. The UUID is the only stable name, so it is the
only one accepted. The rank ORDER is the order of the ``[[cards]]`` entries.

**The resolved placement is passed to CUDA as UUIDs too.** CUDA accepts
``CUDA_VISIBLE_DEVICES=GPU-<uuid>,GPU-<uuid>``, which removes the last place
an index could be misread -- notably ``CUDA_DEVICE_ORDER``, whose default
(FASTEST_FIRST) is not PCI order, so even a correct NVML index is the wrong
string to hand CUDA. Nothing in this package ever writes a bare integer into
CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.turnkey.refusal import (
    REFUSE_CONFIG_INCOMPLETE,
    REFUSE_CONFIG_UNPARSABLE,
    REFUSE_LOG_PATH_SHARED,
    REFUSE_REPO_IS_WORKTREE,
    raise_refusal,
)

__all__ = [
    "CardSpec",
    "WheelPin",
    "PlanSpec",
    "ServingSpec",
    "WatchdogSpec",
    "PreflightSpec",
    "StackConfig",
    "load",
    "loads",
]


@dataclasses.dataclass(frozen=True)
class CardSpec:
    """One physical card, addressed the only stable way it can be."""

    uuid: str
    #: Human label for logs and refusals ("5090", "3080-a"). Never used to
    #: address hardware.
    label: str = ""
    #: Optional expected model name; when set, preflight compares it against
    #: NVML and refuses on mismatch. This catches "the UUID resolved but the
    #: card in that slot was swapped".
    expect_name: str = ""

    def __post_init__(self):
        if not self.uuid.startswith("GPU-"):
            raise_refusal(
                REFUSE_CONFIG_INCOMPLETE, f"cards[{self.label or '?'}].uuid",
                self.uuid, "a GPU-<uuid> string as printed by "
                "`nvidia-smi --query-gpu=uuid --format=csv`",
                remedy="never use an index here; indices move across reboots")


@dataclasses.dataclass(frozen=True)
class WheelPin:
    """#384 wheel-shadow guard.

    The defect: two distributions provide the same import name, and a plain
    ``pip install`` of one silently removes the arm the other provided (the
    INT8 arm, concretely). Nothing fails at install time and nothing fails at
    boot -- the loss shows up later as a quality regression, which is the
    worst possible place to discover it.

    The guard is therefore a BOOT-TIME check: the named distribution must be
    installed at the pinned version, and every module in ``must_import`` must
    import from inside that distribution's install root.
    """

    dist: str = ""
    version: str = ""
    must_import: Tuple[str, ...] = ()
    #: When set, the resolved module's __file__ must live under this prefix.
    #: This is what actually catches a shadow: the import succeeds either
    #: way, but from the wrong tree.
    expect_prefix: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.dist or self.must_import)


@dataclasses.dataclass(frozen=True)
class PlanSpec:
    """How the boot obtains its cuts/reserves.

    ``mode="pinned"``
        Load a plan file solved earlier and REFUSE if its fingerprint does not
        match the world booting now. This is the production mode: a boot must
        be reproducible, and re-solving on every boot makes the stack's shape
        a function of whatever the planner decided this morning.

    ``mode="solve"``
        Run the planner at boot. Honest but non-reproducible; intended for
        bring-up on new hardware.

    There is deliberately no third mode that falls back from pinned to solve.
    A stale pin means the operator changed something; discovering WHAT is the
    operator's job, and a silent re-solve would hide exactly the change worth
    seeing.
    """

    mode: str = "pinned"
    path: str = ""
    #: Refuse if the pinned plan is older than this many days, even when the
    #: fingerprint still matches. 0 disables the age check.
    max_age_days: int = 0

    def __post_init__(self):
        if self.mode not in ("pinned", "solve"):
            raise_refusal(REFUSE_CONFIG_INCOMPLETE, "plan.mode", self.mode,
                          "one of: pinned, solve")
        if self.mode == "pinned" and not self.path:
            raise_refusal(REFUSE_CONFIG_INCOMPLETE, "plan.path", "<empty>",
                          "a path to the pinned plan json")


@dataclasses.dataclass(frozen=True)
class ServingSpec:
    """One serving lane."""

    name: str
    port: int
    #: The boot command, as an argv LIST. A string would invite shell quoting
    #: bugs into the one place that must be byte-reproducible.
    argv: Tuple[str, ...] = ()
    #: Extra environment, applied on top of the resolved base env.
    env: Tuple[Tuple[str, str], ...] = ()
    #: Cards this lane occupies, as indices INTO the config's card list --
    #: i.e. rank order, resolved to UUIDs before anything touches CUDA.
    cards: Tuple[int, ...] = ()
    #: Per-instance boot log. #375 defect 3: never shared between lanes.
    boot_log: str = ""
    #: Seconds to allow between exec and first healthy generation. JIT
    #: cold-cache boots (#172/#615) are legitimately slow; this is the number
    #: that separates "still compiling" from "wedged", so it is explicit
    #: rather than a constant hidden in the watchdog.
    ready_timeout_s: int = 1800
    enabled: bool = True

    def __post_init__(self):
        if not self.argv:
            raise_refusal(REFUSE_CONFIG_INCOMPLETE, f"serving.{self.name}.argv",
                          "<empty>", "a non-empty argv list")
        if not self.boot_log:
            raise_refusal(REFUSE_CONFIG_INCOMPLETE,
                          f"serving.{self.name}.boot_log", "<empty>",
                          "a per-instance boot log path (#375 defect 3)")


@dataclasses.dataclass(frozen=True)
class WatchdogSpec:
    """#604 policy. See watchdog.py for the state machine these feed."""

    #: Cheap liveness poll.
    poll_s: int = 20
    #: A real generation probe is expensive (it occupies the model), so it
    #: runs on its own slower cadence. This is the probe that separates
    #: "HTTP 200" from "actually generating" -- the #622 family proved a
    #: server can serve 200s while every generation hangs.
    generation_probe_s: int = 120
    generation_timeout_s: int = 60
    #: Consecutive failed generation probes before the lane is declared
    #: wedged. >1 because a single probe can lose to a long legitimate batch.
    wedge_confirmations: int = 3
    #: Backoff ladder between restarts, seconds. The last value repeats.
    backoff_s: Tuple[int, ...] = (30, 60, 120, 300, 600)
    #: Restarts allowed inside the window before the watchdog gives up and
    #: stays loud instead of thrashing a broken lane.
    max_restarts: int = 5
    restart_window_s: int = 3600
    enabled: bool = True
    #: See Policy.generation_probe_enabled. Retired by user order
    #: 2026-08-14; False unless a config explicitly says otherwise.
    generation_probe_enabled: bool = False
    #: #799: consume the scheduler's published admission-wedge verdict. ON by
    #: default -- it is the passive replacement for the retired probe, and a
    #: detector nobody consumes is the defect being fixed, not the safe choice.
    wedge_signal_enabled: bool = True
    #: Directory the schedulers publish their verdicts into. Empty means
    #: ``wedge_status.DEFAULT_STATUS_DIR``; both sides must agree, so a config
    #: that overrides it must also set SGLANG_WEDGE_STATUS_DIR in [env].
    wedge_status_dir: str = ""


@dataclasses.dataclass(frozen=True)
class PreflightSpec:
    host_headroom_gib: int = 15
    disk_paths: Tuple[Tuple[str, int], ...] = ()
    #: A card carrying more than this much foreign VRAM fails the orphan
    #: check. Not zero: the driver itself holds a small carve-out.
    card_busy_mib: int = 512
    check_ports: Tuple[int, ...] = ()
    #: Ports that must NEVER be touched or probed destructively. 30099 is the
    #: local router -- law: never killed from an agent session.
    protected_ports: Tuple[int, ...] = (30099,)
    #: Refuse the boot when this rig has no cached VRAM-ledger calibration.
    #: DEFAULT FALSE, deliberately: the ledger is the sizing authority and is
    #: on by default, but an unpriced term makes it fall back to the inherited
    #: heuristic and SAY SO rather than fail, so a fresh rig still boots. A
    #: rig that wants the exact numbers guaranteed sets this to true and gets
    #: a named refusal instead of a quiet fallback.
    require_vram_calibration: bool = False


@dataclasses.dataclass(frozen=True)
class StackConfig:
    name: str
    repo: str
    venv: str
    log_dir: str
    cards: Tuple[CardSpec, ...]
    serving: Tuple[ServingSpec, ...]
    #: Acknowledge that [stack].repo is a git worktree. See
    #: :func:`assert_repo_stable` -- being a worktree is not itself the
    #: hazard, being an unacknowledged one is.
    allow_worktree: bool = False
    wheel: WheelPin = dataclasses.field(default_factory=WheelPin)
    plan: PlanSpec = dataclasses.field(default_factory=lambda: PlanSpec(
        mode="solve"))
    watchdog: WatchdogSpec = dataclasses.field(default_factory=WatchdogSpec)
    preflight: PreflightSpec = dataclasses.field(default_factory=PreflightSpec)
    #: Base environment every lane inherits. The canonical rig env block
    #: lives here rather than in a shell script so that one file describes
    #: the boot.
    base_env: Tuple[Tuple[str, str], ...] = ()

    # -- derived ----------------------------------------------------------

    def card_by_uuid(self, uuid: str) -> Optional[CardSpec]:
        for c in self.cards:
            if c.uuid == uuid:
                return c
        return None

    def lane(self, name: str) -> Optional[ServingSpec]:
        for s in self.serving:
            if s.name == name:
                return s
        return None

    def enabled_lanes(self) -> Tuple[ServingSpec, ...]:
        return tuple(s for s in self.serving if s.enabled)

    def visible_devices(self, lane: ServingSpec) -> str:
        """The CUDA_VISIBLE_DEVICES string for a lane -- UUIDs, in rank order.

        Deliberately returns UUIDs and not indices. See the module docstring:
        an index is only correct relative to an enumeration order, and the
        default CUDA order is not the NVML one.
        """
        out = []
        for i in lane.cards:
            if i < 0 or i >= len(self.cards):
                raise_refusal(
                    REFUSE_CONFIG_INCOMPLETE, f"serving.{lane.name}.cards",
                    i, f"an index into the {len(self.cards)} configured cards")
            out.append(self.cards[i].uuid)
        return ",".join(out)

    def env_for(self, lane: ServingSpec) -> Dict[str, str]:
        """Base env + lane env + the resolved device string."""
        env = dict(self.base_env)
        env.update(dict(lane.env))
        env["CUDA_VISIBLE_DEVICES"] = self.visible_devices(lane)
        # PYTHONPATH must point at THIS checkout's python/ tree. The
        # worktree-PYTHONPATH trap: a lane launched without it silently runs
        # the code of whatever tree happens to be installed, and every
        # measurement taken against it is a measurement of the wrong build.
        env.setdefault("PYTHONPATH", os.path.join(self.repo, "python"))
        return env


# --- parsing --------------------------------------------------------------


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise_refusal(REFUSE_CONFIG_INCOMPLETE, f"{where}.{key}", "<missing>",
                      "a value; the turnkey path does not default it")
    return d[key]


def _pairs(d: Any) -> Tuple[Tuple[str, str], ...]:
    if not d:
        return ()
    return tuple((str(k), str(v)) for k, v in dict(d).items())


def loads(text: str, *, source: str = "<string>") -> StackConfig:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise_refusal(REFUSE_CONFIG_UNPARSABLE, source, str(e), "valid TOML")
    return _build(raw, source)


def load(path: str) -> StackConfig:
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
    except OSError as e:
        raise_refusal(REFUSE_CONFIG_UNPARSABLE, path, str(e),
                      "a readable stack config")
    return loads(text, source=path)


def _build(raw: dict, source: str) -> StackConfig:
    stack = _req(raw, "stack", source)
    repo = _req(stack, "repo", "stack")

    cards_raw = raw.get("cards") or []
    if not cards_raw:
        raise_refusal(REFUSE_CONFIG_INCOMPLETE, "cards", "<empty>",
                      "at least one [[cards]] entry addressed by UUID")
    cards = tuple(
        CardSpec(uuid=_req(c, "uuid", "cards"), label=c.get("label", ""),
                 expect_name=c.get("expect_name", ""))
        for c in cards_raw)

    seen = set()
    for c in cards:
        if c.uuid in seen:
            raise_refusal(REFUSE_CONFIG_INCOMPLETE, "cards.uuid", c.uuid,
                          "each card listed once")
        seen.add(c.uuid)

    serving_raw = raw.get("serving") or {}
    lanes: List[ServingSpec] = []
    for name, s in serving_raw.items():
        lanes.append(ServingSpec(
            name=name,
            port=int(_req(s, "port", f"serving.{name}")),
            argv=tuple(str(a) for a in _req(s, "argv", f"serving.{name}")),
            env=_pairs(s.get("env")),
            cards=tuple(int(i) for i in s.get("cards", ())),
            boot_log=s.get("boot_log", ""),
            ready_timeout_s=int(s.get("ready_timeout_s", 1800)),
            enabled=bool(s.get("enabled", True)),
        ))
    if not lanes:
        raise_refusal(REFUSE_CONFIG_INCOMPLETE, "serving", "<empty>",
                      "at least one [serving.<name>] lane")

    # #375 defect 3 -- shared boot-log path between lanes.
    by_log: Dict[str, str] = {}
    for lane in lanes:
        prev = by_log.get(lane.boot_log)
        if prev is not None:
            raise_refusal(
                REFUSE_LOG_PATH_SHARED, lane.boot_log,
                f"lanes {prev} and {lane.name}", "one boot log per lane",
                remedy="interleaved output proves nothing about either lane")
        by_log[lane.boot_log] = lane.name

    ports: Dict[int, str] = {}
    for lane in lanes:
        prev_p = ports.get(lane.port)
        if prev_p is not None:
            raise_refusal(REFUSE_CONFIG_INCOMPLETE, f"serving.{lane.name}.port",
                          lane.port, f"a free port; {prev_p} already uses it")
        ports[lane.port] = lane.name

    w = raw.get("wheel") or {}
    wheel = WheelPin(
        dist=w.get("dist", ""), version=w.get("version", ""),
        must_import=tuple(w.get("must_import", ())),
        expect_prefix=w.get("expect_prefix", ""))

    # No [plan] table at all means no plan has been pinned yet, so the honest
    # mode is "solve". Defaulting to "pinned" here would refuse every config
    # that has not yet recorded a pin -- turning the absence of an optional
    # feature into a boot failure. Once [plan] IS present, mode="pinned"
    # requires its path (PlanSpec.__post_init__), because a pin without a
    # file is a claim with nothing behind it.
    p = raw.get("plan")
    if p is None:
        plan = PlanSpec(mode="solve", path="")
    else:
        plan = PlanSpec(mode=p.get("mode", "pinned"), path=p.get("path", ""),
                        max_age_days=int(p.get("max_age_days", 0)))

    wd = raw.get("watchdog") or {}
    watchdog = WatchdogSpec(
        poll_s=int(wd.get("poll_s", 20)),
        generation_probe_s=int(wd.get("generation_probe_s", 120)),
        generation_timeout_s=int(wd.get("generation_timeout_s", 60)),
        wedge_confirmations=int(wd.get("wedge_confirmations", 3)),
        backoff_s=tuple(int(x) for x in wd.get("backoff_s",
                                               (30, 60, 120, 300, 600))),
        max_restarts=int(wd.get("max_restarts", 5)),
        restart_window_s=int(wd.get("restart_window_s", 3600)),
        generation_probe_enabled=bool(
            wd.get("generation_probe_enabled", False)),
        wedge_signal_enabled=bool(wd.get("wedge_signal_enabled", True)),
        wedge_status_dir=str(wd.get("wedge_status_dir", "") or ""),
        enabled=bool(wd.get("enabled", True)))
    if not watchdog.backoff_s:
        raise_refusal(REFUSE_CONFIG_INCOMPLETE, "watchdog.backoff_s", "<empty>",
                      "at least one backoff value")
    if watchdog.wedge_confirmations < 1:
        raise_refusal(REFUSE_CONFIG_INCOMPLETE, "watchdog.wedge_confirmations",
                      watchdog.wedge_confirmations, ">= 1")

    pf = raw.get("preflight") or {}
    preflight = PreflightSpec(
        host_headroom_gib=int(pf.get("host_headroom_gib", 15)),
        disk_paths=tuple((str(k), int(v))
                         for k, v in dict(pf.get("disk_free_gib", {})).items()),
        card_busy_mib=int(pf.get("card_busy_mib", 512)),
        check_ports=tuple(int(x) for x in pf.get("check_ports", ())),
        protected_ports=tuple(int(x)
                              for x in pf.get("protected_ports", (30099,))),
        require_vram_calibration=bool(
            pf.get("require_vram_calibration", False)))

    for port in preflight.check_ports:
        if port in preflight.protected_ports:
            raise_refusal(
                REFUSE_CONFIG_INCOMPLETE, "preflight.check_ports", port,
                "a port that is not protected",
                remedy="protected ports host foreign services; the turnkey "
                       "path must never expect them free")

    cfg = StackConfig(
        name=stack.get("name", "htsglang"),
        repo=repo,
        allow_worktree=bool(stack.get("allow_worktree", False)),
        venv=_req(stack, "venv", "stack"),
        log_dir=_req(stack, "log_dir", "stack"),
        cards=cards,
        serving=tuple(lanes),
        wheel=wheel,
        plan=plan,
        watchdog=watchdog,
        preflight=preflight,
        base_env=_pairs(raw.get("env")),
    )

    # Resolve the card indices eagerly so a bad index refuses at parse time
    # rather than at exec time.
    for lane in cfg.enabled_lanes():
        cfg.visible_devices(lane)

    return cfg


def assert_repo_stable(repo: str, allow_worktree: bool = False) -> None:
    """Refuse a stack rooted somewhere that can vanish under the service.

    The motivating defect is real and was live on this rig on 2026-08-12: the
    planner daemon serving :8780 ran with cwd ``/spinning/wt-631-routea``, a
    worktree that an unrelated agent can delete with ``git worktree remove``.
    A service rooted there stops being bootable without anything reporting it.

    The first draft of this check simply refused every worktree, and measuring
    it against the machine proved that too strict: ``/spinning/htsglang-gpu``
    -- which holds the venv and is already named by the installed router unit
    -- is ITSELF a worktree of ``/spinning/htsglang``. Being a worktree is
    therefore not the hazard; being an UNACKNOWLEDGED or BROKEN one is.

    So there are two distinct verdicts:

    * a ``.git`` pointer that does not resolve is refused unconditionally --
      the checkout is already unusable, and no acknowledgement can fix that;
    * a resolvable worktree is refused only until the config says
      ``allow_worktree = true``, which records that a human looked at the
      path and accepted its lifetime. That is a decision, written down --
      not a default and not a guess.
    """
    dotgit = os.path.join(repo, ".git")
    if not os.path.exists(dotgit):
        raise_refusal(REFUSE_REPO_IS_WORKTREE, repo, "no .git",
                      "a git checkout root")
    if os.path.isdir(dotgit):
        return

    # A worktree: .git is a file holding "gitdir: <path>".
    try:
        with open(dotgit) as fh:
            pointer = fh.read().strip()
    except OSError as e:
        raise_refusal(REFUSE_REPO_IS_WORKTREE, repo, f"unreadable .git: {e}",
                      "a readable git pointer")
    target = pointer.split("gitdir:", 1)[-1].strip() if "gitdir:" in pointer \
        else ""
    if not target or not os.path.exists(target):
        raise_refusal(
            REFUSE_REPO_IS_WORKTREE, repo,
            f"worktree pointer does not resolve ({pointer!r})",
            "a worktree whose gitdir exists",
            remedy="the worktree was removed or pruned; re-create it or "
                   "point [stack].repo at the main checkout")
    if not allow_worktree:
        raise_refusal(
            REFUSE_REPO_IS_WORKTREE, repo,
            f"a git worktree (gitdir {target})",
            "the main checkout, or [stack].allow_worktree = true",
            remedy="a worktree can be removed out from under the service; "
                   "set allow_worktree only for a long-lived one")


#: Kept as an alias so the older name in any operator notes still resolves.
assert_repo_not_worktree = assert_repo_stable
