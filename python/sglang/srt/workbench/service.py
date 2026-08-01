# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The workbench, assembled (DESIGN #347).

One object the HTTP surface talks to, built from the server arguments. The
split is the same one #341 draws: everything above this line is protocol and
everything below it is execution, which is what lets the whole surface be
exercised on a card-less host with mock tenants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from sglang.srt.training.feasibility import MachineResources
from sglang.srt.training.tenant import (
    IdleMonitor,
    local_activity_source,
    registry_activity_source,
)
from sglang.srt.workbench.arb import ArbDirectory
from sglang.srt.workbench.log import WorkLog
from sglang.srt.workbench.scheduler import Workbench, WorkbenchConfig
from sglang.srt.workbench.tenant import IdleWorkTenant

logger = logging.getLogger(__name__)

#: Registered in this order when ``--workbench-tenants`` is not given.
#: ``boot_matrix`` (#349) is NOT in the default set: it boots full TP servers
#: and needs a model path, so it is opt-in via ``--workbench-tenants`` with a
#: configured model rather than a silent background consumer.
DEFAULT_TENANTS = ("training", "fp8_tuner", "card_probe")


class WorkbenchError(RuntimeError):
    """Base for the errors the HTTP surface turns into status codes."""

    status_code = 400
    code = "workbench_error"


class WorkbenchDisabled(WorkbenchError):
    """The surface is served but the workbench is switched off."""

    status_code = 503
    code = "workbench_disabled"


class UnknownTenant(WorkbenchError):
    """No tenant by that name is registered."""

    status_code = 404
    code = "unknown_tenant"


class WorkbenchService:
    """Tenant registration, the scheduler, and the payloads the routes return."""

    def __init__(
        self,
        config: WorkbenchConfig,
        *,
        tenants: Optional[Sequence[IdleWorkTenant]] = None,
        monitor: Optional[IdleMonitor] = None,
        reservation_store: Any = None,
        machine_resolver: Optional[Callable[[], MachineResources]] = None,
        arb: Optional[ArbDirectory] = None,
        log: Optional[WorkLog] = None,
    ) -> None:
        self.config = config
        self.reservation_store = reservation_store
        root = Path(config.artifact_root)
        self.machine_resolver = machine_resolver or (
            lambda: _probe_machine(root, reservation_store)
        )
        self.monitor = monitor or IdleMonitor(
            [local_activity_source, registry_activity_source()],
            grace_seconds=config.grace_seconds,
        )
        if arb is None and config.arb_dir:
            arb = ArbDirectory(
                config.arb_dir,
                session=config.arb_session,
                accounted=_ledger_accounting(reservation_store),
            )
        self.arb = arb
        self.workbench = Workbench(
            list(tenants or ()),
            config=config,
            monitor=self.monitor,
            reservation_store=reservation_store,
            machine_resolver=self.machine_resolver,
            log=log,
            arb=arb,
        )

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self.config.enabled:
            self.workbench.start()

    async def stop(self) -> None:
        await self.workbench.stop()

    # -- payloads -----------------------------------------------------------

    def require_enabled(self) -> None:
        if not self.config.enabled:
            raise WorkbenchDisabled(
                "The idle workbench is not enabled on this server. Restart "
                "with --enable-idle-workbench to run queued idle work."
            )

    def snapshot(self) -> dict[str, Any]:
        body = self.workbench.snapshot()
        body["enabled"] = self.config.enabled
        return body

    def events(self, *, after: int = 0, limit: int = 200) -> dict[str, Any]:
        entries, has_more = self.workbench.log.after(after, limit=limit)
        return {
            "object": "list",
            "data": [e.to_json() for e in entries],
            "has_more": has_more,
            "last_seq": entries[-1].seq if entries else int(after),
        }

    def pause(self, *, paused: bool, tenant: Optional[str] = None) -> dict[str, Any]:
        self.require_enabled()
        if tenant is None:
            self.workbench.pause(paused)
            return {"paused": self.workbench.paused, "tenant": None}
        target = self._tenant(tenant)
        target.pause() if paused else target.resume()
        self.workbench.log.append(
            target.name, "info", "paused by request" if paused else "resumed by request"
        )
        if paused and self.workbench.snapshot().get("running") == target.name:
            self.workbench.force_preempt.set()
        self.workbench.wake()
        return {"paused": target.paused, "tenant": target.name}

    def enqueue(self, *, tenant: str, item: Mapping[str, Any]) -> dict[str, Any]:
        self.require_enabled()
        target = self._tenant(tenant)
        try:
            key = target.enqueue(item)
        except NotImplementedError as exc:
            raise WorkbenchError(str(exc)) from None
        except ValueError as exc:
            raise WorkbenchError(str(exc)) from None
        self.workbench.log.append(
            target.name, "info", f"enqueued {key}", data=dict(item)
        )
        self.workbench.wake()
        return {"tenant": target.name, "item": key, "pending": target.pending()}

    def _tenant(self, name: str) -> IdleWorkTenant:
        try:
            return self.workbench.tenant(name)
        except KeyError as exc:
            raise UnknownTenant(str(exc)) from None


# ---------------------------------------------------------------------------
# Assembly from server arguments
# ---------------------------------------------------------------------------


def default_artifact_root() -> Path:
    import os

    explicit = os.environ.get("HTSGLANG_WORKBENCH_ROOT")
    if explicit:
        return Path(explicit)
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "htsglang" / "workbench"


def build_config(server_args: Any) -> WorkbenchConfig:
    import os

    root = (
        Path(server_args.workbench_artifact_root)
        if getattr(server_args, "workbench_artifact_root", None)
        else default_artifact_root()
    )
    arb_dir = str(getattr(server_args, "workbench_arb_dir", "") or "")
    if not arb_dir:
        arb_dir = os.environ.get("HTSGLANG_GPU_ARB_DIR", "")
    return WorkbenchConfig(
        enabled=bool(getattr(server_args, "enable_idle_workbench", False)),
        artifact_root=root,
        grace_seconds=float(
            getattr(server_args, "workbench_idle_grace_seconds", 120.0)
        ),
        poll_seconds=float(getattr(server_args, "workbench_poll_seconds", 2.0)),
        preempt_timeout_s=float(
            getattr(server_args, "workbench_preempt_timeout_s", 60.0)
        ),
        segment_timeout_s=float(
            getattr(server_args, "workbench_segment_timeout_s", 1800.0)
        ),
        arb_dir=arb_dir,
        arb_heartbeat_s=float(getattr(server_args, "workbench_arb_heartbeat_s", 300.0)),
    )


def build_tenants(
    server_args: Any,
    config: WorkbenchConfig,
    *,
    training_service: Any = None,
) -> list[IdleWorkTenant]:
    """Instantiate the tenants ``--workbench-tenants`` names.

    An unknown name is a startup error rather than a silent skip: a typo that
    quietly disables a tenant produces a rig that looks idle-managed and is
    not, which is worse than not booting.
    """
    from sglang.srt.workbench.tenants.boot_matrix import BootMatrixTenant
    from sglang.srt.workbench.tenants.card_probe import CardProbeTenant
    from sglang.srt.workbench.tenants.fp8_tuner import Fp8BlockTunerTenant
    from sglang.srt.workbench.tenants.training import TrainingWorkTenant

    raw = str(getattr(server_args, "workbench_tenants", "") or "")
    names = [n.strip() for n in raw.split(",") if n.strip()] or list(DEFAULT_TENANTS)
    root = Path(config.artifact_root)
    out: list[IdleWorkTenant] = []
    for name in names:
        if name == "training":
            if training_service is None:
                logger.warning(
                    "idle workbench: no training service to register; the "
                    "'training' tenant is skipped"
                )
                continue
            out.append(TrainingWorkTenant(training_service))
        elif name == "fp8_tuner":
            queue = getattr(server_args, "workbench_tuner_queue", None)
            out.append(
                Fp8BlockTunerTenant(
                    artifact_root=root / "fp8_tuner",
                    queue_path=Path(queue) if queue else None,
                    card_selector=str(
                        getattr(server_args, "workbench_tuner_card", "largest")
                        or "largest"
                    ),
                )
            )
        elif name == "card_probe":
            out.append(
                CardProbeTenant(
                    max_age_s=float(
                        getattr(server_args, "workbench_probe_max_age_s", 604800.0)
                    )
                )
            )
        elif name == "boot_matrix":
            out.append(
                BootMatrixTenant(
                    artifact_root=root / "boot_matrix",
                    model_path=str(
                        getattr(server_args, "workbench_boot_matrix_model", "") or ""
                    ),
                )
            )
        else:
            raise ValueError(
                f"--workbench-tenants names an unknown tenant {name!r}; known "
                f"tenants are {', '.join((*DEFAULT_TENANTS, 'boot_matrix'))}"
            )
    return out


def build_service(
    server_args: Any, *, training_service: Any = None
) -> WorkbenchService:
    """The whole assembly, from parsed server arguments."""
    config = build_config(server_args)
    reservation_store = None
    if config.enabled:
        try:
            from sglang.srt.registry.ledger import ReservationStore

            reservation_store = ReservationStore()
        except Exception as exc:  # noqa: BLE001 - the workbench runs without it
            logger.warning(
                "idle workbench: no VRAM ledger (%s: %s); segments will run "
                "without a cross-process reservation",
                type(exc).__name__,
                exc,
            )
    tenants = build_tenants(server_args, config, training_service=training_service)
    return WorkbenchService(
        config, tenants=tenants, reservation_store=reservation_store
    )


def _probe_machine(root: Path, store: Any) -> MachineResources:
    from sglang.srt.training.feasibility import probe_machine

    return probe_machine(artifact_root=root, store=store)


def _ledger_accounting(store: Any):
    """Bytes the ledger says are legitimately held, per NVML index.

    Passed to the arbitration directory so a resident serving engine does not
    look like a foreign process squatting on the card. Unaccounted memory
    still refuses the claim -- that is the case the check exists for.
    """

    def accounted(indices: Sequence[int]) -> dict[int, int]:
        if store is None:
            return {}
        from sglang.srt.registry import nvml

        wanted = {int(i) for i in indices}
        out: dict[int, int] = {}
        for device in nvml.list_devices():
            if device.index not in wanted:
                continue
            ledger = store.read(device.uuid)
            out[device.index] = sum(e.reserved_bytes for e in ledger.gpu_resident)
        return out

    return accounted


__all__ = [
    "DEFAULT_TENANTS",
    "UnknownTenant",
    "WorkbenchDisabled",
    "WorkbenchError",
    "WorkbenchService",
    "build_config",
    "build_service",
    "build_tenants",
    "default_artifact_root",
]
