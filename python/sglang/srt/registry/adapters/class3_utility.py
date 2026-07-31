# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``Class3UtilityAdapter``: single-pass and utility tenants (#333 §6, §7.6).

Two shapes, one adapter, because from the registry's side they are the same
thing -- a process that holds a declared number of bytes on named cards and
can be started and stopped:

``pooling``
    An ``srt`` server booted with ``--is-embedding``. This is the shape that
    makes M1's acceptance gate possible at all: ``is_generation`` is
    process-global, so a generation engine and an embedding engine cannot be
    one process. The registry resolves that by not fighting it -- two tenants,
    one control plane, requests served concurrently from one logical endpoint.

``process``
    An opaque argv with a declared budget. This is how the M2 video-enhance
    executor becomes a registry tenant without the registry learning what a
    frame is. Its own ledger reservation, written when it ran standalone, is
    the same reservation the registry now manages: one store, one invariant.

Class 3's ladder is the trivial one (§6.3): ``HOT`` or ``COLD``. What is
expensive here is the engine *build*, which is a disk-cache problem and not a
residency one, so no ``WARM_GPU`` rung is invented to look symmetric.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from typing import Any, Mapping

from sglang.srt.registry.adapter import (
    AdapterContext,
    AdapterError,
    EstimateError,
    Health,
    register_adapter,
)
from sglang.srt.registry.adapters.class1_srt import (
    DEFAULT_BOOT_TIMEOUT_S,
    DEFAULT_READY_POLL_S,
    Class1SrtAdapter,
)
from sglang.srt.registry.adapters.process import (
    ChildProcess,
    ProcessTenantError,
    http_json,
    http_ok,
    wait_for,
)
from sglang.srt.registry.ledger import MIB
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
)

logger = logging.getLogger(__name__)

ADAPTER_NAME = "class3_utility"

POOLING_MODE = "pooling"
PROCESS_MODE = "process"


class _PoolingTenant(Class1SrtAdapter):
    """An ``srt`` pooling server. Same process mechanics, different class.

    Inheriting rather than copying is deliberate: an embedding server *is* an
    ``srt`` server, and the boot, health, memory-tag and NVML-measurement
    mechanics are identical. What differs is the class number, the launch flag
    and the probe -- and only those are overridden.
    """

    klass = int(EngineClass.UTILITY)

    def build_argv(self) -> list[str]:
        argv = super().build_argv()
        if "--is-embedding" not in argv:
            argv.append("--is-embedding")
        return argv

    def promote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_GPU:
            # §6.3: a pooling engine's pools are small and its graphs cheap;
            # a WARM_GPU rung would cost a recapture and save almost nothing.
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 3 has no WARM_GPU rung "
                "(§6.3); the ladder is HOT / COLD"
            )
        super().promote(target)

    def demote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_GPU:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 3 has no WARM_GPU rung (§6.3)"
            )
        super().demote(target)

    def embed_probe(self, text: str = "hello") -> int:
        """Embed one string; return the vector length. Used by the harness."""
        result = http_json(
            f"{self.base_url}/v1/embeddings",
            method="POST",
            payload={
                "model": self.launch.get("served_model_name", "embedding"),
                "input": text,
            },
            timeout=120.0,
        )
        if isinstance(result, dict) and result.get("data"):
            return len(result["data"][0]["embedding"])
        raise AdapterError(f"unexpected embedding response: {result!r}")


class _OpaqueProcessTenant:
    """An arbitrary process with a declared budget and a health URL."""

    klass = int(EngineClass.UTILITY)

    def __init__(self, spec: EngineSpec, context: AdapterContext) -> None:
        self.spec = spec
        self.context = context
        self.launch: Mapping[str, Any] = dict(spec.launch)
        self._state = ResidencyState.COLD
        self._process: ChildProcess | None = None
        self._cards: tuple[str, ...] = ()
        if not self.launch.get("argv"):
            raise EstimateError(
                f"engine {spec.engine_id!r}: mode {PROCESS_MODE!r} needs launch.argv"
            )
        if self.launch.get("budget_mib") is None:
            raise EstimateError(
                f"engine {spec.engine_id!r}: mode {PROCESS_MODE!r} needs "
                "launch.budget_mib, the absolute per-card budget in MiB"
            )

    def estimate(self, spec: EngineSpec, cards: tuple[str, ...]) -> ResourceProfile:
        per_card = int(self.launch["budget_mib"]) * MIB
        return ResourceProfile(
            posts={c: {"declared_budget": per_card} for c in cards},
            peak_bytes={c: per_card for c in cards},
            notes=(
                "opaque process tenant: the budget is declared, not derived. "
                "The same absolute-MiB semantics as --rank-gpu-memory-mib -- the "
                "whole budget, no ceiling applied on top.",
            ),
        )

    def bind(self, cards: tuple[str, ...]) -> None:
        self._cards = tuple(cards)

    def state(self) -> ResidencyState:
        return self._state

    def pids(self) -> tuple[int, ...]:
        return self._process.child_pids() if self._process is not None else ()

    def promote(self, target: ResidencyState) -> None:
        if target != ResidencyState.HOT:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 3 process tenants are HOT or "
                f"COLD (§6.3); {target.value} is not a rung here"
            )
        if self._state == ResidencyState.HOT:
            return
        argv = self.launch["argv"]
        if isinstance(argv, str):
            argv = shlex.split(argv)
        argv = [str(a).replace("{python}", sys.executable) for a in argv]
        env = dict(self.launch.get("env") or {})
        if self._cards:
            # One physical GPU per process, named by UUID: inside the tenant
            # cuda:0 is unambiguous and no index mapping can go wrong.
            env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(self._cards))
        log_path = (
            os.path.join(self.context.work_dir, f"{self.spec.engine_id}.log")
            if self.context.work_dir
            else None
        )
        self._process = ChildProcess(argv=argv, env=env, log_path=log_path)
        self._process.start()
        health_url = self.launch.get("health_url")
        if health_url:
            try:
                wait_for(
                    lambda: http_ok(health_url),
                    timeout_s=float(
                        self.launch.get("boot_timeout_s", DEFAULT_BOOT_TIMEOUT_S)
                    ),
                    poll_s=DEFAULT_READY_POLL_S,
                    on_dead=lambda: (
                        None
                        if self._process is None or self._process.running
                        else f"exit code {self._process.returncode}"
                    ),
                    what=f"engine {self.spec.engine_id!r} health",
                )
            except ProcessTenantError as exc:
                self._process.stop()
                self._process = None
                raise AdapterError(str(exc)) from None
        self._state = ResidencyState.HOT

    def demote(self, target: ResidencyState) -> None:
        if target != ResidencyState.COLD:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 3 process tenants are HOT or "
                f"COLD (§6.3); {target.value} is not a rung here"
            )
        if self._process is not None:
            self._process.stop()
            self._process = None
        self._state = ResidencyState.COLD

    def measured(self) -> Mapping[str, int]:
        pids = set(self.pids())
        if not pids or not self._cards:
            return {}
        from sglang.srt.registry.nvml import process_bytes_on_uuid  # noqa: PLC0415

        out: dict[str, int] = {}
        for card in self._cards:
            try:
                per_pid = process_bytes_on_uuid(card)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.debug("registry: measured() on %s failed: %s", card, exc)
                continue
            out[card] = sum(b for pid, b in per_pid.items() if pid in pids)
        return out

    def health(self) -> Health:
        if self._state == ResidencyState.COLD:
            return Health(ok=True, detail="cold, no process")
        if self._process is None or not self._process.running:
            return Health(ok=False, detail="process is gone")
        url = self.launch.get("health_url")
        if url and not http_ok(url):
            return Health(ok=False, detail=f"no health answer from {url}")
        return Health(ok=True, detail=f"pid {self._process.pid}")


def build(spec: EngineSpec, context: AdapterContext) -> Any:
    mode = str(spec.launch.get("mode", POOLING_MODE))
    if mode == POOLING_MODE:
        return _PoolingTenant(spec, context)
    if mode == PROCESS_MODE:
        return _OpaqueProcessTenant(spec, context)
    raise EstimateError(
        f"engine {spec.engine_id!r}: unknown class-3 mode {mode!r}; "
        f"use {POOLING_MODE!r} or {PROCESS_MODE!r}"
    )


register_adapter(ADAPTER_NAME, build)
