# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``Class1SrtAdapter``: an autoregressive ``srt`` server as a registry tenant.

Class 1 exists; §4 of DESIGN #333 records only what changes. What changes is
that the server becomes a *tenant*: its budget comes from the ledger rather
than from a global flag, its residency moves along the §3.4 ladder, and the
registry -- not the operator -- decides when it boots.

Ladder, per §4.3, and no more than that:

===========  ==========================================================
``HOT``      process up, weights and pools and graphs resident
``WARM_GPU`` process up, ``kv_cache`` and ``cuda_graph`` tags released;
             weights stay on the device, so promotion is a resume
``COLD``     no process. With #89 hibernate the weights were parked to
             disk first, so the next boot restores them instead of
             re-reading and re-quantising the checkpoint
===========  ==========================================================

``WARM_HOST`` is refused, loudly. Sharded, quantised, post-processed weights
are not a single ``.to("cpu")``, and §4.3 puts it out of scope; an adapter
that silently mapped it onto ``COLD`` would make the ladder a lie.

Budget. The spec declares an absolute per-rank MiB budget, exactly the
``--rank-gpu-memory-mib`` semantics: the whole budget, no implicit ceiling, no
safety factor, no rounding down. A card hosting two ranks of the same engine
therefore reserves twice that. Leaving headroom is the operator's
responsibility; the ledger only refuses the physically impossible.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import time
from typing import Any, Mapping

from sglang.srt.registry.adapter import (
    AdapterContext,
    AdapterError,
    EstimateError,
    Health,
    register_adapter,
)
from sglang.srt.registry.adapters.process import (
    ChildProcess,
    ProcessTenantError,
    cuda_runtime_library_path,
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

ADAPTER_NAME = "class1_srt"

#: A cold boot of a large quantised model is minutes, not seconds. The default
#: is generous because the failure mode of a short timeout is worse than a slow
#: promotion: the registry tears down a server that was about to be ready.
DEFAULT_BOOT_TIMEOUT_S = 900.0
DEFAULT_READY_POLL_S = 1.0

#: Memory-saver tags. ``WARM_GPU`` keeps weights and releases everything the
#: server can rebuild without reading the checkpoint again.
WARM_GPU_TAGS = ("kv_cache", "cuda_graph")
ALL_TAGS = ("weights", "kv_cache", "cuda_graph")


class Class1SrtAdapter:
    """One ``srt`` server process tree, driven along the residency ladder."""

    klass = int(EngineClass.AUTOREGRESSIVE)

    def __init__(self, spec: EngineSpec, context: AdapterContext) -> None:
        self.spec = spec
        self.context = context
        self.launch: Mapping[str, Any] = dict(spec.launch)
        self._state = ResidencyState.COLD
        self._process: ChildProcess | None = None
        self._cards: tuple[str, ...] = ()
        self._last_error: str | None = None
        self._validate_launch()

    # -- configuration -----------------------------------------------------

    def _validate_launch(self) -> None:
        if not self.launch.get("model_path"):
            raise EstimateError(
                f"engine {self.spec.engine_id!r}: launch.model_path is required"
            )
        tp = int(self.launch.get("tp_size", 1))
        if tp < 1:
            raise EstimateError(f"engine {self.spec.engine_id!r}: tp_size must be >= 1")
        for forbidden, why in (
            ("pp_size", "pipeline parallelism"),
            ("dp_size", "data parallelism"),
            ("ep_size", "expert parallelism"),
        ):
            value = self.launch.get(forbidden)
            if value not in (None, 1):
                # Same hard limit #89 and the rank-placement feature already
                # draw. A registry that half-supported PP would produce a
                # reservation that is right for one stage and wrong for the
                # rest.
                raise EstimateError(
                    f"engine {self.spec.engine_id!r}: {why} ({forbidden}={value}) is "
                    "outside the V1 scope of the registry, which is pure "
                    "single-node tensor parallelism"
                )
        rank_cards = self.launch.get("rank_cards")
        if rank_cards is not None and len(rank_cards) != tp:
            raise EstimateError(
                f"engine {self.spec.engine_id!r}: launch.rank_cards has "
                f"{len(rank_cards)} entries but tp_size is {tp}; there is exactly "
                "one card per rank"
            )
        if (
            self.launch.get("rank_gpu_memory_mib") is None
            and self.launch.get("gpu_memory_utilization") is None
        ):
            raise EstimateError(
                f"engine {self.spec.engine_id!r}: declare a budget with "
                "launch.rank_gpu_memory_mib (absolute MiB per rank, preferred: "
                "it is unambiguous on unlike cards) or launch.gpu_memory_"
                "utilization (a fraction of each card's NVML total)"
            )

    @property
    def tp_size(self) -> int:
        return int(self.launch.get("tp_size", 1))

    def rank_cards(self, cards: tuple[str, ...]) -> tuple[str, ...]:
        """Card UUID per rank. The registry-level ``--rank-gpu-id``.

        Explicit ``launch.rank_cards`` wins, because it is the only way to
        express two ranks sharing one card. Otherwise ranks map onto the
        placed cards in order, which requires as many cards as ranks.
        """
        declared = self.launch.get("rank_cards")
        if declared:
            return tuple(str(c) for c in declared)
        if len(cards) == self.tp_size:
            return tuple(cards)
        if len(cards) == 1:
            return tuple(cards) * self.tp_size
        raise EstimateError(
            f"engine {self.spec.engine_id!r}: {self.tp_size} ranks cannot be placed "
            f"on {len(cards)} card(s) without launch.rank_cards saying which rank "
            "goes where"
        )

    # -- planning ----------------------------------------------------------

    def estimate(self, spec: EngineSpec, cards: tuple[str, ...]) -> ResourceProfile:
        ranks = self.rank_cards(cards)
        unknown = set(ranks) - set(cards)
        if unknown:
            raise EstimateError(
                f"engine {spec.engine_id!r}: launch.rank_cards names card(s) "
                f"{sorted(unknown)} that are not in the placement {list(cards)}"
            )
        mib = self.launch.get("rank_gpu_memory_mib")
        peak: dict[str, int] = {}
        posts: dict[str, dict[str, int]] = {}
        for card in cards:
            on_card = sum(1 for r in ranks if r == card)
            if on_card == 0:
                continue
            if mib is not None:
                per_rank = int(mib) * MIB
            else:
                total = self.context.card_totals.get(card)
                if total is None:
                    raise EstimateError(
                        f"engine {spec.engine_id!r}: gpu_memory_utilization needs the "
                        f"NVML total of card {card}, which the registry could not "
                        "resolve"
                    )
                per_rank = int(float(self.launch["gpu_memory_utilization"]) * total)
            peak[card] = per_rank * on_card
            posts[card] = {
                # Pre-boot there is exactly one honest post: what the operator
                # declared. The §4.2 breakdown (weights/kv/graphs/ctx) is
                # measured, not predicted, and arrives through measured().
                "declared_rank_budget": per_rank
                * on_card,
            }
        notes = [
            f"{self.tp_size} rank(s) over {len(peak)} card(s); "
            f"{'absolute MiB budget' if mib is not None else 'fraction of NVML total'}",
            "reservation is the peak the tenant may reach; under #287 a rung may "
            "move freely inside it and a rung that would exceed it is rejected at "
            "plan time (§3.8)",
        ]
        return ResourceProfile(posts=posts, peak_bytes=peak, notes=tuple(notes))

    # -- residency ---------------------------------------------------------

    def state(self) -> ResidencyState:
        return self._state

    def pids(self) -> tuple[int, ...]:
        return self._process.child_pids() if self._process is not None else ()

    def bind(self, cards: tuple[str, ...]) -> None:
        """Tell the adapter which cards the arbiter placed it on."""
        self._cards = tuple(cards)

    def promote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_HOST:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 1 has no WARM_HOST rung "
                "(§4.3) -- sharded, quantised, post-processed weights are not a "
                "single .to('cpu'). The ladder here is HOT / WARM_GPU / COLD."
            )
        if target == self._state:
            return
        if target == ResidencyState.COLD:
            raise AdapterError("COLD is a demotion, not a promotion")
        if self._state == ResidencyState.COLD:
            self._boot()
            self._state = (
                ResidencyState.WARM_GPU
                if target is ResidencyState.WARM_GPU
                else ResidencyState.HOT
            )
            if target is ResidencyState.WARM_GPU:
                self._release(WARM_GPU_TAGS)
            return
        if self._state == ResidencyState.WARM_GPU and target == ResidencyState.HOT:
            self._resume(WARM_GPU_TAGS)
            self._state = ResidencyState.HOT
            return
        raise AdapterError(
            f"engine {self.spec.engine_id!r}: no promotion path "
            f"{self._state.value} -> {target.value}"
        )

    def demote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_HOST:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 1 has no WARM_HOST rung (§4.3)"
            )
        if target == self._state:
            return
        if target == ResidencyState.WARM_GPU:
            if self._state != ResidencyState.HOT:
                raise AdapterError(
                    f"engine {self.spec.engine_id!r}: cannot demote "
                    f"{self._state.value} -> WARM_GPU"
                )
            self._release(WARM_GPU_TAGS)
            self._state = ResidencyState.WARM_GPU
            return
        if target == ResidencyState.COLD:
            self._park_and_stop()
            self._state = ResidencyState.COLD
            return
        raise AdapterError(f"unsupported demotion target {target.value}")

    # -- process control ---------------------------------------------------

    @property
    def base_url(self) -> str:
        host = self.launch.get("host", "127.0.0.1")
        return f"http://{host}:{int(self.launch['port'])}"

    def build_argv(self) -> list[str]:
        """The launch command. Explicit, logged, and reproducible by hand."""
        launch = self.launch
        argv = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(launch["model_path"]),
            "--host",
            str(launch.get("host", "127.0.0.1")),
            "--port",
            str(int(launch["port"])),
            "--tp-size",
            str(self.tp_size),
        ]
        if launch.get("served_model_name"):
            argv += ["--served-model-name", str(launch["served_model_name"])]
        if launch.get("rank_gpu_memory_mib") is not None:
            # --rank-gpu-id and --rank-gpu-memory-mib are each mandatory when
            # the other is set, and together they replace the global
            # utilisation flag rather than complementing it.
            argv += [
                "--rank-gpu-id",
                ",".join(str(i) for i in self.visible_rank_indices()),
                "--rank-gpu-memory-mib",
                str(int(launch["rank_gpu_memory_mib"])),
            ]
        elif launch.get("gpu_memory_utilization") is not None:
            argv += [
                "--gpu-memory-utilization",
                str(float(launch["gpu_memory_utilization"])),
            ]
        if launch.get("hibernate_dir"):
            argv += [
                "--enable-weights-disk-backup",
                "--hibernate-dir",
                str(launch["hibernate_dir"]),
            ]
        if launch.get("enable_memory_saver", True):
            # The WARM_GPU rung releases the kv_cache and cuda_graph tags, and
            # #89's park releases the weights tag. Without the memory saver
            # those releases are no-ops that free nothing, so the rung would
            # report success and hold every byte. Opt out only for an engine
            # that will never leave HOT.
            argv.append("--enable-memory-saver")
        extra = launch.get("extra_args") or []
        if isinstance(extra, str):
            extra = shlex.split(extra)
        argv += [str(a) for a in extra]
        return argv

    def visible_rank_indices(self) -> tuple[int, ...]:
        """``--rank-gpu-id`` values, as indices into this tenant's visible set.

        The card is a UUID everywhere in the registry; the CLI wants an
        integer. Rather than translate a UUID into "the" device index -- there
        is no such thing, CUDA enumerates FASTEST_FIRST and NVML by PCI bus,
        and picking the wrong one silently boots the engine on the wrong card
        -- the child is pinned with ``CUDA_VISIBLE_DEVICES`` to exactly this
        tenant's cards, in placement order. Index i is then placement[i] by
        construction, in every enumeration order there is.
        """
        cards = self._cards or tuple(self.spec.placement)
        order = {uuid: i for i, uuid in enumerate(cards)}
        return tuple(order[u] for u in self.rank_cards(cards))

    def visible_devices(self) -> str:
        return ",".join(self._cards or tuple(self.spec.placement))

    def _boot(self) -> None:
        log_path = None
        if self.context.work_dir:
            log_path = os.path.join(
                self.context.work_dir, f"{self.spec.engine_id}.server.log"
            )
        env = {"CUDA_VISIBLE_DEVICES": self.visible_devices()}
        if self.launch.get("enable_memory_saver", True):
            extra = cuda_runtime_library_path()
            if extra:
                inherited = os.environ.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{extra}{os.pathsep}{inherited}" if inherited else extra
                )
        env.update(self.launch.get("env") or {})
        self._process = ChildProcess(
            argv=self.build_argv(),
            env=env,
            log_path=log_path,
        )
        self._process.start()
        timeout = float(self.launch.get("boot_timeout_s", DEFAULT_BOOT_TIMEOUT_S))
        try:
            wait_for(
                lambda: http_ok(f"{self.base_url}/health"),
                timeout_s=timeout,
                poll_s=DEFAULT_READY_POLL_S,
                on_dead=lambda: (
                    None
                    if self._process is None or self._process.running
                    else f"exit code {self._process.returncode}"
                    + (f", log {log_path}" if log_path else "")
                ),
                what=f"engine {self.spec.engine_id!r} to answer /health",
            )
        except ProcessTenantError as exc:
            self._last_error = str(exc)
            self._process.stop()
            self._process = None
            raise AdapterError(str(exc)) from None
        self._last_error = None

    def _park_and_stop(self) -> None:
        if self._process is None:
            return
        if self.launch.get("hibernate_dir") and self._state in (
            ResidencyState.HOT,
            ResidencyState.WARM_GPU,
        ):
            # #89: park the FINAL post-transform weights so the next boot is a
            # disk read instead of a checkpoint load. GGUF-scoped upstream; a
            # non-GGUF engine simply stops, which is correct and slower.
            try:
                http_json(f"{self.base_url}/hibernate", method="POST", timeout=600.0)
            except Exception as exc:  # noqa: BLE001 - stopping must not be blocked
                logger.warning(
                    "registry: engine %s could not hibernate (%s); stopping anyway. "
                    "The next promotion will be a full load.",
                    self.spec.engine_id,
                    exc,
                )
        self._process.stop()
        self._process = None

    def _release(self, tags: tuple[str, ...]) -> None:
        self._post_memory("release_memory_occupation", tags)

    def _resume(self, tags: tuple[str, ...]) -> None:
        self._post_memory("resume_memory_occupation", tags)

    def _post_memory(self, endpoint: str, tags: tuple[str, ...]) -> None:
        if self._process is None:
            raise AdapterError(
                f"engine {self.spec.engine_id!r} has no process to {endpoint}"
            )
        try:
            http_json(
                f"{self.base_url}/{endpoint}",
                method="POST",
                payload={"tags": list(tags)},
                timeout=float(self.launch.get("memory_op_timeout_s", 300.0)),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an adapter error
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: {endpoint} {list(tags)} failed: {exc}"
            ) from None

    # -- observability -----------------------------------------------------

    def measured(self) -> Mapping[str, int]:
        """Device bytes per card, from NVML, per pid of this tenant's group.

        Read from the driver rather than asked of the tenant: a tenant that is
        mid-boot, wedged, or lying is still accounted for, and the same code
        works for every class.
        """
        pids = set(self.pids())
        if not pids or not self._cards:
            return {}
        from sglang.srt.registry.nvml import process_bytes_on_uuid  # noqa: PLC0415

        out: dict[str, int] = {}
        for card in self._cards:
            try:
                per_pid = process_bytes_on_uuid(card)
            except Exception as exc:  # noqa: BLE001 - measurement is best effort
                logger.debug("registry: measured() on %s failed: %s", card, exc)
                continue
            out[card] = sum(b for pid, b in per_pid.items() if pid in pids)
        return out

    def health(self) -> Health:
        if self._state == ResidencyState.COLD:
            return Health(ok=True, detail="cold, no process")
        if self._process is None or not self._process.running:
            return Health(ok=False, detail=self._last_error or "process is gone")
        if http_ok(f"{self.base_url}/health"):
            return Health(ok=True, detail=f"serving on {self.base_url}")
        return Health(ok=False, detail=f"no /health answer from {self.base_url}")

    def generate_probe(self, prompt: str = "Hello", max_new_tokens: int = 8) -> str:
        """A minimal generation, used by the acceptance harness, not by policy."""
        started = time.monotonic()
        result = http_json(
            f"{self.base_url}/generate",
            method="POST",
            payload={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            },
            timeout=120.0,
        )
        logger.info(
            "registry: probe on %s took %.0f ms",
            self.spec.engine_id,
            (time.monotonic() - started) * 1000.0,
        )
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)


register_adapter(ADAPTER_NAME, Class1SrtAdapter)
