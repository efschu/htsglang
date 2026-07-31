# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The training service: one object the HTTP surface talks to (#341-M1).

Everything above this line is protocol and everything below it is execution.
The serving adapters (:mod:`sglang.srt.entrypoints.openai.serving_files`,
:mod:`...serving_finetune`) parse and shape; this module decides. That split
is what lets the whole surface be tested against a live server with no GPU:
the service is constructed with a mock backend and a synthetic machine, and
the routes above it never know.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from sglang.srt.training import feasibility as feas
from sglang.srt.training.backends import (
    BackendProbe,
    BackendUnavailable,
    RunSpec,
    TrainingBackend,
    get_backend,
    resolve_backend,
)
from sglang.srt.training.feasibility import (
    MIB,
    FeasibilityDecision,
    MachineResources,
    ModelProfile,
    ModelProfileError,
    TrainingDemandSpec,
    TrainingMethod,
    parse_method,
)
from sglang.srt.training.store import (
    EXTENSION_KEY,
    FINE_TUNE_PURPOSE,
    FileStore,
    Hyperparameters,
    InvalidJobState,
    JobStatus,
    JobStore,
    StoreError,
    TenantState,
    TrainingJob,
    new_id,
    now_ts,
)
from sglang.srt.training.tenant import (
    IdleMonitor,
    PlannedRun,
    TenantConfig,
    TrainingTenant,
    local_activity_source,
    registry_activity_source,
)

logger = logging.getLogger(__name__)


class InfeasibleRequest(StoreError):
    """The job cannot run on this machine. Carries the ladder."""

    status_code = 400
    code = "insufficient_resources"

    def __init__(self, decision: FeasibilityDecision) -> None:
        super().__init__(decision.render())
        self.decision = decision


class TenantDisabled(StoreError):
    """The surface is served but the tenant is switched off."""

    status_code = 503
    code = "training_tenant_disabled"


class BackendRejected(StoreError):
    """No executor can run this job."""

    status_code = 400
    code = "backend_unavailable"

    def __init__(self, probe: BackendProbe) -> None:
        super().__init__(probe.reason)
        self.probe = probe


@dataclass
class TrainingServiceConfig(TenantConfig):
    """Tenant knobs plus the ones only the service needs."""

    #: Directory that model names are resolved against when the request does
    #: not carry an absolute path. Empty means "paths only".
    model_root: str = ""
    max_file_bytes: int = 2 * 1024 * 1024 * 1024
    #: Events an SSE consumer may fall behind by before it is dropped (#344).
    event_stream_timeout_s: float = 60.0

    def to_json(self) -> dict[str, Any]:
        body = super().to_json()
        body.update(
            {
                "model_root": self.model_root,
                "max_file_bytes": self.max_file_bytes,
                "event_stream_timeout_s": self.event_stream_timeout_s,
            }
        )
        return body


def parse_extension(body: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the fork's knobs out of a vanilla-shaped request.

    Three carriers, in precedence order, because different clients can reach
    different ones: the ``x-htsglang`` object (SDK ``extra_body``), then
    ``metadata`` keys prefixed ``x-htsglang.`` (any client, since metadata is
    a plain string map on the wire), then nothing. A client that knows none of
    them gets the defaults and a working job.
    """
    extension: dict[str, Any] = {}
    metadata = body.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if str(key).startswith(f"{EXTENSION_KEY}."):
                extension[str(key)[len(EXTENSION_KEY) + 1 :]] = value
    block = body.get(EXTENSION_KEY)
    if isinstance(block, Mapping):
        extension.update(block)
    elif isinstance(block, str):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, Mapping):
                extension.update(parsed)
        except json.JSONDecodeError:
            raise StoreError(
                f"{EXTENSION_KEY} was sent as a string but is not valid JSON"
            ) from None
    return extension


def _resolve_hyperparameters(body: Mapping[str, Any]) -> Hyperparameters:
    """``auto`` resolved to the number the executor will actually be given."""
    raw = dict(body.get("hyperparameters") or {})
    method = body.get("method")
    if isinstance(method, Mapping):
        # The newer surface nests them under method.supervised.hyperparameters.
        for kind in ("supervised", "dpo", "reinforcement"):
            nested = method.get(kind)
            if isinstance(nested, Mapping) and isinstance(
                nested.get("hyperparameters"), Mapping
            ):
                raw = {**nested["hyperparameters"], **raw}
                break

    def number(key: str, default: float) -> float:
        value = raw.get(key, "auto")
        if value in (None, "auto"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            raise StoreError(
                f"hyperparameters.{key} must be a number or 'auto', got {value!r}"
            ) from None

    return Hyperparameters(
        n_epochs=max(1, int(number("n_epochs", 3))),
        batch_size=max(1, int(number("batch_size", 1))),
        learning_rate_multiplier=number("learning_rate_multiplier", 1.0),
    )


class TrainingService:
    """Files, jobs, feasibility, backends and the idle tenant, assembled."""

    #: Base learning rate the multiplier is applied to.
    BASE_LEARNING_RATE = 5e-5

    def __init__(
        self,
        config: Optional[TrainingServiceConfig] = None,
        *,
        jobs: Optional[JobStore] = None,
        files: Optional[FileStore] = None,
        monitor: Optional[IdleMonitor] = None,
        reservation_store: Any = None,
        machine_resolver: Optional[Callable[[], MachineResources]] = None,
        model_profiler: Callable[[str], ModelProfile] = feas.profile_model,
        backend_factory: Optional[
            Callable[[str, TrainingMethod], TrainingBackend]
        ] = None,
    ) -> None:
        self.config = config or TrainingServiceConfig()
        root = Path(self.config.artifact_root)
        self.jobs = jobs or JobStore()
        self.files = files or FileStore(
            root / "files", max_bytes=self.config.max_file_bytes
        )
        self.reservation_store = reservation_store
        self.machine_resolver = machine_resolver or (
            lambda: feas.probe_machine(artifact_root=root, store=self.reservation_store)
        )
        self.model_profiler = model_profiler
        self.backend_factory = backend_factory or self._default_backend_factory
        self.monitor = monitor or IdleMonitor(
            [local_activity_source, registry_activity_source()],
            grace_seconds=self.config.grace_seconds,
        )
        self.tenant = TrainingTenant(
            self.jobs,
            config=self.config,
            monitor=self.monitor,
            planner=self.plan,
            reservation_store=reservation_store,
            machine_resolver=self.machine_resolver,
        )

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        Path(self.config.artifact_root).mkdir(parents=True, exist_ok=True)
        if self.config.enabled:
            self.tenant.start()

    async def stop(self) -> None:
        await self.tenant.stop()

    def snapshot(self) -> dict[str, Any]:
        machine = self.machine_resolver()
        return {
            "tenant": self.tenant.snapshot(),
            "machine": machine.to_json(),
            "backends": [
                get_backend(name).probe().to_json() for name in ("llamafactory", "mock")
            ],
            "jobs": len(self.jobs),
        }

    # -- files --------------------------------------------------------------

    def create_file(self, *, filename: str, content: bytes, purpose: str):
        return self.files.create(filename=filename, content=content, purpose=purpose)

    def delete_file(self, file_id: str) -> None:
        if file_id in self.jobs.files_in_use():
            raise InvalidJobState(
                f"file {file_id} is the training file of a job that has not "
                "finished; cancel the job before deleting its data"
            )
        self.files.delete(file_id)

    # -- jobs ---------------------------------------------------------------

    def create_job(self, body: Mapping[str, Any]) -> TrainingJob:
        if not self.config.enabled:
            # A named rejection, not a 404. "This route does not exist" and
            # "this route exists and is switched off" are different problems
            # with different fixes, and only one of them is the operator's.
            raise TenantDisabled(
                "The idle training tenant is not enabled on this server, so "
                "fine-tuning jobs are not accepted. Restart with "
                "--enable-training-tenant to turn it on."
            )
        model = str(body.get("model") or "").strip()
        if not model:
            raise StoreError("'model' is required")
        training_file = str(body.get("training_file") or "").strip()
        if not training_file:
            raise StoreError("'training_file' is required")
        stored = self.files.get(training_file)
        if stored.purpose != FINE_TUNE_PURPOSE:
            raise StoreError(
                f"file {training_file} has purpose {stored.purpose!r}; a "
                f"fine-tuning job needs a file uploaded with purpose "
                f"{FINE_TUNE_PURPOSE!r}",
            )
        validation_file = body.get("validation_file")
        if validation_file:
            self.files.get(str(validation_file))

        extension = parse_extension(body)
        hyperparameters = _resolve_hyperparameters(body)
        method = parse_method(extension.get("method", self.config.default_method))
        backend_name = str(extension.get("backend", self.config.default_backend))

        job = TrainingJob(
            id=new_id("ftjob"),
            created_at=now_ts(),
            model=model,
            training_file=training_file,
            validation_file=str(validation_file) if validation_file else None,
            seed=int(body.get("seed") or 0),
            suffix=(str(body["suffix"]) if body.get("suffix") else None),
            hyperparameters=hyperparameters,
            metadata=_string_map(body.get("metadata")),
            method=_method_object(body.get("method"), method),
            status=JobStatus.VALIDATING_FILES,
            training_method=method.value,
            backend=backend_name,
            request_extension=dict(extension),
        )
        self.jobs.create(job)
        self.jobs.append_event(
            job,
            "info",
            f"validating training file {training_file} "
            f"({stored.line_count} records, {stored.bytes} bytes)",
        )

        base_model_path = self._resolve_base_model(model, extension)
        job.base_model_path = base_model_path
        job.extension["requested_backend"] = backend_name

        decision = self.evaluate(job, method=method, extension=extension)
        job.feasibility = decision.to_json()
        if not decision.fits:
            # Fail fast at submission with the arithmetic, rather than
            # accepting a job that will fail hours later in an idle window.
            self.jobs.append_event(job, "error", decision.message)
            job.status = JobStatus.FAILED
            job.tenant_state = TenantState.DONE
            job.finished_at = now_ts()
            job.error = {
                "code": "insufficient_resources",
                "message": decision.message,
                "param": None,
            }
            raise InfeasibleRequest(decision)

        job.status = JobStatus.QUEUED
        job.tenant_state = TenantState.WAITING_FOR_IDLE
        job.extension["feasibility_headroom_mib"] = (
            (decision.option.headroom_bytes // MIB) if decision.option else 0
        )
        self.jobs.append_event(
            job,
            "info",
            f"queued: {decision.message}",
        )
        self.tenant.wake()
        return job

    def cancel_job(self, job_id: str) -> TrainingJob:
        job = self.jobs.get(job_id)
        if job.status.is_terminal:
            raise InvalidJobState(
                f"job {job_id} is already {job.status.value} and cannot be cancelled"
            )
        job.cancel_requested = True
        self.jobs.append_event(job, "info", "cancellation requested by the client")
        current = self.tenant._current  # noqa: SLF001 - same package, one owner
        if current is None or current[0].id != job.id:
            # Not executing: the cancel is final immediately, so a client that
            # polls right after cancelling sees the terminal state.
            job.status = JobStatus.CANCELLED
            job.tenant_state = TenantState.DONE
            job.finished_at = now_ts()
        self.tenant.wake()
        return job

    # -- planning -----------------------------------------------------------

    def _resolve_base_model(self, model: str, extension: Mapping[str, Any]) -> str:
        explicit = str(extension.get("base_model_path") or "").strip()
        candidates = [c for c in (explicit, model) if c]
        if self.config.model_root:
            candidates.append(str(Path(self.config.model_root) / model))
        for candidate in candidates:
            if Path(candidate).is_dir():
                return candidate
        raise StoreError(
            f"base model {model!r} could not be resolved to a directory on this "
            f"server. Tried: {', '.join(repr(c) for c in candidates)}. Pass an "
            f"absolute path as 'model', or set base_model_path in the "
            f"{EXTENSION_KEY} block, or start the server with "
            f"--training-model-root.",
        )

    def demand_spec(
        self, extension: Mapping[str, Any], job: TrainingJob
    ) -> TrainingDemandSpec:
        def number(key: str, default: float) -> float:
            value = extension.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                raise StoreError(
                    f"{EXTENSION_KEY}.{key} must be a number, got {value!r}"
                ) from None

        return TrainingDemandSpec(
            sequence_length=int(number("sequence_length", 2048)),
            micro_batch_size=int(
                number("micro_batch_size", job.hyperparameters.batch_size)
            ),
            gradient_checkpointing=bool(extension.get("gradient_checkpointing", True)),
            checkpoints_retained=int(number("checkpoints_retained", 2)),
        )

    def evaluate(
        self,
        job: TrainingJob,
        *,
        method: TrainingMethod,
        extension: Mapping[str, Any],
    ) -> FeasibilityDecision:
        try:
            profile = self.model_profiler(job.base_model_path)
        except ModelProfileError as exc:
            raise StoreError(str(exc)) from None
        return feas.evaluate(
            model=profile,
            method=method,
            spec=self.demand_spec(extension, job),
            machine=self.machine_resolver(),
        )

    def plan(self, job: TrainingJob) -> PlannedRun:
        """Turn a queued job into a launchable run. Called by the tenant."""
        extension = dict(job.request_extension)
        method = parse_method(job.training_method)
        backend = self.backend_factory(
            str(job.extension.get("requested_backend") or self.config.default_backend),
            method,
        )
        decision = self.evaluate(job, method=method, extension=extension)
        job.feasibility = decision.to_json()
        if not decision.fits:
            raise InfeasibleRequest(decision)

        machine = self.machine_resolver()
        index_of = {c.uuid: c.index for c in machine.cards}
        visible = ",".join(
            str(index_of[uuid]) for uuid in decision.chosen_cards if uuid in index_of
        )
        output_dir = Path(self.config.artifact_root) / "jobs" / job.id
        spec = RunSpec(
            job_id=job.id,
            base_model_path=job.base_model_path,
            method=method,
            dataset_path=self.files.get(job.training_file).path,
            output_dir=output_dir,
            n_epochs=job.hyperparameters.n_epochs,
            micro_batch_size=job.hyperparameters.batch_size,
            learning_rate=self.BASE_LEARNING_RATE
            * job.hyperparameters.learning_rate_multiplier,
            sequence_length=decision.spec.sequence_length,
            seed=job.seed,
            gradient_checkpointing=decision.spec.gradient_checkpointing,
            save_steps=int(extension.get("save_steps", self.config.save_steps)),
            validation_path=(
                self.files.get(job.validation_file).path
                if job.validation_file
                else None
            ),
            resume_from=job.resume_from,
            cuda_visible_devices=visible,
            extra=extension,
        )
        return PlannedRun(
            backend=backend,
            spec=spec,
            card_uuids=decision.chosen_cards,
            per_card_bytes=decision.per_card_bytes,
        )

    def _default_backend_factory(
        self, name: str, method: TrainingMethod
    ) -> TrainingBackend:
        try:
            backend, _probes = resolve_backend(name, method)
        except BackendUnavailable as exc:
            raise BackendRejected(exc.probe) from None
        except KeyError as exc:
            raise StoreError(str(exc)) from None
        probe = backend.probe()
        if not probe.available:
            raise BackendRejected(probe)
        return backend


def _string_map(value: Any) -> Optional[dict[str, str]]:
    if not isinstance(value, Mapping):
        return None
    return {str(k): str(v) for k, v in value.items()}


def _method_object(raw: Any, method: TrainingMethod) -> dict[str, Any]:
    """Echo back a spec-shaped ``method`` object.

    OpenAI's ``method.type`` vocabulary is ``supervised | dpo |
    reinforcement`` and does not have a slot for LoRA versus full finetune --
    those are ours and live in the extension block. The object is echoed
    faithfully so a client that sent one sees it back unchanged.
    """
    if isinstance(raw, Mapping) and raw.get("type"):
        return dict(raw)
    return {
        "type": "supervised",
        "supervised": {"hyperparameters": {}},
        EXTENSION_KEY: {"training_method": method.value},
    }
