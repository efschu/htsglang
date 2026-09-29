# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""File and job store behind the OpenAI fine-tuning surface (#341-M1).

Two resources, one module, because they are one lifecycle: a fine-tuning job
is a reference to an uploaded file plus a state machine, and a file whose job
is still running may not be deleted out from under it.

**Why the state machine is exactly OpenAI's.** ``validating_files -> queued ->
running -> succeeded | failed | cancelled`` is the protocol's vocabulary, and
every client -- the official SDKs, LangChain, the fork's own frontend -- polls
against it. Our tenant semantics (DESIGN #341 D4) add preemption, which is not
in that vocabulary and must not be added to it: a preempted job has not
stopped, it is a job whose wall-clock is longer than its compute time. It
therefore stays ``running``, and the fact that it is currently parked shows up
in the namespaced ``x-htsglang`` block and in the event stream, where a client
that does not know the fork simply ignores it.

**Why files live on disk and jobs live in memory.** The uploaded JSONL is an
input a backend subprocess must be able to open by path, so it is a file. Job
records are the state of work owned by this process; a restart kills the
subprocesses, so persisting the records would only produce jobs that claim to
be running and are not. Recovering across restarts needs the executor to
outlive the server, which is M3 territory (see DESIGN_341 open items).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

#: Namespaced extension key, identical to the one the error envelope uses.
EXTENSION_KEY = "x-htsglang"

#: Purposes the files endpoint accepts. ``fine-tune`` is the only one this
#: surface can act on; the rest are rejected by name rather than silently
#: stored, so a client that uploaded a batch file learns immediately.
FINE_TUNE_PURPOSE = "fine-tune"
FINE_TUNE_RESULTS_PURPOSE = "fine-tune-results"

#: Upload ceiling. A JSONL training set that exceeds this is a dataset the
#: caller should stage on the host filesystem and reference by path through
#: the extension block, not push through a JSON API.
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024


class StoreError(RuntimeError):
    """Base for store-level failures that map onto an OpenAI error body."""

    status_code = 400
    code = "invalid_request_error"
    param: Optional[str] = None


class NotFound(StoreError):
    status_code = 404
    code = "not_found"


class InvalidFile(StoreError):
    status_code = 400
    code = "invalid_file"
    param = "file"


class InvalidJobState(StoreError):
    status_code = 409
    code = "invalid_state"


class JobStatus(str, Enum):
    """The protocol's vocabulary, and nothing else."""

    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class TenantState(str, Enum):
    """Where a job sits in the idle-tenant machine (DESIGN #341 D4).

    This is the fork's own axis and rides in the extension block. It is
    orthogonal to :class:`JobStatus`: a job that is ``running`` by protocol is
    either ``training`` or ``preempted`` here, and the difference is the whole
    reason the tenant exists.
    """

    #: Accepted, not yet started, waiting for the rig to go idle.
    WAITING_FOR_IDLE = "waiting_for_idle"
    #: Holding a VRAM lease and executing.
    TRAINING = "training"
    #: Checkpointed and released on serving demand; will resume.
    PREEMPTED = "preempted"
    #: Finished, no lease held.
    DONE = "done"


def new_id(prefix: str, sep: str = "-") -> str:
    return f"{prefix}{sep}{secrets.token_hex(12)}"


def now_ts() -> int:
    return int(time.time())


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(name: str) -> str:
    """A caller-supplied filename reduced to something safe to join.

    The upload names a file on our disk. ``../`` in a filename is the oldest
    bug in file-upload APIs and the fix is not to escape it but to refuse to
    treat the caller's string as a path at all: only the basename survives,
    and only from a restricted alphabet.
    """
    base = os.path.basename(name or "").strip() or "upload.jsonl"
    cleaned = _SAFE_NAME.sub("_", base)
    return cleaned[:200]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@dataclass
class StoredFile:
    """One uploaded file, spec-shaped."""

    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: str
    path: Path
    status: str = "processed"
    status_details: Optional[str] = None
    #: Number of JSONL records, for a fine-tune file. ``None`` when the file
    #: was not parsed as JSONL.
    line_count: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.id,
            "object": "file",
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
            "status": self.status,
            "status_details": self.status_details,
            # OpenAI returns ``expires_at`` for files with a retention policy.
            # Ours do not expire, and a null is the spec-shaped way to say so.
            "expires_at": None,
        }
        if self.line_count is not None:
            body[EXTENSION_KEY] = {
                "line_count": self.line_count,
                "path": str(self.path),
            }
        return body


def validate_jsonl(raw: bytes) -> int:
    """Parse a fine-tune upload and return its record count.

    Validation happens at upload rather than at job start because the protocol
    has a status for it (``status: error`` on the file, ``validating_files``
    on the job) and because a syntax error found three hours into an idle
    window is a wasted window.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidFile(
            f"training file is not valid UTF-8 (byte offset {exc.start})"
        ) from None
    count = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidFile(
                f"training file line {lineno} is not valid JSON: {exc.msg}"
            ) from None
        if not isinstance(record, dict):
            raise InvalidFile(
                f"training file line {lineno} is a {type(record).__name__}, "
                "but every line must be a JSON object"
            )
        if not ({"messages", "prompt", "text", "instruction"} & set(record)):
            raise InvalidFile(
                f"training file line {lineno} has none of the recognised keys "
                "'messages', 'prompt', 'text' or 'instruction'; it cannot be "
                "turned into a training example"
            )
        count += 1
    if count == 0:
        raise InvalidFile("training file contains no records")
    return count


class FileStore:
    """Uploaded files, on disk, addressed by OpenAI-shaped ids."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self._files: dict[str, StoredFile] = {}

    def create(self, *, filename: str, content: bytes, purpose: str) -> StoredFile:
        if purpose not in (FINE_TUNE_PURPOSE, FINE_TUNE_RESULTS_PURPOSE):
            raise InvalidFile(
                f"purpose {purpose!r} is not served here. This endpoint stores "
                f"training data for the fine-tuning API, so the accepted "
                f"purposes are {FINE_TUNE_PURPOSE!r} and "
                f"{FINE_TUNE_RESULTS_PURPOSE!r}."
            )
        if len(content) > self.max_bytes:
            raise InvalidFile(
                f"file is {len(content)} bytes, over the {self.max_bytes}-byte "
                "upload limit. Stage the dataset on the server filesystem and "
                f"reference it through the {EXTENSION_KEY} block instead."
            )
        line_count = None
        if purpose == FINE_TUNE_PURPOSE:
            line_count = validate_jsonl(content)

        file_id = new_id("file")
        target_dir = self.root / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(filename)
        path = target_dir / safe
        path.write_bytes(content)
        stored = StoredFile(
            id=file_id,
            filename=safe,
            bytes=len(content),
            created_at=now_ts(),
            purpose=purpose,
            path=path,
            line_count=line_count,
        )
        self._files[file_id] = stored
        logger.info(
            "training: stored %s (%s, %d bytes, %s records)",
            file_id,
            safe,
            stored.bytes,
            line_count,
        )
        return stored

    def get(self, file_id: str) -> StoredFile:
        stored = self._files.get(file_id)
        if stored is None:
            raise NotFound(f"No such File object: {file_id}")
        return stored

    def list(self, *, purpose: Optional[str] = None) -> list[StoredFile]:
        items = [f for f in self._files.values() if purpose in (None, f.purpose)]
        items.sort(key=lambda f: f.created_at, reverse=True)
        return items

    def delete(self, file_id: str) -> None:
        stored = self.get(file_id)
        shutil.rmtree(stored.path.parent, ignore_errors=True)
        del self._files[file_id]

    def content(self, file_id: str) -> bytes:
        return self.get(file_id).path.read_bytes()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class JobEvent:
    """One entry in a job's event log, spec-shaped."""

    id: str
    created_at: int
    level: str
    message: str
    data: Optional[dict[str, Any]] = None
    type: str = "message"
    #: Monotonic sequence number. ``created_at`` has one-second resolution and
    #: several events land inside one second, so pagination cursors and
    #: stream resumption key off this rather than the timestamp.
    seq: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "fine_tuning.job.event",
            "created_at": self.created_at,
            "level": self.level,
            "message": self.message,
            "data": self.data,
            "type": self.type,
        }


@dataclass
class JobCheckpoint:
    """One saved checkpoint, spec-shaped."""

    id: str
    created_at: int
    fine_tuned_model_checkpoint: str
    fine_tuning_job_id: str
    step_number: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "fine_tuning.job.checkpoint",
            "created_at": self.created_at,
            "fine_tuned_model_checkpoint": self.fine_tuned_model_checkpoint,
            "fine_tuning_job_id": self.fine_tuning_job_id,
            "step_number": self.step_number,
            "metrics": dict(self.metrics),
        }


@dataclass
class Hyperparameters:
    """The three the protocol carries, resolved from ``auto``.

    ``auto`` is a legal value on the wire for all three. Resolving it here
    rather than in the backend keeps the number the job reports and the number
    the executor was given the same one.
    """

    n_epochs: int = 3
    batch_size: int = 1
    learning_rate_multiplier: float = 1.0

    def to_json(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate_multiplier": self.learning_rate_multiplier,
        }


@dataclass
class TrainingJob:
    """One fine-tuning job: the protocol fields plus the tenant's own state."""

    id: str
    created_at: int
    model: str
    training_file: str
    seed: int
    hyperparameters: Hyperparameters
    validation_file: Optional[str] = None
    suffix: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    method: Optional[dict[str, Any]] = None
    organization_id: str = "org-htsglang"

    status: JobStatus = JobStatus.VALIDATING_FILES
    error: Optional[dict[str, Any]] = None
    fine_tuned_model: Optional[str] = None
    finished_at: Optional[int] = None
    trained_tokens: Optional[int] = None
    result_files: list[str] = field(default_factory=list)
    estimated_finish: Optional[int] = None

    # -- the fork's axis, all of it under the extension key ----------------
    tenant_state: TenantState = TenantState.WAITING_FOR_IDLE
    training_method: str = "lora"
    backend: str = ""
    base_model_path: str = ""
    output_dir: Optional[str] = None
    cards: tuple[str, ...] = ()
    reserved_bytes_per_card: int = 0
    preemptions: int = 0
    resume_from: Optional[str] = None
    last_step: int = 0
    feasibility: Optional[dict[str, Any]] = None
    #: The caller's own ``x-htsglang`` knobs, kept verbatim so a resumed
    #: run is planned from the same numbers as the first attempt.
    request_extension: dict[str, Any] = field(default_factory=dict)
    #: What the surface adds to the response block. Keys starting with an
    #: underscore are internal bookkeeping and are not serialised.
    extension: dict[str, Any] = field(default_factory=dict)

    events: list[JobEvent] = field(default_factory=list)
    checkpoints: list[JobCheckpoint] = field(default_factory=list)
    #: Set when a cancel arrives; the scheduler observes it at the next
    #: boundary. Cancelling is a request, not a kill.
    cancel_requested: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "fine_tuning.job",
            "created_at": self.created_at,
            "error": self.error,
            "fine_tuned_model": self.fine_tuned_model,
            "finished_at": self.finished_at,
            "hyperparameters": self.hyperparameters.to_json(),
            "model": self.model,
            "organization_id": self.organization_id,
            "result_files": list(self.result_files),
            "seed": self.seed,
            "status": self.status.value,
            "trained_tokens": self.trained_tokens,
            "training_file": self.training_file,
            "validation_file": self.validation_file,
            "estimated_finish": self.estimated_finish,
            "integrations": None,
            "metadata": self.metadata,
            "method": self.method,
            EXTENSION_KEY: self.extension_block(),
        }

    def extension_block(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "tenant_state": self.tenant_state.value,
            "training_method": self.training_method,
            "backend": self.backend,
            "base_model_path": self.base_model_path,
            "preemptions": self.preemptions,
            "last_step": self.last_step,
            "cards": list(self.cards),
            "reserved_mib_per_card": self.reserved_bytes_per_card // (1 << 20),
        }
        if self.output_dir:
            block["output_dir"] = self.output_dir
        if self.resume_from:
            block["resume_from"] = self.resume_from
        if self.feasibility is not None:
            block["feasibility"] = self.feasibility
        block.update({k: v for k, v in self.extension.items() if not k.startswith("_")})
        if self.request_extension:
            block["request"] = dict(self.request_extension)
        return block


class JobStore:
    """Job records, their event logs, and the subscribers on those logs.

    Single-event-loop by construction: every mutation happens on the FastAPI
    loop, so there is no lock. The scheduler is an asyncio task on that same
    loop and the backends are subprocesses whose output is read by it, which
    is what makes that assumption hold rather than merely convenient.
    """

    def __init__(self, *, max_events_per_job: int = 20000) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._seq = 0
        self.max_events_per_job = int(max_events_per_job)

    # -- records ------------------------------------------------------------

    def create(self, job: TrainingJob) -> TrainingJob:
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> TrainingJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFound(f"No such FineTuningJob object: {job_id}")
        return job

    def list(self) -> list[TrainingJob]:
        items = list(self._jobs.values())
        items.sort(key=lambda j: (j.created_at, j.id), reverse=True)
        return items

    def queued(self) -> list[TrainingJob]:
        """Jobs the scheduler may still pick up, oldest first.

        A preempted job is ``running`` by protocol but is not executing, so it
        belongs in this list -- that is precisely the resume path.
        """
        items = [
            j
            for j in self._jobs.values()
            if not j.status.is_terminal
            and j.tenant_state in (TenantState.WAITING_FOR_IDLE, TenantState.PREEMPTED)
        ]
        items.sort(key=lambda j: (j.created_at, j.id))
        return items

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def files_in_use(self) -> set[str]:
        return {
            f
            for job in self._jobs.values()
            if not job.status.is_terminal
            for f in (job.training_file, job.validation_file)
            if f
        }

    # -- events -------------------------------------------------------------

    def append_event(
        self,
        job: TrainingJob,
        level: str,
        message: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
        event_type: str = "message",
    ) -> JobEvent:
        self._seq += 1
        event = JobEvent(
            id=new_id("ftevent"),
            created_at=now_ts(),
            level=level,
            message=message,
            data=dict(data) if data is not None else None,
            type=event_type,
            seq=self._seq,
        )
        job.events.append(event)
        if len(job.events) > self.max_events_per_job:
            # Metric events arrive per step; an unbounded log is a leak with a
            # long fuse. The head is dropped rather than the tail because a
            # consumer that just attached wants the recent end.
            del job.events[: len(job.events) - self.max_events_per_job]
        for queue in list(self._subscribers.get(job.id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A subscriber that cannot keep up is not a reason to stall
                # training. It will notice the gap through the terminal event.
                logger.debug("training: dropped event for a slow subscriber")
        return event

    def events_after(
        self, job: TrainingJob, *, after: Optional[str], limit: int
    ) -> tuple[list[JobEvent], bool]:
        """Cursor pagination as the SDK's ``SyncCursorPage`` expects it."""
        start = 0
        if after:
            for index, event in enumerate(job.events):
                if event.id == after:
                    start = index + 1
                    break
            else:
                raise NotFound(f"No such event cursor: {after}")
        window = job.events[start : start + limit]
        has_more = start + limit < len(job.events)
        return window, has_more

    def subscribe(self, job_id: str, *, maxsize: int = 1024) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(job_id)
        if not subscribers:
            return
        with contextlib.suppress(ValueError):
            subscribers.remove(queue)
        if not subscribers:
            self._subscribers.pop(job_id, None)

    def subscriber_count(self, job_id: str) -> int:
        return len(self._subscribers.get(job_id, ()))

    # -- checkpoints --------------------------------------------------------

    def append_checkpoint(
        self,
        job: TrainingJob,
        *,
        step_number: int,
        path: str,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> JobCheckpoint:
        checkpoint = JobCheckpoint(
            id=new_id("ftckpt", sep="_"),
            created_at=now_ts(),
            fine_tuned_model_checkpoint=path,
            fine_tuning_job_id=job.id,
            step_number=int(step_number),
            metrics=dict(metrics or {}),
        )
        job.checkpoints.append(checkpoint)
        job.last_step = max(job.last_step, int(step_number))
        job.resume_from = path
        return checkpoint


def page(items: Iterable[Any], *, has_more: bool = False) -> dict[str, Any]:
    """The list envelope every OpenAI collection endpoint returns."""
    data = [i.to_json() if hasattr(i, "to_json") else i for i in items]
    body: dict[str, Any] = {"object": "list", "data": data, "has_more": has_more}
    if data:
        body["first_id"] = data[0].get("id")
        body["last_id"] = data[-1].get("id")
    return body
