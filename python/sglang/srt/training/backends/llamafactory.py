# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""LLaMA-Factory, wrapped as a subprocess (DESIGN #341 D1).

LLaMA-Factory is Apache-2.0 and has the broadest method ladder of the three
suites D1 names, which is why it is the LLM backend for M1. Nothing from it is
copied into this tree: this module writes a YAML config, spawns
``llamafactory-cli train``, reads its stdout, and knows where it puts
checkpoints.

## Resolved open item: CLI subprocess, not LLaMA-Factory's API mode

DESIGN_341 left open whether LLaMA-Factory's own API mode could serve as the
internal executor interface. It cannot, and the reason is that its API mode is
not a training API. ``llamafactory-cli api`` starts an OpenAI-compatible
*chat* server for a model that has already been trained; the only programmatic
training entry point is the in-process call
``llamafactory.train.tuner.run_exp(args)``. So the real choice is subprocess
CLI versus importing LLaMA-Factory into the serving process, and three
properties decide it:

1. **Preemption.** The tenant contract is checkpoint-and-release on serving
   demand (D4). A subprocess is stopped with a signal and its VRAM is returned
   by the kernel when it exits. An in-process trainer holds a CUDA context and
   an allocator arena in the *serving* process; releasing it means unloading
   torch state the server also uses, which is not a thing that can be made to
   work reliably.
2. **Dependency isolation.** LLaMA-Factory pins transformers, peft, trl,
   accelerate and bitsandbytes. Importing it into the server makes the serving
   stack's versions and the training stack's versions one constraint set, so
   an upgrade for training can break inference.
3. **Blast radius.** A CUDA OOM or a segfault in a training step kills a
   subprocess. In-process it kills the server.

The cost of the subprocess is the interface: preemption granularity is
``save_steps``, because the trainer's own checkpointing is the only place it
can be stopped cleanly. That is a real limitation and it is stated in the
event stream when a preempt happens, not hidden.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from sglang.srt.training.backends import (
    BackendEvent,
    BackendProbe,
    BackendRun,
    EventSink,
    RunOutcome,
    RunSpec,
    RunStatus,
    TrainingBackend,
)
from sglang.srt.training.feasibility import TrainingMethod

logger = logging.getLogger(__name__)

#: ``finetuning_type`` per rung of our ladder. QLoRA is LoRA plus
#: ``quantization_bit``; offloaded full-FT is full plus a DeepSpeed stage-3
#: offload config, which the operator supplies through ``extra``.
_FINETUNING_TYPE = {
    TrainingMethod.LORA: "lora",
    TrainingMethod.QLORA: "lora",
    TrainingMethod.FREEZE: "freeze",
    TrainingMethod.FULL: "full",
    TrainingMethod.FULL_OFFLOAD: "full",
}

#: HF Trainer logs a Python dict per logging step. Matching the whole braced
#: span and parsing it with ``literal_eval`` is more robust than a per-key
#: regex, which silently stops finding fields when a version adds one.
_LOG_DICT = re.compile(r"\{['\"](?:loss|train_runtime|eval_loss)['\"].*\}")

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")


def _cli_command() -> tuple[list[str], str]:
    """How to invoke LLaMA-Factory here, and a label for the probe."""
    executable = shutil.which("llamafactory-cli")
    if executable:
        return [executable], executable
    return [
        sys.executable,
        "-m",
        "llamafactory.cli",
    ], f"{sys.executable} -m llamafactory.cli"


def _installed_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        for name in ("llamafactory", "llama-factory", "llmtuner"):
            try:
                return version(name)
            except PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001 - metadata lookup must never be fatal
        return None
    return None


def _importable() -> bool:
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec("llamafactory") is not None
    except (ImportError, ValueError):
        return False


def dataset_info_for(records_style: str) -> dict[str, Any]:
    """The ``dataset_info.json`` entry that matches the uploaded JSONL.

    LLaMA-Factory resolves datasets by name through a ``dataset_info.json`` in
    ``dataset_dir``. Generating it per job rather than mutating a shared one
    keeps concurrent jobs from fighting over a single registry file.
    """
    if records_style == "sharegpt":
        return {
            "htsglang_job": {
                "file_name": "train.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        }
    if records_style == "text":
        return {
            "htsglang_job": {
                "file_name": "train.jsonl",
                "columns": {"prompt": "text"},
            }
        }
    return {
        "htsglang_job": {
            "file_name": "train.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }


def detect_records_style(path: Path) -> str:
    """Read the first record and name its shape."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "messages" in record:
                    return "sharegpt"
                if "text" in record:
                    return "text"
                return "alpaca"
    except (OSError, json.JSONDecodeError):
        pass
    return "alpaca"


def build_config(spec: RunSpec, *, dataset_dir: Path) -> dict[str, Any]:
    """The YAML LLaMA-Factory is handed. Pure, so tests can assert on it."""
    config: dict[str, Any] = {
        "model_name_or_path": spec.base_model_path,
        "stage": str(spec.extra.get("stage", "sft")),
        "do_train": True,
        "finetuning_type": _FINETUNING_TYPE[spec.method],
        "dataset": "htsglang_job",
        "dataset_dir": str(dataset_dir),
        "template": str(spec.extra.get("template", "default")),
        "cutoff_len": int(spec.sequence_length),
        "output_dir": str(spec.output_dir),
        "overwrite_output_dir": spec.resume_from is None,
        "per_device_train_batch_size": int(spec.micro_batch_size),
        "gradient_accumulation_steps": int(
            spec.extra.get("gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(spec.learning_rate),
        "num_train_epochs": float(spec.n_epochs),
        "lr_scheduler_type": str(spec.extra.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(spec.extra.get("warmup_ratio", 0.03)),
        "logging_steps": int(spec.extra.get("logging_steps", 5)),
        # The preemption granularity, so it is set from the tenant's number
        # rather than left at the trainer's default.
        "save_steps": int(spec.save_steps),
        "save_strategy": "steps",
        "gradient_checkpointing": bool(spec.gradient_checkpointing),
        "bf16": True,
        "seed": int(spec.seed),
        "plot_loss": False,
        "report_to": "none",
    }
    if spec.method in (TrainingMethod.LORA, TrainingMethod.QLORA):
        config["lora_target"] = str(spec.extra.get("lora_target", "all"))
        config["lora_rank"] = int(spec.extra.get("lora_rank", 16))
        config["lora_alpha"] = int(spec.extra.get("lora_alpha", 32))
    if spec.method is TrainingMethod.QLORA:
        config["quantization_bit"] = int(spec.extra.get("quantization_bit", 4))
        config["quantization_method"] = str(
            spec.extra.get("quantization_method", "bnb")
        )
    if spec.method is TrainingMethod.FREEZE:
        config["freeze_trainable_layers"] = int(
            spec.extra.get("freeze_trainable_layers", 2)
        )
    if spec.method is TrainingMethod.FULL_OFFLOAD:
        deepspeed = spec.extra.get("deepspeed")
        if deepspeed:
            config["deepspeed"] = str(deepspeed)
    if spec.validation_path is not None:
        config["val_size"] = float(spec.extra.get("val_size", 0.1))
        config["eval_strategy"] = "steps"
        config["eval_steps"] = int(spec.extra.get("eval_steps", spec.save_steps))
    if spec.resume_from:
        config["resume_from_checkpoint"] = spec.resume_from
    for key, value in (spec.extra.get("llamafactory") or {}).items():
        config[key] = value
    return config


def write_yaml(config: dict[str, Any], path: Path) -> None:
    """Serialise the config. YAML if available, JSON otherwise.

    LLaMA-Factory accepts a ``.json`` config as readily as a ``.yaml`` one, so
    a host without PyYAML is not a reason to fail a job.
    """
    try:
        import yaml  # noqa: PLC0415

        path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    except ImportError:
        path.with_suffix(".json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )


def latest_checkpoint(output_dir: Path) -> Optional[str]:
    """The highest-numbered ``checkpoint-N`` under ``output_dir``."""
    best: Optional[tuple[int, Path]] = None
    if not output_dir.is_dir():
        return None
    for child in output_dir.iterdir():
        match = _CHECKPOINT_DIR.match(child.name)
        if child.is_dir() and match:
            step = int(match.group(1))
            if best is None or step > best[0]:
                best = (step, child)
    return str(best[1]) if best else None


def parse_log_line(line: str) -> Optional[dict[str, Any]]:
    """Turn one HF Trainer log line into metrics, or ``None``."""
    match = _LOG_DICT.search(line)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


class LlamaFactoryRun(BackendRun):
    """One ``llamafactory-cli train`` process."""

    def __init__(
        self,
        spec: RunSpec,
        sink: EventSink,
        process: asyncio.subprocess.Process,
        *,
        log_path: Path,
    ) -> None:
        self.spec = spec
        self._sink = sink
        self._process = process
        self._log_path = log_path
        self._step = _step_of(spec.resume_from)
        self._seen_checkpoints: set[str] = set()
        self._stop_reason: Optional[RunStatus] = None
        self._trained_tokens: Optional[int] = None
        self._task = asyncio.get_running_loop().create_task(self._drive())

    async def wait(self) -> RunOutcome:
        return await self._task

    async def preempt(self, *, timeout_s: float = 120.0) -> RunOutcome:
        self._stop_reason = RunStatus.PREEMPTED
        self._sink(
            BackendEvent(
                level="info",
                message=(
                    "serving demand arrived; stopping the trainer. Work since the "
                    f"last save (every {self.spec.save_steps} steps) is redone on "
                    "resume -- that is the preemption granularity of a subprocess "
                    "executor"
                ),
            )
        )
        return await self._signal_and_wait(signal.SIGTERM, timeout_s)

    async def cancel(self, *, timeout_s: float = 60.0) -> RunOutcome:
        self._stop_reason = RunStatus.CANCELLED
        return await self._signal_and_wait(signal.SIGTERM, timeout_s)

    async def _signal_and_wait(self, sig: int, timeout_s: float) -> RunOutcome:
        if self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                self._process.send_signal(sig)
        try:
            return await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Bounded teardown: the serving tenant must not wait on a trainer
            # that will not leave.
            self._sink(
                BackendEvent(
                    level="warn",
                    message=(
                        f"trainer did not exit within {timeout_s}s of SIGTERM; "
                        "escalating to SIGKILL"
                    ),
                )
            )
            with contextlib.suppress(ProcessLookupError, OSError):
                self._process.kill()
            return await self._task

    async def _drive(self) -> RunOutcome:
        assert self._process.stdout is not None
        with self._log_path.open("ab") as log:
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    break
                log.write(raw)
                self._consume(raw.decode("utf-8", errors="replace").rstrip())
        returncode = await self._process.wait()

        checkpoint = latest_checkpoint(self.spec.output_dir)
        if self._stop_reason is RunStatus.PREEMPTED:
            return RunOutcome(
                status=RunStatus.PREEMPTED,
                last_checkpoint=checkpoint,
                last_step=self._step,
            )
        if self._stop_reason is RunStatus.CANCELLED:
            return RunOutcome(
                status=RunStatus.CANCELLED,
                last_checkpoint=checkpoint,
                last_step=self._step,
            )
        if returncode != 0:
            tail = _tail(self._log_path)
            return RunOutcome(
                status=RunStatus.FAILED,
                last_checkpoint=checkpoint,
                last_step=self._step,
                error=(
                    f"llamafactory-cli exited with code {returncode}. "
                    f"Last output: {tail}"
                ),
            )
        artifact = self.spec.output_dir / "adapter_model.safetensors"
        return RunOutcome(
            status=RunStatus.SUCCEEDED,
            fine_tuned_model=self.spec.output_dir.name,
            artifact_path=(
                str(artifact) if artifact.exists() else str(self.spec.output_dir)
            ),
            last_checkpoint=checkpoint,
            last_step=self._step,
            trained_tokens=self._trained_tokens,
        )

    def _consume(self, line: str) -> None:
        metrics = parse_log_line(line)
        if metrics is not None:
            step = int(metrics.get("step") or metrics.get("global_step") or 0)
            if step:
                self._step = step
            self._sink(
                BackendEvent(
                    level="info",
                    message=line,
                    data={k: v for k, v in metrics.items()},
                    type="metrics",
                    step=step or None,
                )
            )
        elif line.strip():
            level = "error" if _looks_like_error(line) else "info"
            self._sink(BackendEvent(level=level, message=line))
        self._notice_checkpoints()

    def _notice_checkpoints(self) -> None:
        found = latest_checkpoint(self.spec.output_dir)
        if found and found not in self._seen_checkpoints:
            self._seen_checkpoints.add(found)
            step = _step_of(found)
            self._step = max(self._step, step)
            self._sink(
                BackendEvent(
                    level="info",
                    message=f"checkpoint saved at step {step}",
                    data={"step": step},
                    type="metrics",
                    checkpoint_path=found,
                    step=step,
                )
            )


def _step_of(path: Optional[str]) -> int:
    if not path:
        return 0
    match = _CHECKPOINT_DIR.match(Path(path).name)
    return int(match.group(1)) if match else 0


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in (
            "traceback (most recent call last)",
            "error:",
            "cuda out of memory",
        )
    )


def _tail(path: Path, *, limit: int = 600) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return "(no log)"
    return data[-limit:].decode("utf-8", errors="replace").strip()


class LlamaFactoryBackend(TrainingBackend):
    """LLM SFT/LoRA/QLoRA/freeze/full through ``llamafactory-cli train``."""

    name = "llamafactory"

    def probe(self) -> BackendProbe:
        command, label = _cli_command()
        installed = _importable()
        version = _installed_version()
        if not installed and shutil.which("llamafactory-cli") is None:
            return BackendProbe(
                name=self.name,
                available=False,
                reason=(
                    "LLaMA-Factory is not installed in this server's Python "
                    "environment: neither the 'llamafactory' module nor the "
                    "'llamafactory-cli' executable was found."
                ),
                executable=label,
                remedies=(
                    f"install it into the interpreter running this server "
                    f"({sys.executable}): pip install llamafactory",
                    "then restart the server so the probe sees it",
                    "or submit the job with the 'mock' backend to exercise the "
                    "job surface without training",
                ),
            )
        return BackendProbe(
            name=self.name,
            available=True,
            version=version or "unknown",
            executable=label,
        )

    def supported_methods(self) -> tuple[TrainingMethod, ...]:
        return (
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FREEZE,
            TrainingMethod.FULL,
            TrainingMethod.FULL_OFFLOAD,
        )

    def prepare(self, spec: RunSpec) -> tuple[list[str], Path]:
        """Materialise the run directory and return the argv and config path."""
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = spec.output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / "train.jsonl"
        if spec.dataset_path.resolve() != target.resolve():
            shutil.copyfile(spec.dataset_path, target)
        style = detect_records_style(target)
        (dataset_dir / "dataset_info.json").write_text(
            json.dumps(dataset_info_for(style), indent=2), encoding="utf-8"
        )
        config = build_config(spec, dataset_dir=dataset_dir)
        config_path = spec.output_dir / "train_config.yaml"
        write_yaml(config, config_path)
        written = (
            config_path if config_path.exists() else config_path.with_suffix(".json")
        )
        command, _ = _cli_command()
        return [*command, "train", str(written)], written

    async def launch(self, spec: RunSpec, sink: EventSink) -> BackendRun:
        self.check(spec)
        argv, config_path = self.prepare(spec)
        env = dict(os.environ)
        if spec.cuda_visible_devices:
            # Process-level isolation, as everywhere else in the fork: the
            # child sees exactly the cards this job leased.
            env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")
        sink(
            BackendEvent(
                level="info",
                message=(
                    f"launching {' '.join(argv)} with CUDA_VISIBLE_DEVICES="
                    f"{env.get('CUDA_VISIBLE_DEVICES', '')!r}"
                ),
                data={"config": str(config_path)},
            )
        )
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(spec.output_dir),
            start_new_session=True,
        )
        return LlamaFactoryRun(
            spec, sink, process, log_path=spec.output_dir / "train.log"
        )
