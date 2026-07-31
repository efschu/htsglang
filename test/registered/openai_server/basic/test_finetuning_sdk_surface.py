# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#341-M1: the fine-tuning surface, driven by the official ``openai`` SDK.

DESIGN #341 D3 says external suites must be able to point at this fork and
submit training jobs without knowing anything about it. The only way to
demonstrate that is to make the vanilla client do it: these tests use the same
``openai`` package a user would install, against a real uvicorn on a real
port, running the real FastAPI app with the real routes and the real error
handlers.

What is mocked and what is not:

* **Real**: the socket, the HTTP layer, the routes, the serving adapters, the
  job store, the feasibility formula, the idle monitor, the tenant scheduler,
  the preempt/resume loop, the SSE framing, the OpenAI error envelopes, and
  the SDK's own response validation against its typed models.
* **Mocked**: the *machine* (a synthetic 80 GiB card, because CI has none),
  the *executor* (:class:`MockBackend`, which runs the whole run lifecycle
  with the arithmetic replaced), and the inference engine the harness always
  mocks. No ledger is configured, so the tenant runs without a cross-process
  reservation and says so in the event log.

Requires no GPU. ``CUDA_VISIBLE_DEVICES=99`` is the intended way to run it.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import openai
import requests
from openai_sdk_harness import TOKENIZER_NAME, live_server

from sglang.srt.training.backends.mock import MockBackend
from sglang.srt.training.feasibility import GIB, CardResources, MachineResources
from sglang.srt.training.service import TrainingService, TrainingServiceConfig
from sglang.srt.training.tenant import DemandSample, IdleMonitor
from sglang.test.ci.ci_register import register_cpu_ci

# The engine, the machine and the executor are all mocked; this needs no card.
register_cpu_ci(est_time=45, suite="base-a-test-cpu")


TRAINING_JSONL = "".join(
    json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }
    )
    + "\n"
    for i in range(12)
).encode()


SYNTHETIC_MACHINE = MachineResources(
    cards=(
        CardResources(
            uuid="GPU-synthetic-0",
            index=0,
            name="Synthetic 80GB",
            total_bytes=80 * GIB,
            available_bytes=80 * GIB,
        ),
    ),
    ram_total_bytes=256 * GIB,
    ram_available_bytes=200 * GIB,
    disk_free_bytes=2000 * GIB,
    disk_path="/synthetic",
)

TINY_MACHINE = MachineResources(
    cards=(
        CardResources(
            uuid="GPU-tiny-0",
            index=0,
            name="Synthetic 6GB",
            total_bytes=6 * GIB,
            available_bytes=6 * GIB,
        ),
    ),
    ram_total_bytes=16 * GIB,
    ram_available_bytes=8 * GIB,
    disk_free_bytes=20 * GIB,
    disk_path="/synthetic",
)


def write_base_model(root: Path, *, params: int = 7_600_000_000) -> Path:
    """A model directory the feasibility gate can actually profile.

    The weights are not written -- a 15 GB file per test run is not a thing a
    CI host should do -- but the safetensors index is, with a real
    ``total_size``. That is the field the profiler prefers precisely because
    it is the checkpoint's own account of its size, so the gate sees a 7.6B
    model and prices it as one.
    """
    directory = root / "base-model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 12288,
                "vocab_size": 151936,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": params * 2}, "weight_map": {}})
    )
    return directory


def build_service(
    root: Path, *, machine=SYNTHETIC_MACHINE, busy=None, enabled: bool = True
) -> TrainingService:
    demand = busy if busy is not None else (lambda: False)
    return TrainingService(
        TrainingServiceConfig(
            enabled=enabled,
            artifact_root=root / "artifacts",
            grace_seconds=0.0,
            poll_seconds=0.02,
            default_backend="mock",
            save_steps=10,
        ),
        monitor=IdleMonitor(
            [lambda: DemandSample(source="fake_serving", busy=demand())],
            grace_seconds=0.0,
        ),
        machine_resolver=lambda: machine,
        backend_factory=lambda name, method: MockBackend(step_seconds=0.004),
    )


def wait_until(predicate, timeout_s: float, what: str):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"{what} did not happen within {timeout_s}s")


class FineTuningSDKSurfaceTest(unittest.TestCase):
    """Files, jobs, events, checkpoints and cancel, through the real SDK."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.model_dir = write_base_model(root)
        cls.busy = {"value": False}
        cls.service = build_service(root, busy=lambda: cls.busy["value"])
        cls._server = live_server(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME),
            training_service=cls.service,
        )
        cls.base_url = cls._server.__enter__()
        cls.client = openai.OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._tmp.cleanup()

    def upload(self, *, purpose: str = "fine-tune"):
        return self.client.files.create(
            file=("train.jsonl", TRAINING_JSONL), purpose=purpose
        )

    def submit(self, **extension):
        uploaded = self.upload()
        block = {"sequence_length": 512, "total_steps": 60, "save_steps": 10}
        block.update(extension)
        return self.client.fine_tuning.jobs.create(
            model=str(self.model_dir),
            training_file=uploaded.id,
            extra_body={"x-htsglang": block},
        )

    # -- files --------------------------------------------------------------

    def test_files_create_returns_a_spec_shaped_file_object(self):
        uploaded = self.upload()
        self.assertTrue(uploaded.id.startswith("file-"))
        self.assertEqual(uploaded.object, "file")
        self.assertEqual(uploaded.purpose, "fine-tune")
        self.assertEqual(uploaded.status, "processed")
        self.assertEqual(uploaded.bytes, len(TRAINING_JSONL))
        self.assertEqual(uploaded.filename, "train.jsonl")

    def test_files_list_and_retrieve_round_trip(self):
        uploaded = self.upload()
        listed = self.client.files.list()
        self.assertIn(uploaded.id, [f.id for f in listed.data])
        fetched = self.client.files.retrieve(uploaded.id)
        self.assertEqual(fetched.id, uploaded.id)

    def test_files_content_returns_what_was_uploaded(self):
        uploaded = self.upload()
        content = self.client.files.content(uploaded.id)
        self.assertEqual(content.content, TRAINING_JSONL)

    def test_files_delete(self):
        uploaded = self.upload()
        deleted = self.client.files.delete(uploaded.id)
        self.assertTrue(deleted.deleted)
        with self.assertRaises(openai.NotFoundError):
            self.client.files.retrieve(uploaded.id)

    def test_unknown_file_is_a_typed_404(self):
        with self.assertRaises(openai.NotFoundError) as ctx:
            self.client.files.retrieve("file-does-not-exist")
        self.assertIn("No such File object", str(ctx.exception))

    def test_invalid_jsonl_is_a_typed_400_naming_the_line(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.files.create(
                file=("bad.jsonl", b'{"messages": []}\n{oops}\n'), purpose="fine-tune"
            )
        self.assertIn("line 2", str(ctx.exception))

    def test_unsupported_purpose_is_refused_by_name(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.files.create(file=("x.jsonl", b"{}"), purpose="assistants")
        self.assertIn("fine-tune", str(ctx.exception))

    # -- jobs ---------------------------------------------------------------

    def test_create_job_returns_a_spec_shaped_job(self):
        job = self.submit()
        self.assertTrue(job.id.startswith("ftjob-"))
        self.assertEqual(job.object, "fine_tuning.job")
        self.assertIn(job.status, ("queued", "running"))
        self.assertEqual(job.model, str(self.model_dir))
        self.assertEqual(job.seed, 0)
        self.assertEqual(job.result_files, [])
        self.assertIsNotNone(job.hyperparameters)

    def test_extension_block_carries_the_fork_axis(self):
        job = self.submit(method="qlora")
        block = job.model_extra["x-htsglang"]
        self.assertEqual(block["training_method"], "qlora")
        self.assertIn(block["tenant_state"], ("waiting_for_idle", "training"))
        self.assertIn("feasibility", block)
        self.assertTrue(block["feasibility"]["fits"])
        # A vanilla client never looks here; the typed fields are all present.
        self.assertEqual(job.object, "fine_tuning.job")

    def test_method_can_also_ride_in_metadata_for_clients_without_extra_body(self):
        uploaded = self.upload()
        job = self.client.fine_tuning.jobs.create(
            model=str(self.model_dir),
            training_file=uploaded.id,
            metadata={
                "x-htsglang.method": "freeze",
                "x-htsglang.sequence_length": "512",
                "x-htsglang.total_steps": "20",
            },
        )
        self.assertEqual(job.model_extra["x-htsglang"]["training_method"], "freeze")

    def test_retrieve_and_list_jobs(self):
        job = self.submit()
        fetched = self.client.fine_tuning.jobs.retrieve(job.id)
        self.assertEqual(fetched.id, job.id)
        listed = self.client.fine_tuning.jobs.list(limit=50)
        self.assertIn(job.id, [j.id for j in listed.data])

    def test_unknown_job_is_a_typed_404(self):
        with self.assertRaises(openai.NotFoundError):
            self.client.fine_tuning.jobs.retrieve("ftjob-nope")

    def test_missing_training_file_is_a_typed_400(self):
        with self.assertRaises(openai.NotFoundError):
            self.client.fine_tuning.jobs.create(
                model=str(self.model_dir), training_file="file-nope"
            )

    def test_unresolvable_model_is_refused_with_what_was_tried(self):
        uploaded = self.upload()
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.fine_tuning.jobs.create(
                model="some-model-that-is-not-here", training_file=uploaded.id
            )
        self.assertIn("could not be resolved", str(ctx.exception))

    def test_job_runs_to_completion_and_reports_a_model(self):
        job = self.submit(total_steps=40)
        final = wait_until(
            lambda: (
                lambda j: (
                    j if j.status in ("succeeded", "failed", "cancelled") else None
                )
            )(self.client.fine_tuning.jobs.retrieve(job.id)),
            60,
            f"job {job.id} reaching a terminal state",
        )
        self.assertEqual(final.status, "succeeded", final.model_extra)
        self.assertTrue(final.fine_tuned_model.startswith("ft:"))
        self.assertIsNotNone(final.finished_at)
        self.assertGreater(final.trained_tokens or 0, 0)

    def test_events_list_paginates_with_a_cursor(self):
        job = self.submit(total_steps=40)
        wait_until(
            lambda: len(
                self.client.fine_tuning.jobs.list_events(job.id, limit=100).data
            )
            >= 4,
            60,
            "events accumulating",
        )
        first = self.client.fine_tuning.jobs.list_events(job.id, limit=2)
        self.assertEqual(len(first.data), 2)
        self.assertEqual(first.data[0].object, "fine_tuning.job.event")
        second = self.client.fine_tuning.jobs.list_events(
            job.id, after=first.data[-1].id, limit=2
        )
        self.assertNotEqual(first.data[0].id, second.data[0].id)

    def test_checkpoints_are_listed_as_spec_shaped_objects(self):
        job = self.submit(total_steps=40, save_steps=10)
        wait_until(
            lambda: self.client.fine_tuning.jobs.checkpoints.list(job.id).data,
            60,
            "a checkpoint appearing",
        )
        checkpoints = self.client.fine_tuning.jobs.checkpoints.list(job.id)
        first = checkpoints.data[0]
        self.assertEqual(first.object, "fine_tuning.job.checkpoint")
        self.assertEqual(first.fine_tuning_job_id, job.id)
        self.assertGreater(first.step_number, 0)
        self.assertIn("checkpoint-", first.fine_tuned_model_checkpoint)

    def test_cancel_reaches_a_terminal_state(self):
        job = self.submit(total_steps=100000)
        wait_until(
            lambda: self.client.fine_tuning.jobs.retrieve(job.id).status == "running",
            30,
            "job starting",
        )
        cancelled = self.client.fine_tuning.jobs.cancel(job.id)
        self.assertIn(cancelled.status, ("running", "cancelled"))
        final = wait_until(
            lambda: (lambda j: j if j.status == "cancelled" else None)(
                self.client.fine_tuning.jobs.retrieve(job.id)
            ),
            30,
            "cancellation landing",
        )
        self.assertEqual(final.status, "cancelled")

    def test_cancelling_a_finished_job_is_a_conflict(self):
        job = self.submit(total_steps=20)
        wait_until(
            lambda: self.client.fine_tuning.jobs.retrieve(job.id).status == "succeeded",
            60,
            "job finishing",
        )
        with self.assertRaises(openai.ConflictError):
            self.client.fine_tuning.jobs.cancel(job.id)

    # -- the event stream ---------------------------------------------------

    def test_events_stream_is_sse_and_terminates(self):
        """``stream=true`` is the tap the OpenAI CLI uses. Raw, not via SDK."""
        job = self.submit(total_steps=20)
        response = requests.get(
            f"{self.base_url}/v1/fine_tuning/jobs/{job.id}/events",
            params={"stream": "true"},
            stream=True,
            timeout=90,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        frames = []
        done = False
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None or not raw.strip():
                continue
            if raw.startswith(":"):
                continue  # keepalive
            self.assertTrue(raw.startswith("data: "), raw)
            payload = raw[len("data: ") :]
            if payload == "[DONE]":
                done = True
                break
            frames.append(json.loads(payload))
        response.close()
        self.assertTrue(done, "stream must end with [DONE]")
        self.assertTrue(frames)
        self.assertEqual(frames[0]["object"], "fine_tuning.job.event")

    def test_a_stream_consumer_that_leaves_does_not_touch_the_job(self):
        """#344 client-liveness: the tap is cleaned up, the job is not."""
        job = self.submit(total_steps=200)
        response = requests.get(
            f"{self.base_url}/v1/fine_tuning/jobs/{job.id}/events",
            params={"stream": "true"},
            stream=True,
            timeout=30,
        )
        next(response.iter_lines(decode_unicode=True))
        response.close()
        wait_until(
            lambda: self.service.jobs.subscriber_count(job.id) == 0,
            30,
            "the subscriber being released",
        )
        # The job is a fire-and-forget object: losing its listener is not a
        # cancellation, and it must still finish.
        final = wait_until(
            lambda: (lambda j: j if j.status == "succeeded" else None)(
                self.client.fine_tuning.jobs.retrieve(job.id)
            ),
            120,
            "the job finishing without its listener",
        )
        self.assertEqual(final.status, "succeeded")

    # -- the tenant ---------------------------------------------------------

    def test_serving_demand_preempts_and_idle_resumes(self):
        """D4 through the HTTP surface, against a fake serving-demand signal."""
        job = self.submit(total_steps=600, save_steps=10)
        wait_until(
            lambda: self.client.fine_tuning.jobs.retrieve(job.id).status == "running",
            30,
            "job starting",
        )
        try:
            self.busy["value"] = True
            preempted = wait_until(
                lambda: (
                    lambda j: (
                        j
                        if j.model_extra["x-htsglang"]["tenant_state"] == "preempted"
                        else None
                    )
                )(self.client.fine_tuning.jobs.retrieve(job.id)),
                60,
                "preemption",
            )
            block = preempted.model_extra["x-htsglang"]
            # Preemption is NOT a protocol state: a vanilla client still sees
            # a job that is running, only for longer.
            self.assertEqual(preempted.status, "running")
            self.assertEqual(block["preemptions"], 1)
            self.assertGreater(block["last_step"], 0)
            self.assertIn("checkpoint-", block["resume_from"])
        finally:
            self.busy["value"] = False
        final = wait_until(
            lambda: (lambda j: j if j.status == "succeeded" else None)(
                self.client.fine_tuning.jobs.retrieve(job.id)
            ),
            180,
            "resumption and completion",
        )
        self.assertEqual(final.model_extra["x-htsglang"]["last_step"], 600)
        messages = [
            e.message
            for e in self.client.fine_tuning.jobs.list_events(job.id, limit=100).data
        ]
        self.assertTrue(any("preempting" in m for m in messages))
        self.assertTrue(any("resuming from" in m for m in messages))

    def test_tenant_endpoint_reports_the_machine_and_the_probes(self):
        body = requests.get(f"{self.base_url}/v1/fine_tuning/tenant", timeout=10).json()
        self.assertTrue(body["tenant"]["config"]["enabled"])
        self.assertEqual(body["machine"]["cards"][0]["name"], "Synthetic 80GB")
        self.assertIn("llamafactory", [p["backend"] for p in body["backends"]])


class InfeasibleRequestTest(unittest.TestCase):
    """A rejection that carries the arithmetic and the ladder."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.model_dir = write_base_model(root)
        cls.service = build_service(root, machine=TINY_MACHINE)
        cls._server = live_server(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME),
            training_service=cls.service,
        )
        cls.base_url = cls._server.__enter__()
        cls.client = openai.OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._tmp.cleanup()

    def test_full_finetune_on_a_small_card_is_rejected_with_the_ladder(self):
        uploaded = self.client.files.create(
            file=("train.jsonl", TRAINING_JSONL), purpose="fine-tune"
        )
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.fine_tuning.jobs.create(
                model=str(self.model_dir),
                training_file=uploaded.id,
                extra_body={"x-htsglang": {"method": "full", "sequence_length": 8192}},
            )
        body = ctx.exception.response.json()["error"]
        self.assertEqual(body["code"], "insufficient_resources")
        self.assertIn("Method ladder against this machine", body["message"])
        extension = body["x-htsglang"]
        self.assertFalse(extension["feasibility"]["fits"])
        ladder = extension["feasibility"]["ladder"]
        self.assertEqual(len(ladder), 5)
        # Cheapest-first, and every rung carries its own posts.
        self.assertEqual(
            [r["per_card_mib"] for r in ladder],
            sorted(r["per_card_mib"] for r in ladder),
        )
        for rung in ladder:
            self.assertIn("weights_mib", rung["posts"])
            self.assertIn("logits_mib", rung["posts"])
        self.assertTrue(extension["what_would_make_it_work"])
        self.assertIn("Synthetic 6GB", body["message"])


class DisabledTenantTest(unittest.TestCase):
    """The routes exist and say what is switched off."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls._server = live_server(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        )
        cls.base_url = cls._server.__enter__()
        cls.client = openai.OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def test_job_creation_names_the_flag(self):
        uploaded = self.client.files.create(
            file=("train.jsonl", TRAINING_JSONL), purpose="fine-tune"
        )
        with self.assertRaises(openai.APIStatusError) as ctx:
            self.client.fine_tuning.jobs.create(
                model="/nonexistent", training_file=uploaded.id
            )
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("--enable-training-tenant", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
