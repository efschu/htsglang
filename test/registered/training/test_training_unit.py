# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #341-M1 pieces below the HTTP surface: formula, store, tenant.

Everything here runs on a CPU-only host with no card, no ledger and no
training suite installed, which is deliberate: the feasibility gate is a
formula over numbers, and a formula that can only be tested on the machine it
was written for is exactly the rig constant DESIGN #341 D2 forbids. Every
machine in these tests is synthetic, and the assertions are about arithmetic
and about ordering, not about this rig.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.training import feasibility as feas
from sglang.srt.training.backends import RunSpec, RunStatus, get_backend
from sglang.srt.training.backends.llamafactory import (
    build_config,
    detect_records_style,
    latest_checkpoint,
    parse_log_line,
)
from sglang.srt.training.backends.mock import MockBackend
from sglang.srt.training.feasibility import (
    GIB,
    LADDER,
    CardResources,
    MachineResources,
    ModelProfile,
    TrainingDemandSpec,
    TrainingMethod,
)
from sglang.srt.training.service import (
    TenantDisabled,
    TrainingService,
    TrainingServiceConfig,
    parse_extension,
)
from sglang.srt.training.store import (
    FileStore,
    InvalidFile,
    JobStatus,
    JobStore,
    TenantState,
    validate_jsonl,
)
from sglang.srt.training.tenant import DemandSample, IdleMonitor
from sglang.test.ci.ci_register import register_cpu_ci

# No card is touched anywhere in this file; that is the point of it.
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


TRAINING_JSONL = b"".join(
    json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
    ).encode()
    + b"\n"
    for i in range(8)
)


def card(name: str, total_gib: float, *, index: int, available_gib=None):
    total = int(total_gib * GIB)
    available = total if available_gib is None else int(available_gib * GIB)
    return CardResources(
        uuid=f"GPU-{name}",
        index=index,
        name=name,
        total_bytes=total,
        available_bytes=available,
    )


def machine(*cards, ram_gib: float = 128.0, disk_gib: float = 500.0):
    return MachineResources(
        cards=tuple(cards),
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB * 0.8),
        disk_free_bytes=int(disk_gib * GIB),
        disk_path="/synthetic",
    )


def model(params_b: float, *, hidden=4096, layers=32, heads=32, vocab=152064):
    return ModelProfile(
        path="/synthetic/model",
        params=int(params_b * 1e9),
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        vocab_size=vocab,
        stored_dtype_bytes=2.0,
        weight_bytes_on_disk=int(params_b * 1e9 * 2),
        source="synthetic",
    )


def write_model_dir(root: Path, *, hidden=2048, layers=24, vocab=151936) -> Path:
    directory = root / "base-model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "intermediate_size": hidden * 4,
                "vocab_size": vocab,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (directory / "model.safetensors").write_bytes(b"\0" * 4096)
    return directory


class FeasibilityFormulaTest(unittest.TestCase):
    """D2: a formula over the actual machine, with an actionable rejection."""

    def test_ladder_is_ordered_by_computed_cost_not_by_name(self):
        spec = TrainingDemandSpec(sequence_length=1024, micro_batch_size=1)
        decision = feas.evaluate(
            model=model(7),
            method=TrainingMethod.FULL,
            spec=spec,
            machine=machine(card("A", 24, index=0)),
        )
        totals = [o.posts.device_total for o in decision.ladder]
        self.assertEqual(totals, sorted(totals), "ladder must be cheapest-first")
        self.assertEqual(
            {o.method for o in decision.ladder}, set(LADDER), "every rung priced"
        )

    def test_qlora_is_cheaper_than_lora_which_is_cheaper_than_full(self):
        spec = TrainingDemandSpec(sequence_length=1024)
        by_method = {
            m: feas.compute_posts(model(7), m, spec).device_total for m in LADDER
        }
        self.assertLess(by_method[TrainingMethod.QLORA], by_method[TrainingMethod.LORA])
        self.assertLess(by_method[TrainingMethod.LORA], by_method[TrainingMethod.FULL])
        self.assertLess(
            by_method[TrainingMethod.FULL_OFFLOAD], by_method[TrainingMethod.FULL]
        )

    def test_same_request_fits_on_a_bigger_machine(self):
        """The formula is the machine's, not the rig's."""
        spec = TrainingDemandSpec(sequence_length=1024)
        small = feas.evaluate(
            model=model(32),
            method=TrainingMethod.FULL,
            spec=spec,
            machine=machine(card("small", 20, index=0)),
        )
        big = feas.evaluate(
            model=model(32),
            method=TrainingMethod.FULL,
            spec=spec,
            machine=machine(card("big", 640, index=0), ram_gib=2048, disk_gib=4096),
        )
        self.assertFalse(small.fits)
        self.assertTrue(big.fits, big.render())

    def test_rejection_names_the_cheapest_rung_that_would_fit(self):
        spec = TrainingDemandSpec(sequence_length=1024)
        decision = feas.evaluate(
            model=model(7),
            method=TrainingMethod.FULL,
            spec=spec,
            machine=machine(card("RTX 3080", 20, index=0)),
        )
        self.assertFalse(decision.fits)
        rendered = decision.render()
        self.assertIn("Method ladder against this machine", rendered)
        self.assertIn("MiB/card", rendered)
        self.assertIn("short by", rendered)
        alternatives = decision.fitting_alternatives()
        self.assertTrue(alternatives, rendered)
        self.assertIn(alternatives[0].method.value, " ".join(decision.remedies))

    def test_posts_are_all_named_and_sum_to_the_total(self):
        posts = feas.compute_posts(
            model(7), TrainingMethod.LORA, TrainingDemandSpec(sequence_length=2048)
        )
        named = (
            posts.weights
            + posts.gradients
            + posts.optimizer
            + posts.activations
            + posts.logits
            + posts.cuda_context
        )
        self.assertEqual(named, posts.device_total)
        self.assertGreater(posts.logits, 0, "the fp32 logit upcast is a real post")

    def test_offloaded_full_moves_the_optimizer_to_host_ram(self):
        spec = TrainingDemandSpec(sequence_length=512)
        offload = feas.compute_posts(model(7), TrainingMethod.FULL_OFFLOAD, spec)
        self.assertEqual(offload.optimizer, 0)
        self.assertEqual(offload.gradients, 0)
        self.assertGreater(offload.host_offload, 0)

    def test_no_visible_card_is_reported_as_such_not_as_a_shortfall(self):
        decision = feas.evaluate(
            model=model(1),
            method=TrainingMethod.LORA,
            spec=TrainingDemandSpec(),
            machine=MachineResources(probe_error="NvmlUnavailableError: no driver"),
        )
        self.assertFalse(decision.fits)
        self.assertIn("No CUDA device is visible", decision.message)
        self.assertIn("no driver", decision.message)

    def test_disk_shortfall_is_its_own_rejection_and_is_priced_per_rung(self):
        """A full-FT checkpoint is the model; a LoRA checkpoint is not."""
        decision = feas.evaluate(
            model=model(7),
            method=TrainingMethod.FULL,
            spec=TrainingDemandSpec(sequence_length=256, checkpoints_retained=2),
            machine=machine(card("H100", 80, index=0), ram_gib=512, disk_gib=8),
        )
        self.assertFalse(decision.fits)
        self.assertIn("disk short by", decision.message)
        # LoRA on the same disk is fine, and the ladder must say so rather
        # than failing the whole request on one rung's checkpoint size.
        lora = next(o for o in decision.ladder if o.method is TrainingMethod.LORA)
        self.assertTrue(lora.fits, decision.render())

    def test_profile_reads_the_model_directory_not_the_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_model_dir(Path(tmp))
            profile = feas.profile_model(directory)
            self.assertGreater(profile.params, 0)
            self.assertEqual(profile.hidden_size, 2048)
            self.assertEqual(profile.num_layers, 24)
            self.assertIn(
                profile.source, ("weight_files_on_disk", "config_architecture")
            )

    def test_profile_refuses_a_directory_it_cannot_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(feas.ModelProfileError):
                feas.profile_model(Path(tmp) / "nope")


class FileStoreTest(unittest.TestCase):
    def test_valid_jsonl_counts_records(self):
        self.assertEqual(validate_jsonl(TRAINING_JSONL), 8)

    def test_bad_json_names_the_line(self):
        with self.assertRaises(InvalidFile) as ctx:
            validate_jsonl(b'{"messages": []}\n{not json}\n')
        self.assertIn("line 2", str(ctx.exception))

    def test_record_without_a_recognised_key_is_refused(self):
        with self.assertRaises(InvalidFile) as ctx:
            validate_jsonl(b'{"whatever": 1}\n')
        self.assertIn("line 1", str(ctx.exception))

    def test_filename_traversal_is_reduced_to_a_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            stored = store.create(
                filename="../../etc/passwd", content=TRAINING_JSONL, purpose="fine-tune"
            )
            self.assertEqual(stored.filename, "passwd")
            self.assertTrue(str(stored.path).startswith(tmp))

    def test_wrong_purpose_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            with self.assertRaises(InvalidFile) as ctx:
                store.create(filename="a.jsonl", content=b"{}", purpose="batch")
            self.assertIn("fine-tune", str(ctx.exception))


class ExtensionParsingTest(unittest.TestCase):
    def test_extension_object_wins_over_metadata(self):
        parsed = parse_extension(
            {
                "metadata": {"x-htsglang.method": "lora", "unrelated": "1"},
                "x-htsglang": {"method": "qlora", "save_steps": 10},
            }
        )
        self.assertEqual(parsed["method"], "qlora")
        self.assertEqual(parsed["save_steps"], 10)
        self.assertNotIn("unrelated", parsed)

    def test_metadata_carrier_alone_works(self):
        parsed = parse_extension({"metadata": {"x-htsglang.method": "freeze"}})
        self.assertEqual(parsed["method"], "freeze")

    def test_vanilla_request_yields_no_extension(self):
        self.assertEqual(parse_extension({"model": "m", "training_file": "f"}), {})


class IdleMonitorTest(unittest.TestCase):
    def test_a_source_with_no_opinion_does_not_veto(self):
        clock = [1000.0]
        monitor = IdleMonitor(
            [lambda: DemandSample(source="blind")],
            grace_seconds=60.0,
            clock=lambda: clock[0],
        )
        verdict = monitor.sample()
        self.assertTrue(verdict.idle)
        self.assertIn("no activity has ever been observed", verdict.reason())

    def test_recent_activity_blocks_the_idle_window(self):
        clock = [1000.0]
        monitor = IdleMonitor(
            [lambda: DemandSample(source="local", last_activity_ts=990.0)],
            grace_seconds=60.0,
            clock=lambda: clock[0],
        )
        self.assertFalse(monitor.sample().idle)
        clock[0] = 1100.0
        self.assertTrue(monitor.sample().idle)

    def test_an_explicitly_busy_source_blocks_regardless_of_the_grace(self):
        monitor = IdleMonitor(
            [lambda: DemandSample(source="engine", busy=True)],
            grace_seconds=0.0,
            clock=lambda: 1000.0,
        )
        verdict = monitor.sample()
        self.assertFalse(verdict.idle)
        self.assertIn("engine", verdict.reason())

    def test_a_raising_source_is_skipped_not_fatal(self):
        def broken():
            raise RuntimeError("registry on fire")

        monitor = IdleMonitor([broken], grace_seconds=0.0, clock=lambda: 1.0)
        self.assertTrue(monitor.sample().idle)


class LlamaFactoryWrapperTest(unittest.TestCase):
    """The wrapper's pure parts. No suite is installed on this host."""

    def test_probe_rejects_informatively_when_not_installed(self):
        probe = get_backend("llamafactory").probe()
        if probe.available:
            self.skipTest("LLaMA-Factory is installed here; the reject path is moot")
        self.assertIn("not installed", probe.reason)
        self.assertTrue(any("pip install" in r for r in probe.remedies))

    def test_config_carries_the_method_and_the_preemption_granularity(self):
        spec = RunSpec(
            job_id="ftjob-1",
            base_model_path="/models/base",
            method=TrainingMethod.QLORA,
            dataset_path=Path("/tmp/train.jsonl"),
            output_dir=Path("/tmp/out"),
            save_steps=17,
            sequence_length=1024,
        )
        config = build_config(spec, dataset_dir=Path("/tmp/out/dataset"))
        self.assertEqual(config["finetuning_type"], "lora")
        self.assertEqual(config["quantization_bit"], 4)
        self.assertEqual(config["save_steps"], 17)
        self.assertEqual(config["cutoff_len"], 1024)
        self.assertEqual(config["model_name_or_path"], "/models/base")

    def test_resume_sets_the_checkpoint_and_stops_overwriting(self):
        spec = RunSpec(
            job_id="ftjob-1",
            base_model_path="/models/base",
            method=TrainingMethod.LORA,
            dataset_path=Path("/tmp/train.jsonl"),
            output_dir=Path("/tmp/out"),
            resume_from="/tmp/out/checkpoint-50",
        )
        config = build_config(spec, dataset_dir=Path("/tmp/out/dataset"))
        self.assertEqual(config["resume_from_checkpoint"], "/tmp/out/checkpoint-50")
        self.assertFalse(config["overwrite_output_dir"])

    def test_trainer_log_line_becomes_metrics(self):
        parsed = parse_log_line(
            "{'loss': 1.2345, 'grad_norm': 0.5, 'learning_rate': 4.9e-05, 'epoch': 0.12}"
        )
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["loss"], 1.2345)

    def test_a_plain_line_is_not_metrics(self):
        self.assertIsNone(parse_log_line("Loading checkpoint shards: 50%"))

    def test_latest_checkpoint_picks_the_highest_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (10, 200, 30):
                (root / f"checkpoint-{step}").mkdir()
            (root / "not-a-checkpoint").mkdir()
            self.assertTrue(latest_checkpoint(root).endswith("checkpoint-200"))

    def test_records_style_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_bytes(TRAINING_JSONL)
            self.assertEqual(detect_records_style(path), "sharegpt")
            path.write_text('{"instruction": "a", "output": "b"}\n')
            self.assertEqual(detect_records_style(path), "alpaca")


class MockBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_produces_checkpoints_and_an_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            spec = RunSpec(
                job_id="ftjob-mock",
                base_model_path="/models/base",
                method=TrainingMethod.LORA,
                dataset_path=Path(tmp) / "train.jsonl",
                output_dir=Path(tmp) / "out",
                save_steps=10,
                extra={"total_steps": 40},
            )
            run = await MockBackend(step_seconds=0.001).launch(spec, events.append)
            outcome = await run.wait()
            self.assertIs(outcome.status, RunStatus.SUCCEEDED)
            self.assertTrue(Path(outcome.artifact_path).exists())
            checkpoints = [e for e in events if e.checkpoint_path]
            self.assertEqual(len(checkpoints), 4)
            self.assertTrue(any(e.type == "metrics" for e in events))

    async def test_preempt_stops_at_a_checkpoint_and_resume_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = RunSpec(
                job_id="ftjob-mock",
                base_model_path="/models/base",
                method=TrainingMethod.LORA,
                dataset_path=Path(tmp) / "train.jsonl",
                output_dir=Path(tmp) / "out",
                save_steps=10,
                extra={"total_steps": 100},
            )
            backend = MockBackend(step_seconds=0.005)
            run = await backend.launch(spec, lambda e: None)
            await asyncio.sleep(0.06)
            outcome = await run.preempt(timeout_s=5)
            self.assertIs(outcome.status, RunStatus.PREEMPTED)
            self.assertIsNotNone(outcome.last_checkpoint)
            first_stop = outcome.last_step
            self.assertGreater(first_stop, 0)
            self.assertLess(first_stop, 100)

            resumed = await backend.launch(
                RunSpec(**{**spec.__dict__, "resume_from": outcome.last_checkpoint}),
                lambda e: None,
            )
            final = await resumed.wait()
            self.assertIs(final.status, RunStatus.SUCCEEDED)
            self.assertEqual(final.last_step, 100)

    async def test_cancel_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = RunSpec(
                job_id="ftjob-mock",
                base_model_path="/models/base",
                method=TrainingMethod.LORA,
                dataset_path=Path(tmp) / "train.jsonl",
                output_dir=Path(tmp) / "out",
                extra={"total_steps": 1000},
            )
            run = await MockBackend(step_seconds=0.005).launch(spec, lambda e: None)
            await asyncio.sleep(0.02)
            outcome = await run.cancel(timeout_s=5)
            self.assertIs(outcome.status, RunStatus.CANCELLED)


class TenantPreemptResumeTest(unittest.IsolatedAsyncioTestCase):
    """D4 end to end, against a fake serving-demand signal."""

    def build(self, tmp: Path, *, demand):
        directory = write_model_dir(tmp)
        config = TrainingServiceConfig(
            enabled=True,
            artifact_root=tmp / "artifacts",
            grace_seconds=0.0,
            poll_seconds=0.02,
            default_backend="mock",
            save_steps=10,
        )
        service = TrainingService(
            config,
            monitor=IdleMonitor([demand], grace_seconds=0.0, clock=lambda: 1000.0),
            machine_resolver=lambda: machine(card("synthetic", 80, index=3)),
            backend_factory=lambda name, method: MockBackend(step_seconds=0.004),
        )
        return service, directory

    async def test_job_runs_preempts_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            busy = {"value": False}
            service, directory = self.build(
                root, demand=lambda: DemandSample(source="fake", busy=busy["value"])
            )
            service.start()
            try:
                stored = service.create_file(
                    filename="train.jsonl",
                    content=TRAINING_JSONL,
                    purpose="fine-tune",
                )
                job = service.create_job(
                    {
                        "model": str(directory),
                        "training_file": stored.id,
                        "x-htsglang": {
                            "method": "lora",
                            "sequence_length": 512,
                            "save_steps": 10,
                            "total_steps": 400,
                        },
                    }
                )
                self.assertIs(job.status, JobStatus.QUEUED)

                await self.until(lambda: job.tenant_state is TenantState.TRAINING, 5)
                self.assertIs(job.status, JobStatus.RUNNING)

                # Serving demand arrives.
                busy["value"] = True
                await self.until(lambda: job.tenant_state is TenantState.PREEMPTED, 10)
                # Protocol status is unchanged: preemption is not a state.
                self.assertIs(job.status, JobStatus.RUNNING)
                self.assertEqual(job.preemptions, 1)
                self.assertIsNotNone(job.resume_from)
                preempted_at = job.last_step
                self.assertGreater(preempted_at, 0)

                # The rig goes idle again.
                busy["value"] = False
                await self.until(lambda: job.status is JobStatus.SUCCEEDED, 20)
                self.assertIs(job.tenant_state, TenantState.DONE)
                self.assertEqual(job.last_step, 400)
                self.assertTrue(job.fine_tuned_model.startswith("ft:"))
                self.assertGreaterEqual(len(job.checkpoints), 2)
                messages = [e.message for e in job.events]
                self.assertTrue(any("preempting" in m for m in messages), messages[:5])
                self.assertTrue(any("resuming from" in m for m in messages))
            finally:
                await service.stop()

    async def test_cancel_while_running_reaches_a_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service, directory = self.build(
                root, demand=lambda: DemandSample(source="fake")
            )
            service.start()
            try:
                stored = service.create_file(
                    filename="train.jsonl",
                    content=TRAINING_JSONL,
                    purpose="fine-tune",
                )
                job = service.create_job(
                    {
                        "model": str(directory),
                        "training_file": stored.id,
                        "x-htsglang": {"sequence_length": 512, "total_steps": 5000},
                    }
                )
                await self.until(lambda: job.tenant_state is TenantState.TRAINING, 5)
                service.cancel_job(job.id)
                await self.until(lambda: job.status is JobStatus.CANCELLED, 10)
            finally:
                await service.stop()

    async def test_disabled_tenant_rejects_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingService(
                TrainingServiceConfig(enabled=False, artifact_root=Path(tmp))
            )
            with self.assertRaises(TenantDisabled) as ctx:
                service.create_job({"model": "m", "training_file": "file-x"})
            self.assertIn("--enable-training-tenant", str(ctx.exception))

    async def until(self, predicate, timeout_s: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"condition not reached within {timeout_s}s")


class JobStoreTest(unittest.TestCase):
    def test_event_cursor_pagination(self):
        from sglang.srt.training.store import Hyperparameters, TrainingJob, new_id

        store = JobStore()
        job = store.create(
            TrainingJob(
                id=new_id("ftjob"),
                created_at=0,
                model="m",
                training_file="file-1",
                seed=0,
                hyperparameters=Hyperparameters(),
            )
        )
        for index in range(5):
            store.append_event(job, "info", f"event {index}")
        first, has_more = store.events_after(job, after=None, limit=2)
        self.assertEqual(len(first), 2)
        self.assertTrue(has_more)
        second, has_more = store.events_after(job, after=first[-1].id, limit=10)
        self.assertEqual(len(second), 3)
        self.assertFalse(has_more)
        self.assertEqual(second[0].message, "event 2")


if __name__ == "__main__":
    unittest.main()
