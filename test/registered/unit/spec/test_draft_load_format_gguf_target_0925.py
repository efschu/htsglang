"""A GGUF target's safetensors DRAFT loads in the draft's own format, on every build path.

Boot weg2rc4gg (27B line, RC4 9738626129, 2026-09-25 04:41:16Z), group P, PP2:

    File ".../speculative/draft_worker_common.py", line 239, in build_draft_tp_worker
    ...
    File ".../model_loader/loader.py", line 2253, in _prepare_weights
    ValueError: .../Qwen3.8-27B-DFlash2-W8-lued is not a file.

Mechanism. ``server_args._handle_load_format`` turns the TARGET's ``auto`` into
``gguf`` (the unsloth IQ4_XS file). Every draft runner built its ``LoadConfig``
from ``server_args.load_format``, so the DFlash2-lued-W8 draft -- a safetensors
DIRECTORY -- went to ``GGUFModelLoader``, which takes one ``.gguf`` FILE only.
``--speculative-draft-load-format auto`` did not help on P: the only place that
applied it was a process-wide ``server_args.override(load_format=...)`` on the
SPECULATIVE branch of ``Scheduler.maybe_init_draft_worker``, and P is draft-KV
only (spec algorithm NONE) -- it builds its draft through
``_maybe_init_draft_kv_producer`` -> ``DFlashDraftKvProducer`` ->
``DFlashWorkerV2`` -> ``build_draft_tp_worker``. On D the same override rewrote
the TARGET's ``load_format`` for the rest of the process.

Fix under test: the draft runner decides its own format
(``ModelRunner._this_runners_load_format`` ->
``configs.load_config.resolve_draft_load_format``): the flag when given, else
the target's format, except a GGUF target's ``gguf`` for a draft that is not a
GGUF file -> ``auto``. The process-wide override is gone.

The build paths run for real: ``DFlashDraftKvProducer.__init__`` (P) and
``Scheduler.maybe_init_draft_worker`` on its speculative branch (D), through
``DFlashWorkerV2.__init__``, ``build_draft_tp_worker``, ``TpModelWorker.__init__``
and its ``_init_model_config`` (a real draft ``ModelConfig`` from a tiny
config). ``ModelRunner.__init__`` is the one stand-in: it takes the load format
the way ``load_model`` takes it (pinned below), builds the real ``LoadConfig``,
asks the real ``get_model_loader`` and, if that is the GGUF loader, runs its
real first step ``_prepare_weights`` -- which is where the boot died. Group
collectives (tp/pp group, SpecTpSync, draft_pp_scope) are stubbed: no GPU, no
process group, no weights.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.configs.load_config as load_config_mod
import sglang.srt.distributed.parallel_state as parallel_state
import sglang.srt.model_executor.model_runner as MR
import sglang.srt.speculative.dflash_draft_kv_producer as dkp
import sglang.srt.speculative.dflash_worker_v2 as dfw
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    GGUFModelLoader,
    get_model_loader,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _StopAtLoaderChoice(Exception):
    """Raised by the runner stand-in once the loader is chosen: the build stops
    there, and the record says what the draft runner would have loaded with."""


def _runner_stand_in(self, *args, **kwargs):
    """Stands in for ``ModelRunner.__init__`` (every ``TpModelWorker`` passes
    keyword arguments). It takes the load format exactly as ``load_model``
    takes it -- ``self._this_runners_load_format()`` on the fix, the raw
    ``server_args.load_format`` on the base (both pinned by
    ``test_the_stand_in_takes_the_format_the_way_load_model_does``) -- builds
    the real LoadConfig and asks the real loader factory."""
    self.server_args = kwargs["server_args"]
    self.model_config = kwargs["model_config"]
    self.tp_rank = kwargs.get("tp_rank", 0)
    self.is_draft_worker = kwargs.get("is_draft_worker", False)
    self.is_phase_flip_tp_stack = kwargs.get("is_phase_flip_tp_stack", False)
    # verbatim from ModelRunner.__init__ (pinned below)
    self.is_draft_model_runner = (
        self.is_draft_worker and not self.is_phase_flip_tp_stack
    )
    decide = getattr(MR.ModelRunner, "_this_runners_load_format", None)
    decided = decide(self) if decide is not None else self.server_args.load_format
    load_config = LoadConfig(load_format=decided)
    loader = get_model_loader(load_config, self.model_config)
    record = {
        "decided": decided,
        "load_format": load_config.load_format,
        "loader": type(loader),
        "model_path": self.model_config.model_path,
        "is_draft_model_runner": self.is_draft_model_runner,
    }
    if isinstance(loader, GGUFModelLoader):
        # the boot's own first step in GGUFModelLoader.load_model
        loader._prepare_weights(self.model_config.model_path)
    raise _StopAtLoaderChoice(record)


class _Group:
    is_last_rank = True
    rank_in_group = 2
    world_size = 3


class _TargetWorker:
    """The target side a draft build reads: its model config's context length
    and its device."""

    device = "cpu"
    model_runner = SimpleNamespace(model_config=SimpleNamespace(context_len=4096))


def _ranks():
    return SimpleNamespace(
        gpu_id=0, tp_rank=0, dp_rank=None, moe_ep_rank=0, attn_cp_rank=0, moe_dp_rank=0
    )


@contextlib.contextmanager
def _build_scope(target_args):
    """The collectives a draft build enters, stubbed; ModelRunner replaced; the
    target's ServerArgs published the way the scheduler process publishes them
    (set_global_server_args_for_scheduler) before any draft is built."""
    get_context().set_server_args(target_args)
    with contextlib.ExitStack() as stack:
        stack.callback(setattr, get_context(), "_server_args", None)
        stack.enter_context(
            mock.patch.object(MR.ModelRunner, "__init__", _runner_stand_in)
        )
        stack.enter_context(
            mock.patch.object(parallel_state, "get_pp_group", lambda: _Group())
        )
        stack.enter_context(
            mock.patch.object(parallel_state, "draft_pp_scope", contextlib.nullcontext)
        )
        stack.enter_context(mock.patch.object(dfw, "get_tp_group", lambda: None))
        stack.enter_context(mock.patch.object(dfw, "SpecTpSync", lambda *a, **k: None))
        yield


class _DraftDir:
    """A tiny safetensors-style draft DIRECTORY (config only: the build stops at
    the loader choice)."""

    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="draft_dir_")
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 1,
            "vocab_size": 256,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        }
        with open(os.path.join(self.path, "config.json"), "w") as f:
            json.dump(cfg, f)


def _target_args(draft_dir, *, target_format="gguf", draft_flag=None, kv_only=False):
    """The target's RESOLVED ServerArgs: ``gguf`` is what _handle_load_format
    leaves on a .gguf target (weg2rc4gg: model_path=...UD-IQ4_XS.gguf)."""
    args = ServerArgs(model_path="dummy")
    args.override(
        "test.target",
        load_format=target_format,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=draft_dir,
        speculative_draft_load_format=draft_flag,
        speculative_draft_kv_only=kv_only,
    )
    return args


class ProducerPath(CustomTestCase):
    """P: spec algorithm NONE, draft-KV producer (the path that died)."""

    def _build(self, args):
        scheduler = SimpleNamespace(
            server_args=args, ps=_ranks(), nccl_port=0, tp_worker=_TargetWorker()
        )
        with _build_scope(args):
            with self.assertRaises(_StopAtLoaderChoice) as cm:
                dkp.DFlashDraftKvProducer(scheduler, SpeculativeAlgorithm.DFLASH)
        return cm.exception.args[0]

    def test_gguf_target_draft_directory_loads_auto_without_any_flag(self):
        draft = _DraftDir()
        rec = self._build(_target_args(draft.path, kv_only=True))
        self.assertTrue(rec["is_draft_model_runner"])
        self.assertEqual(rec["model_path"], draft.path)
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertIs(rec["loader"], DefaultModelLoader)

    def test_the_flag_reaches_the_producer_draft(self):
        draft = _DraftDir()
        rec = self._build(_target_args(draft.path, kv_only=True, draft_flag="auto"))
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertIs(rec["loader"], DefaultModelLoader)

    def test_non_gguf_target_is_byte_identical(self):
        """INT8 / FP8 / NVFP4: the target is a directory, load_format stays
        ``auto``, the flag is unset (server_args echo of the RC4 boots). The
        draft runner hands the target's own object on, unchanged."""
        draft = _DraftDir()
        args = _target_args(draft.path, target_format="auto", kv_only=True)
        rec = self._build(args)
        self.assertIs(rec["decided"], args.load_format)
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertIs(rec["loader"], DefaultModelLoader)


class SpecPath(CustomTestCase):
    """D: speculative branch of Scheduler.maybe_init_draft_worker."""

    def _build(self, args):
        scheduler = SimpleNamespace(
            server_args=args,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            ps=_ranks(),
            nccl_port=0,
            tp_worker=_TargetWorker(),
        )
        with _build_scope(args):
            with self.assertRaises(_StopAtLoaderChoice) as cm:
                Scheduler.maybe_init_draft_worker(scheduler)
        return cm.exception.args[0]

    def test_gguf_target_draft_directory_loads_auto_without_any_flag(self):
        draft = _DraftDir()
        args = _target_args(draft.path)
        rec = self._build(args)
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertIs(rec["loader"], DefaultModelLoader)
        self.assertEqual(args.load_format, "gguf")

    def test_the_flag_does_not_rewrite_the_targets_load_format(self):
        """The old process-wide override made the TARGET's load_format 'auto'
        for the rest of the process; every later reader of it (the weg2 wake
        disk-reload of a .gguf target) then read the draft's format."""
        draft = _DraftDir()
        args = _target_args(draft.path, draft_flag="auto")
        rec = self._build(args)
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertIs(rec["loader"], DefaultModelLoader)
        self.assertEqual(args.load_format, "gguf")

    def test_non_gguf_target_is_byte_identical(self):
        draft = _DraftDir()
        args = _target_args(draft.path, target_format="auto")
        rec = self._build(args)
        self.assertIs(rec["decided"], args.load_format)
        self.assertIs(rec["loader"], DefaultModelLoader)
        self.assertEqual(args.load_format, "auto")


class EagleShape(CustomTestCase):
    """The EAGLE family (eagle_worker_v2, multi_layer_eagle_worker_v2,
    standalone_worker_v2, draft_kv_producer) hands the TARGET's ServerArgs
    straight to TpModelWorker(is_draft_worker=True) -- no draft copy."""

    def _build(self, args, **extra):
        from sglang.srt.managers.tp_worker import TpModelWorker

        with _build_scope(args):
            with self.assertRaises(_StopAtLoaderChoice) as cm:
                TpModelWorker(
                    server_args=args,
                    gpu_id=0,
                    tp_rank=0,
                    moe_ep_rank=0,
                    pp_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    dp_rank=None,
                    nccl_port=0,
                    is_draft_worker=True,
                    context_length=4096,
                    **extra,
                )
        return cm.exception.args[0]

    def test_gguf_target_draft_directory_loads_auto(self):
        draft = _DraftDir()
        args = _target_args(draft.path)
        rec = self._build(args)
        self.assertEqual(rec["load_format"], LoadFormat.AUTO)
        self.assertEqual(args.load_format, "gguf")

    def test_an_explicit_flag_is_honoured_without_the_scheduler_override(self):
        draft = _DraftDir()
        rec = self._build(
            _target_args(draft.path, target_format="auto", draft_flag="dummy")
        )
        self.assertEqual(rec["load_format"], LoadFormat.DUMMY)


class TheDecision(CustomTestCase):
    """configs.load_config.resolve_draft_load_format and the runner method."""

    def _resolve(self):
        fn = getattr(load_config_mod, "resolve_draft_load_format", None)
        self.assertIsNotNone(
            fn, "configs.load_config.resolve_draft_load_format is missing"
        )
        return fn

    def test_the_table(self):
        resolve = self._resolve()
        draft = _DraftDir()
        gguf_file = os.path.join(tempfile.mkdtemp(), "draft.gguf")
        with open(gguf_file, "wb") as f:
            f.write(b"GGUF" + b"\0" * 16)
        ns = lambda fmt, flag=None: SimpleNamespace(  # noqa: E731
            load_format=fmt, speculative_draft_load_format=flag
        )
        auto = "auto"
        self.assertEqual(resolve(ns("gguf"), draft.path), "auto")
        self.assertEqual(
            resolve(ns("gguf"), gguf_file), "gguf"
        )  # a GGUF drafter / the target's own MTP head
        self.assertEqual(resolve(ns("gguf"), None), "gguf")
        self.assertIs(resolve(ns(auto), draft.path), auto)
        self.assertEqual(resolve(ns("safetensors"), draft.path), "safetensors")
        self.assertEqual(resolve(ns("gguf", "dummy"), draft.path), "dummy")
        self.assertEqual(resolve(ns("auto", "gguf"), gguf_file), "gguf")
        self.assertEqual(resolve(ns(LoadFormat.GGUF), draft.path), "auto")

    def test_target_and_phase_flip_stack_keep_the_target_format(self):
        decide = getattr(MR.ModelRunner, "_this_runners_load_format", None)
        self.assertIsNotNone(decide, "ModelRunner._this_runners_load_format is missing")
        draft = _DraftDir()
        args = _target_args(draft.path, draft_flag="auto")
        for is_draft in (False, True):
            runner = MR.ModelRunner.__new__(MR.ModelRunner)
            runner.server_args = args
            runner.model_config = SimpleNamespace(model_path=draft.path)
            runner.is_draft_model_runner = is_draft
            got = decide(runner)
            if is_draft:
                self.assertEqual(got, "auto")
            else:
                self.assertIs(got, args.load_format)

    def test_the_stand_in_takes_the_format_the_way_load_model_does(self):
        load_model = inspect.getsource(MR.ModelRunner.load_model)
        init = inspect.getsource(MR.ModelRunner.__init__)
        self.assertIn("load_format=self._this_runners_load_format(),", load_model)
        self.assertNotIn("load_format=self.server_args.load_format,", load_model)
        self.assertIn(
            "self.is_draft_model_runner = is_draft_worker and not is_phase_flip_tp_stack",
            init,
        )

    def test_no_process_wide_draft_load_format_override(self):
        code = "\n".join(
            line
            for line in inspect.getsource(
                Scheduler.maybe_init_draft_worker
            ).splitlines()
            if not line.strip().startswith("#")
        )
        self.assertNotIn("speculative_draft_load_format", code)
        self.assertNotIn(".override(", code)


if __name__ == "__main__":
    unittest.main()
