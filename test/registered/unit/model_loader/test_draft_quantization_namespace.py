# SPDX-License-Identifier: Apache-2.0
"""#318: a draft block stored dense inside a quantized checkpoint.

``Qwen3.6-27B-...-MTP-Preserved-GPTQ-Int4`` carries 2399 tensors. 400 of them
are ``qweight``/``qzeros``/``scales``/``g_idx`` quadruples -- and the 15 under
``mtp.`` are none of those: GPTQModel left the MTP block in BF16 and, unlike
the AWQ and FP8 siblings of the same base model, wrote no
``modules_to_not_convert`` entry saying so.

sglang then built the NEXTN drafter from the target's ``gptq`` method: a Marlin
skeleton whose parameters are ``qweight``/``qzeros``/``scales``/``g_idx``, fed
seven plain ``mtp.*.weight`` names per rank. Every one of them missed, each
behind a deduplicated ``logger.warning_once`` on the CHECKPOINT-name side, and
the load reported success. The drafter kept its uninitialized weights: accept
1.0052, 0 of 573 drafts accepted over 191 verify ticks, bit-identical in both
arms of the measurement.

This is #290's failure signature with the two sides swapped (there: a packed
GGUF stream into a dense skeleton), which is why the three fixes pinned here
belong together:

* the draft's quantization is decided from the CHECKPOINT NAMESPACE, not from
  geometry -- #316's shape guard is right not to fire, MTP shapes are
  Marlin-legal;
* ``--speculative-draft-model-quantization unquant`` becomes a real escape
  hatch instead of a value indistinguishable from "not given";
* a draft parameter nothing ever wrote is a named error, wherever it happens.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest

import torch

from sglang.srt.configs.model_config import (
    ModelConfig,
    _draft_checkpoint_is_dense,
    _quant_cfg_excludes_draft_namespace,
)
from sglang.srt.model_loader.weight_utils import raise_on_unloaded_draft_parameters
from sglang.srt.utils.common import (
    checkpoint_namespace_is_dense,
    checkpoint_weight_names,
)

GPTQ_QUANT_CFG = {
    "bits": 4,
    "checkpoint_format": "gptq",
    "desc_act": False,
    "group_size": 128,
    "quant_method": "gptq",
    "sym": True,
}

# The seven projections that missed, and the norms that landed.
_MTP_DENSE_NAMES = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]


def _packed(name: str) -> list:
    stem = name[: -len(".weight")]
    return [f"{stem}.{suffix}" for suffix in ("qweight", "qzeros", "scales", "g_idx")]


def _target_names(packed: bool = True) -> list:
    names = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    for layer in range(2):
        base = f"model.language_model.layers.{layer}"
        names.append(f"{base}.input_layernorm.weight")
        for proj in ("self_attn.qkv_proj", "mlp.down_proj"):
            if packed:
                names.extend(_packed(f"{base}.{proj}.weight"))
            else:
                names.append(f"{base}.{proj}.weight")
    return names


def _write_checkpoint(directory: str, names, quant_cfg=None) -> str:
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in names
                }
            },
            f,
        )
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    if quant_cfg is not None:
        config["quantization_config"] = quant_cfg
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)
    return directory


class TestTheNamespaceProbe(unittest.TestCase):
    """`checkpoint_namespace_is_dense` reads names, never weights."""

    def test_the_mtp_block_of_the_gptq_checkpoint_reads_dense(self):
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=True) + _MTP_DENSE_NAMES)
            self.assertTrue(checkpoint_namespace_is_dense(d, prefixes=("mtp.",)))
            # ... while the file as a whole is emphatically not dense. The two
            # verdicts disagreeing IS the bug.
            self.assertFalse(checkpoint_namespace_is_dense(d, prefixes=("",)))

    def test_a_packed_mtp_block_reads_packed(self):
        packed_mtp = []
        for name in _MTP_DENSE_NAMES:
            if name.endswith(("q_proj.weight", "down_proj.weight")):
                packed_mtp.extend(_packed(name))
            else:
                packed_mtp.append(name)
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=True) + packed_mtp)
            self.assertFalse(checkpoint_namespace_is_dense(d, prefixes=("mtp.",)))

    def test_an_absent_namespace_has_no_verdict(self):
        """None, not False: the caller must not read "packed" into "absent"."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=True))
            self.assertIsNone(checkpoint_namespace_is_dense(d, prefixes=("mtp.",)))

    def test_a_missing_checkpoint_has_no_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(checkpoint_weight_names(d))
            self.assertIsNone(checkpoint_namespace_is_dense(d))

    def test_a_single_file_checkpoint_is_read_from_its_header(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            save_file(
                {"mtp.layers.0.mlp.down_proj.weight": torch.zeros(2, 2)},
                os.path.join(d, "model.safetensors"),
            )
            self.assertEqual(
                checkpoint_weight_names(d), {"mtp.layers.0.mlp.down_proj.weight"}
            )
            self.assertTrue(checkpoint_namespace_is_dense(d, prefixes=("mtp.",)))

    def test_a_gguf_file_is_not_a_safetensors_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "drafter.gguf")
            with open(path, "wb") as f:
                f.write(b"GGUF")
            self.assertIsNone(checkpoint_weight_names(path))
            self.assertIsNone(checkpoint_namespace_is_dense(path))

    def test_a_standalone_dense_draft_directory_reads_dense(self):
        """Shape 2: the draft has its own directory, so every tensor is its own."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=False))
            self.assertTrue(_draft_checkpoint_is_dense(d))

    def test_a_standalone_packed_draft_directory_reads_packed(self):
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=True))
            self.assertFalse(_draft_checkpoint_is_dense(d))


class TestTheProducerSIgnoreListIsRespected(unittest.TestCase):
    """A quantizer that DID write the exemption keeps its current path."""

    def test_the_awq_sibling_form_is_recognised(self):
        self.assertTrue(
            _quant_cfg_excludes_draft_namespace(
                {
                    "quant_method": "awq",
                    "modules_to_not_convert": [
                        "mtp",
                        "model.language_model.layers.45.linear_attn.in_proj_a",
                    ],
                }
            )
        )

    def test_the_quark_sibling_form_is_recognised(self):
        self.assertTrue(
            _quant_cfg_excludes_draft_namespace(
                {"quant_method": "quark", "exclude_layers": ["mtp.layers.0.mlp.*"]}
            )
        )

    def test_a_list_without_the_draft_namespace_is_not_an_exemption(self):
        self.assertFalse(
            _quant_cfg_excludes_draft_namespace(
                {
                    "quant_method": "awq",
                    "modules_to_not_convert": [
                        "model.language_model.layers.45.linear_attn.in_proj_a"
                    ],
                }
            )
        )

    def test_the_gptq_checkpoint_declares_no_exemption_at_all(self):
        """The root cause, as one assertion."""
        self.assertFalse(_quant_cfg_excludes_draft_namespace(GPTQ_QUANT_CFG))


def _verify(model_path, **attrs):
    """Run `_verify_quantization` on a config whose hf side is a stub."""
    cfg = ModelConfig.__new__(ModelConfig)
    cfg.model_path = model_path
    cfg.quantization = attrs.pop("quantization", None)
    cfg.quantization_explicitly_unset = attrs.pop(
        "quantization_explicitly_unset", False
    )
    cfg.quantization_inherited = attrs.pop("quantization_inherited", False)
    cfg.is_draft_model = attrs.pop("is_draft_model", False)
    quant_cfg = attrs.pop("quant_cfg", GPTQ_QUANT_CFG)
    assert not attrs, attrs
    cfg.hf_config = types.SimpleNamespace(quantization_config=dict(quant_cfg))
    cfg._verify_quantization()
    return cfg.quantization


class TestTheDraftQuantizationDecision(unittest.TestCase):
    """What the draft model is BUILT as, given what is on disk."""

    def test_a_dense_mtp_block_builds_the_draft_dense(self):
        """THE regression. Without this the drafter runs on torch.empty."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(
                d, _target_names(packed=True) + _MTP_DENSE_NAMES, GPTQ_QUANT_CFG
            )
            self.assertIsNone(
                _verify(d, is_draft_model=True, quantization_inherited=True)
            )

    def test_a_packed_mtp_block_stays_quantized(self):
        packed_mtp = [
            n
            for name in _MTP_DENSE_NAMES
            for n in (_packed(name) if name.endswith("proj.weight") else [name])
        ]
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(
                d, _target_names(packed=True) + packed_mtp, GPTQ_QUANT_CFG
            )
            self.assertIn(
                _verify(d, is_draft_model=True, quantization_inherited=True),
                ("gptq", "gptq_marlin"),
            )

    def test_a_checkpoint_that_exempts_mtp_itself_is_untouched(self):
        """The AWQ/FP8/Quark siblings keep the path they load on today.

        Their MTP block is already built dense by the per-layer skip logic, so
        the probe must not reach a second, differently-shaped decision for
        them -- the drafter's own lm_head/embed modules would change build.
        """
        awq_cfg = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "modules_to_not_convert": ["mtp"],
        }
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=True) + _MTP_DENSE_NAMES, awq_cfg)
            self.assertIsNotNone(
                _verify(
                    d,
                    is_draft_model=True,
                    quantization_inherited=True,
                    quant_cfg=awq_cfg,
                )
            )

    def test_an_explicit_draft_method_wins_over_the_probe(self):
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(
                d, _target_names(packed=True) + _MTP_DENSE_NAMES, GPTQ_QUANT_CFG
            )
            self.assertEqual(
                _verify(
                    d,
                    is_draft_model=True,
                    quantization_inherited=False,
                    quantization="gptq",
                ),
                "gptq",
            )

    def test_the_target_model_is_never_probed(self):
        """Only the DRAFT path may be overturned; the target keeps gptq."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, _target_names(packed=False), GPTQ_QUANT_CFG)
            self.assertIsNotNone(_verify(d, is_draft_model=False))

    def test_an_explicit_opt_out_survives_the_checkpoint_config(self):
        """`unquant` on a fully PACKED checkpoint still builds dense.

        This is the escape hatch itself: before the fix `_verify_quantization`
        handed the method straight back from `quantization_config`, so the flag
        changed nothing at all.
        """
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(
                d,
                _target_names(packed=True)
                + [n for name in _MTP_DENSE_NAMES for n in _packed(name)],
                GPTQ_QUANT_CFG,
            )
            self.assertIsNone(
                _verify(
                    d,
                    is_draft_model=True,
                    quantization_explicitly_unset=True,
                )
            )

    def test_without_the_opt_out_the_same_checkpoint_quantizes(self):
        """The falsifier for the test above: same input, flag off."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(
                d,
                _target_names(packed=True)
                + [n for name in _MTP_DENSE_NAMES for n in _packed(name)],
                GPTQ_QUANT_CFG,
            )
            self.assertIsNotNone(
                _verify(d, is_draft_model=True, quantization_inherited=True)
            )


def _server_args(**kwargs):
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs.__new__(ServerArgs)
    sa.tokenizer_path = "t"
    sa.model_path = "m"
    sa.served_model_name = "m"
    sa.device = "cuda"
    sa.random_seed = 0
    sa.mm_process_config = {}
    sa.quantization = kwargs.pop("quantization", None)
    sa.speculative_draft_model_quantization = kwargs.pop(
        "speculative_draft_model_quantization", None
    )
    assert not kwargs, kwargs
    return sa


class TestTheUnquantFlagIsRealNow(unittest.TestCase):
    """`--speculative-draft-model-quantization unquant` used to be a no-op.

    ServerArgs resolved "not given" and "unquant" to the same None (6413 vs
    6424), so nothing downstream could tell them apart and ModelConfig
    re-derived the method from the checkpoint. The two records added here are
    what makes the flag decidable.
    """

    def _resolve(self, **kwargs):
        from sglang.srt.server_args import ServerArgs

        sa = _server_args(**kwargs)
        ServerArgs._handle_missing_default_values(sa)
        return sa

    def test_unquant_is_recorded_as_an_explicit_opt_out(self):
        sa = self._resolve(
            quantization="gptq", speculative_draft_model_quantization="unquant"
        )
        self.assertIsNone(sa.speculative_draft_model_quantization)
        self.assertTrue(sa._speculative_draft_model_quantization_explicitly_unset)

    def test_not_given_is_not_an_opt_out_even_though_it_resolves_the_same(self):
        """The exact confusion that made the flag a no-op."""
        sa = self._resolve(quantization=None)
        self.assertIsNone(sa.speculative_draft_model_quantization)
        self.assertFalse(sa._speculative_draft_model_quantization_explicitly_unset)
        self.assertFalse(sa._speculative_draft_model_quantization_explicitly_set)

    def test_an_inherited_method_is_marked_inherited(self):
        sa = self._resolve(quantization="gptq")
        self.assertEqual(sa.speculative_draft_model_quantization, "gptq")
        self.assertFalse(sa._speculative_draft_model_quantization_explicitly_set)

    def test_an_explicit_method_is_marked_explicit(self):
        sa = self._resolve(
            quantization="gptq", speculative_draft_model_quantization="awq"
        )
        self.assertEqual(sa.speculative_draft_model_quantization, "awq")
        self.assertTrue(sa._speculative_draft_model_quantization_explicitly_set)

    def test_the_target_flag_keeps_its_own_record(self):
        sa = self._resolve(quantization="unquant")
        self.assertIsNone(sa.quantization)
        self.assertTrue(sa._quantization_explicitly_unset)


class _Drafter(torch.nn.Module):
    """A drafter skeleton: one projection, one norm, one shared vocab head."""

    def __init__(self, quantized: bool):
        super().__init__()
        if quantized:
            self.qkv_proj = torch.nn.ParameterDict(
                {
                    key: torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
                    for key in ("qweight", "qzeros", "scales", "g_idx")
                }
            )
        else:
            self.qkv_proj = torch.nn.ParameterDict(
                {"weight": torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)}
            )
        self.norm = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.embed_tokens = torch.nn.ParameterDict(
            {"weight": torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)}
        )
        self.lm_head = torch.nn.ParameterDict(
            {"weight": torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)}
        )


class TestSkippedDraftParametersAreLoud(unittest.TestCase):
    """The seatbelt: what nothing wrote must not pass for loaded."""

    def test_a_dense_checkpoint_against_a_packed_skeleton_raises(self):
        """The #318 boot, as an error instead of an accept rate."""
        model = _Drafter(quantized=True)
        with self.assertRaises(ValueError) as ctx:
            raise_on_unloaded_draft_parameters(
                model, {"norm"}, model_path="/ckpt/gptq-int4"
            )
        message = str(ctx.exception)
        self.assertIn("unloaded", message)
        self.assertIn("qkv_proj.qweight", message)
        self.assertIn("/ckpt/gptq-int4", message)
        # It must name BOTH directions -- the packed-stream/dense-skeleton one
        # is #290 and reaches the same code path.
        self.assertIn("qweight", message)

    def test_the_matching_pair_loads_clean(self):
        model = _Drafter(quantized=False)
        raise_on_unloaded_draft_parameters(model, {"qkv_proj.weight", "norm"})

    def test_the_target_provided_vocab_modules_are_exempt(self):
        """embed_tokens / lm_head arrive from the target AFTER the load."""
        model = _Drafter(quantized=False)
        loaded = {"qkv_proj.weight", "norm"}
        self.assertNotIn("embed_tokens.weight", loaded)
        raise_on_unloaded_draft_parameters(model, loaded)

    def test_a_model_that_reports_nothing_is_left_alone(self):
        """Most draft archs return None from load_weights; they are unchanged."""
        model = _Drafter(quantized=True)
        raise_on_unloaded_draft_parameters(model, None)
        raise_on_unloaded_draft_parameters(model, set())

    def test_a_model_that_reports_nothing_says_so_out_loud(self):
        """#514/#505-A1-01: still allowed, no longer silent.

        Audit #505 measured this guard's reach and found it skipping most draft
        classes, with the skip indistinguishable from a pass. A guard that
        silently declines to run reproduces exactly the condition it exists to
        remove -- an unwritten draft parameter whose only symptom is an accept
        rate near zero. The unchecked state stays permitted; it is now logged
        once per class.
        """
        from sglang.srt.model_loader import weight_utils

        weight_utils._DRAFT_LOAD_UNCHECKED_SEEN.clear()
        model = _Drafter(quantized=True)
        with self.assertLogs(weight_utils.logger, level="WARNING") as captured:
            raise_on_unloaded_draft_parameters(model, None, model_path="/ckpt/x")
        joined = "\n".join(captured.output)
        self.assertIn("completeness NOT checked", joined)
        self.assertIn("_Drafter", joined)
        self.assertIn("accept rate", joined)

        # Once per class, not once per rank per load.
        with self.assertNoLogs(weight_utils.logger, level="WARNING"):
            raise_on_unloaded_draft_parameters(model, None, model_path="/ckpt/x")

        # The empty-set arm is the same story and must also speak.
        weight_utils._DRAFT_LOAD_UNCHECKED_SEEN.clear()
        with self.assertLogs(weight_utils.logger, level="WARNING") as captured:
            raise_on_unloaded_draft_parameters(model, set(), model_path="/ckpt/x")
        self.assertIn("EMPTY set", "\n".join(captured.output))

    def test_the_gguf_loader_checks_draft_completeness_too(self):
        """#514/#505-A1-01: GGUFModelLoader discarded load_weights' return, so
        a GGUF draft -- the fork's own #113 territory, where a packed-name
        mismatch is precisely the failure mode -- got no check at all."""
        import inspect

        from sglang.srt.model_loader.loader import GGUFModelLoader

        import ast
        import textwrap

        source = textwrap.dedent(inspect.getsource(GGUFModelLoader.load_model))
        self.assertIn("raise_on_unloaded_draft_parameters", source)

        tree = ast.parse(source)
        # Every `model.load_weights(...)` here must have its result BOUND --
        # a bare Expr statement is the discard that disabled the check.
        discarded = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "load_weights"
        ]
        self.assertEqual(
            discarded,
            [],
            "GGUFModelLoader discards load_weights' return value again "
            f"(line {discarded}); the draft completeness check cannot run "
            "without it (#505-A1-01)",
        )

    def test_the_mtp_wrappers_return_their_base_class_report(self):
        """One missing `return` in a thin MTP wrapper disables the guard for
        that whole draft class, and nothing anywhere says so. Structural pin:
        a wrapper whose load_weights only CALLS super() drops the report."""
        import ast
        import inspect

        from sglang.srt.models import qwen3_next_mtp

        tree = ast.parse(inspect.getsource(qwen3_next_mtp))
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
        ]
        self.assertTrue(handlers, "qwen3_next_mtp has no load_weights")
        for handler in handlers:
            supers = [
                node
                for node in ast.walk(handler)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load_weights"
                and isinstance(node.func.value, ast.Call)
                and getattr(node.func.value.func, "id", None) == "super"
            ]
            if not supers:
                continue
            returned = [
                node
                for node in ast.walk(handler)
                if isinstance(node, ast.Return) and node.value is not None
            ]
            self.assertTrue(
                returned,
                "qwen3_next_mtp.load_weights calls super().load_weights but "
                "returns nothing, so raise_on_unloaded_draft_parameters reads "
                "None and skips this draft class entirely (#505-A1-01)",
            )

    def test_the_escape_hatch_downgrades_the_error(self):
        from sglang.srt.environ import envs

        model = _Drafter(quantized=True)
        with envs.SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS.override(True):
            raise_on_unloaded_draft_parameters(model, {"norm"})

    def test_the_loader_only_checks_draft_models(self):
        """The target keeps loading exactly as before, reported or not."""
        import inspect

        from sglang.srt.model_loader.loader import DefaultModelLoader

        source = inspect.getsource(DefaultModelLoader.load_weights_and_postprocess)
        self.assertIn("is_draft_model", source)
        # ... and the check runs BEFORE process_weights_after_loading, so the
        # diagnostic is the unloaded parameter and not a repack kernel dying on
        # the empty tensor it left behind.
        self.assertLess(
            source.index("raise_on_unloaded_draft_parameters("),
            source.index("quant_method.process_weights_after_loading"),
        )


#: The parameters `Qwen3_5ForCausalLMMTP` builds for a Qwen3.6-27B MTP block:
#: `fc` + the two pre-FC norms, a single `Qwen3_5AttentionDecoderLayer` (which
#: holds `qkv_proj`/`o_proj`/`q_norm`/`k_norm` directly -- hence the loader's
#: `.self_attn` strip), the final norm, and the two vocab modules the target
#: hands over afterwards.
_MTP_SKELETON_STEMS = [
    "fc",
    "pre_fc_norm_embedding",
    "pre_fc_norm_hidden",
    "model.norm",
    "model.layers.0.input_layernorm",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.q_norm",
    "model.layers.0.k_norm",
]
_MTP_SKELETON_PROJECTIONS = [
    "model.layers.0.qkv_proj",
    "model.layers.0.o_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
]
_MTP_TARGET_PROVIDED = ["model.embed_tokens.weight", "lm_head.weight"]


class _StubParam:
    def __init__(self, name, record):
        self.name = name
        self._record = record

    def weight_loader(self, *args, **kwargs):
        self._record.add(self.name)


class _StubDrafter:
    """Enough of a model for the real `load_weights` to run against.

    `Qwen3_5ForCausalLMMTP.load_weights` touches exactly `self.config` (for
    `num_experts`) and `self.named_parameters()`, so the actual mapping code --
    the `mtp.` -> `model.` rewrite, the `.self_attn` strip, the stacked
    q/k/v and gate/up fusion -- can be exercised without a CUDA context, a
    process group or 16 GiB of weights.
    """

    def __init__(self, quantized: bool):
        self.config = types.SimpleNamespace()
        self.written = set()
        names = list(_MTP_TARGET_PROVIDED)
        names += [f"{stem}.weight" for stem in _MTP_SKELETON_STEMS]
        for stem in _MTP_SKELETON_PROJECTIONS:
            if quantized:
                names += [
                    f"{stem}.{suffix}"
                    for suffix in ("qweight", "qzeros", "scales", "g_idx")
                ]
            else:
                names.append(f"{stem}.weight")
        self._params = {name: _StubParam(name, self.written) for name in names}

    def named_parameters(self):
        return list(self._params.items())


def _mtp_weight_stream():
    return [(name, torch.zeros(1)) for name in _MTP_DENSE_NAMES]


class TestTheRealNameMappingAgainstBothSkeletons(unittest.TestCase):
    """The load itself, replayed: which parameters actually get written."""

    @staticmethod
    def _replay(quantized: bool):
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

        drafter = _StubDrafter(quantized=quantized)
        loaded = Qwen3_5ForCausalLMMTP.load_weights(drafter, _mtp_weight_stream())
        return drafter, loaded

    def test_the_fixture_matches_the_real_checkpoint(self):
        """Guard the fixture: if the index changes, this test says so."""
        path = (
            "/spinning/llm_stuff/club-3090/models-cache/"
            "Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4"
        )
        if not os.path.isdir(path):
            self.skipTest(f"checkpoint absent: {path}")
        names = checkpoint_weight_names(path)
        self.assertEqual(
            sorted(n for n in names if n.startswith("mtp.")), sorted(_MTP_DENSE_NAMES)
        )

    def test_the_dense_skeleton_takes_every_parameter(self):
        """The fix, measured: 0 skipped parameters."""
        drafter, _ = self._replay(quantized=False)
        expected = {f"{stem}.weight" for stem in _MTP_SKELETON_STEMS}
        expected |= {f"{stem}.weight" for stem in _MTP_SKELETON_PROJECTIONS}
        self.assertEqual(drafter.written, expected)

    def test_the_packed_skeleton_takes_nothing_but_the_norms(self):
        """The bug, measured: the four projections never get written.

        Sixteen parameters (4 modules x qweight/qzeros/scales/g_idx) stay at
        their allocation value, which is what accept 1.0052 was made of.
        """
        drafter, _ = self._replay(quantized=True)
        self.assertEqual(
            drafter.written, {f"{stem}.weight" for stem in _MTP_SKELETON_STEMS}
        )

    def test_the_guard_is_silent_on_the_dense_skeleton(self):
        drafter, loaded = self._replay(quantized=False)
        raise_on_unloaded_draft_parameters(drafter, loaded, model_path="/ckpt")

    def test_the_guard_names_the_packed_skeletons_gap(self):
        """The regression against the 1.005 signature."""
        drafter, loaded = self._replay(quantized=True)
        with self.assertRaises(ValueError) as ctx:
            raise_on_unloaded_draft_parameters(drafter, loaded, model_path="/ckpt")
        message = str(ctx.exception)
        self.assertIn("16 parameter(s)", message)
        self.assertIn("down_proj.qweight", message)
        self.assertIn("_StubDrafter", message)


if __name__ == "__main__":
    unittest.main()
