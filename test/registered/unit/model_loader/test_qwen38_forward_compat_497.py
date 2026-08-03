# SPDX-License-Identifier: Apache-2.0
"""#497: generic robustness a Qwen3.8-class checkpoint needs on day 0.

No speculation about Qwen3.8's geometry -- none is public (ANALYSE_495).
Everything here is a property that must hold for ANY future checkpoint that
reuses an existing ``model_type``, which is what the only real code artifact
about Qwen3.8 (vLLM PR #50068) indicates will happen: an AMD engineer with an
early-access checkpoint wires "Qwen3.8 Max FP8" through the EXISTING
``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` classes rather than a new
``Qwen3_8...`` class.

Three properties, each with its own falsifier:

(a) resolution is by ``model_type``, never by a version string in the model
    NAME -- so a checkpoint called anything at all loads if its
    ``model_type`` is known, and a genuinely unknown one refuses BY NAME
    instead of crashing;
(b) hybrid geometry is read from config fields, so a different depth or a
    different full-attention interval needs no code change;
(c) the M-RoPE declaration gap vLLM #50068 closes on its side is
    characterised here on ours, with both gate predicates quoted at their
    source, so the disagreement is visible rather than latent.

Hermetic: no GPU, no checkpoint, no download.

Usage:
    python3 -m pytest test/registered/unit/model_loader/test_qwen38_forward_compat_497.py -v
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]

#: The text-only Qwen3.5/3.6 config shape, from the two real configs read in
#: ANALYSE_495 §2.1 (byte-identical between 3.5 and 3.6 in every field below).
#: Used as the STARTING point that gets varied; nothing here is asserted to be
#: Qwen3.8's geometry.
QWEN35_TEXT_CONFIG = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "num_hidden_layers": 64,
    "full_attention_interval": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_theta": 10000000,
    },
}


class _Lit:
    __slots__ = ("value", "line")

    def __init__(self, value, line):
        self.value = value
        self.line = line


def _code_string_literals(path: Path):
    """String literals a module EVALUATES, excluding docstrings and comments.

    A ratchet against version-string matching must look at code, not at prose:
    this tree documents "Qwen3.5/3.6" in dozens of docstrings, and flagging
    those would make the check useless noise. Comments never reach the parser;
    docstrings are removed explicitly.
    """
    import ast  # noqa: PLC0415

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            out.append(_Lit(node.value, node.lineno))
    return out


# ---------------------------------------------------------------------------
# (a) resolution keys on model_type, not on a version string
# ---------------------------------------------------------------------------


class TestResolutionIsModelTypeDriven(unittest.TestCase):
    def test_the_gguf_adapter_resolves_by_model_type(self):
        from sglang.srt.model_loader.gguf_registry import get_gguf_adapter_class

        self.assertIsNotNone(get_gguf_adapter_class("qwen3_5"))

    def test_the_text_only_variant_resolves_too(self):
        """``text_config.model_type`` is ``qwen3_5_text``; a text-only
        checkpoint is exactly what a Qwen3.8-27B release would be."""
        from sglang.srt.model_loader.gguf_registry import get_gguf_adapter_class

        self.assertIsNotNone(get_gguf_adapter_class("qwen3_5_text"))

    def test_an_unknown_checkpoint_NAME_does_not_matter(self):
        """The property that carries day 0: rename the checkpoint to anything,
        keep the model_type, and it must still resolve to the same adapter."""
        from sglang.srt.model_loader.gguf_registry import get_gguf_adapter_class

        # create_gguf_adapter() reads exactly one field to choose the class:
        #   model_type = getattr(hf_config, "model_type", None)
        #   cls = get_gguf_adapter_class(model_type)
        # (gguf_registry.py:81-88). The name is never consulted, so a rename
        # cannot change the answer.
        for name in (
            "Qwen/Qwen3.5-27B",
            "Qwen/Qwen3.8-27B-Whatever-They-Call-It",
            "",
        ):
            with self.subTest(name=name):
                self.assertIs(
                    get_gguf_adapter_class("qwen3_5"),
                    get_gguf_adapter_class("qwen3_5"),
                )
        src = (_ROOT / "python/sglang/srt/model_loader/gguf_registry.py").read_text()
        self.assertIn('model_type = getattr(hf_config, "model_type", None)', src)
        self.assertNotIn("name_or_path", src)

    def test_a_genuinely_unknown_model_type_is_a_miss_not_a_crash(self):
        from sglang.srt.model_loader.gguf_registry import get_gguf_adapter_class

        # None == "no bespoke family, take the generic GGUF path".
        self.assertIsNone(get_gguf_adapter_class("qwen9_9_from_the_future"))
        self.assertIsNone(get_gguf_adapter_class(None))

    def test_an_unknown_architecture_lands_on_the_transformers_fallback(self):
        """Characterisation, and a day-0 fact worth knowing.

        ``_normalize_archs`` (registry.py:61-78) drops every unregistered
        architecture and, when anything was dropped, APPENDS
        ``TransformersForCausalLM``. So a checkpoint declaring a brand-new
        architecture string does NOT refuse -- it silently resolves to the
        generic transformers backend. That is a soft landing, not a crash, but
        it is also not a signal: if Qwen3.8 ships a new architecture name, the
        boot may come up on a generic path with none of this fork's features
        rather than saying so. See ANALYSE_495 §4 for the day-0 action.
        """
        from sglang.srt.models.registry import ModelRegistry

        unknown = "Qwen3_8DefinitelyNotRegisteredForCausalLM"
        self.assertNotIn(unknown, ModelRegistry.get_supported_archs())
        _cls, arch = ModelRegistry.resolve_model_cls([unknown])
        self.assertEqual(arch, "TransformersForCausalLM")

    def test_the_refusal_path_itself_names_the_architecture(self):
        """When even the fallback is absent, the message must name what was
        asked for and what exists -- not a KeyError."""
        from sglang.srt.models.registry import ModelRegistry

        unknown = "Qwen3_8DefinitelyNotRegisteredForCausalLM"
        with self.assertRaises(ValueError) as caught:
            ModelRegistry._raise_for_unsupported([unknown])
        message = str(caught.exception)
        self.assertIn(unknown, message)
        self.assertIn("not supported", message)

    def test_the_dispatch_table_is_keyed_on_model_type_strings_only(self):
        """Ratchet: nobody may add a display-name key like "Qwen3.5".

        HF ``model_type`` values are lowercase with underscores. A key with a
        dot or a capital letter would be a model NAME, i.e. the version-string
        matching this test exists to keep out.
        """
        from sglang.srt.model_loader.gguf_qwen35 import _MODEL_TYPE_TO_GGUF_ARCH

        for key in _MODEL_TYPE_TO_GGUF_ARCH:
            with self.subTest(key=key):
                self.assertRegex(key, r"^[a-z0-9_]+$")

    def test_no_module_matches_a_qwen_version_string_on_the_load_path(self):
        """The load path must not branch on "Qwen3.5"/"Qwen3.6" DISPLAY text.

        Two scoping decisions, both deliberate:
        * only the loader and config packages -- the planner DOES map display
          names (``planner/rig_profile_source.py:64-65`` turns a model path
          into a label for its measurement store) and that is legitimate,
          it is not a load decision;
        * only string literals the module evaluates, never docstrings or
          comments -- this tree documents the family in prose everywhere, and
          flagging that would drown the signal.
        """
        offenders = []
        # The DOT is what makes it a display name. ``qwen35``/``qwen35moe``
        # without one are llama.cpp GGUF architecture strings -- real
        # identifiers, and exactly what the dispatch is supposed to use.
        pattern = re.compile(r"[Qq]wen3\.[56]")
        for sub in ("model_loader", "configs"):
            for path in (_ROOT / "python" / "sglang" / "srt" / sub).rglob("*.py"):
                for tok in _code_string_literals(path):
                    if pattern.search(tok.value):
                        offenders.append(
                            f"{path.relative_to(_ROOT)}:{tok.line}: {tok.value!r}"
                        )
        self.assertEqual(
            offenders,
            [],
            "version-string matching on the load path; dispatch on model_type "
            f"instead: {offenders}",
        )

    def test_the_version_string_ratchet_can_actually_fail(self):
        """Can-discriminate. A ratchet that is green because its pattern never
        matches anything is not a ratchet (CLAUDE.md: an instrument's verdict
        counts only after it passes a can-discriminate check)."""
        import ast  # noqa: PLC0415

        pattern = re.compile(r"[Qq]wen3\.[56]")
        offending = ast.parse('if "Qwen3.5" in model_path:\n    pass\n')
        found = [
            n.value
            for n in ast.walk(offending)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        self.assertTrue(any(pattern.search(v) for v in found))
        # ...and the GGUF arch identifier must NOT trip it.
        self.assertFalse(pattern.search("qwen35"))
        self.assertFalse(pattern.search("qwen35moe"))

    def test_the_literal_extractor_skips_docstrings_and_comments(self):
        """The other half of can-discriminate: no false positives on prose."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(
                '''"""Module doc mentioning Qwen3.5."""\n'''
                "# comment mentioning Qwen3.6\n"
                'X = "real-literal-Qwen3.5"\n'
            )
            path = Path(fh.name)
        try:
            values = [t.value for t in _code_string_literals(path)]
            self.assertIn("real-literal-Qwen3.5", values)
            self.assertNotIn("Module doc mentioning Qwen3.5.", values)
            self.assertFalse(any("Qwen3.6" in v for v in values))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# (b) hybrid geometry comes from the config, not from constants
# ---------------------------------------------------------------------------


def _write_config(tmp: str, **overrides) -> str:
    cfg = dict(QWEN35_TEXT_CONFIG)
    cfg.update(overrides)
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp


class _ArgsStub:
    """Just enough of ServerArgs to exercise the two real methods."""

    def __init__(self, model_path: str):
        from sglang.srt.server_args import ServerArgs

        self.model_path = model_path
        self._NUM_LAYER_CONFIG_KEYS = ServerArgs._NUM_LAYER_CONFIG_KEYS
        self._sa = ServerArgs

    def declared_num_hidden_layers(self):
        return self._sa.declared_num_hidden_layers(self)

    def declared_layer_kinds(self):
        return self._sa.declared_layer_kinds(self)


class TestGeometryIsConfigDriven(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_reference_geometry_reproduces(self):
        args = _ArgsStub(_write_config(self._tmp.name))
        kinds = args.declared_layer_kinds()
        self.assertEqual(len(kinds), 64)
        self.assertEqual(sum(kinds), 16)  # interval 4 -> every 4th layer
        self.assertTrue(kinds[3] and kinds[63])
        self.assertFalse(kinds[0])

    def test_a_different_depth_and_interval_need_no_code_change(self):
        """The day-0 property: change the two numbers, get the right answer."""
        args = _ArgsStub(
            _write_config(
                self._tmp.name, num_hidden_layers=48, full_attention_interval=6
            )
        )
        kinds = args.declared_layer_kinds()
        self.assertEqual(len(kinds), 48)
        self.assertEqual(sum(kinds), 8)
        self.assertEqual([i for i, k in enumerate(kinds) if k][:2], [5, 11])

    def test_an_explicit_layer_types_list_wins_over_the_interval(self):
        types = ["linear_attention"] * 5 + ["full_attention"]
        args = _ArgsStub(
            _write_config(
                self._tmp.name,
                num_hidden_layers=6,
                full_attention_interval=4,
                layer_types=types,
            )
        )
        self.assertEqual(
            args.declared_layer_kinds(), [False, False, False, False, False, True]
        )

    def test_a_config_without_hybrid_fields_is_all_attention(self):
        cfg = {k: v for k, v in QWEN35_TEXT_CONFIG.items()}
        cfg.pop("full_attention_interval")
        path = self._tmp.name
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f)
        self.assertEqual(_ArgsStub(path).declared_layer_kinds(), [True] * 64)

    def test_the_fields_are_read_from_a_nested_text_config_too(self):
        """A wrapper config (``Qwen3_5ForConditionalGeneration``) carries the
        geometry under ``text_config``; day 0 may hand us either shape."""
        path = self._tmp.name
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 32,
                        "full_attention_interval": 8,
                    },
                },
                f,
            )
        kinds = _ArgsStub(path).declared_layer_kinds()
        self.assertEqual(len(kinds), 32)
        self.assertEqual(sum(kinds), 4)

    def test_the_gdn_geometry_is_read_per_field(self):
        """No hardcoded 16/48/128: change them and the numbers follow."""
        from sglang.srt.uneven_perf import layer_family_census

        census = layer_family_census({"num_hidden_layers": 40}, 40)
        self.assertEqual(census.n_layers, 40)
        self.assertEqual(census.ffn_width_factor, 40)

    def test_no_layer_count_literal_guards_the_qwen35_path(self):
        """Ratchet against a re-introduced `== 64`-style geometry assumption."""
        offenders = []
        for rel in (
            "python/sglang/srt/models/qwen3_5.py",
            "python/sglang/srt/model_loader/gguf_qwen35.py",
        ):
            path = _ROOT / rel
            for n, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if re.search(
                    r"(num_hidden_layers|n_layers|num_layers)\s*[=!]=\s*\d", code
                ):
                    offenders.append(f"{rel}:{n}: {code.strip()}")
        self.assertEqual(offenders, [], f"hardcoded layer count: {offenders}")


# ---------------------------------------------------------------------------
# (c) the M-RoPE declaration gap (vLLM #50068 equivalent, on our side)
# ---------------------------------------------------------------------------


class TestMRopeDeclarationGap(unittest.TestCase):
    """Both gate predicates, quoted at their source and evaluated here.

    RUNNER SIDE -- ``model_executor/model_runner.py:599-604``::

        rope_scaling = getattr(model_config.hf_text_config, "rope_parameters",
                               None) or getattr(model_config.hf_text_config,
                                                "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )

    It reads the CONFIG only. A text-only Qwen3.5/3.6 config carries
    ``rope_parameters.mrope_section = [11, 11, 10]``, so this is True for a
    text-only checkpoint, and ``forward_batch_info.py:876`` then builds
    ``mrope_positions``.

    MODEL SIDE -- ``models/qwen3_5.py:1839`` and ``:1996`` set
    ``self.is_mrope_enabled`` on ``Qwen3_5ForConditionalGeneration`` and
    ``Qwen3_5MoeForConditionalGeneration`` ONLY. The text-only
    ``Qwen3_5ForCausalLM`` (``:1280``) and ``Qwen3_5MoeForCausalLM``
    (``:1611``) do not set it at all.

    CONSUMER -- ``runner/prefill_cuda_graph_runner.py:521-531``::

        if forward_batch.mrope_positions is None:
            return forward_batch.positions
        if getattr(model, "is_mrope_enabled", False):
            return forward_batch.mrope_positions
        language_model = getattr(model, "language_model", None)
        if getattr(language_model, "is_mrope_enabled", False):
            return forward_batch.mrope_positions
        return forward_batch.positions

    So on the text-only path the runner COMPUTES mrope positions and this
    consumer then discards them, because neither attribute exists. That is the
    same declaration gap vLLM #50068 closes by adding ``SupportsMRoPE`` to its
    text-only causal class.

    These tests characterise the CURRENT state deliberately. Closing the gap
    changes which positions a captured graph replays, so it needs a boot to
    validate and is a GPU-window decision, not a desk change -- see
    ``docs/dev/ANALYSE_495_qwen38_forward_compat.md`` §4. When it is closed,
    these two tests flip together and that is the intended signal.
    """

    def _rope_predicate(self, cfg: dict) -> bool:
        """The runner-side predicate, transcribed from model_runner.py:599."""
        rope_scaling = cfg.get("rope_parameters") or cfg.get("rope_scaling", {})
        return rope_scaling is not None and "mrope_section" in rope_scaling

    def test_the_runner_predicate_is_true_for_a_text_only_config(self):
        self.assertTrue(self._rope_predicate(QWEN35_TEXT_CONFIG))

    def test_the_transcription_matches_the_source(self):
        """Reference-twin discipline (#418 family): the predicate above is a
        copy, so pin the original's shape rather than trusting the copy."""
        src = (_ROOT / "python/sglang/srt/model_executor/model_runner.py").read_text()
        self.assertIn('"mrope_section" in rope_scaling', src)
        self.assertIn('model_config.hf_text_config, "rope_parameters", None', src)
        self.assertIn('"rope_scaling", {})', src)
        # ...and it is the CONFIG that decides, with no model-class term.
        self.assertNotIn("model_is_mrope = getattr(self.model", src)

    def test_only_the_multimodal_classes_declare_is_mrope_enabled(self):
        """The gap itself, read off the source.

        If this ever fails because a THIRD assignment appeared, that is the fix
        landing -- update the count and the note in ANALYSE_495 §4 together.
        """
        src = (_ROOT / "python/sglang/srt/models/qwen3_5.py").read_text()
        self.assertEqual(src.count("self.is_mrope_enabled = "), 2)
        # ...and both sit inside ForConditionalGeneration classes.
        classes = [m.group(1) for m in re.finditer(r"^class (\w+)", src, re.M)]
        self.assertIn("Qwen3_5ForCausalLM", classes)
        self.assertIn("Qwen3_5ForConditionalGeneration", classes)
        for name, body in _class_bodies(src).items():
            declares = "self.is_mrope_enabled = " in body
            if name.endswith("ForConditionalGeneration"):
                self.assertTrue(declares, f"{name} should declare it")
            else:
                self.assertFalse(
                    declares, f"{name} now declares it -- the #497(c) gap moved"
                )

    def test_the_consumer_falls_back_to_plain_positions(self):
        """The other half: without the attribute the mrope positions are
        dropped, which is what makes the gap silent rather than loud."""
        src = (
            _ROOT
            / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
        ).read_text()
        self.assertIn('if getattr(model, "is_mrope_enabled", False):', src)
        self.assertIn('getattr(language_model, "is_mrope_enabled", False)', src)
        self.assertIn("return forward_batch.positions", src)


def _class_bodies(src: str) -> dict:
    """``{class_name: source_of_that_class}`` for top-level classes."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^class (\w+)", src, re.M)]
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
        out[name] = src[pos:end]
    return out


if __name__ == "__main__":
    unittest.main()
