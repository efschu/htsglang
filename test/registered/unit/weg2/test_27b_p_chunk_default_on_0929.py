"""27B default on: --p-chunk-policy dynamic on the INT8 checkpoint (registry
qwen27b chunk.policy on chunk.default_formats; user rule 29.09. ~10:15Z).
The NF dynpf finding (D found only part of the prefix after P->D at bs4x16k)
is rooted NF-specific (QSA index sidecar missing in arena/L3; the 27B has no
QSA sidecar) -- operator 29.09.

Beleg: chunkab rc9j, A fixed dkr27bbar1chunka09260010 vs B dynamic
dkr27bint8chunkBbar109260034: 128k 36.59 -> 31.62 s (-13.6 %), 32k 4.68 ->
4.35 s, 2k/8k equal, needle 3/3 per step.
"""

import pytest

from sglang.srt.weg2 import form as FM

_MC = "/spinning/llm_stuff/club-3090/models-cache/"
INT8 = _MC + "Qwen3.8-27B-INT8-gdncov-vocabembed"
FP8 = _MC + "Qwen3.8-27B-FP8"
NVFP4 = _MC + "Qwen3.8-27B-NVFP4-RadixArk"
GGUF = _MC + "Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf"
NF_INT4 = _MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


def _defaults(profile, argv=(), model=None):
    from sglang.srt.weg2 import launcher as L

    words = ["--tree", "/t", "--tag", "x"]
    if profile:
        words += ["--profile", profile]
    if model is not None:
        words += ["--model", model]
    words += list(argv)
    ns = L.build_parser().parse_args(words)
    L.apply_profile_arg_defaults(ns, words)
    return ns


def test_parser_default_is_the_code_default_fixed():
    from sglang.srt.weg2 import launcher as L

    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
    assert ns.p_chunk_policy == L.P_CHUNK_POLICY_DEFAULT == "fixed"


def test_rows():
    assert FM.PROFILES["qwen27b"].chunk.policy == "dynamic"
    assert FM.PROFILES["qwen27b"].chunk.default_formats == ("int8",)
    assert FM.PROFILES["nextflash"].chunk.policy == "fixed"          # NF unchanged
    assert FM.PROFILES["nextflash"].chunk.default_formats == ()


G512 = ("--p-prefill-graph", "512")   # the measured form's P prefill graph (27b.env)


@pytest.mark.parametrize("profile,model,argv,want", [
    ("qwen27b", INT8, G512, "dynamic"),                               # 27B INT8: default on
    (None, None, G512, "dynamic"),                                    # launcher defaults = 27B INT8
    ("qwen27b", INT8, G512 + ("--p-chunk-policy", "fixed"), "fixed"),  # explicit wins
    ("qwen27b", INT8, G512 + ("--p-chunk-policy=fixed",), "fixed"),
    ("qwen27b", FP8, G512, "fixed"), ("qwen27b", GGUF, G512, "fixed"),  # unmeasured formats
    ("qwen27b", NVFP4, G512, "fixed"),                                # states its own flags
    ("nextflash", NF_INT4, G512, "fixed"),                            # NF unchanged
    ("nextflash", NF_INT4, ("--p-chunk-policy", "dynamic"), "dynamic"),  # NF H92 arm: explicit
])
def test_chunk_policy_cli_default_follows_the_row(profile, model, argv, want):
    ns = _defaults(profile, argv, model=model)
    assert ns.p_chunk_policy == want


@pytest.mark.parametrize("argv,want", [
    ((), "fixed"),                                   # no graph: baseline 4096 > max 2048
    (("--p-prefill-graph", "4096"), "fixed"),        # bucket above max 2048
    (("--p-prefill-graph", "2048"), "dynamic"),
    (("--p-chunk-fixed", "1024"), "dynamic"),        # a baseline under the max, no graph
    (("--p-chunk-max", "4096"), "dynamic"),          # a max that takes the 4096 baseline
])
def test_dynamic_default_only_where_the_launcher_can_run_it(argv, want):
    """The launcher refuses dynamic when its baseline width passes --p-chunk-max
    (apply_p_chunk_policy); the registry default never produces that refusal."""
    assert _defaults("qwen27b", argv, model=INT8).p_chunk_policy == want


def test_the_default_dynamic_form_passes_apply_p_chunk_policy():
    from sglang.srt.weg2 import launcher as L

    ns = _defaults("qwen27b", G512, model=INT8)
    L.apply_p_prefill_graph(ns)
    L.apply_p_chunk_policy(ns, None, [])
    assert L._P_CHUNK["policy"] == "dynamic"


def test_dynamic_takes_the_int8_measurement_form():
    """The profile's dynamic form was --p-chunk-max 2048 --p-chunk-model
    builtin-int8 --p-chunk-mscale int8: the parser defaults ARE that form."""
    ns = _defaults("qwen27b", G512, model=INT8)
    assert (ns.p_chunk_policy, ns.p_chunk_max, ns.p_chunk_model, ns.p_chunk_mscale) == (
        "dynamic", 2048, "builtin-int8", "int8")


@pytest.mark.parametrize("profile,model,want", [
    ("qwen27b", INT8, "int8"), ("qwen27b", INT8 + "/", "int8"),
    ("qwen27b", "/elsewhere/Qwen3.8-27B-INT8-gdncov-vocabembed", "int8"),
    ("qwen27b", FP8, "fp8"), ("qwen27b", NVFP4, "nvfp4"), ("qwen27b", GGUF, "gguf"),
    ("qwen27b", _MC + "Qwen3.8-27B-INT8-abl-wxp", ""),   # a derivative: unknown -> code default
    ("nextflash", NF_INT4, "int4-mixed"), ("nextflash", INT8, ""), (None, INT8, ""),
])
def test_format_of_is_the_registry_format_by_calibration_identity(profile, model, want):
    assert FM.format_of(profile, model) == want
