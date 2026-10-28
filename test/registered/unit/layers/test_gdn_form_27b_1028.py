"""27b-gdn-form-1028: the 27B's GDN layers as the REAL class builds them, read by ``gdn_form`` (census line 27B-GDN-FORM).

NF's GDN fusions (FORMHP-GDN-IN b166989d8, FORMHP-GDN-OUT 67c1f8242) failed their boot self-test on the card (formb169 21:46:32 TP0:
"module is not the fusable pair (lane / TP / bias / norm form)") because the detection required tp_size == 1 and Form B builds the GDN
layers with tp_size 3 and the split [all, 0, 0]; their unit tests used a SimpleNamespace with ``tp_size = 1`` set by hand. 14a71cc5e
replaced the rule by "the projection's forward IS quant_method.apply on this rank". These tests build ``Qwen3_5GatedDeltaNet`` itself --
the real class, the real 27B text config and the real INT8 checkpoint quant config (copied verbatim from
Qwen3.8-27B-INT8-gdncov-vocabembed/config.json), the shard plan of the production D boots (--rank-tp-ratio 58,25,25, TP 3), every
rank -- plus a P stage (TP 1) and the FP8 checkpoint's block-128 method, and pin:

  (1) the form each rank builds (heads, K / N, reduce / gather, norm variant, quant method per projection);
  (2) NF's structural rule passes on every 27B rank, the old ``tp_size == 1`` rule would refuse all three D ranks, and the only
      refusal left is the lane (the 27B's projections are not the formhp e4m3 W8A16 GEMV NF's kernels embed);
  (3) the census line and its switch, and that ``Qwen3_5ForCausalLM.__init__`` logs it right after make_layers;
  (4) the refusals name their reason (all-reducing out_proj, gathering in_proj, a rank without heads).
CPU only: the card capability is pinned (rank 0 = sm_120 5090, ranks 1/2 = sm_86 3080), nothing is launched.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

import ast
import os
import subprocess
import sys
import textwrap

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))

_MK = textwrap.dedent(
    """
    import sys, types, torch
    RANK, TP, RATIOS, QUANT = int(sys.argv[1]), int(sys.argv[2]), eval(sys.argv[3]), sys.argv[4]
    torch.cuda.get_device_capability = lambda *a, **k: (12, 0) if RANK == 0 else (8, 6)
    torch.cuda.current_device = lambda: "cpu"
    import flliper.srt.models.qwen3_5 as q
    from flliper.srt.configs.qwen3_5 import Qwen3_5TextConfig
    from flliper.srt.distributed.utils import set_tp_partition_ratios
    from flliper.srt.layers import gdn_form as gf
    TEXT = dict(hidden_size=5120, num_hidden_layers=4, full_attention_interval=4, linear_num_key_heads=16, linear_num_value_heads=48,
                linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, output_gate_type="swish",
                hidden_act="silu", rms_norm_eps=1e-06, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                intermediate_size=17408, vocab_size=248320, attn_output_gate=True, partial_rotary_factor=0.25)
    INT8 = {"config_groups": {"INT8": {"format": "int-quantized", "input_activations": {"actorder": None, "block_structure": None,
            "dynamic": True, "group_size": None, "num_bits": 8, "observer": None, "observer_kwargs": {}, "scale_dtype": None,
            "strategy": "token", "symmetric": True, "type": "int", "zp_dtype": None}, "output_activations": None, "targets": ["Linear"],
            "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": None, "num_bits": 8,
            "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None, "strategy": "channel", "symmetric": True,
            "type": "int", "zp_dtype": None}}}, "format": "int-quantized", "global_compression_ratio": None,
            "ignore": ["re:.*(vision|visual).*", "lm_head", "re:.*norm.*", "re:.*conv1d.*", "re:.*linear_attn\\\\.in_proj_a.*",
                       "re:.*linear_attn\\\\.in_proj_b.*"], "kv_cache_scheme": None, "quant_method": "compressed-tensors",
            "quantization_status": "compressed", "transform_config": {}, "version": "0.18.1.a20260806"}
    def quant_config():
        if QUANT == "int8":
            from flliper.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
            return CompressedTensorsConfig.from_config(INT8)
        if QUANT == "fp8":
            from flliper.srt.layers.quantization.fp8 import Fp8Config
            # the FP8 checkpoint keeps in_proj_a / in_proj_b BF16 (D boot 00:53:08: ba=UnquantizedLinearMethod; block 128 cannot cut N 2 x 12)
            return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128],
                             ignored_layers=["model.language_model.layers.1.linear_attn.in_proj_b",
                                             "model.language_model.layers.1.linear_attn.in_proj_a"],
                             packed_modules_mapping=dict(q.Qwen3_5ForCausalLM.packed_modules_mapping))
        return None
    cfg = Qwen3_5TextConfig(**TEXT)
    cfg.torch_dtype = torch.bfloat16
    import flliper.srt.layers.quantization.fp8 as fp8_mod
    q.get_parallel = fp8_mod.get_parallel = lambda: types.SimpleNamespace(attn_tp_rank=RANK, attn_tp_size=TP, attn_dcp_size=1,
                                                                          tp_size=TP, tp_rank=RANK)
    set_tp_partition_ratios(RATIOS)
    torch.set_default_dtype(torch.bfloat16)
    def build(layer_id=1):
        return q.Qwen3_5GatedDeltaNet(cfg, layer_id=layer_id, quant_config=quant_config(),
                                      prefix=f"model.language_model.layers.{layer_id}.linear_attn")
    """
)


def _run(code: str, *, rank: int, tp: int = 3, ratios=(58, 25, 25), quant: str = "int8"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    argv = [sys.executable, "-c", _MK + textwrap.dedent(code), str(rank), str(tp), repr(None if ratios is None else list(ratios)), quant]
    p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600)
    out = [line for line in p.stdout.splitlines() if line.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-4000:]
    return ast.literal_eval(out[-1][4:])


_FORM = """
    f = gf.read_form(build())
    print("RES", [f._asdict(), gf.gdn_out_why(f), gf.gdn_in_why(f)])
"""

_CT = "CompressedTensorsLinearMethod/CompressedTensorsW8A8Int8"


def test_int8_d_ranks_build_the_expected_form():
    # --rank-tp-ratio 58,25,25 over 16 k-head units: 8 / 4 / 4 k heads, 24 / 12 / 12 v heads
    for rank, hk in ((0, 8), (1, 4), (2, 4)):
        f, why_out, why_in = _run(_FORM, rank=rank)
        hv = 3 * hk
        assert (f["tp"], f["tp_rank"], f["k_heads"], f["v_heads"], f["k_dim"], f["v_dim"]) == (3, rank, hk, hv, 128, 128), f
        assert (f["out_k"], f["qkvz_n"]) == (hv * 128, 2 * hk * 128 + 2 * hv * 128), f
        assert (f["out_in_parallel"], f["out_reduce"], f["out_bias"], f["qkvz_gather"], f["qkvz_bias"]) == (True, False, False, False, False), f
        assert (f["qkvz"], f["out"], f["ba"]) == (_CT, _CT, "UnquantizedLinearMethod"), f
        assert (f["norm"], f["norm_act"], f["norm_before_gate"], f["norm_group"], f["norm_bias"], f["eps"]) == (
            "RMSNorm", "swish", True, 0, False, 1e-06), f
        # NF's structural rule passes; the only refusal is the lane
        assert why_out == f"lane {_CT}" and why_in == f"lane {_CT}", (why_out, why_in)


def test_the_old_tp_size_rule_would_refuse_every_27b_d_rank():
    # the first NF detection (67c1f8242 / b166989d8 _module_args): int(layer.tp_size) != 1 -> refuse. The real 27B D ranks carry tp 3.
    for rank in (0, 1, 2):
        f, _o, _i = _run(_FORM, rank=rank)
        assert f["tp"] != 1, f


def test_p_stage_builds_all_heads_at_tp_1():
    f, why_out, why_in = _run(_FORM, rank=0, tp=1, ratios=None)
    assert (f["tp"], f["k_heads"], f["v_heads"], f["out_k"], f["out_reduce"]) == (1, 16, 48, 6144, False), f
    assert why_out == f"lane {_CT}" and why_in == f"lane {_CT}", (why_out, why_in)


def test_fp8_block_checkpoint_form():
    f, why_out, why_in = _run(_FORM, rank=1, quant="fp8")
    assert (f["qkvz"], f["out"], f["ba"]) == ("Fp8LinearMethod", "Fp8LinearMethod", "UnquantizedLinearMethod"), f
    assert why_out == "lane Fp8LinearMethod" and why_in == "lane Fp8LinearMethod", (why_out, why_in)


def test_refusals_name_their_reason():
    r = _run(
        """
        m = build()
        out = []
        m.out_proj.reduce_results = True
        out.append(gf.gdn_out_why(gf.read_form(m)))
        m.out_proj.reduce_results = False
        m.in_proj_qkvz.gather_output = True
        out.append(gf.gdn_in_why(gf.read_form(m)))
        m.in_proj_qkvz.gather_output = False
        m.local_num_v_heads = 0
        f = gf.read_form(m)
        out += [gf.gdn_out_why(f), gf.gdn_in_why(f)]
        print("RES", out)
        """,
        rank=2,
    )
    assert r == ["out_proj all-reduces (tp 3)", "in_proj_qkvz all-gathers (tp 3)", "no v heads on this rank", "no heads on this rank"], r


def test_census_line_and_switch():
    r = _run(
        """
        import os
        layers = torch.nn.ModuleList([build(0), build(1), build(2), torch.nn.Identity()])
        line = gf.log_census(layers=layers, gdn_cls=q.Qwen3_5GatedDeltaNet, pp_rank=0, pp_size=1)
        os.environ["FLLIPER_GDN_FORM_CENSUS"] = "0"
        off = gf.log_census(layers=layers, gdn_cls=q.Qwen3_5GatedDeltaNet, pp_rank=0, pp_size=1)
        none = gf.log_census(layers=torch.nn.ModuleList([torch.nn.Identity()]), gdn_cls=q.Qwen3_5GatedDeltaNet, pp_rank=0, pp_size=1)
        print("RES", [line, off, none])
        """,
        rank=1,
    )
    line, off, none = r
    assert line.startswith("27B-GDN-FORM pp=0/1 gdn=3 forms=1 [n=3 layers=0..2 cls=Qwen3_5GatedDeltaNet tp=3 rank=1 heads_k/v=4/12"), line
    assert f"out={_CT} K=1536 in_parallel=True reduce=False bias=False" in line, line
    assert f"gdn_in=skip:lane_{_CT} gdn_out=skip:lane_{_CT}" in line, line
    assert off == "" and none == "", (off, none)


def test_qwen3_5_logs_the_census_after_make_layers():
    src = open(os.path.join(_PY, "flliper", "srt", "models", "qwen3_5.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5ForCausalLM")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    line_of = {ast.unparse(n.func): n.lineno for n in ast.walk(init) if isinstance(n, ast.Call)}
    assert "gdn_form.log_census" in line_of and "make_layers" in line_of, line_of
    assert line_of["make_layers"] < line_of["gdn_form.log_census"], line_of
    gdn = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5GatedDeltaNet")
    text = ast.unparse(gdn)
    # the forms the census reads: out_proj never all-reduces inside, qkvz never gathers (the defaults of the projections)
    assert "reduce_results=False" in text and "input_is_parallel=True" in text, "out_proj arguments changed"
