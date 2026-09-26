"""27B line G2 (2026-09-25): the weg2 launcher on a GGUF ``--model``.

A GGUF launch names the ``.gguf`` FILE. The server has read the sibling
``config.json`` for that since #481a (``ServerArgs.declared_config_path``), the
launcher did not: ``_model_config`` joined ``config.json`` onto the file path and
died on ``<...>.gguf/config.json``, and so did every other launcher reader of the
config (quant method, P-cut config, tied-head flag, digest, depth). Also:

* the content digest (``host_ledger.checkpoint_digest``) needs an index a GGUF
  does not have -- and a names-only digest would give UD-IQ4_XS and UD-Q8_K_XL
  of one model the same identity (and the same P-cut calibration);
* a qwen35 GGUF loads its tokenizer ONLY from sibling files, and the unsloth
  directory has none: both groups would die in their tokenizer managers after
  the cards are taken. ``--tokenizer-path`` goes to BOTH groups, or the launch
  is refused by name first (W163);
* the loader runs the FILE's depth when it differs from the config (depth
  reconciliation); the launcher's layer map would describe another model (W162).

Desk only: tiny GGUFs written with gguf-py in tmp dirs; the real unsloth files
are read (header only) when present.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402

from sglang.srt import server_args as SA  # noqa: E402
from sglang.srt.model_loader import gguf_shards  # noqa: E402
from sglang.srt.weg2 import host_ledger as HL  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import ring_table as RT  # noqa: E402

N_LAYERS = 64
REAL_DIR = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-GGUF-unsloth"
REAL_IQ4 = os.path.join(REAL_DIR, "Qwen3.8-27B-UD-IQ4_XS.gguf")
REAL_Q8 = os.path.join(REAL_DIR, "Qwen3.8-27B-UD-Q8_K_XL.gguf")
REAL_TOKENIZER_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
)
ARGV_MODEL = REAL_TOKENIZER_DIR  # the model the existing argv tests build with
BUDGETS = [28904, 17704, 17672]


def _config(n_layers=N_LAYERS, tie=False, quant_method=None):
    text = {
        "num_hidden_layers": n_layers,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(n_layers)
        ],
        "vocab_size": 8,
        "tie_word_embeddings": tie,
    }
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": text}
    if quant_method:
        cfg["quantization_config"] = {"quant_method": quant_method}
    return cfg


def _write_config(directory, **kw):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "config.json")
    with open(path, "w") as fh:
        json.dump(_config(**kw), fh)
    return path


def _write_gguf(
    path, arch="qwen35", block_count=N_LAYERS + 1, nextn=1, attn_dtype=np.float16
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = gguf.GGUFWriter(path, arch)
    w.add_block_count(block_count)
    if nextn is not None:
        w.add_uint32(f"{arch}.nextn_predict_layers", nextn)
    w.add_tensor("token_embd.weight", np.zeros((8, 4), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=attn_dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(autouse=True)
def _fresh_caches():
    """No test sees another's header facts, split resolution or tokenizer.
    (Tolerant of a tree without them, so the base shows its real failures.)"""

    def clear():
        getattr(HL, "_GGUF_HEADER_FACTS", {}).clear()
        gguf_shards._RESOLVED_CACHE.clear()
        if hasattr(L, "_TOKENIZER_PATH"):
            L._TOKENIZER_PATH["path"] = None

    clear()
    yield
    clear()


@pytest.fixture
def gguf_ckpt(tmp_path):
    """A qwen35 GGUF (64 backbone blocks + 1 MTP block) with its sibling config."""
    d = tmp_path / "ggufdir"
    cfg = _write_config(str(d))
    return _write_gguf(str(d / "m-IQ4_XS.gguf")), cfg


# --- the config: the server's rule, called -----------------------------------


def test_a_gguf_file_reads_its_sibling_config(gguf_ckpt):
    """RED on e34b90ffe8: FileNotFoundError on <...>.gguf/config.json."""
    path, cfg = gguf_ckpt
    assert L._model_config(path) == _config()
    assert L.model_config_path(path) == cfg


def test_a_directory_reads_exactly_as_before(tmp_path):
    cfg = _write_config(str(tmp_path / "st"))
    assert L.model_config_path(str(tmp_path / "st")) == cfg
    assert L._model_config(str(tmp_path / "st")) == _config()


def test_the_launcher_asks_the_servers_rule(gguf_ckpt, tmp_path):
    path, _ = gguf_ckpt
    _write_config(str(tmp_path / "st"))
    bare = str(tmp_path / "bare" / "x.gguf")
    _write_gguf(bare)
    for p in (path, str(tmp_path / "st"), bare, str(tmp_path / "nowhere")):
        server = SA.ServerArgs.declared_config_path(types.SimpleNamespace(model_path=p))
        assert server == SA.declared_config_path_for(p)
        if server is None:
            with pytest.raises(FileNotFoundError):
                L.model_config_path(p)
        else:
            assert L.model_config_path(p) == server


def test_no_config_anywhere_names_both_places(tmp_path):
    bare = _write_gguf(str(tmp_path / "bare" / "x.gguf"))
    with pytest.raises(OSError) as ei:
        L._model_config(bare)
    msg = str(ei.value)
    assert os.path.join(bare, "config.json") in msg
    assert os.path.join(str(tmp_path / "bare"), "config.json") in msg


def test_depth_and_layer_kinds_of_a_gguf(gguf_ckpt):
    """RED on e34b90ffe8 (no config found)."""
    path, _ = gguf_ckpt
    assert L.model_num_layers(path) == N_LAYERS
    kinds = L.model_layer_kinds(path)
    assert len(kinds) == N_LAYERS and sum(kinds) == N_LAYERS // 4


@pytest.mark.parametrize(
    "blocks,nextn,ok",
    [
        (N_LAYERS + 1, 1, True),
        (N_LAYERS, None, True),
        (N_LAYERS, 0, True),
        (N_LAYERS - 1, None, False),
        (N_LAYERS + 1, None, False),
    ],
)
def test_a_gguf_whose_depth_is_not_its_configs_is_w162(tmp_path, blocks, nextn, ok):
    d = tmp_path / "g"
    _write_config(str(d))
    path = _write_gguf(str(d / "m.gguf"), block_count=blocks, nextn=nextn)
    if ok:
        assert L.model_num_layers(path) == N_LAYERS
    else:
        with pytest.raises(L.Weg2LaunchRefused, match="W162 Weg2GgufDepthRefused"):
            L.model_num_layers(path)


def test_the_depth_is_the_one_the_loader_reconciles(gguf_ckpt):
    """block_count minus the NEXTN blocks, as gguf_registry.reconcile_sibling_config."""
    path, _ = gguf_ckpt
    assert L.gguf_backbone_depth(path) == N_LAYERS
    src = inspect.getsource(
        __import__(
            "sglang.srt.model_loader.gguf_registry", fromlist=["x"]
        ).reconcile_sibling_config
    )
    assert 'n_blocks -= kv("nextn_predict_layers") or 0' in src


def test_the_quant_method_of_a_gguf_is_gguf(gguf_ckpt, tmp_path):
    """RED on e34b90ffe8: "" (config unreadable), so p_cut_calibration_line took
    the GGUF for the INT8 incumbent family and said nothing."""
    path, _ = gguf_ckpt
    assert L.checkpoint_quant_method(path) == "gguf"
    _write_config(str(tmp_path / "f8"), quant_method="fp8")
    assert L.checkpoint_quant_method(str(tmp_path / "f8")) == "fp8"
    assert L.checkpoint_quant_method(str(tmp_path / "nothing")) == ""
    env, line = L.fp8_layout_decision("gguf", "exchange", False)
    assert env == {} and line is None


def test_an_uncalibrated_gguf_is_named(gguf_ckpt, monkeypatch, tmp_path):
    path, _ = gguf_ckpt
    monkeypatch.setattr(HL, "CALIB_DIR", str(tmp_path / "no-calib"), raising=False)
    line = L.p_cut_calibration_line(path, L.checkpoint_quant_method(path))
    assert line and "UNCALIBRATED gguf checkpoint" in line


def _config_joins(tree):
    """(function, argument source) of every os.path.join(<arg>, "config.json")."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "os.path.join"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "config.json"
            ):
                out.append((fn.name, ast.unparse(node.args[0])))
    return out


def test_no_reader_joins_config_json_onto_the_model_path():
    """The sweep: every launcher-side reader of the MODEL's config goes through
    the server's rule. Left: the fallbacks inside the two resolvers, and the
    DFlash draft (a directory by --spec-form contract)."""
    allowed = {
        ("model_config_path", "str(model)"),
        ("_config_path", "model_path"),
        ("_weg2_arena_ledger_terms", "_dp"),
    }
    for mod in (L, HL, RT):
        tree = ast.parse(inspect.getsource(mod))
        found = {
            (fn, arg)
            for fn, arg in _config_joins(tree)
            if "model" in arg or arg == "_dp"
        }
        assert found <= allowed, (mod.__name__, sorted(found - allowed))
    assert "model_config_path(model)" in inspect.getsource(L.solve_p_cut)


def test_the_tied_head_flag_of_a_gguf(tmp_path):
    """RED on e34b90ffe8: False (config unreadable) -- the ring priced a head a
    tied GGUF does not have."""
    d = tmp_path / "tied"
    _write_config(str(d), tie=True)
    path = _write_gguf(str(d / "m.gguf"))
    assert RT._tie_word_embeddings(path) is True
    assert RT._tie_word_embeddings(str(d)) is True


# --- the digest ----------------------------------------------------------------


def _write_index(directory, names):
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": {n: "a.safetensors" for n in names}}, fh)


def test_a_directory_digest_is_bit_identical(tmp_path):
    import hashlib

    d = str(tmp_path / "st")
    _write_config(d)
    names = ["model.layers.0.w", "lm_head.weight"]
    _write_index(d, names)
    tc = _config()["text_config"]
    c = hashlib.sha256(
        json.dumps(tc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    i = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    dg, why = HL.checkpoint_digest(d)
    assert dg == hashlib.sha256((c + i).encode()).hexdigest()
    assert why.startswith(
        "content digest over config.text_config + sorted tensor NAMES"
    )


def test_a_gguf_has_a_content_digest(gguf_ckpt):
    """RED on e34b90ffe8: (None, 'checkpoint digest unavailable ...')."""
    path, _ = gguf_ckpt
    dg, why = HL.checkpoint_digest(path)
    assert dg and len(dg) == 64, why
    assert "gguf content digest" in why and "2 tensors in 1 part(s)" in why


def test_two_quantizations_of_one_model_digest_apart(tmp_path):
    d = tmp_path / "two"
    _write_config(str(d))
    a = _write_gguf(str(d / "m-IQ4_XS.gguf"), attn_dtype=np.float16)
    b = _write_gguf(str(d / "m-Q8_K_XL.gguf"), attn_dtype=np.float32)
    assert HL.checkpoint_digest(a)[0] != HL.checkpoint_digest(b)[0]


def test_the_digest_is_the_content_not_the_path(gguf_ckpt, tmp_path):
    path, _ = gguf_ckpt
    other = tmp_path / "copy"
    _write_config(str(other))
    shutil.copy(path, str(other / "renamed.gguf"))
    dg = HL.checkpoint_digest(path)[0]
    assert dg is not None
    assert HL.checkpoint_digest(str(other / "renamed.gguf"))[0] == dg


def test_a_gguf_without_its_config_has_no_digest(tmp_path):
    bare = _write_gguf(str(tmp_path / "bare" / "x.gguf"))
    dg, why = HL.checkpoint_digest(bare)
    assert dg is None and "sibling config.json" in why


def test_checkpoint_layers_of_a_gguf(gguf_ckpt):
    """RED on e34b90ffe8: None."""
    path, _ = gguf_ckpt
    assert HL.checkpoint_layers(path) == N_LAYERS


def test_a_launch_reads_the_header_once(gguf_ckpt, monkeypatch):
    """gguf-py parses every metadata field on open (8.5 s on the real IQ4_XS):
    depth, arch and five digest sites share ONE read (plus the loader's own
    split resolution)."""
    path, _ = gguf_ckpt
    opened = []
    real = gguf.GGUFReader

    def counting(p, *a, **k):
        opened.append(p)
        return real(p, *a, **k)

    monkeypatch.setattr(gguf, "GGUFReader", counting)
    for _ in range(5):
        HL.checkpoint_digest(path)
    L.model_num_layers(path)
    L.gguf_backbone_depth(path)
    with pytest.raises(L.Weg2LaunchRefused):
        L.tokenizer_path_decision(path, "")
    assert len(opened) == 2, opened


# --- the tokenizer --------------------------------------------------------------


def _common(group="both"):
    return L.common_flags(ARGV_MODEL, 1, 1200, "{}", 69632, "write_through", group)


def _tokenizer_dir(tmp_path, name="tok"):
    d = tmp_path / name
    d.mkdir()
    (d / "tokenizer.json").write_text("{}")
    return str(d)


def test_no_flag_and_no_gguf_adds_nothing(tmp_path):
    before = _common()
    _write_config(str(tmp_path / "st"))
    ns = types.SimpleNamespace(model=str(tmp_path / "st"), tokenizer_path="")
    assert L.tokenizer_path_decision(ns.model, "") == (None, None)
    assert L.apply_tokenizer_path(ns) is None
    assert _common() == before
    assert "--tokenizer-path" not in before


@pytest.mark.skipif(not os.path.isdir(ARGV_MODEL), reason="argv model absent")
def test_an_explicit_tokenizer_goes_to_both_groups(tmp_path):
    tok = _tokenizer_dir(tmp_path)
    ns = types.SimpleNamespace(model=ARGV_MODEL, tokenizer_path=tok)
    line = L.apply_tokenizer_path(ns)
    assert line == f"WEG2 TOKENIZER --tokenizer-path {tok} (both groups)"
    for argv in (
        _common("P"),
        _common("D"),
        L.argv_p("py", ARGV_MODEL, BUDGETS, 1, 1200, "{}", []),
        L.argv_d("py", ARGV_MODEL, BUDGETS, 1, 1200, "{}", []),
    ):
        i = argv.index("--model-path")
        assert argv[i : i + 4] == ["--model-path", ARGV_MODEL, "--tokenizer-path", tok]
        assert argv.count("--tokenizer-path") == 1


@pytest.mark.parametrize("which", ["missing", "empty"])
def test_a_tokenizer_dir_without_tokenizer_files_is_w163(tmp_path, which):
    bad = str(tmp_path / "nope")
    if which == "empty":
        os.makedirs(bad)
    with pytest.raises(L.Weg2LaunchRefused, match="W163 Weg2TokenizerRefused"):
        L.tokenizer_path_decision(str(tmp_path), bad)


def test_a_qwen35_gguf_without_a_tokenizer_is_w163(gguf_ckpt):
    path, _ = gguf_ckpt
    with pytest.raises(L.Weg2LaunchRefused, match="W163 Weg2TokenizerRefused") as ei:
        L.tokenizer_path_decision(path, "")
    assert "--tokenizer-path" in str(ei.value) and "qwen35" in str(ei.value)


def test_a_qwen35_gguf_with_sibling_tokenizer_needs_no_flag(gguf_ckpt):
    path, _ = gguf_ckpt
    with open(os.path.join(os.path.dirname(path), "tokenizer.json"), "w") as fh:
        fh.write("{}")
    tok, line = L.tokenizer_path_decision(path, "")
    assert tok is None and "sibling files" in line


def test_a_gguf_of_a_transformers_arch_needs_no_flag(tmp_path):
    d = tmp_path / "llama"
    _write_config(str(d))
    path = _write_gguf(str(d / "m.gguf"), arch="llama")
    tok, line = L.tokenizer_path_decision(path, "")
    assert tok is None and "transformers" in line


def test_the_arch_decision_is_the_servers_peek(gguf_ckpt, tmp_path):
    from sglang.srt.model_loader.gguf_registry import sibling_config_gguf_archs
    from sglang.srt.utils.hf_transformers.config import _peek_bespoke_gguf_arch

    path, _ = gguf_ckpt
    llama = _write_gguf(str(tmp_path / "llama" / "m.gguf"), arch="llama")
    for p in (path, llama):
        arch = HL.gguf_header_facts(p).arch
        assert _peek_bespoke_gguf_arch(p) == (
            arch if arch in sibling_config_gguf_archs() else None
        )


def test_the_cli_flag_and_its_install_order():
    ns = L.build_parser().parse_args(["--tree", "t", "--tag", "x"])
    assert ns.tokenizer_path == ""
    src = inspect.getsource(L.main)
    assert src.index("apply_tokenizer_path(ns)") < src.index("common_flags(")


# --- the real checkpoint (header only; skipped where absent) -----------------------


@pytest.mark.skipif(not os.path.isfile(REAL_IQ4), reason="unsloth GGUF absent")
def test_the_real_unsloth_iq4_xs():
    assert L.model_config_path(REAL_IQ4) == os.path.join(REAL_DIR, "config.json")
    assert L.model_num_layers(REAL_IQ4) == N_LAYERS
    assert L.checkpoint_quant_method(REAL_IQ4) == "gguf"
    facts = HL.gguf_header_facts(REAL_IQ4)
    assert (facts.arch, facts.block_count, facts.nextn_predict_layers) == (
        "qwen35",
        65,
        1,
    )
    with pytest.raises(L.Weg2LaunchRefused, match="W163"):
        L.tokenizer_path_decision(REAL_IQ4, "")
    if os.path.isdir(REAL_TOKENIZER_DIR):
        assert (
            L.tokenizer_path_decision(REAL_IQ4, REAL_TOKENIZER_DIR)[0]
            == REAL_TOKENIZER_DIR
        )
    dg, why = HL.checkpoint_digest(REAL_IQ4)
    assert dg, why
    if os.path.isfile(REAL_Q8):
        assert HL.checkpoint_digest(REAL_Q8)[0] not in (None, dg)


def test_the_h95_seat_table_reads_the_config_of_a_gguf(tmp_path, monkeypatch):
    """H95 (8623408ef9) read ``os.path.join(ns.model, "config.json")``: for a
    GGUF (the FILE) that is ``<x>.gguf/config.json`` and the D seat table was
    dropped with NotADirectoryError. Behavioural companion of the sweep above:
    the table reaches its config through ``model_config_path``."""
    import types

    d = tmp_path / "ckpt"
    _write_config(str(d), quant_method="gguf")
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["text_config"]["num_experts_per_tok"] = 8
    cfg_path.write_text(json.dumps(cfg))
    gguf = d / "model.gguf"
    gguf.write_bytes(b"GGUF")

    class _Reached(Exception):
        pass

    seen = {}

    def _vram(ns, er, plan_kwargs, cfg, form):
        seen["top_k"] = (cfg.get("text_config") or cfg).get("num_experts_per_tok")
        raise _Reached()

    monkeypatch.setattr(L, "d_stated_seats", lambda ns: 2)
    monkeypatch.setattr(L, "d_replayssm_spec_plan_form", lambda ns: object())
    monkeypatch.setattr(L, "d_seat_vram_plan_form", _vram)
    er = types.SimpleNamespace(POOL_GRAPH_MODE_ENV="POOL_MODE")
    lines = L.d_seat_table_lines(types.SimpleNamespace(model=str(gguf)), er,
                                 {"env_d": {"POOL_MODE": "pool"}}, "D")
    assert seen.get("top_k") == 8, lines
