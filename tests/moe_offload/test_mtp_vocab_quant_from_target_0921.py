"""#66 (fnFL2v66, 21.09.): the MTP's vocab follows the TARGET's quantization.

``_mtp_quant_config`` nulls the config whenever the checkpoint's ``mtp.``
tensors are dense -- correct for the mtp layers, wrong for ``embed_tokens``,
whose rows come from the target half of the same checkpoint.  Minachist packs
the vocab (AutoRound group_3 ``re:.*embed_tokens``), so the nulled config
built a bf16 table and the producer's resident load refused by name:

    checkpoint tensor model.language_model.embed_tokens.weight_packed has no
    parameter on the built embedding (built: ['embed_tokens.weight'])

That refusal is RIGHT -- int codes without their scale would load as token
soup.  What had to change is the build.
"""

import inspect
import types

from sglang.srt.models import qwen4_exp, qwen4_exp_mtp


def test_the_mtp_hands_the_unmodified_config_through_for_the_vocab():
    src = inspect.getsource(qwen4_exp_mtp.Qwen4ExpForCausalLMMTP.__init__)
    # captured BEFORE the nulling, passed to the model build
    assert src.index("embed_quant_config = quant_config") < src.index(
        "quant_config = _mtp_quant_config(quant_config)"
    )
    assert "embed_quant_config=embed_quant_config," in src


def test_the_builder_prefers_the_vocab_config_and_keeps_the_old_rule():
    src = inspect.getsource(qwen4_exp.Qwen4ExpModel._build_embed_tokens)
    assert 'getattr(self, "_embed_quant_config", None) or quant_config' in src
    # the rule itself is unchanged: quantize iff the config NAMES the vocab
    assert "vocab_named_in_targets(raw, name)" in src


def test_the_attribute_is_set_before_the_builder_runs():
    """super().__init__ is what calls _build_embed_tokens."""
    src = inspect.getsource(qwen4_exp.Qwen4ExpModel.__init__)
    assert src.index("self._embed_quant_config = embed_quant_config") < src.index(
        "super().__init__("
    )


def test_a_named_vocab_quantizes_and_an_unnamed_one_does_not(monkeypatch):
    """The decision table, on the two real export shapes."""
    seen = {}

    class _Cfg:
        def __init__(self, cfg):
            self.config = cfg

    def fake_vocab_named(raw, name):
        return raw.get("named", False)

    monkeypatch.setattr(qwen4_exp, "vocab_named_in_targets", fake_vocab_named)

    def build(embed_cfg, own_cfg):
        stub = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_first_rank=True),
            _embed_quant_config=embed_cfg,
        )
        monkeypatch.setattr(qwen4_exp, "skip_on_worker", lambda *a: None)
        captured = {}

        def fake_embed(*a, **kw):
            captured["quant_config"] = kw.get("quant_config")
            return "embedding"

        monkeypatch.setattr(qwen4_exp, "VocabParallelEmbedding", fake_embed)
        monkeypatch.setattr(qwen4_exp, "is_dp_attention_enabled", lambda: False)
        monkeypatch.setattr(qwen4_exp, "form_a_dense_is_unsharded", lambda: False)
        cfg = types.SimpleNamespace(vocab_size=8, hidden_size=4)
        qwen4_exp.Qwen4ExpModel._build_embed_tokens(stub, cfg, own_cfg, prefix="mtp")
        return captured["quant_config"]

    # Minachist: mtp dense (own config nulled) but the vocab IS named
    named = _Cfg({"named": True})
    assert build(named, None) is named
    # cyankiwi: vocab not named -> dense, exactly as before
    unnamed = _Cfg({"named": False})
    assert build(unnamed, None) is None
    # no embed config (every non-MTP caller): the own config decides
    assert build(None, named) is named
    seen.clear()
