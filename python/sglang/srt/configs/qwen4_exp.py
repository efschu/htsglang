"""Qwen4Exp ("Qwen3.8-Flash-Next") model configuration.

PROVENANCE — this is upstream code, ADOPTED, not fork-authored. The body below
is byte-identical to upstream's ``python/sglang/srt/configs/qwen4_exp.py`` from
sgl-project/sglang PR #36497 "Introduce Qwen 3.8 Flash Next" (JustinTong0323).
Verified against TWO independent copies of the same 169 lines: the PR head
``78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`` (raw.githubusercontent, fetched
2026-08-30) and this repo's own ``upstream/qwen4-main-squashed``
(``99c9362e6685db579c469f6e0e566b08827b3477``); ``diff -u`` between them is
empty. Re-check with:
    git show upstream/qwen4-main-squashed:python/sglang/srt/configs/qwen4_exp.py

Keep it adopted. Every fork-local deviation is fenced with a ``FORK-LOCAL``
comment so the next rebase onto upstream stays a diff instead of archaeology.
Complete list of deviations: (1) this docstring; (2) the wrapper attribute
forwarding at the end of ``Qwen4ExpConfig.__init__``.

How the hybrid layout is declared — DO NOT add a second mechanism.
``Qwen4ExpTextConfig`` subclasses :class:`Qwen3NextConfig`, and
``ModelRunner.hybrid_gdn_config`` (model_executor/model_runner.py:3338-3351)
tests ``isinstance(hf_config.get_text_config(), Qwen3NextConfig | ...)``. The
subclass relation ALONE therefore registers this model with the hybrid GDN
backend, the mamba state pool (mem_cache/kv_cache_builder.py:465) and the
attention registry — exactly as ``Qwen3_5TextConfig`` (configs/qwen3_5.py:15)
already does. ``linear_layer_ids``, ``full_attention_layer_ids`` and
``mamba2_cache_params`` are inherited unchanged; the GDN state geometry they
read off this config is ``linear_num_value_heads`` 48 x
``linear_value_head_dim`` 128 = 6144 intermediate, ``linear_num_key_heads`` 16
groups, state size ``linear_key_head_dim`` 128, ``linear_conv_kernel_dim`` 4,
``mamba_ssm_dtype`` "float32" (picked up by configs/mamba_utils.py:79-145,
which reads it from ``text_config`` for VL models).

The QSA indexer keys stay ``indexer_n_heads`` / ``indexer_kv_heads`` /
``indexer_head_dim`` / ``indexer_budget`` / ``indexer_compress_ratio`` and flow
through ``**kwargs`` onto the text config unchanged. They MUST NOT be renamed to
the fork's ``index_*`` spelling: upstream
``layers/attention/qsa/config.py::parse_qsa_profile`` treats ``index_*`` as a
DIFFERENT (tokenwise) indexer variant and raises "Ambiguous QSA config" when
both spellings are present. This checkpoint's values are accepted by upstream's
validator: ``indexer_kv_heads`` 1 (MQA required), ``indexer_compress_ratio`` 4
(>= 2 required), ``indexer_budget`` 2048 divisible by it, and
``budget // ratio`` = 512, which is in ``fast_topk_v2``'s permitted set
{512, 2048}.

``ple_layer_ids: [2]`` (config) vs PLE tensors that measurably live under
``model.language_model.layers.1.ple.*``: upstream resolves the disagreement with
``short_conv_layer_ids`` = ``{id - 1}`` -> ``[1]``, i.e. ``ple_layer_ids`` names
CONSUMER layers and the owner is one layer below. Upstream and the measured
tensor names agree, so no fork override is needed here.

PLE ("N-gram Embedding") — the geometry these config fields actually describe,
read out of upstream's own implementation
(``models/qwen4_exp.py`` at upstream/qwen4-main-squashed, class
``Qwen4ExpPLEEmbedding``: ``_build_head_vocab_and_offsets`` :532-546,
``padded_vocab_size`` :481-484, shard loader ``load_qwen4_exp_ple_shard``
:1863-1926) and then CHECKED against the measured checkpoint.

Measured: 128 tensors
``...layers.1.ple.ple_embedding.ngram_embedding.shard_<i>.weight``, each BF16
(2_500_012, 160).

  ngram_heads    = (ngram_size - 1) * heads_per_ngram = (3 - 1) * 8 = 16
  head_dim       = ple_embed_dim // ngram_heads       = 2560 // 16   = 160
                                                       == measured width
  head vocab i   = the (ple_layer_index*16 + i + 1)-th PRIME strictly after
                   ngram_vocab_size_base - 1 = 19_999_999
  total          = sum of those 16 primes
  padded         = ceil(total / make_ngram_vocab_size_divisible_by) * 128
  shard rows     = padded // split_ngram_parts

With ``ple_layer_index = 0`` the 16 primes run 20_000_003 .. 20_000_171,
total = 320_001_446, padded = 320_001_536, and
320_001_536 // 128 = 2_500_012 — EXACTLY the measured shard height, and
320_001_536 x 160 x 2 B = 95.3688 GiB against 95.368 GiB measured. So the 128
checkpoint shards are a UNIFORM ROW SPLIT of ONE padded table; they are not 16
separately sized embedders. ``ple_layer_index`` is load-bearing for the hash:
at index 1 or 2 the primes shift and the padded total becomes 320_005_120 /
320_009_088, neither of which matches the checkpoint. DESK-PROVEN, exact.

DO NOT REUSE ``layers/n_gram_embedding.py::NgramEmbedding`` FOR THIS MODEL.
That layer is LongCat's and its moduli are consecutive ODD INTEGERS
(n_gram_embedding.py:71-80, ``mod = over_embedding_m + 2*((n-2)*k + head) + 1``),
while Qwen4-Exp's are PRIMES. Its 16-embedder concatenation also cannot tile the
checkpoint's 128 uniform shards. Wiring it here would not raise — it would
mis-hash every lookup and yield plausible-looking logits, i.e. the failure that
gets misattributed to quantisation for days. Upstream's own path
(``kernels/ops/qwen4_ple.py``, ``mem_cache/ple_state_pool.py``,
``--ple-offload-embedding``) is the only correct one, and the two must never
both be wired.

``ngram_heads_vocab_sizes`` / ``ngram_heads_offsets`` / ``layer_multipliers``
are persistent buffers: upstream RECOMPUTES them from these config fields and
ALSO loads them from the checkpoint when present, under a shape check
(models/qwen4_exp.py:1755-1775). That is the belt-and-braces — the config
fields above must therefore stay consistent with the checkpoint's buffers or
that shape check is the thing that fires.
"""

from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig


class Qwen4ExpVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen4_exp"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen4ExpTextConfig(Qwen3NextConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=128,
        ple_offload_embedding=False,
        ple_embedding_dtype=None,
        index_share_for_mtp_iteration=True,
        rope_parameters=None,
        layer_types=None,
        **kwargs,
    ):
        if hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {hc_count}.")
        # Qwen3.5/Qwen4-Exp checkpoints may provide RoPE settings under
        # rope_parameters. Normalize it before parent init so Qwen3Next shared
        # config logic sees the expected rope_scaling and rope_theta fields.
        if rope_parameters is not None:
            if kwargs.get("rope_scaling") is None:
                kwargs["rope_scaling"] = rope_parameters
            if kwargs.get("rope_theta") is None and "rope_theta" in rope_parameters:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if (
                kwargs.get("partial_rotary_factor") is None
                and "partial_rotary_factor" in rope_parameters
            ):
                kwargs["partial_rotary_factor"] = rope_parameters[
                    "partial_rotary_factor"
                ]
        super().__init__(**kwargs)
        if self.rope_scaling is None:
            self.rope_scaling = rope_parameters or {}
        self.rope_parameters = rope_parameters or self.rope_scaling
        self.hc_count = hc_count
        # ModelConfig sizes the speculative hidden width off `hc_mult`
        # (the DeepSeek-V4 mHC field); Qwen4-Exp spells it `hc_count`.
        self.hc_mult = hc_count
        self.hc_lowrank = hc_lowrank
        self.layer_types = layer_types
        self.ple_layer_ids = ple_layer_ids or []
        self.ple_embed_dim = ple_embed_dim or self.hidden_size
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.ple_offload_embedding = ple_offload_embedding
        # "float8_e4m3fn" keeps fp8 PLE tables fp8-resident; text_config-scoped.
        self.ple_embedding_dtype = ple_embedding_dtype
        # MTP draft decode steps reuse the draft-extend indexer selection
        # (GLM-5.2 IndexShare); default on for Qwen4-Exp, checkpoint config
        # or --json-model-override-args can disable it.
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration

    @property
    def layers_block_type(self):
        if self.layer_types is not None:
            return [
                (
                    "attention"
                    if layer_type in ("full_attention", "qwen_sparse_attention")
                    else layer_type
                )
                for layer_type in self.layer_types
            ]
        return super().layers_block_type

    @property
    def short_conv_layer_ids(self):
        if not self.ple_layer_ids:
            return []
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self):
        if not self.short_conv_layer_ids:
            return None
        ple_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        ple_channels = self.hidden_size * self.hc_count
        return ple_channels, ple_state_len

    @property
    def ngram_context_len(self):
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        rope_parameters=None,
        **kwargs,
    ):
        # The nested text config is authoritative; old exports also copied this
        # value to the top level.
        if text_config is not None:
            kwargs.pop("split_ngram_parts", None)

        # Backward compatibility: older Qwen4-Exp checkpoints were text-only
        # and stored text attributes at the top level.
        text_kwargs = (
            dict(kwargs)
            if text_config is None
            and "hidden_size" in kwargs
            and "num_hidden_layers" in kwargs
            else {}
        )
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.rope_parameters = rope_parameters or getattr(
            self.text_config, "rope_parameters", {}
        )
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

        # ---- FORK-LOCAL (the only code deviation from upstream) ------------
        # Upstream leaves the wrapper without the text attributes and relies on
        # utils/hf_transformers/config.py:149-157 to copy them during
        # ``HfModelConfigParser.parse``. That copy is guarded by
        # ``isinstance(model, str)``, so a config constructed directly
        # in-process (layout planner, ``--json-model-override-args``, tests) has
        # none of them, and any consumer handed the WRAPPER instead of the text
        # config raises AttributeError. Mirror the loader's own behaviour here
        # rather than adding a second mechanism elsewhere.
        # The list is exactly the names ``ModelConfig`` accesses with no getattr
        # default: configs/model_config.py:988 head_dim, :1171
        # num_attention_heads, :1172 num_key_value_heads, :1178
        # full_attention_interval, :1190 hidden_size, :1198 num_hidden_layers,
        # :1235 vocab_size, :970 max_position_embeddings, and layer_types
        # (:2285 / :2307 / :2326), plus the token ids
        # utils/hf_transformers/common.py:273-278 propagates in both directions
        # anyway. Never overwrite a value the wrapper already carries.
        for _attr in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "layer_types",
            "full_attention_interval",
            "rms_norm_eps",
            "hidden_act",
            "use_cache",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        ):
            _val = getattr(self.text_config, _attr, None)
            if _val is not None and getattr(self, _attr, None) is None:
                setattr(self, _attr, _val)
        # ---- end FORK-LOCAL -----------------------------------------------
