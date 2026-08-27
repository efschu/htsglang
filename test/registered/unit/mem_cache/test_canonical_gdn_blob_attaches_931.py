"""#931 -- the canonical GDN blob must ATTACH, and a hybrid that cannot must refuse.

WHAT WAS DARK, and for how long. ``a38f39f1ee`` (#706 slice 2, 2026-08-17) built
the geometry-neutral ``{hash}.mamba`` blob and ``build_mamba_window`` to cut it
for either phase. ``HiCacheController._canonical_mamba_window`` opened with

    cache_params = getattr(model_config, "mamba2_cache_params", None)
    mamba_layer_ids = list(getattr(cache_params, "layers", None) or [])
    if not mamba_layer_ids:
        return None

and ``mamba2_cache_params`` is a property of the CHECKPOINT config
(``Qwen3NextConfig``, configs/qwen3_next.py:288, ``layers=self.linear_layer_ids``),
never of sglang's ``ModelConfig`` -- ``grep -c mamba2_cache_params
configs/model_config.py`` is 0. The getattr missed on every model, the list was
always empty, and the blob has NEVER attached on any boot.

THE MEASUREMENT, boot 2g (2026-08-27, boot_2f_698cd396ce_0827_0704.log). Format
explicitly armed: ``phase_flip_canonical_kv_page=True``,
``phase_flip_writeback=True``, ``hicache_storage_backend='file'``, page_size 1,
so the pairing guard at server_args.py:8716 held. And yet::

    "#706 canonical KV page active"   3 lines (one per rank)
    "#706 canonical GDN blob active"  0 lines
    the NotImplementedError refusal   0 lines

The KV half canonical, the GDN half silently absent -- the exact split
``_canonical_mamba_window``'s own docstring calls fatal: "a canonical KV page
beside a phase-local GDN blob delivers ZERO usable prefix ... Silently running
KV-only would look like the feature was on while every cross-phase lookup
missed." That is the origin of #928: KV crossed the flip geometry-neutrally
while the recurrent anchor stayed phase-local, so a resume after the cutover
read another phase's bytes.

WHY THE #706 SUITE DID NOT CATCH IT. ``test_canonical_mamba_blob_706.py`` and
``test_mamba_phase_uniform_706.py`` pin the FORMAT -- extents, completeness
markers, key shape. Nothing asked whether the code path that builds the window
is ever entered. Desk-written, never executed: the format is proven and the
road to it was never driven. This file is the road.

THE FIX IN ONE SENTENCE: the same "RESOLVED, not guessed" ladder the attention
half received on the same day (``resolve_attn_layer_ids``), for the linear half
(``resolve_linear_layer_ids``), so an empty list is a PROVEN dense model and an
unresolvable hybrid RAISES instead of skipping.

WHAT EACH TEST HOLDS DOWN
  1. a Qwen3-Next-shaped hybrid resolves its GDN layer ids   -- the defect;
  2. the wrapped source (mamba2_cache_params.layers) resolves -- the second rung;
  3. a proven-dense model returns []                          -- mutant guard: a
     ladder that raises for everything is an outage;
  4. a hybrid with kinds but no id list RAISES                -- the
     silently-wrong-into-refusal conversion the docstring demanded;
  5. ``_canonical_mamba_window`` gets PAST its early return    -- the attach
     itself, which is what was actually broken; asserting on the resolver alone
     would have left the caller free to keep its own bare getattr.

NOT COVERED HERE, and named rather than implied: the end-to-end content proof
(archive a PP-phase anchor through the fence, load it back in the TP phase,
compare against the reference state) needs live pools and a storage backend and
is not hermetically cheap. It is a boot-proof item; the coherence probe in the
measurement window exercises exactly that path.
"""

import unittest

from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalPageError,
    resolve_linear_layer_ids,
)

# Qwen3-Next: 48 of 64 layers are GDN, attention every 4th.
_BLOCK_TYPES = [
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
]
_LINEAR_IDS = [i for i, t in enumerate(_BLOCK_TYPES) if t == "linear_attention"]


class _CacheParams:
    def __init__(self, layers) -> None:
        self.layers = layers


class _TextConfig:
    """The checkpoint config. Attributes are set selectively per test, because
    which of them exists is the whole question."""

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _HfConfig:
    architectures = ["Qwen3NextForCausalLM"]


class _ModelConfig:
    """sglang's ModelConfig: note it deliberately has NO mamba2_cache_params
    and NO linear_layer_ids, which is exactly why the old lookup missed."""

    def __init__(self, text_config, *, is_hybrid=False) -> None:
        self.hf_text_config = text_config
        self.hf_config = _HfConfig()
        self.num_hidden_layers = 64
        self.is_hybrid = is_hybrid


class TestCanonicalGdnBlobAttaches931(unittest.TestCase):
    def test_a_qwen3_next_shaped_hybrid_resolves_its_gdn_layers(self):
        cfg = _ModelConfig(
            _TextConfig(
                layers_block_type=_BLOCK_TYPES,
                linear_layer_ids=_LINEAR_IDS,
                full_attention_layer_ids=[3, 7, 11],
            )
        )
        self.assertEqual(
            resolve_linear_layer_ids(cfg),
            _LINEAR_IDS,
            "the GDN layer ids live on the CHECKPOINT config, not on "
            "ModelConfig; reading them off ModelConfig returns nothing on "
            "every model and leaves the canonical blob permanently dark",
        )

    def test_the_wrapped_source_resolves_too(self):
        """mamba2_cache_params.layers, when the primitive is absent."""
        cfg = _ModelConfig(
            _TextConfig(
                layers_block_type=_BLOCK_TYPES,
                mamba2_cache_params=_CacheParams(_LINEAR_IDS),
            )
        )
        self.assertEqual(resolve_linear_layer_ids(cfg), _LINEAR_IDS)

    def test_a_proven_dense_model_has_no_linear_layers(self):
        """MUTANT GUARD. A ladder that refuses everything is an outage."""
        cfg = _ModelConfig(_TextConfig())
        self.assertEqual(
            resolve_linear_layer_ids(cfg),
            [],
            "no layer kinds and no hybrid flag is a POSITIVE dense proof, and "
            "a dense model legitimately has no blob",
        )

    def test_a_hybrid_that_cannot_be_resolved_refuses(self):
        """The conversion the caller's own docstring demanded: a hybrid whose
        blob cannot be made canonical must not run with canonical KV pages
        alone."""
        cfg = _ModelConfig(_TextConfig(layers_block_type=_BLOCK_TYPES))
        with self.assertRaises(CanonicalPageError) as caught:
            resolve_linear_layer_ids(cfg)
        self.assertIn("linear_layer_ids", str(caught.exception))

    def test_the_window_builder_is_reached_for_a_hybrid(self):
        """THE ATTACH ITSELF. The early return is what fired on every boot, so
        the test has to prove the code gets past it -- a green resolver with a
        caller still holding its own bare getattr would look identical from
        outside."""
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache import canonical_page_store

        class _Sentinel(Exception):
            pass

        def _boom(*a, **kw):
            raise _Sentinel()

        class _Hybrid:
            mamba_pool = object()

        class _Self:
            mem_pool_device_hybrid = _Hybrid()
            tp_size = 1
            tp_rank = 0

        cfg = _ModelConfig(
            _TextConfig(
                layers_block_type=_BLOCK_TYPES,
                linear_layer_ids=_LINEAR_IDS,
            )
        )

        original = canonical_page_store.derive_mamba_blob_spec
        canonical_page_store.derive_mamba_blob_spec = _boom
        try:
            with self.assertRaises(
                _Sentinel,
                msg="_canonical_mamba_window returned early instead of "
                "building a window: the blob never attaches, and a canonical "
                "KV page beside a phase-local GDN blob serves the other "
                "phase's recurrent bytes (#928)",
            ):
                HiCacheController._canonical_mamba_window(_Self(), None, cfg)
        finally:
            canonical_page_store.derive_mamba_blob_spec = original


if __name__ == "__main__":
    unittest.main()
