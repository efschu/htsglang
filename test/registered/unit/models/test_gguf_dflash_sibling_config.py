"""A DFLASH-draft GGUF must have its config read from the sibling config.json.

The DFLASH drafter was built in two halves and only one of them was wired:

* the WEIGHT loader knows it -- ``model_loader/gguf_dflash.py`` builds the
  drafter's name map and ``loader.py`` dispatches on
  ``is_dflash_gguf_config(model_config.hf_config)``;
* the CONFIG peek did not -- ``_peek_bespoke_gguf_arch`` only recognised the
  archs harvested from the ``gguf_registry`` FAMILIES (qwen35, qwen35moe,
  gemma4), and the drafter is deliberately not a family: its config carries
  ``model_type: "qwen3"``, so registering it would capture every plain Qwen3
  GGUF. Its GGUF header says ``dflash-draft``, which nothing looked at.

The halves only meet through the config: the loader's dispatch reads
``architectures: ["DFlashDraftModel"]``, which exists solely in the sibling
config.json. Without the peek, the file went to transformers' GGUF reader and
raised ``GGUF model with architecture dflash-draft is not supported yet``
before the drafter's own loader ever ran -- round 7c boot C, ~2 min into the
boot.

The fixture is a synthetic header-only GGUF, so this runs anywhere: the real
1.8 GiB drafter is exercised by the second class when it happens to be on the
machine.
"""

import json
import pathlib
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# The released drafter's geometry, scaled down. Only the ratios that
# reconcile_sibling_config checks have to be self-consistent.
_GEOMETRY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 2,
}

_SIBLING_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "vocab_size": 100,
    "num_target_layers": 8,
    "dflash_config": {"mask_token_id": 7, "target_layer_ids": [0, 1]},
    **_GEOMETRY,
}

_REAL_DRAFT = pathlib.Path(
    "/spinning/llm_stuff/club-3090/models-cache/qwen3.6-27b-dflash-gguf"
    "/Qwen3.6-27B-DFlash-Q8_0.gguf"
)


def _write_gguf(directory: pathlib.Path, arch: str) -> pathlib.Path:
    """A minimal GGUF carrying ``arch`` in its header plus the geometry KVs."""
    import gguf
    import numpy as np

    path = directory / "tiny-draft.gguf"
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_block_count(_GEOMETRY["num_hidden_layers"])
    writer.add_embedding_length(_GEOMETRY["hidden_size"])
    writer.add_feed_forward_length(_GEOMETRY["intermediate_size"])
    writer.add_head_count(_GEOMETRY["num_attention_heads"])
    writer.add_head_count_kv(_GEOMETRY["num_key_value_heads"])
    writer.add_key_length(_GEOMETRY["head_dim"])
    writer.add_tensor("dflash_fc.weight", np.zeros((64, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    (directory / "config.json").write_text(json.dumps(_SIBLING_CONFIG))
    return path


class TestDflashGgufSiblingConfig(CustomTestCase):
    def test_dflash_arch_is_a_sibling_config_arch(self):
        """The ratchet: the header string the peek dispatches on."""
        from sglang.srt.model_loader.gguf_dflash import DFLASH_GGUF_ARCHS
        from sglang.srt.model_loader.gguf_registry import sibling_config_gguf_archs

        archs = sibling_config_gguf_archs()
        for arch in DFLASH_GGUF_ARCHS:
            self.assertIn(
                arch,
                archs,
                msg=(
                    f"{arch!r} must reach the sibling-config peek, or the "
                    "drafter dies in transformers' GGUF reader before its own "
                    "loader runs."
                ),
            )
        # The families must not have been dropped on the way in.
        for arch in ("qwen35", "gemma4"):
            self.assertIn(arch, archs)

    def test_get_config_reads_the_sibling_config_json(self):
        from sglang.srt.utils.hf_transformers.config import _peek_bespoke_gguf_arch
        from sglang.srt.utils.hf_transformers_utils import get_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(pathlib.Path(tmp), "dflash-draft")

            self.assertEqual(_peek_bespoke_gguf_arch(str(path)), "dflash-draft")

            config = get_config(str(path), trust_remote_code=True)
            self.assertEqual(config.architectures, ["DFlashDraftModel"])
            for attr, want in _GEOMETRY.items():
                self.assertEqual(getattr(config, attr), want, msg=attr)
            # Straight out of config.json -- the GGUF metadata carries neither.
            self.assertEqual(config.num_target_layers, 8)

    def test_the_loader_half_now_sees_a_dflash_config(self):
        """The two halves meet: the config the peek produces is the one the
        weight loader dispatches on."""
        from sglang.srt.model_loader.gguf_dflash import (
            build_dflash_name_map,
            is_dflash_gguf_config,
        )
        from sglang.srt.utils.hf_transformers_utils import get_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(pathlib.Path(tmp), "dflash-draft")
            config = get_config(str(path), trust_remote_code=True)

        self.assertTrue(is_dflash_gguf_config(config))
        name_map = build_dflash_name_map(config)
        self.assertEqual(name_map["dflash_fc.weight"], "fc.weight")
        self.assertIn("blk.1.post_attention_norm.weight", name_map)

    def test_an_unknown_gguf_arch_is_not_routed_to_the_sibling_config(self):
        """Negative control against an over-broad fix.

        Sending EVERY GGUF to a sibling config.json would make the test above
        pass while silently changing how ordinary GGUFs are read. Only the
        registered archs may take that route.
        """
        from sglang.srt.utils.hf_transformers.config import _peek_bespoke_gguf_arch

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(pathlib.Path(tmp), "some-unregistered-arch")
            self.assertIsNone(_peek_bespoke_gguf_arch(str(path)))


@unittest.skipUnless(
    _REAL_DRAFT.is_file(), f"drafter artifact not on this machine: {_REAL_DRAFT}"
)
class TestDflashGgufSiblingConfigOnTheRealDrafter(CustomTestCase):
    """The released Qwen3.6-27B-DFlash-Q8_0.gguf, config only -- no GPU, no
    weights loaded."""

    def test_full_config_chain_on_the_released_drafter(self):
        import gguf

        from sglang.srt.model_loader.gguf_dflash import (
            audit_dflash_name_map,
            build_dflash_name_map,
            is_dflash_gguf_config,
        )
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
        from sglang.srt.utils.hf_transformers_utils import get_config

        config = get_config(str(_REAL_DRAFT), trust_remote_code=True)
        self.assertEqual(config.architectures, ["DFlashDraftModel"])
        self.assertTrue(is_dflash_gguf_config(config))

        # The name map has to match the file exactly -- 58 tensors, no leftovers
        # on either side. This is the CPU gate round 7c ran by hand.
        names = [t.name for t in gguf.GGUFReader(str(_REAL_DRAFT)).tensors]
        self.assertIsNone(audit_dflash_name_map(build_dflash_name_map(config), names))

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.assertEqual(draft_config.require_num_layers(), 5)
        self.assertEqual(draft_config.num_target_layers, 64)

    def test_model_config_builds_for_the_drafter(self):
        """What the draft worker actually does, and what the alias hook's probe
        only anticipates: ModelConfig on the drafter path."""
        from sglang.srt.configs.model_config import ModelConfig

        config = ModelConfig(
            model_path=str(_REAL_DRAFT), trust_remote_code=True, is_draft_model=True
        )
        self.assertEqual(config.hf_config.architectures, ["DFlashDraftModel"])
        self.assertEqual(config.num_hidden_layers, 5)


if __name__ == "__main__":
    unittest.main()
