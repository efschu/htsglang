"""A runtime dtype override must reach everything that derives a dtype (#171).

`_get_and_verify_dtype()` is not the last word on the dtype. ModelRunner
re-decides it inside `load_model()` when the device has no bfloat16
(`_needs_float16_fallback()` -- sm75, gfx900) and assigns
`model_config.dtype = torch.float16`. That happens long after the HF config was
parsed, so the HF config still says bfloat16.

Anything that derives its own dtype from the HF config then disagrees with the
model. The mamba conv-state cache is the case that bites: it stores recent
*input activations* of the causal conv1d, `mamba_utils.mamba2_state_dtype()`
builds it from the config, and on sm75 the GDN in_proj emits float16 into a
bfloat16 cache -- `Index put requires source and destination dtypes match`, on
the boot path of every hybrid GDN Qwen3.5. (`SGLANG_MAMBA_CONV_DTYPE=float16`
is the workaround this removes.)

The fix is a pin on assignment: `ModelConfig.dtype` is a property whose setter
writes the resolved dtype back onto the HF config(s), so a runtime override
propagates by construction. These tests set the dtype the way ModelRunner does
and check what the conv cache would be built from.

Pure CPU tests: no model is loaded and no device is touched.
"""

import os
import unittest

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.mamba_utils import mamba2_state_dtype
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bf16_checkpoint_config() -> PretrainedConfig:
    """A hybrid-GDN-shaped HF config as a bfloat16 checkpoint provides it."""
    cfg = PretrainedConfig()
    cfg.dtype = torch.bfloat16
    return cfg


def _model_config_without_loading(hf_config) -> ModelConfig:
    """A ModelConfig carrying `hf_config`, without touching the filesystem.

    `ModelConfig.__init__` reads a checkpoint; only the dtype plumbing is under
    test here, so the instance is built directly around the two attributes the
    setter pins onto.
    """
    mc = ModelConfig.__new__(ModelConfig)
    mc.hf_config = hf_config
    mc.hf_text_config = hf_config
    return mc


class TestRuntimeDtypeOverrideReachesTheConvCache(unittest.TestCase):
    def setUp(self):
        # The conv dtype has an explicit env override that outranks everything;
        # it must be absent for the config-derived default to be observable.
        env = envs.SGLANG_MAMBA_CONV_DTYPE
        prior = os.environ.get(env.name)

        def restore():
            if prior is None:
                env.clear()
            else:
                os.environ[env.name] = prior

        self.addCleanup(restore)
        env.clear()

    # ---- the defect this exists to prevent ------------------------------
    def test_sm75_float16_fallback_reaches_the_conv_state_dtype(self):
        """The bug, end to end: a bf16 checkpoint whose runtime dtype was
        overridden to float16 must NOT get a bfloat16 conv-state cache."""
        hf_config = _bf16_checkpoint_config()
        mc = _model_config_without_loading(hf_config)

        # Resolution as ModelConfig.__init__ performs it: bf16 checkpoint,
        # bf16 runtime.
        mc.dtype = torch.bfloat16
        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.bfloat16)

        # The sm75 fallback, verbatim from ModelRunner.load_model().
        mc.dtype = torch.float16

        self.assertIs(mc.dtype, torch.float16)
        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.float16)

    def test_the_pin_reaches_a_separate_text_config(self):
        """VL/hybrid configs carry the runtime dtype on the text sub-config,
        and the mamba cache is built from that one."""
        hf_config = _bf16_checkpoint_config()
        text_config = _bf16_checkpoint_config()
        mc = ModelConfig.__new__(ModelConfig)
        mc.hf_config = hf_config
        mc.hf_text_config = text_config

        mc.dtype = torch.float16

        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.float16)
        self.assertIs(mamba2_state_dtype(text_config).conv, torch.float16)

    # ---- the other causes, and the paths that must not move -------------
    def test_the_awq_downcast_case_goes_through_the_same_setter(self):
        """The FIRST cause the config path was written for -- no checkpoint
        dtype at all, resolved to float16 -- now flows through this one setter
        instead of a separate pin in __init__. Same outcome, one code path."""
        hf_config = PretrainedConfig()
        mc = _model_config_without_loading(hf_config)

        mc.dtype = torch.float16

        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.float16)

    def test_an_explicit_env_override_still_outranks_the_runtime_dtype(self):
        """SGLANG_MAMBA_CONV_DTYPE is the documented escape hatch and keeps
        precedence -- including against the runtime override."""
        hf_config = _bf16_checkpoint_config()
        mc = _model_config_without_loading(hf_config)
        mc.dtype = torch.float16

        with envs.SGLANG_MAMBA_CONV_DTYPE.override("bfloat16"):
            self.assertIs(mamba2_state_dtype(hf_config).conv, torch.bfloat16)

        # ... and without it, the runtime dtype is back in charge.
        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.float16)

    def test_a_bf16_run_keeps_a_bf16_conv_cache(self):
        """Regression criterion: the default path must not move. No override,
        bf16 checkpoint, bf16 runtime -> bf16 cache."""
        hf_config = _bf16_checkpoint_config()
        mc = _model_config_without_loading(hf_config)

        mc.dtype = torch.bfloat16

        self.assertIs(mamba2_state_dtype(hf_config).conv, torch.bfloat16)
        self.assertIs(mamba2_state_dtype(hf_config).temporal, torch.float32)

    def test_dtype_is_readable_and_is_what_was_assigned(self):
        """The property must behave as the plain attribute it replaces."""
        mc = _model_config_without_loading(_bf16_checkpoint_config())
        mc.dtype = torch.float16
        self.assertIs(mc.dtype, torch.float16)

    def test_a_config_that_refuses_the_pin_does_not_break_the_assignment(self):
        """Some configs are read-only proxies; pinning is best-effort and must
        never take the load path down."""

        class _Frozen:
            def __setattr__(self, name, value):
                raise AttributeError("read-only")

        mc = ModelConfig.__new__(ModelConfig)
        object.__setattr__(mc, "hf_config", _Frozen())
        object.__setattr__(mc, "hf_text_config", None)

        mc.dtype = torch.float16
        self.assertIs(mc.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
