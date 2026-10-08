# fLLiper public APIs

# Rename transition (RENAME_PLAN 8.7 step 2): canonical environment spelling BEFORE anything below
# reads it. See _compat_boot.py. (The rig-state directory is linked by the server entry points, not
# at import: an import must not write into $HOME.)
from ._compat_boot import bridge_environ as _bridge_environ

_bridge_environ()
del _bridge_environ

# Install stubs early for platforms where certain dependencies are unavailable
# (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
# get_device_properties).  This must run before any downstream imports.
import platform as _platform
import sys as _sys

if _sys.platform == "darwin" and _platform.machine() == "arm64":
    try:
        import torch as _torch

        if _torch.backends.mps.is_available():
            from flliper._triton_stub import install as _install_triton_stub

            _install_triton_stub()
            del _install_triton_stub

            from flliper._mps_stub import install as _install_mps_stub

            _install_mps_stub()
            del _install_mps_stub
        del _torch
    except ImportError:
        pass
del _platform
del _sys

# #237 root ticket: ARM the transformers compatibility patches instead of
# applying them eagerly. Applying them meant importing transformers here, in
# every process -- and transformers reaches torch._dynamo, which imports
# triton. Arming installs a post-import hook, so the patches land inside
# transformers' own import (before any caller can use it) and a process that
# never imports transformers never pays for it.
from flliper.srt.utils.hf_transformers_patches import arm as _arm_hf_patches

_arm_hf_patches()
del _arm_hf_patches

# Frontend Language APIs
from flliper.global_config import global_config
from flliper.lang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from flliper.lang.backend.runtime_endpoint import RuntimeEndpoint
from flliper.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# Lazy import some libraries
from flliper.utils import LazyImport
from flliper.version import __version__

Anthropic = LazyImport("flliper.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("flliper.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("flliper.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("flliper.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("flliper.lang.backend.vertexai", "VertexAI")

# Runtime Engine APIs
ServerArgs = LazyImport("flliper.srt.server_args", "ServerArgs")
Engine = LazyImport("flliper.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "Crusoe",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
