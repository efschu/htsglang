# [Deprecated] nsa_backend.py is a thin re-export shim for backward compatibility.
# Use dsa_backend.py instead. This file will be removed in a future release.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa_backend is deprecated; "
    "use flliper.srt.layers.attention.dsa_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa_backend import *  # noqa: F401, F403
from flliper.srt.layers.attention.dsa_backend import (  # noqa: F401
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAFlashMLAMetadata,
    DSAIndexerMetadata,
    DSAMetadata,
    NativeSparseAttnBackend,
    NativeSparseAttnMultiStepBackend,
    NSAFlashMLAMetadata,
    NSAIndexerMetadata,
    NSAMetadata,
)
