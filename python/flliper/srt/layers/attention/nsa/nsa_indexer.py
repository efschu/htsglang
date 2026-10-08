# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.nsa_indexer is deprecated; "
    "use flliper.srt.layers.attention.dsa.dsa_indexer instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.dsa_indexer import *  # noqa: F401, F403
