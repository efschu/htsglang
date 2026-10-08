# [Deprecated] Re-export shim for backward compatibility. Use dsa.dequant_k_cache instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.dequant_k_cache is deprecated; "
    "use flliper.srt.layers.attention.dsa.dequant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.dequant_k_cache import *  # noqa: F401, F403
