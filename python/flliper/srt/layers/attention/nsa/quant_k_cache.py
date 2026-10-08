# [Deprecated] Re-export shim for backward compatibility. Use dsa.quant_k_cache instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.quant_k_cache is deprecated; "
    "use flliper.srt.layers.attention.dsa.quant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.quant_k_cache import *  # noqa: F401, F403
