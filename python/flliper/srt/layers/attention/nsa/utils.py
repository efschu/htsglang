# [Deprecated] Re-export shim for backward compatibility. Use dsa.utils instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.utils is deprecated; "
    "use flliper.srt.layers.attention.dsa.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.utils import *  # noqa: F401, F403
