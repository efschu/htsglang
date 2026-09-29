"""Deprecated import path for ``flliper.benchmark.serving``.

``python -m flliper.bench_serving`` and ``from flliper.bench_serving import ...``
still work, but the implementation now lives in ``flliper.benchmark.serving``.
Update references to the new path.
"""

import warnings

from flliper.benchmark.serving import *  # noqa: F401,F403
from flliper.benchmark.serving import cli_main

warnings.warn(
    "`flliper.bench_serving` is deprecated and will be removed in a future "
    "release; use `flliper.benchmark.serving` instead "
    "(e.g. `python -m flliper.benchmark.serving`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
