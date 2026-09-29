"""Deprecated import path for ``flliper.benchmark.offline_throughput``.

``python -m flliper.bench_offline_throughput`` and
``from flliper.bench_offline_throughput import ...`` still work, but the
implementation now lives in ``flliper.benchmark.offline_throughput``.
Update references to the new path.
"""

import warnings

from flliper.benchmark.offline_throughput import *  # noqa: F401,F403
from flliper.benchmark.offline_throughput import cli_main

warnings.warn(
    "`flliper.bench_offline_throughput` is deprecated and will be removed in a "
    "future release; use `flliper.benchmark.offline_throughput` instead "
    "(e.g. `python -m flliper.benchmark.offline_throughput`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
