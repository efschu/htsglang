"""Deprecated import path for ``flliper.benchmark.one_batch_server``.

``python -m flliper.bench_one_batch_server`` and
``from flliper.bench_one_batch_server import ...`` still work, but the
implementation now lives in ``flliper.benchmark.one_batch_server``.
Update references to the new path.
"""

import warnings

from flliper.benchmark.one_batch_server import *  # noqa: F401,F403
from flliper.benchmark.one_batch_server import cli_main

warnings.warn(
    "`flliper.bench_one_batch_server` is deprecated and will be removed in a "
    "future release; use `flliper.benchmark.one_batch_server` instead "
    "(e.g. `python -m flliper.benchmark.one_batch_server`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
