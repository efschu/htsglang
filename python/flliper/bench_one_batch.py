"""Deprecated import path for ``flliper.benchmark.one_batch``.

``python -m flliper.bench_one_batch`` and ``from flliper.bench_one_batch import ...``
still work, but the implementation now lives in ``flliper.benchmark.one_batch``.
Update references to the new path.
"""

import warnings

from flliper.benchmark.one_batch import *  # noqa: F401,F403
from flliper.benchmark.one_batch import cli_main

warnings.warn(
    "`flliper.bench_one_batch` is deprecated and will be removed in a future "
    "release; use `flliper.benchmark.one_batch` instead "
    "(e.g. `python -m flliper.benchmark.one_batch`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()
