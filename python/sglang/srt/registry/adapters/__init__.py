# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The three lane adapters of #333 §7.6.

Importing this package registers them. The registry itself imports nothing
from here directly -- it goes through
:func:`sglang.srt.registry.adapter.build_adapter`, which is what keeps the
arbiter free of any reference to a model architecture.
"""
