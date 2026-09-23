"""barlink control words are allocated untagged (Wake-Parallel, xsn317/318)."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def test_control_words_are_allocated_outside_the_memory_saver_tags():
    import importlib
    b = importlib.import_module("sglang.srt.distributed.device_communicators.barlink_bar1")
    src = open(b.__file__).read()
    i = src.index("with _untagged_alloc():")
    blk = src[i:i + 300]
    assert "self._round_dev = torch.zeros(3" in blk and "self._ctl_dev = torch.zeros(2" in blk
    # the helper is a real context manager also without the saver
    with b._untagged_alloc():
        pass
