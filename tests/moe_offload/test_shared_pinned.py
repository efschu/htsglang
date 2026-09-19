"""Task #47 step 1: a tmpfs-backed MAP_SHARED buffer is seen byte-identically
by a second process, sizes are enforced, and teardown removes the store."""
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.layers.moe import shared_pinned as sp


def test_two_processes_share_the_same_bytes(tmp_path):
    path = str(tmp_path / "L3-w13")
    t, created = sp.shared_pinned_empty(path, (4, 8), torch.bfloat16, register=False)
    assert created and t.shape == (4, 8) and t.dtype == torch.bfloat16
    t.copy_(torch.arange(32, dtype=torch.bfloat16).view(4, 8))
    # a second, independent process attaches and reads what we wrote
    code = f"""
import torch, sys
sys.path.insert(0, {repr(os.path.join(os.path.dirname(sp.__file__), '..', '..', '..', '..'))})
from sglang.srt.layers.moe import shared_pinned as sp
t, created = sp.shared_pinned_empty({path!r}, (4, 8), torch.bfloat16, register=False)
assert not created
print(int(t.float().sum().item()), t[3, 7].item())
t[0, 0] = 99.0
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split() == [str(sum(range(32))), "31.0"]
    assert t[0, 0].item() == 99.0  # the other process's write is visible here


def test_a_different_size_is_refused(tmp_path):
    path = str(tmp_path / "L0-w2")
    sp.shared_pinned_empty(path, (2, 2), torch.float32, register=False)
    with pytest.raises(ValueError):
        sp.shared_pinned_empty(path, (2, 3), torch.float32, register=False)


def test_unlink_store_removes_the_epoch(tmp_path):
    d = tmp_path / "nf-experts-1"
    for i in range(3):
        sp.shared_pinned_empty(str(d / f"L{i}-w13"), (2,), torch.uint8, register=False)
    assert sp.unlink_store(str(d)) == 3 and not d.exists()
    assert sp.unlink_store(str(d)) == 0
