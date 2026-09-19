"""fn4i (19.09.): a host-resident target embedding (no .weight) shares modules with the draft."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from types import SimpleNamespace

from sglang.srt.speculative.eagle_worker_v2 import target_shares_vocab_modules


def test_dense_target_hands_over_tensors():
    assert not target_shares_vocab_modules(SimpleNamespace(weight=1), SimpleNamespace(weight=1))


def test_gguf_head_and_host_embedding_share_modules():
    assert target_shares_vocab_modules(SimpleNamespace(qweight=1), SimpleNamespace(weight=1))
    assert target_shares_vocab_modules(SimpleNamespace(weight=1), SimpleNamespace(table=1))
    assert not target_shares_vocab_modules(SimpleNamespace(weight=1), None)
