"""upstream 44fd17b696 (the ``prepare_mamba_track_for_verify`` hunk) on the
fork's DFLASH verify.

Hermetic, CPU. The DFLASH verify commit (``_update_target_mamba_state_after_verify``)
writes a tracked Mamba state only ``if batch.mamba_track_indices is not None``.
Spec batches skip the track refresh in ``prepare_for_decode`` and
``ScheduleBatch.filter_batch`` / ``merge_batch`` null the three track fields, so
without a rebuild right before the TARGET_VERIFY forward the indices are None:
the GPU writes no track while the scheduler (``_mamba_prefix_cache_update``)
still flips the ping-pong slot and records ``mamba_last_track_seqlen`` -- a
silent mis-anchor in the radix tree. EAGLE (eagle_utils) and NGRAM
(ngram_worker) call ``spec_utils.prepare_mamba_track_for_verify``; DFLASH did
not.

Pins:
1. behaviour: after a filter/merge-style nulling the helper rebuilds the track
   indices from the requests and leaves mask/seqlens None, and the DFLASH
   commit then hands exactly those indices plus the crossing steps to the
   backend (without the rebuild it hands None / no steps);
2. wiring (AST; the verify function is too large to drive hermetically): the
   DFLASH verify calls the helper before ``prepare_for_verify`` (which
   snapshots the track fields into the verify ForwardBatch) and before the
   Mamba commit, in the same function;
3. mutant: with the call removed from the source the wiring check is red.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.runtime_context as rc
from sglang.srt.speculative import dflash_worker_v2 as dfw
from sglang.srt.speculative import spec_utils

HELPER = "prepare_mamba_track_for_verify"


class _Backend:
    def __init__(self):
        self.calls = []

    def update_mamba_state_after_mtp_verify(self, **kw):
        self.calls.append(kw)


def _fake_self(backend):
    return SimpleNamespace(
        _need_mamba_verify_commit=True,
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend, model=object())
        ),
    )


@pytest.fixture
def extra_buffer_on(monkeypatch):
    sa = SimpleNamespace(
        mamba_cache_chunk_size=64,
        mamba_track_interval=256,
        enable_mamba_extra_buffer=lambda: True,
    )
    monkeypatch.setattr(rc, "get_server_args", lambda: sa)
    monkeypatch.setattr(spec_utils, "get_server_args", lambda: sa)

    # set_mamba_track_indices_from_reqs pins host memory (CUDA only); the CPU
    # test drops the flag, nothing else about the helper changes.
    real_tensor = torch.tensor

    def _tensor(*a, **kw):
        kw.pop("pin_memory", None)
        return real_tensor(*a, **kw)

    monkeypatch.setattr(torch, "tensor", _tensor)
    return sa


def _batch(next_idx, pre):
    """A DFLASH running batch right after filter/merge: track fields None."""
    bs = len(next_idx)
    # req_pool_indices[i] = i; ping-pong buffer row i = [100 + 10*i, 101 + 10*i]
    mapping = torch.tensor([[100 + 10 * i, 101 + 10 * i] for i in range(bs)])
    return SimpleNamespace(
        reqs=[SimpleNamespace(mamba_next_track_idx=n) for n in next_idx],
        req_pool_indices=torch.arange(bs),
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        seq_lens=torch.tensor(pre, dtype=torch.int64),
        tree_cache=SimpleNamespace(page_size=1),
    )


def _commit(batch, pre, commit):
    backend = _Backend()
    pre_t = torch.tensor(pre, dtype=torch.int64)
    commit_t = torch.tensor(commit, dtype=torch.int32)
    dfw.DFlashWorkerV2._update_target_mamba_state_after_verify(
        _fake_self(backend),
        batch=batch,
        seq_lens_pre_verify=pre_t,
        seq_lens_post_verify=pre_t + commit_t.to(pre_t.dtype),
        commit_lens=commit_t,
    )
    assert len(backend.calls) == 1
    return backend.calls[0]


def test_without_rebuild_the_commit_tracks_nothing(extra_buffer_on):
    """The base behaviour (red on the pre-fix wiring): None indices after
    filter/merge -> the backend gets no destination and no steps."""
    b = _batch([0, 1], [250, 100])
    kw = _commit(b, [250, 100], [8, 3])
    assert kw["mamba_track_indices"] is None
    assert kw["mamba_steps_to_track"] is None


def test_rebuild_then_commit_tracks_the_crossing(extra_buffer_on):
    b = _batch([0, 1], [250, 100])
    # stale extend-time leftovers must not survive the rebuild
    b.mamba_track_mask = torch.tensor([True, True])
    b.mamba_track_seqlens = torch.tensor([256, 256])
    spec_utils.prepare_mamba_track_for_verify(b)
    assert b.mamba_track_indices.tolist() == [100, 111]  # row i, next_idx i
    assert b.mamba_track_mask is None
    assert b.mamba_track_seqlens is None
    kw = _commit(b, [250, 100], [8, 3])
    assert kw["mamba_track_indices"].tolist() == [100, 111]
    # req0 250 -> 258 crosses 256 (state after step 5); req1 100 -> 103 none
    assert kw["mamba_steps_to_track"].tolist() == [5, -1]


def test_rebuild_is_inert_without_extra_buffer(monkeypatch):
    sa = SimpleNamespace(enable_mamba_extra_buffer=lambda: False)
    monkeypatch.setattr(spec_utils, "get_server_args", lambda: sa)
    b = _batch([0], [10])
    spec_utils.prepare_mamba_track_for_verify(b)
    assert b.mamba_track_indices is None


# --- wiring -----------------------------------------------------------------


def _verify_fn(src):
    tree = ast.parse(textwrap.dedent(src))
    fns = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "forward_batch_generation"
    ]
    assert len(fns) == 1
    return fns[0]


def _calls(fn, name):
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if fname == name:
                out.append(n)
    return out


def _check_wiring(src):
    fn = _verify_fn(src)
    rebuilds = _calls(fn, HELPER)
    assert rebuilds, "DFLASH verify never rebuilds the mamba track indices"
    first = min(c.lineno for c in rebuilds)
    prep = _calls(fn, "prepare_for_verify")
    commit = _calls(fn, "_update_target_mamba_state_after_verify")
    assert prep and commit
    assert first < min(c.lineno for c in prep), (
        "rebuild must precede prepare_for_verify (it snapshots the fields)"
    )
    assert first < min(c.lineno for c in commit)


def test_verify_rebuilds_track_indices_before_the_verify_forward():
    _check_wiring(inspect.getsource(dfw.DFlashWorkerV2))


def test_mutant_call_removed_is_red():
    src = inspect.getsource(dfw.DFlashWorkerV2)
    assert src.count(f"{HELPER}(batch)") == 1
    mutant = src.replace(f"{HELPER}(batch)", "pass")
    with pytest.raises(AssertionError, match="never rebuilds"):
        _check_wiring(mutant)


def test_mutant_call_after_prepare_for_verify_is_red():
    src = inspect.getsource(dfw.DFlashWorkerV2)
    anchor = "batch.seq_lens_cpu = seq_lens_cpu_backup"
    assert src.count(anchor) == 1
    late = src.replace(f"{HELPER}(batch)", "pass").replace(
        anchor, f"{anchor}\n        {HELPER}(batch)"
    )
    with pytest.raises(AssertionError, match="must precede prepare_for_verify"):
        _check_wiring(late)
