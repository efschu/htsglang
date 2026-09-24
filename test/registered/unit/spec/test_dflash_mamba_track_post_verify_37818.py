"""upstream #37818 (+ DFlash hunk of #35412) on the fork's DFLASH verify.

Hermetic, CPU. Pins two things:

1. ``DFlashWorkerV2._update_target_mamba_state_after_verify`` measures the
   track-boundary crossing against the POST-verify lengths. At the call site
   ``batch.seq_lens`` is still the pre-verify value (the scheduler advances it
   from ``new_seq_lens`` later), so the old comparison ``pre // I !=
   batch.seq_lens // I`` was always False and DFLASH decode never wrote a
   tracked Mamba state -- while the scheduler still flipped the ping-pong slot
   and recorded ``mamba_last_track_seqlen`` for the radix insert.
2. The verify call site hands ``seq_lens_post_verify=new_seq_lens`` and has
   ``new_seq_lens`` bound before that call on every accept path (AST wiring
   check -- the verify function is too large to drive hermetically).

The grid is the fork's ``spec_utils.mamba_track_grid`` (lcm of tree page,
mamba chunk and track interval); under the weighted uneven DCP the tree page
stays natural, so it equals the raw interval on the 27B D group.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.runtime_context as rc
from sglang.srt.speculative import dflash_worker_v2 as dfw


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
        server_args=SimpleNamespace(mamba_track_interval=256),
    )


@pytest.fixture
def grid_args(monkeypatch):
    sa = SimpleNamespace(mamba_cache_chunk_size=64, mamba_track_interval=256)
    monkeypatch.setattr(rc, "get_server_args", lambda: sa)
    return sa


def _run(pre, commit, *, page=1):
    backend = _Backend()
    pre_t = torch.tensor(pre, dtype=torch.int64)
    commit_t = torch.tensor(commit, dtype=torch.int32)
    batch = SimpleNamespace(
        mamba_track_indices=torch.tensor([7 + i for i in range(len(pre))]),
        # the call site's truth: batch.seq_lens is STILL the pre-verify value
        seq_lens=pre_t.clone(),
        tree_cache=SimpleNamespace(page_size=page),
    )
    dfw.DFlashWorkerV2._update_target_mamba_state_after_verify(
        _fake_self(backend),
        batch=batch,
        seq_lens_pre_verify=pre_t,
        seq_lens_post_verify=pre_t + commit_t.to(pre_t.dtype),
        commit_lens=commit_t,
    )
    assert len(backend.calls) == 1
    return backend.calls[0]


def test_crossing_inside_accept_run_is_tracked(grid_args):
    # req0: 250 -> 258 crosses 256, the state after token 5 of the block
    # covers exactly 256 tokens; req1: 100 -> 103 crosses nothing.
    kw = _run([250, 100], [8, 3])
    assert kw["last_correct_step_indices"].tolist() == [7, 2]
    assert kw["mamba_steps_to_track"].tolist() == [5, -1]
    assert kw["mamba_track_indices"].tolist() == [7, 8]


def test_crossing_on_last_committed_token(grid_args):
    # 248 -> 256: the boundary is the last committed state (index 7)
    kw = _run([248], [8])
    assert kw["mamba_steps_to_track"].tolist() == [7]


def test_no_crossing_tracks_nothing(grid_args):
    kw = _run([256], [8])  # 256 -> 264 stays inside [256, 512)
    assert kw["mamba_steps_to_track"].tolist() == [-1]


def test_grid_follows_the_tree_page(grid_args):
    # a coarser tree page (stock even-DCP widening) moves the grid to 512:
    # 250 -> 258 no longer lands on a radix node, 510 -> 514 does.
    kw = _run([250, 510], [8, 4], page=512)
    assert kw["mamba_steps_to_track"].tolist() == [-1, 1]


def _verify_function_source():
    src = inspect.getsource(dfw.DFlashWorkerV2)
    tree = ast.parse(textwrap.dedent(src))
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_update_target_mamba_state_after_verify"
            ):
                hits.append((fn, node))
    return hits


def test_call_site_hands_post_verify_lengths():
    hits = _verify_function_source()
    assert hits, "verify path no longer calls the Mamba commit"
    for fn, call in hits:
        kws = {k.arg: k.value for k in call.keywords}
        assert "seq_lens_post_verify" in kws, (
            f"{fn.name}: Mamba commit called without seq_lens_post_verify"
        )
        val = kws["seq_lens_post_verify"]
        assert isinstance(val, ast.Name) and val.id == "new_seq_lens"
        # new_seq_lens must be (re)bound from prefix_lens + commit_lens before
        # the call on the eager/sampling accept paths (Triton binds it itself).
        bound_before = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "new_seq_lens" for t in n.targets)
            and n.lineno < call.lineno
            and "commit_lens" in ast.unparse(n.value)
        ]
        assert bound_before, f"{fn.name}: new_seq_lens not derived before the call"
