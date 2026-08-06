"""Pin-test for #616c: fused accept-broadcast (single collective, not three).

Bug summary: Three tensors (predict, accept_index, num_correct_drafts) were
broadcast as three separate NCCL collectives.  predict and accept_index share
the same wire size (both int32, both bs * draft_token_num elements), so a
pairing shift of exactly one silently delivered predict's payload into
accept_index's buffer.  The GPU coredump showed accept_index holding predict's
values (zero-init, not -1 fill), confirming the swap.

The fix packs all three into ONE collective via pack_accept_payload /
unpack_accept_payload around a single capture_safe_tp_broadcast call.

This test parses the source of eagle_sample at the AST level (no GPU, no
distributed runtime) and asserts the fused-broadcast invariant holds.  A
revert to three separate broadcasts will fail these assertions.
"""

import ast
import inspect
import textwrap

import pytest


@pytest.fixture()
def _eagle_sample_tree():
    """Return the AST of eagle_sample without importing CUDA deps."""
    import sglang.srt.speculative.eagle_utils as eagle_utils

    src = inspect.getsource(eagle_utils.eagle_sample)
    cleaned = textwrap.dedent(src)
    return ast.parse(cleaned)


def _find_broadcast_calls(tree: ast.Module) -> list[ast.Call]:
    """Return all capture_safe_tp_broadcast Call nodes in the function body."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "capture_safe_tp_broadcast"
    ]


def _find_func_calls(tree: ast.Module, name: str) -> list[ast.Call]:
    """Return all Call nodes that call `name` by simple Name lookup."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _is_one_element_tuple(call: ast.Call) -> bool:
    """Check if a broadcast call's second arg is a 1-element tuple."""
    if len(call.args) < 2:
        return False
    arg = call.args[1]
    if not isinstance(arg, ast.Tuple):
        return False
    return len(arg.elts) == 1


class TestAcceptBroadcastFusedPin:
    """Source-level pins -- every assert names WHY it matters."""

    def test_at_least_one_fused_broadcast(self, _eagle_sample_tree: ast.Module):
        """At least one capture_safe_tp_broadcast in eagle_sample must use a
        1-element tuple (the fused payload).

        Why pinned: the original code had separate broadcasts per tensor.
        Two same-size collectives (predict and accept_index) could silently
        swap payloads via a pairing-shift -- accept_index received predict's
        zero-init data instead of its -1 fill, proven by a GPU coredump.
        A single fused broadcast eliminates that pairing entirely.  If all
        broadcast calls use multi-element tuples (the old unfused pattern),
        the #616c bug is live again.
        """
        calls = _find_broadcast_calls(_eagle_sample_tree)
        assert len(calls) >= 1, (
            "capture_safe_tp_broadcast must appear at least once in eagle_sample. "
            "Zero calls means the accept-sync was removed entirely."
        )
        fused = [c for c in calls if _is_one_element_tuple(c)]
        assert len(fused) >= 1, (
            f"Expected at least 1 fused (1-element tuple) broadcast, found "
            f"{len(fused)} of {len(calls)} total.  The fused payload pattern "
            f"(_packed,) is the core fix for #616c: without it, predict and "
            f"accept_index silently swap via same-size collective pairing."
        )

    def test_all_broadcasts_are_fused(self, _eagle_sample_tree: ast.Module):
        """Every capture_safe_tp_broadcast call must use a 1-element tuple.

        Why pinned: if even ONE broadcast call uses a multi-element tuple
        (e.g. (predict, accept_index, num_correct_drafts)), that call site
        restores the #616c pairing-shift bug.  All call sites must carry the
        fused topology uniformly.  A future revert of a single call site
        (while keeping others fused) would be hard to detect without this
        blanket check.
        """
        calls = _find_broadcast_calls(_eagle_sample_tree)
        unfused = [c for c in calls if not _is_one_element_tuple(c)]
        assert len(unfused) == 0, (
            f"{len(unfused)} of {len(calls)} broadcast call(s) use a non-fused "
            f"(multi-element tuple or non-tuple) payload.  Any multi-element "
            f"tuple restores the #616c bug: two same-size collectives can swap. "
            f"All broadcasts must pass a single fused buffer."
        )

    def test_pack_and_unpack_called_at_least_once(self, _eagle_sample_tree: ast.Module):
        """pack_accept_payload and unpack_accept_payload must each be called at
        least once in eagle_sample.

        Why pinned: the pack/unpack pair is the mechanism that fuses the three
        tensors.  Zero calls means the pack/unpack machinery was removed
        (revert to separate broadcasts = silent same-size swap).  The count
        is >= 1 because there are multiple code paths (main verify and
        weightless-receive), each with its own pack-broadcast-unpack triplet.
        The important invariant is that the fusion machinery exists at all.
        """
        for func_name in ("pack_accept_payload", "unpack_accept_payload"):
            calls = _find_func_calls(_eagle_sample_tree, func_name)
            assert len(calls) >= 1, (
                f"Expected at least 1 {func_name}() call in eagle_sample, "
                f"found {len(calls)}.  "
                f"{func_name} is the fusion mechanism for #616c: removing it "
                f"reverts to separate broadcasts (silent same-size swap)."
            )
