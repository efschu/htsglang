"""#965 -- one match reading, EIGHT fields, and an invalidator that cleared three.

THE DEFECT THIS PINS, and the reason it is a class and not a bug.

`Req.init_next_round_input` reads the radix tree ONCE and unpacks that single
`match_result` into EIGHT co-derived attributes in one tuple assignment
(`schedule_batch.py`, the `(self.prefix_indices, self.last_node,
self.last_host_node, self.best_match_node, self.host_hit_length,
self.swa_host_hit_length, self.mamba_host_hit_length,
self.mamba_branching_seqlen) = (...)` statement). They are ONE reading of ONE
geometry wearing eight names.

`Req.truncate_prefix_to` moves the prefix those eight were derived from. Its own
docstring says exactly why that matters -- "the geometry was DERIVED from the
prefix that just moved, so it is now a stale reading rather than a report" --
and that sentence is true, word for word, of every field in the tuple. It wrote
`prefix_indices`, and (over two separate windows, each paid for by a boot)
`cache_protected_len` (#930) and `extend_range` (#958). The rest were left
standing.

THE PATH HAS NOTHING IN BETWEEN. `scheduler.py` calls
`req.init_next_round_input(...)`, which derives all eight; a dozen lines later
both arms of the `pp_size > 1` fork call `req.truncate_prefix_to(told)`; and
then `adder.add_one_req(req)` reads the stale ones. No `match_prefix`, no
re-derivation.

WHAT THE STALE READINGS DO, at the reader (`schedule_policy.PrefillAdder.
add_one_req`):

  * `real_input_tokens = cand_extend_input_len - req.host_hit_length` subtracts
    a host hit that is no longer part of this prefix, under-counting the input
    against the budget gates.
  * `req.needs_host_load_back()` is still True, so `init_load_back` runs with a
    stale `best_match_node`/`host_hit_length` and does
    `req.prefix_indices = torch.cat([req.prefix_indices, new_indices])`. The
    result covers `[0, told)` and then `[L_dev, L_dev+H)` with a HOLE between
    them, while `prepare_for_extend` sizes the cross-stage tensor off
    `len(prefix_indices)` as though it were contiguous. That is the
    silently-wrong-context class `phase_flip_runtime` names as the one that
    must never ship.
  * `req.cache_protected_len = prefix_len` on that branch re-raises precisely
    the value the truncation lowered nine lines earlier -- undoing #930.

WHY A GROUP CONTRACT AND NOT A SEVENTH ONE-FIELD FIX. Two of the eight have
already been fixed one at a time, each discovered by a boot, each costing a
window. Fixing a third the same way buys the same lesson a third time. The
invalidation is therefore expressed as ONE contract over the whole tuple, and
`test_every_co_derived_field_is_accounted_for` FAILS when a ninth field joins
the group -- so the next person to widen the match result is told, by a red
test at desk time, that they must decide its invalidation.

`last_node` IS DELIBERATELY NOT NULLED, and that is the one asymmetry in the
contract. It is not a reading, it is a RESOURCE HANDLE: `cache_unfinished_req`
took `inc_lock_ref` on it and `req.last_node` is the only remaining reference
to that ref. Nulling it here would leak the lock ref -- the node becomes
permanently unevictable, which is a separate open defect on the void path and
must not be manufactured here as well. It is listed in the contract as
`HANDLE`, so the invariant test still accounts for it and a future reader
cannot mistake the omission for an oversight.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest


def _req_class():
    from sglang.srt.managers.schedule_batch import Req

    return Req


class _FakeNode:
    pass


def _req_with_a_full_match(told_from: int = 1250):
    """A Req carrying one complete match reading, without a tree or a GPU."""
    Req = _req_class()
    req = Req.__new__(Req)
    req.prefix_indices = list(range(4096))
    req.last_node = _FakeNode()
    req.last_host_node = _FakeNode()
    req.best_match_node = _FakeNode()
    req.host_hit_length = 512
    req.swa_host_hit_length = 256
    req.mamba_host_hit_length = 128
    req.mamba_branching_seqlen = 3000
    req.cache_protected_len = 4096
    req.extend_range = None
    return req


class TestTheGroupIsInvalidatedAsAGroup:
    def test_host_load_back_is_off_until_a_fresh_match(self):
        """THE CORE READER. `add_one_req` gates `init_load_back` on this, and
        a True here after a truncation is what concatenates a discontiguous
        prefix onto a clamped one."""
        req = _req_with_a_full_match()
        assert req.needs_host_load_back() is True
        req.truncate_prefix_to(1250)
        assert req.needs_host_load_back() is False

    @pytest.mark.parametrize(
        "field, stale_is",
        [
            ("host_hit_length", 0),
            ("swa_host_hit_length", 0),
            ("mamba_host_hit_length", 0),
            ("best_match_node", None),
            ("last_host_node", None),
            ("mamba_branching_seqlen", None),
        ],
    )
    def test_each_derived_reading_is_cleared(self, field, stale_is):
        """One case per field, so a partial fix names which field it missed
        instead of failing as one opaque assertion."""
        req = _req_with_a_full_match()
        req.truncate_prefix_to(1250)
        assert getattr(req, field) == stale_is, (
            f"{field} was derived from the prefix that just moved; leaving it "
            f"makes it a stale reading that add_one_req consumes as a report"
        )

    def test_the_prefix_itself_still_truncates(self):
        req = _req_with_a_full_match()
        req.truncate_prefix_to(1250)
        assert len(req.prefix_indices) == 1250
        assert req.cache_protected_len <= 1250

    def test_the_lock_ref_handle_is_not_dropped(self):
        """DANGER DIRECTION. `last_node` holds an outstanding `inc_lock_ref`;
        nulling it here would leak the ref and make the node permanently
        unevictable. It is invalidated by RELEASE, not by assignment, and that
        release is not this method's job."""
        req = _req_with_a_full_match()
        node = req.last_node
        req.truncate_prefix_to(1250)
        assert req.last_node is node

    def test_a_no_op_truncation_leaves_a_valid_reading_valid(self):
        """DANGER DIRECTION, and it is the invalidator's own documented rule:
        "A no-op truncation leaves a valid geometry valid -- clearing it there
        would void healthy passes for nothing." If the prefix did not move, the
        readings were not derived against anything that changed, and clearing
        them would throw away a host hit every rank still holds."""
        req = _req_with_a_full_match()
        req.prefix_indices = list(range(1250))
        req.truncate_prefix_to(1250)
        assert req.needs_host_load_back() is True
        assert req.host_hit_length == 512
        assert req.best_match_node is not None


class TestTheContractCoversTheWholeTuple:
    """THE RATCHET. This is the part that stops the next field being paid for
    by another boot."""

    #: Every attribute assigned from a single `match_result` reading, and what
    #: `truncate_prefix_to` owes it. Adding a field to the producer without
    #: adding it here fails `test_every_co_derived_field_is_accounted_for`.
    CO_DERIVED = {
        "prefix_indices": "TRUNCATED",
        "last_node": "HANDLE",  # released, never nulled -- see module docstring
        "last_host_node": "CLEARED",
        "best_match_node": "CLEARED",
        "host_hit_length": "CLEARED",
        "swa_host_hit_length": "CLEARED",
        "mamba_host_hit_length": "CLEARED",
        "mamba_branching_seqlen": "CLEARED",
    }

    def _tuple_targets(self, path: pathlib.Path, func_names):
        """Attribute names assigned by a tuple-unpack of a `match_result`.

        Read from the AST rather than by grep: a grep for the field names would
        match the readers too, and the whole point is to detect a field that
        nobody has thought about yet -- which by definition has no name I could
        have written into a pattern.
        """
        tree = ast.parse(path.read_text())
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
                continue
            src = ast.unparse(node.value)
            if "match_result." not in src:
                continue
            for elt in node.targets[0].elts:
                if isinstance(elt, ast.Attribute):
                    found.add(elt.attr)
        return found

    def _schedule_batch_path(self):
        from sglang.srt.managers import schedule_batch

        return pathlib.Path(schedule_batch.__file__)

    def _schedule_policy_path(self):
        from sglang.srt.managers import schedule_policy

        return pathlib.Path(schedule_policy.__file__)

    def test_every_co_derived_field_is_accounted_for(self):
        """A NINTH FIELD MUST FAIL HERE, at desk time.

        Two of these eight were fixed one at a time, each found by a boot
        (#930 `cache_protected_len`, #958 `extend_range`). This is the test
        that makes the third one free.
        """
        found = set()
        found |= self._tuple_targets(self._schedule_batch_path(), None)
        found |= self._tuple_targets(self._schedule_policy_path(), None)

        assert found == set(self.CO_DERIVED), (
            "The set of attributes derived from ONE match_result reading has "
            "changed.\n"
            f"  in the code but not in the contract: {sorted(found - set(self.CO_DERIVED))}\n"
            f"  in the contract but not in the code: {sorted(set(self.CO_DERIVED) - found)}\n"
            "Every one of these is invalidated the moment truncate_prefix_to "
            "moves the prefix they were read against. Decide what the new "
            "field owes -- TRUNCATED, CLEARED, or HANDLE (released, never "
            "nulled) -- implement it in truncate_prefix_to, and record it "
            "here. Do NOT simply add the name."
        )

    def test_the_invalidator_mentions_every_cleared_field(self):
        """Source-level companion to the behavioural cases above: a field the
        contract calls CLEARED must actually be named in the invalidator, so a
        behavioural test that passes for an unrelated reason cannot hide a
        missing line."""
        Req = _req_class()
        src = inspect.getsource(Req.truncate_prefix_to)
        missing = [
            name
            for name, duty in self.CO_DERIVED.items()
            if duty == "CLEARED" and name not in src
        ]
        assert not missing, f"truncate_prefix_to never names: {missing}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
