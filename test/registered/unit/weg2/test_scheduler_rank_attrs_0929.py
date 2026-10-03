"""The Scheduler keeps its ranks in ``self.ps`` (ParallelState), never as
``self.pp_rank`` / ``self.tp_rank`` / ... -- a read of such an attribute
raises AttributeError the first time the line runs. rc12z30x (c0aafd8fc5,
Bootzeit 3) read ``self.pp_rank`` in init_model_worker and killed every
scheduler after graph capture (27B nvfp4form 09:19:59Z); the same class hit
``self.tp_rank`` twice before (see the comments in scheduler.py).

The rule: a ``self.<rank attr>`` read inside scheduler.py is allowed only when
the module also ASSIGNS that attribute on ``self``."""

import ast
import pathlib

import sglang.srt.managers.scheduler as scheduler_mod

RANK_ATTRS = {
    "pp_rank", "tp_rank", "dp_rank", "pp_size", "tp_size", "dp_size",
    "attn_tp_rank", "attn_tp_size", "attn_cp_rank", "attn_dp_rank",
}


def _self_attrs(tree):
    reads, writes = [], set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self" and node.attr in RANK_ATTRS):
            if isinstance(node.ctx, ast.Store):
                writes.add(node.attr)
            else:
                reads.append((node.attr, node.lineno))
    return reads, writes


def test_scheduler_reads_no_unassigned_rank_attr():
    src = pathlib.Path(scheduler_mod.__file__).read_text()
    reads, writes = _self_attrs(ast.parse(src))
    bad = [(a, ln) for a, ln in reads if a not in writes]
    assert not bad, f"self.<rank> read but never assigned (use self.ps.<rank>): {bad}"


def test_the_check_catches_the_z30x_form():
    code = (
        "class S:\n"
        "    def init_model_worker(self):\n"
        "        note(self.pp_rank, self.tp_rank)\n"
    )
    reads, writes = _self_attrs(ast.parse(code))
    assert {a for a, _ in reads} == {"pp_rank", "tp_rank"} and not writes
