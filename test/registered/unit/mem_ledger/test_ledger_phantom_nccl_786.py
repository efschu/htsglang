# SPDX-License-Identifier: Apache-2.0
"""#786: the ledger refused to price NCCL buffers that are never allocated.

Boot 735-standT, 2026-08-20, refused the full per-card reserve with nine
unbounded terms. Three of them read:

    NCCL communicator buffers on <card>: ... NCCL-owned group(s): pp
    (should_build_pynccl(use_pynccl=True, world_size=3, barlink_active=False)
    is True), flip_tp (...), flip_dcp (...)

The same boot log says, four times:

    barlink enabled for group 'pp:0':       requested=bar1, ACHIEVED=bar1
    barlink enabled for group 'flip_tp:0':  requested=bar1, ACHIEVED=bar1
    barlink enabled for group 'flip_dcp:0': requested=bar1, ACHIEVED=bar1
    barlink enabled for group 'world:0':    requested=bar1, ACHIEVED=bar1

Barlink owns every group, so no PyNccl communicator is constructed and those
buffers do not exist. The ledger refused to price memory that is never
allocated, and a refusal is not a warning -- it costs the entire full-demand
path, which is why the corridor verdict has been reporting ``net == raw``.

ROOT CAUSE, an ORDERING defect introduced by #781. The predicate
``parallel_state.should_build_barlink`` (parallel_state.py:489) is
``bool(envs.SGLANG_BARLINK.get()) and world_size > 1``. Before #781 the boot
script set that variable, so it was true by the time the ledger ran. #781 made
``--barlink`` the source of truth and the environment an internal detail this
process publishes from its own argv -- but it published in
``_handle_environment_variables`` near the END of ``__post_init__``, while the
ledger runs in ``_handle_gpu_memory_settings`` well before it. The ledger read
an unset variable.

This is the same shape as the #596 defect already documented inside
``ledger_full_demand_per_gpu``: a value the ledger reads, resolved after the
ledger reads it. The remedy is the same one -- resolve it first.
"""

import inspect
import os
import re

from sglang.srt import server_args as sa_mod
from sglang.srt.mem_ledger import engine as ledger_engine

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _post_init_source() -> str:
    return inspect.getsource(sa_mod.ServerArgs.__post_init__)


def test_the_781_flags_are_published_before_the_ledger_prices_them():
    """THE REGRESSION GUARD, expressed as the ordering it is about.

    Not a mock of the ledger: the defect was never in what the ledger computes,
    it was in WHEN the value it reads becomes true. So the property under test
    is the call order inside ``__post_init__``.
    """
    src = _post_init_source()
    publish = src.find("self._publish_barlink_ownership_env()")
    ledger = src.find("self._handle_gpu_memory_settings(")

    assert publish != -1, "the barlink ownership publish vanished from __post_init__"
    assert ledger != -1, "the ledger-bearing call vanished from __post_init__"
    assert publish < ledger, (
        "the #781 flags must be published BEFORE the VRAM ledger prices the "
        "NCCL term, or should_build_barlink() reads an unset SGLANG_BARLINK "
        "and the ledger refuses over communicator buffers that barlink means "
        "are never allocated (#786)"
    )


def test_uneven_tp_also_runs_after_the_publish():
    """``_handle_uneven_tp`` reaches the ledger too, and runs even earlier."""
    src = _post_init_source()
    publish = src.find("self._publish_barlink_ownership_env()")
    uneven = src.find("self._handle_uneven_tp()")
    assert publish != -1 and uneven != -1
    assert publish < uneven, (
        "_handle_uneven_tp prices its demand from the same ledger, so the "
        "publish has to precede it as well"
    )


def test_the_early_publish_is_narrow_and_idempotent():
    """The hoisted helper must stay small enough to be safe to call twice.

    WHY THIS TEST EXISTS AND WHAT IT CAUGHT. The first version of the fix
    hoisted ``_publish_promoted_781_flags`` itself. That method is 145 lines
    and also resolves debug-cuda-graph and custom-all-reduce, including a
    ``self.debug_cuda_graph = False`` -- so the "idempotent publisher" claim
    in the hoist comment was false, and the change moved far more behaviour
    than the defect warranted. The helper was split out instead.
    """
    src = inspect.getsource(sa_mod.ServerArgs._publish_barlink_ownership_env)
    body = [
        line
        for line in src.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(body) < 40, (
        "the early-published helper has grown; anything hoisted ahead of the "
        "ledger runs before the rest of __post_init__ has resolved, so it "
        "must stay confined to publishing what the ledger reads"
    )
    assert "is not None" in src, (
        "the publish must stay conditional on the flag being set; an "
        "unconditional publish would write a default into the environment "
        "and change behaviour for anyone who has not moved to the flags"
    )
    # It must not mutate self -- a second call would then not be idempotent.
    assert not re.search(r"\bself\.\w+\s*=(?!=)", src), (
        "the helper assigns to a ServerArgs field, so it is no longer "
        "idempotent and cannot safely be called from two places"
    )


def test_the_full_publisher_delegates_rather_than_duplicating():
    """One definition: the late publisher must call the same helper.

    Two copies of ``os.environ["SGLANG_BARLINK"] = ...`` would let the early
    and late values diverge, which is the failure this whole ticket is about.
    """
    src = inspect.getsource(sa_mod.ServerArgs._publish_promoted_781_flags)
    assert "_publish_barlink_ownership_env()" in src
    assert 'os.environ["SGLANG_BARLINK"]' not in src, (
        "the barlink publish is duplicated instead of delegated"
    )


# ---------------------------------------------------------------------------
# The ghost-remedy family (#786, second half).
# ---------------------------------------------------------------------------


def test_no_ledger_refusal_prescribes_a_script_that_does_not_exist():
    """An error message naming a command that is not in the tree is a lie.

    The graph-capture refusal told the operator to run ``ingest-boot-log``
    against a boot log carrying the 'Capture ... begin' lines. No such CLI
    exists anywhere in the repository -- ``grep -rn ingest`` over
    ``mem_ledger/``, ``rigmon/`` and ``scripts/vram_ledger/`` finds only the
    string itself. The one remedy that needed no GPU was unimplemented, which
    is a plausible reason these terms sat unpriced.

    SELECTED BY SHAPE rather than by name, so the next invented remedy is
    caught too: every ``scripts/...py`` path a ledger message prints must
    resolve on disk.
    """
    src = inspect.getsource(ledger_engine)
    referenced = set(re.findall(r"(scripts/[\w/]+\.py)", src))
    assert referenced, "expected the ledger to name at least one real remedy"

    missing = [
        path
        for path in sorted(referenced)
        if not os.path.exists(os.path.join(REPO_ROOT, path))
    ]
    assert not missing, (
        f"ledger refusal messages name script(s) that do not exist: {missing}. "
        "An operator following that instruction gets 'command not found' and "
        "concludes the term cannot be priced."
    )


def test_the_ingest_boot_log_ghost_is_gone_specifically():
    """The concrete instance, pinned so it cannot come back by copy-paste."""
    src = inspect.getsource(ledger_engine)
    assert "ingest-boot-log" not in src, (
        "the graph-capture refusal again prescribes `ingest-boot-log`, which "
        "is not implemented anywhere in this tree (#786)"
    )
