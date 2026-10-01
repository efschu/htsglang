"""Dual-model co-boot: the host-side namespace of two stacks (pure).

Design addendum 8.2: each model runs in its own container (own netns, pid ns,
/dev/shm, memory cgroup), which removes every in-stack collision. What both
containers still share is host-side: the published port, the container name
(cleanup traps match by name prefix -- a d2 cleanup once stopped a foreign
boot), the bind-mounted state/holder/evidence/store directories, and the host
RAM the two memory caps are carved from. ``check_coboot`` names every such
collision; an empty list is the only green.
"""
import dataclasses

import pytest

from sglang.srt.weg2.dual_coboot import (
    CobootRefused,
    StackSpec,
    check_coboot,
    default_specs,
    require_coboot,
)

GIB = 1 << 30


def test_default_specs_do_not_collide():
    a, b = default_specs(acc_root="/acc", tag="t1")
    assert check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB) == []
    assert a.host_port == 30030 and b.host_port != 30030


def test_same_port_is_named():
    a, b = default_specs(acc_root="/acc", tag="t1")
    b = dataclasses.replace(b, host_port=a.host_port)
    errs = check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert any("host_port" in e for e in errs)


def test_name_prefix_overlap_is_named_both_directions():
    a, b = default_specs(acc_root="/acc", tag="t1")
    a = dataclasses.replace(a, container_name="htsglang-acc-27b", cleanup_prefix="htsglang-acc-")
    b = dataclasses.replace(b, container_name="htsglang-acc-nf-x", cleanup_prefix="htsglang-acc-nf-")
    errs = check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert any("cleanup_prefix" in e and "htsglang-acc-nf-x" in e for e in errs)


def test_shared_or_nested_directories_are_named():
    a, b = default_specs(acc_root="/acc", tag="t1")
    b = dataclasses.replace(b, arb_dir=a.arb_dir)
    errs = check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert any("arb_dir" in e for e in errs)
    b2 = dataclasses.replace(default_specs(acc_root="/acc", tag="t1")[1],
                             store_dir=a.store_dir.rstrip("/") + "/sub")
    errs = check_coboot([a, b2], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert any("store_dir" in e for e in errs)


def test_memory_budget_is_one_awake_plus_the_other_asleep():
    # static awake caps of both (90 + 88) can never fit 120 GiB: the budget that
    # must fit is the awake stack's cap plus every OTHER stack's asleep cap,
    # for whichever stack is awake (the arbiter moves the caps at a model flip)
    a, b = default_specs(acc_root="/acc", tag="t1")
    a = dataclasses.replace(a, mem_awake_bytes=90 * GIB, mem_asleep_bytes=20 * GIB)
    b = dataclasses.replace(b, mem_awake_bytes=88 * GIB, mem_asleep_bytes=40 * GIB)
    errs = check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert any("mem" in e and "27b awake" in e for e in errs)
    b = dataclasses.replace(b, mem_asleep_bytes=30 * GIB)
    assert check_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB) == []


def test_asleep_cap_above_awake_cap_is_named():
    a, b = default_specs(acc_root="/acc", tag="t1")
    b = dataclasses.replace(b, mem_asleep_bytes=b.mem_awake_bytes + GIB)
    errs = check_coboot([a, b], host_mem_bytes=500 * GIB, host_reserve_bytes=0)
    assert any("mem_asleep" in e for e in errs)


def test_two_stacks_of_the_same_model_are_refused():
    a, _ = default_specs(acc_root="/acc", tag="t1")
    b = dataclasses.replace(a, container_name="other", host_port=30041,
                            cleanup_prefix="other", arb_dir="/x/arb", state_dir="/x/state",
                            evidence_dir="/x/ev", store_dir="/x/store")
    errs = check_coboot([a, b], host_mem_bytes=500 * GIB, host_reserve_bytes=0)
    assert any("model" in e for e in errs)


def test_require_raises_with_every_collision_named():
    a, b = default_specs(acc_root="/acc", tag="t1")
    b = dataclasses.replace(b, host_port=a.host_port, arb_dir=a.arb_dir)
    with pytest.raises(CobootRefused) as ei:
        require_coboot([a, b], host_mem_bytes=126 * GIB, host_reserve_bytes=6 * GIB)
    assert "host_port" in str(ei.value) and "arb_dir" in str(ei.value)


def test_spec_refuses_a_port_the_rig_reserves():
    with pytest.raises(CobootRefused):
        StackSpec(model="nf", container_name="n", cleanup_prefix="n", host_port=30099,
                  arb_dir="/a", state_dir="/s", evidence_dir="/e", store_dir="/st",
                  mem_awake_bytes=GIB, mem_asleep_bytes=GIB)
