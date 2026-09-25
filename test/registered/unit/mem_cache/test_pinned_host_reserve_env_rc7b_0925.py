"""RC7b: the pinned-host OS reserve is settable per container.

NF host acceptance, 2026-09-25: under a container memory cap of 82g the fixed
10 GiB reserve left nothing for the pins that come late -- `available`
(cap - nonreclaim) was 8.7 GB, minus 10 GiB = 0 usable, so the P->D KV
hand-back's lazy read buffers (0.54 GB) were refused: W53
Weg2StoreHandbackFailed and 413 on long prompts. The 27B ran the same
acceptance at cap 76g with nonreclaim 61-66 GiB, just through.

SGLANG_PINNED_HOST_RESERVE_GIB (GiB, finite >= 0) states the OS's share of
THIS machine; unset keeps 10 GiB (byte-identical natively). Every check reads it
at call time through pinned_host_budget.pinned_host_reserve() and names the
value and its source. Invalid values are refused loudly.
"""

import ast
import pathlib

import pytest

from sglang.srt.mem_cache import pinned_host_budget as B

GIB = 1024**3
ENV = "SGLANG_PINNED_HOST_RESERVE_GIB"


def _post(nbytes):
    return B.PinnedHostPost(name="hicache-read-buffers", flag="--hicache-size", nbytes=int(nbytes))


def test_unset_is_the_native_ten_gib(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert B.pinned_host_reserve() == (10 * GIB, "default 10 GiB")
    assert B.pinned_host_reserve_bytes() == B.PINNED_HOST_RESERVE_BYTES == 10 * GIB


def test_the_env_sets_it(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    assert B.pinned_host_reserve() == (2 * GIB, f"env {ENV}=2")
    monkeypatch.setenv(ENV, " 1.5 ")
    assert B.pinned_host_reserve() == (int(1.5 * GIB), f"env {ENV}=1.5")
    monkeypatch.setenv(ENV, "0")
    assert B.pinned_host_reserve_bytes() == 0


@pytest.mark.parametrize("raw", ["-1", "abc", "10GiB", "nan", "inf", "-0.5"])
def test_invalid_values_are_refused_by_name(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(B.PinnedHostReserveInvalid) as ei:
        B.pinned_host_reserve()
    assert ENV in str(ei.value) and repr(raw) in str(ei.value)


def test_the_capped_container_case_default_refuses_env_two_admits(monkeypatch):
    """The acceptance's own numbers: 8.7 GB available, 0.54 GB of read buffers."""
    monkeypatch.delenv(ENV, raising=False)
    err = B.joint_pinned_host_error([_post(0.54e9)], 82e9, 8.7e9)
    assert err is not None
    assert "10.74 GB OS reserve (default 10 GiB)" in err
    monkeypatch.setenv(ENV, "2")
    assert B.joint_pinned_host_error([_post(0.54e9)], 82e9, 8.7e9) is None


def test_the_message_names_an_env_reserve(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    err = B.joint_pinned_host_error([_post(9e9)], 82e9, 8.7e9)
    assert err is not None and f"2.15 GB OS reserve (env {ENV}=2)" in err


def test_an_explicit_reserve_still_wins_and_says_so(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    err = B.joint_pinned_host_error([_post(0.54e9)], 82e9, 8.7e9, 10 * GIB)
    assert err is not None and "OS reserve (caller)" in err


def test_the_runtime_registry_reads_the_env_at_call_time(monkeypatch):
    B.reset_pinned_posts_for_tests() if hasattr(B, "reset_pinned_posts_for_tests") else None
    monkeypatch.setattr(B, "pinned_host_memory_bytes", lambda: (int(82e9), int(8.7e9)))
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(ValueError):
        B.check_and_register_pinned_post("hicache-read-buffers", "--hicache-size", int(0.54e9))
    monkeypatch.setenv(ENV, "2")
    try:
        B.check_and_register_pinned_post("hicache-read-buffers", "--hicache-size", int(0.54e9))
    finally:
        B.unregister_pinned_post("hicache-read-buffers")


def _code_uses(path, name):
    """Loads of ``name`` in real code (Name/attribute/import), never docstrings."""
    tree = ast.parse(pathlib.Path(path).read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            hits.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == name:
            hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
            hits.append(node.lineno)
    return hits


def test_no_check_reads_the_fixed_constant_any_more():
    """Every place that entered the reserve into a calculation reads the env now:
    the constant survives only as the documented default inside its module."""
    root = pathlib.Path(B.__file__).resolve().parents[1]  # python/sglang/srt
    offenders = {}
    for path in root.rglob("*.py"):
        if path.name == "pinned_host_budget.py":
            continue
        text = path.read_text(errors="replace")
        if "PINNED_HOST_RESERVE_BYTES" not in text:
            continue
        hits = _code_uses(path, "PINNED_HOST_RESERVE_BYTES")
        if hits:
            offenders[str(path.relative_to(root))] = hits
    assert offenders == {}, offenders


def test_the_weg1_planner_sizes_with_the_same_reserve():
    from sglang.srt.planner import weg1_host_sizing as W

    assert _code_uses(W.__file__, "RESERVE_BYTES") == [], "weg1 still computes with its fixed copy"
    assert "pinned_host_reserve" in pathlib.Path(W.__file__).read_text()
