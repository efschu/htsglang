"""#1303 -- sweep dead-tag admin-key files, and do not fall for the self-match.

THE RESIDUE, measured on the box 2026-09-10 10:4xZ: `boot_weg2sn5pre.adminkey`
and `boot_weg2xsn5.adminkey`, both from Sep 9, both with ZERO live processes --
per-boot secrets outliving their boots in a shared directory, which is exactly
what #1275's `drop_admin_key_file` exists to prevent. They survived because
those boots were never torn down, so the teardown path that removes the key
never ran. The sweep at LAUNCH is the second half #1275 needs.

THE FIRST TEST IS THE PROBE, because the probe is what nearly got this wrong.
`pgrep -fc weg2sn5pre` answered **2** for a tag with zero real processes: the
shell running the probe and the pgrep itself both carry the tag in their argv.
Reading that as "alive" is harmless; reading its inverse as "dead" is how a
sweep deletes a LIVE boot's secret. So the naive probe is written out here and
asserted to be WRONG on that input -- if it ever agrees with the corrected one
on the trap, this file has stopped testing anything.
"""

import os

import pytest

from sglang.srt.weg2 import admin_key as ak

# The trap, verbatim: what /proc actually held when the wrong reading was taken.
TRAP_PROCS = [
    (1111, "pgrep -fc weg2sn5pre"),
    (1112, "bash -c pgrep -af weg2sn5pre | grep -v pgrep"),
]
LIVE_PROCS = [
    (2001, "python3 -m sglang.launch_server --tag weg2sn6h --port 30031"),
    (2002, "/bin/bash /spinning/gpu-arb/devtools/mem_timeseries.sh weg2sn6h"),
]
DEAD_TAGS = ("weg2sn5pre", "weg2xsn5")
LIVE_TAG = "weg2sn6h"


def _naive(tag, procs):
    """The probe that was wrong, kept so the test can prove it is wrong.

    This is `pgrep -f <tag>` as a predicate: anything whose command line
    mentions the tag counts as a holder.
    """
    return any(tag in c for _, c in procs)


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

def test_the_naive_probe_is_wrong_on_the_trap_and_the_corrected_one_is_not():
    """RED-FIRST ON THE TRAP. The naive probe must say ALIVE for a tag whose
    only 'processes' are probes asking about it; the corrected probe must say
    dead. If these two ever agree here, the exclusion has been lost."""
    assert _naive("weg2sn5pre", TRAP_PROCS) is True, (
        "the naive probe must reproduce the wrong reading -- it is the mutant"
    )
    assert ak.tag_has_live_holder("weg2sn5pre", TRAP_PROCS) is False


def test_the_two_dead_tags_read_zero_and_a_live_one_reads_nonzero():
    """The measured case: 0 and 0 against >0, which is what licensed the sweep."""
    for tag in DEAD_TAGS:
        assert ak.tag_has_live_holder(tag, TRAP_PROCS) is False, tag
        assert ak.tag_has_live_holder(tag, LIVE_PROCS) is False, tag
    assert ak.tag_has_live_holder(LIVE_TAG, LIVE_PROCS) is True


def test_the_probes_own_pid_is_excluded():
    """A process may not be its own holder."""
    procs = [(4242, f"python x --tag {LIVE_TAG}")]
    assert ak.tag_has_live_holder(LIVE_TAG, procs) is True
    assert ak.tag_has_live_holder(LIVE_TAG, procs, own_pids=(4242,)) is False


def test_an_unreadable_cmdline_is_not_a_holder():
    """An empty/unreadable /proc entry is a race with an exiting process, and
    the only safe reading of a race is 'gone'. The dangerous direction is
    covered by the caller's unconditional keep set, not here."""
    assert ak.tag_has_live_holder(LIVE_TAG, [(9, ""), (10, None)]) is False


def test_an_empty_tag_never_matches_anything():
    """Otherwise every process would hold the empty tag and nothing would ever
    be swept -- fail-safe, but silently useless."""
    assert ak.tag_has_live_holder("", LIVE_PROCS) is False


def test_the_reader_does_not_shell_out():
    """`read_procs` must read /proc directly: a subprocess would put the tag
    into ANOTHER command line and re-create the self-match it exists to
    avoid."""
    import ast
    import inspect
    import textwrap

    src = inspect.getsource(ak.read_procs)
    # READ THE EXECUTED CODE, NOT THE TEXT -- sixth instance of this class
    # today. `read_procs`' own docstring says "rather than shelling out to
    # `pgrep`" in order to REJECT it, so a source-text grep reports the
    # rejection as a use and this very assertion failed on its own subject.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    body = fn.body[1:] if (
        fn.body and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
    ) else fn.body
    names = set()
    for st in body:
        for n in ast.walk(st):
            if isinstance(n, ast.Name):
                names.add(n.id)
            if isinstance(n, ast.Attribute):
                names.add(n.attr)
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                names.add(n.value)
    for bad in ("subprocess", "pgrep", "system", "popen", "Popen", "check_output"):
        assert bad not in names, f"the reader shells out via {bad}"
    assert any("/cmdline" in n for n in names if isinstance(n, str))


def test_the_reader_sees_this_very_process():
    """The untestable half, smoke-tested at least once: it must find us."""
    procs = ak.read_procs()
    assert any(pid == os.getpid() for pid, _ in procs)


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

def _dir(tmp_path, tags):
    d = tmp_path / "weg2"
    d.mkdir()
    for t in tags:
        (d / f"boot_{t}.adminkey").write_text("k" * 44)
    return str(tmp_path)


def test_the_sweep_removes_the_dead_and_keeps_the_live(tmp_path, monkeypatch):
    root = _dir(tmp_path, list(DEAD_TAGS) + [LIVE_TAG])
    monkeypatch.setattr(ak, "read_procs", lambda *a, **k: LIVE_PROCS)
    line = ak.sweep_stale_keys(root, keep_tags=())
    assert "removed=weg2sn5pre,weg2xsn5" in line
    assert f"kept={LIVE_TAG}" in line
    assert not (tmp_path / "weg2" / "boot_weg2sn5pre.adminkey").exists()
    assert not (tmp_path / "weg2" / "boot_weg2xsn5.adminkey").exists()
    assert (tmp_path / "weg2" / f"boot_{LIVE_TAG}.adminkey").exists(), (
        "a key whose tag has a live holder must never be removed"
    )


def test_the_current_boots_tag_is_kept_even_when_it_looks_dead(tmp_path, monkeypatch):
    """THE SAFETY BELT, and it is structural rather than a second check: the
    caller passes its own tag, so even a wrong liveness verdict cannot delete
    the key of the boot doing the sweeping."""
    root = _dir(tmp_path, [LIVE_TAG])
    monkeypatch.setattr(ak, "read_procs", lambda *a, **k: [])   # nothing alive
    line = ak.sweep_stale_keys(root, keep_tags=(LIVE_TAG,))
    assert f"kept={LIVE_TAG}" in line
    assert "removed=none" in line
    assert (tmp_path / "weg2" / f"boot_{LIVE_TAG}.adminkey").exists()


def test_the_line_prints_both_lists_always(tmp_path, monkeypatch):
    """A sweep that printed only its removals would be unauditable in exactly
    the direction that matters."""
    root = _dir(tmp_path, list(DEAD_TAGS) + [LIVE_TAG])
    monkeypatch.setattr(ak, "read_procs", lambda *a, **k: LIVE_PROCS)
    line = ak.sweep_stale_keys(root, keep_tags=(), dry=True)
    assert "WEG2-LAUNCH ADMINKEY SWEEP" in line
    assert "removed=" in line and "kept=" in line
    assert "(DRY)" in line
    # dry means nothing moved
    assert (tmp_path / "weg2" / "boot_weg2sn5pre.adminkey").exists()


def test_the_sweep_never_raises(tmp_path, monkeypatch):
    """It runs on the launch path; it may not block a boot."""
    monkeypatch.setattr(ak, "read_procs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    line = ak.sweep_stale_keys(str(tmp_path), keep_tags=())
    assert "failed" in line and "RuntimeError" in line


def test_a_non_key_file_is_never_touched(tmp_path, monkeypatch):
    root = _dir(tmp_path, DEAD_TAGS)
    other = tmp_path / "weg2" / "boot_weg2sn5pre.json"
    other.write_text("{}")
    monkeypatch.setattr(ak, "read_procs", lambda *a, **k: [])
    ak.sweep_stale_keys(root, keep_tags=())
    assert other.exists(), "the sweep must only ever remove *.adminkey"


def test_the_launcher_calls_it_with_its_own_tag_in_the_keep_set():
    """Desk-written-never-executed: the sweep must actually be wired, and
    wired WITH the belt."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    assert "sweep_stale_keys(GPU_ARB, keep_tags=(ns.tag,)" in src
