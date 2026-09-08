"""#1246 carrier census: the front's ``--carrier-max-tokens`` is READ from the
KV carrier's own prefetch-limit line, and every way of not measuring it is a
named refusal instead of a silent 0.

The defect these cover (boot weg2rg5, /spinning/gpu-arb/weg2/BOOT_weg2rg5_0908.md
"THE FINDING"): the launcher scanned D's log for the upstream warning
``HiCache host KV pool (N tokens) is smaller than the device pool`` and took
``int(0.9 * min(N))``.  That warning is emitted by every host pool below its
device pool -- the mamba/GDN state pool included, which printed the words "KV
pool" about 19 mamba slots -- so ``min`` was 19, the bound was 17, and the front
took CARRIER-EXCEEDS on every request at exit 0.

Hermetic: no CUDA, no server, no boot.  The two REPLAY tests read real boot logs
from /spinning/evidence-665-f1 and skip when they are absent, so the file stays
runnable on a machine without the evidence tree.
"""

import os

import pytest

from sglang.srt.weg2 import carrier_census as cc

# The two real D logs of the boot pair that made the defect visible: same tip
# family, same KV carrier, different ledger arm (rg3 M=1200, rg5 M=600).
RG3_LOG = "/spinning/evidence-665-f1/boot_weg2_weg2rg3_5b015ad139_0908_041053.D.log"
RG5_LOG = "/spinning/evidence-665-f1/boot_weg2_weg2rg5_15a46a611a_0908_050519.D.log"

#: One rank's real line, verbatim from rg5's D log (line 963).
LINE_TPL = (
    "[2026-09-08 05:08:00 TP{rank}] #915 PREFETCH LIMIT now={now} "
    "(fraction={frac} x host size {size}) role={role} pool_id={pool} "
    "phase=pp generation=0 site={site}"
)

#: The mamba warning that poisoned the old census, verbatim from rg5's D log.
MAMBA_WARNING = (
    "[2026-09-08 05:07:59 TP0] HiCache host KV pool (19 tokens) is smaller than "
    "the device pool (20 tokens);L2 cache effectiveness is reduced."
)

#: The KV warning of the same boot, verbatim.
KV_WARNING = (
    "[2026-09-08 05:07:59 TP0] HiCache host KV pool (30518 tokens) is smaller than "
    "the device pool (362253 tokens);L2 cache effectiveness is reduced."
)


def _write(tmp_path, lines, name="D.log"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _limit_lines(values, *, role="staging", size=30518, frac="0.9", site=cc.CENSUS_SITE):
    out = []
    for rank, now in enumerate(values):
        out.append(LINE_TPL.format(rank=rank, now=now, frac=frac, size=size,
                                   role=role, pool=13140197628528 + rank, site=site))
    return out


# ------------------------------------------------------------------ the floor
def test_route_floor_is_the_front_chunk_grant_and_carries_its_derivation():
    """The floor is READ from the module that owns it, and never printed bare."""
    from sglang.srt.weg2 import front

    floor, why = cc.route_floor()
    assert floor == front.CHUNK_TOKENS
    # provenance, not a hand number: the log line has to be able to state where
    # the floor came from and why the route needs it.
    assert "front.CHUNK_TOKENS" in why and "front.py:67" in why
    assert "CARRIER-EXCEEDS" in why and "SHORT" in why


# ------------------------------------------------------------ the rank census
def test_three_agreeing_ranks_give_the_bound_they_agree_on(tmp_path):
    log = _write(tmp_path, _limit_lines([27466, 27466, 27466]))
    c = cc.census(log, expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466
    assert c.per_rank == {0: 27466, 1: 27466, 2: 27466}
    assert c.role == "staging" and c.host_size == 30518 and c.fraction == 0.9
    assert len(c.lines) == 3


def test_the_bound_is_the_enforced_limit_not_a_fraction_recomputed_here(tmp_path):
    """The 0.9 is the RUNTIME's and travels on the line; the census never applies
    it a second time.  A boot whose fraction changed must move the bound."""
    log = _write(tmp_path, _limit_lines([20000, 20000, 20000], frac="0.5", size=40000))
    c = cc.census(log, expected_ranks=3, floor=4096)
    assert c.verdict == "ok"
    assert c.bound == 20000  # not 0.9 * 40000, and not 0.9 * 20000
    assert c.fraction == 0.5


def test_a_missing_rank_refuses_and_is_not_a_bound(tmp_path):
    log = _write(tmp_path, _limit_lines([27466, 27466]))
    c = cc.census(log, expected_ranks=3, floor=4096)
    assert c.verdict == "missing" and not c.ok
    assert c.bound == 0
    assert "[2]" in c.detail  # names the rank that never reported


def test_an_empty_log_is_missing_not_route_disabled(tmp_path):
    log = _write(tmp_path, ["[2026-09-08 05:08:00 TP0] nothing to see here"])
    c = cc.census(log, expected_ranks=3, floor=4096)
    assert c.verdict == "missing"
    assert "NOT measured" in c.detail and "switched off" in c.detail


def test_an_absent_file_is_missing_not_zero(tmp_path):
    c = cc.census(str(tmp_path / "does-not-exist.log"), expected_ranks=3, floor=4096)
    assert c.verdict == "missing" and c.bound == 0


@pytest.mark.parametrize(
    "kwargs, axis",
    [
        (dict(values=[27466, 27466, 24000]), "now"),
        (dict(values=[27466, 27466, 27466], frac_last="0.5"), "fraction"),
        (dict(values=[27466, 27466, 27466], size_last=40000), "host_size"),
        (dict(values=[27466, 27466, 27466], role_last="retention"), "role"),
    ],
)
def test_disagreeing_ranks_refuse_on_the_axis_they_disagree_on(tmp_path, kwargs, axis):
    """Two ranks can print the same budget from a different pool or fraction --
    the front routes the whole group by ONE number, so every term must agree."""
    values = kwargs["values"]
    lines = _limit_lines(values[:-1])
    lines += [LINE_TPL.format(
        rank=len(values) - 1, now=values[-1],
        frac=kwargs.get("frac_last", "0.9"), size=kwargs.get("size_last", 30518),
        role=kwargs.get("role_last", "staging"), pool=999, site=cc.CENSUS_SITE)]
    c = cc.census(_write(tmp_path, lines), expected_ranks=3, floor=4096)
    assert c.verdict == "disagree" and c.bound == 0
    assert f"'{axis}'" in c.detail


# -------------------------------------------------------------- the floor gate
def test_a_bound_below_the_floor_refuses_instead_of_shipping_an_off_switch(tmp_path):
    """The rg5 shape, reproduced: agreeing ranks, a bound no prompt can use."""
    log = _write(tmp_path, _limit_lines([17, 17, 17], size=19))
    c = cc.census(log, expected_ranks=3, floor=4096)
    assert c.verdict == "below_floor" and not c.ok
    assert c.bound == 17  # reported, so the log line can name it
    assert "17" in c.detail and "4096" in c.detail


def test_a_bound_exactly_at_the_floor_is_still_an_off_switch(tmp_path):
    """The round-trip interval CHUNK_TOKENS < prompt <= bound is EMPTY at
    equality, so the test is strict, not >=."""
    c = cc.census(_write(tmp_path, _limit_lines([4096] * 3)), expected_ranks=3, floor=4096)
    assert c.verdict == "below_floor"
    c2 = cc.census(_write(tmp_path, _limit_lines([4097] * 3), name="d2.log"),
                   expected_ranks=3, floor=4096)
    assert c2.verdict == "ok" and c2.bound == 4097


# ------------------------------------------------- the population it reads from
def test_the_mamba_pool_warning_cannot_enter_the_census(tmp_path):
    """THE DEFECT ITSELF.  A log carrying both pool warnings and the KV
    carrier's own limit line yields the carrier's number, not min(19, 30518)."""
    lines = [KV_WARNING, MAMBA_WARNING] + _limit_lines([27466] * 3)
    c = cc.census(_write(tmp_path, lines), expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466
    assert 19 not in c.per_rank.values()
    assert int(0.9 * 19) != c.bound


def test_only_the_init_hicache_site_is_counted(tmp_path):
    """``log_prefetch_limit`` also runs after every cutover rebind.  The census
    is taken at launch, so a later rebind line must not join the sample."""
    lines = _limit_lines([27466] * 3) + _limit_lines([9] * 3, site="cutover_rebind")
    c = cc.census(_write(tmp_path, lines), expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466 and len(c.lines) == 3


def test_prose_mentioning_the_marker_is_not_a_measurement(tmp_path):
    """#995 trap: the fork's own log prose tells the reader to 'read the #915
    PREFETCH LIMIT line' (scheduler_pp_mixin.py:2932)."""
    prose = ("[2026-09-08 05:09:00 TP0] reason=rate_limited persists, read the "
             "#915 PREFETCH LIMIT line for the budget in force")
    c = cc.census(_write(tmp_path, [prose]), expected_ranks=3, floor=4096)
    assert c.verdict == "missing"


def test_no_tp_size_in_argv_is_a_refusal_not_a_guess(tmp_path):
    c = cc.census(_write(tmp_path, _limit_lines([27466] * 3)), expected_ranks=0, floor=4096)
    assert c.verdict == "missing" and "--tp-size" in c.detail


# ------------------------------------------------------- the rank count source
@pytest.mark.parametrize("argv, expected", [
    (["python", "-m", "sglang.launch_server", "--tp-size", "3", "--pp-size", "1"], 3),
    (["python", "--tp-size=6"], 6),
    (["python", "--pp-size", "3"], 0),
    (["python", "--tp-size", "not-a-number"], 0),
])
def test_tp_size_comes_from_the_group_argv(argv, expected):
    assert cc.tp_size_of(argv) == expected


# --------------------------------------------------------------- REPLAY (real)
@pytest.mark.skipif(not os.path.exists(RG3_LOG), reason="evidence log absent")
def test_replay_rg3_yields_27466():
    """rg3 worked; the new census must not change what it got."""
    c = cc.census(RG3_LOG, expected_ranks=3, floor=4096)
    assert c.verdict == "ok"
    assert c.bound == 27466
    assert c.per_rank == {0: 27466, 1: 27466, 2: 27466}
    assert c.role == "staging" and c.host_size == 30518


@pytest.mark.skipif(not os.path.exists(RG5_LOG), reason="evidence log absent")
def test_replay_rg5_yields_27466_not_17():
    """rg5's KV carrier was IDENTICAL to rg3's -- 30518 tokens, limit 27466.
    Only the mamba pool differed, and it was never the carrier.  The old census
    shipped 17 and killed the route; the new one recovers the real bound, so
    this boot form would have made its round trips."""
    c = cc.census(RG5_LOG, expected_ranks=3, floor=4096)
    assert c.verdict == "ok"
    assert c.bound == 27466
    assert c.bound != 17
    assert c.per_rank == {0: 27466, 1: 27466, 2: 27466}


@pytest.mark.skipif(not (os.path.exists(RG3_LOG) and os.path.exists(RG5_LOG)),
                    reason="evidence logs absent")
def test_replay_both_boots_agree_and_the_old_source_did_not():
    """The two boots' KV carriers are the same size; the OLD source (min over
    every 'HiCache host KV pool (N tokens)' line) is what differed."""
    import re

    old = re.compile(r"HiCache host KV pool \((\d+) tokens\)")
    old_pools = {}
    for name, path in (("rg3", RG3_LOG), ("rg5", RG5_LOG)):
        pools = []
        with open(path, errors="replace") as f:
            for line in f:
                m = old.search(line)
                if m:
                    pools.append(int(m.group(1)))
        old_pools[name] = pools

    # the old census: rg3 fine, rg5 poisoned by the 19-token mamba pool
    assert int(0.9 * min(old_pools["rg3"])) == 27466
    assert int(0.9 * min(old_pools["rg5"])) == 17
    assert 19 in old_pools["rg5"] and 19 not in old_pools["rg3"]

    # the new census: both boots read the same carrier
    a = cc.census(RG3_LOG, expected_ranks=3, floor=4096)
    b = cc.census(RG5_LOG, expected_ranks=3, floor=4096)
    assert a.bound == b.bound == 27466


# ------------------------------------------------------ the emitters stop lying
def test_the_pool_warnings_name_their_component():
    """Instrument-text law, Klasse A.  Both emitters of the 'host pool is
    smaller than the device pool' warning must print WHICH pool -- a mamba pool
    saying "KV pool" is what the launcher census believed."""
    import inspect

    from sglang.srt.mem_cache import memory_pool_host
    from sglang.srt.mem_cache.pool_host import base

    for mod in (memory_pool_host, base):
        src = inspect.getsource(mod)
        assert "HiCache host KV pool (%d tokens) is smaller" not in src, (
            f"{mod.__name__} still announces every host pool as a KV pool")
        assert "HiCache host pool %s (%d tokens) is smaller" in src

    mamba_src = inspect.getsource(memory_pool_host.MambaPoolHost.__init__)
    assert "HiCache host pool %s" in mamba_src
    base_src = inspect.getsource(base.HostKVCache.__init__)
    assert 'getattr(self, "budget_label", None) or type(self).__name__' in base_src


def test_the_launcher_no_longer_scrapes_the_warning():
    """One mechanism: the launcher reads the carrier's own limit line and holds
    no copy of the runtime's fraction."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    assert "HiCache host KV pool" not in src
    assert "0.9 * min" not in src
    assert "carrier_census" in src
    assert "W45 Weg2CarrierCensusRefused" in src
    assert "--no-carrier-route" in src
