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


def _limit_lines(values, *, role="staging", size=30518, frac="0.9", site=cc.CENSUS_SITES[0]):
    out = []
    for rank, now in enumerate(values):
        out.append(LINE_TPL.format(rank=rank, now=now, frac=frac, size=size,
                                   role=role, pool=13140197628528 + rank, site=site))
    return out


# ------------------------------------------------------------------ the floor
def test_route_floor_is_in_carrier_est_tokens_not_chunk_tokens():
    """FIX 2, the unit defect.  ``carrier_max_tokens`` is compared against
    ``carrier_est`` (CARRIER_CHARS_PER_TOKEN=2.4), not against the SHORT
    remainder (CHARS_PER_TOKEN=3.0), so the floor is 1.25x CHUNK_TOKENS.  Taking
    CHUNK_TOKENS itself passed every bound in (4096, 5120] as usable while NO
    prompt of ANY length could round-trip under one."""
    from sglang.srt.weg2 import front

    floor, why = cc.route_floor()
    assert floor > front.CHUNK_TOKENS
    assert floor == int(front.CHUNK_TOKENS * front.CHARS_PER_TOKEN
                        / front.CARRIER_CHARS_PER_TOKEN)
    assert floor == 5120  # at today's constants; the line above is the rule


def test_route_floor_carries_its_derivation_and_resolves_its_citations():
    """The floor is never printed bare, and the front line numbers in the
    provenance are RESOLVED, not transcribed (instrument-text law)."""
    import inspect

    from sglang.srt.weg2 import front

    floor, why = cc.route_floor()
    assert str(floor) in why
    assert "CHARS_PER_TOKEN" in why and "CARRIER_CHARS_PER_TOKEN" in why
    assert "CARRIER-EXCEEDS" in why and "SHORT" in why
    # the scope qualifier: SHORT is four conjuncts, so the floor is a
    # conservative refusal in D's serving steady state, not a licence
    assert "four conjuncts" in why and "conservative refusal" in why

    lines = inspect.getsourcelines(front)[0]
    for anchor in ("CHUNK_TOKENS = ",
                   "and carrier_est > self.carrier_max_tokens:",
                   "and remainder <= CHUNK_TOKENS:"):
        cited = cc._front_line(anchor, -1)
        assert cited > 0, f"anchor vanished from front.py: {anchor!r}"
        assert anchor in lines[cited - 1], (
            f"front.py:{cited} does not hold {anchor!r} -- the citation drifted")
        assert f"front.py:{cited}" in why


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
        role=kwargs.get("role_last", "staging"), pool=999, site=cc.CENSUS_SITES[0])]
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
    """The round-trip set is EMPTY at equality, so the test is strict, not >=.
    Driven at the REAL floor, so the boundary the census enforces is the
    boundary the front actually routes by (see the routing test below)."""
    floor, _ = cc.route_floor()
    c = cc.census(_write(tmp_path, _limit_lines([floor] * 3)), expected_ranks=3, floor=floor)
    assert c.verdict == "below_floor"
    c2 = cc.census(_write(tmp_path, _limit_lines([floor + 1] * 3), name="d2.log"),
                   expected_ranks=3, floor=floor)
    assert c2.verdict == "ok" and c2.bound == floor + 1


def test_the_old_chunk_tokens_floor_would_have_passed_an_off_switch(tmp_path):
    """The defect FIX 2 closes, as a value: a group D whose KV host pool is
    5688 tokens yields limit 5119, which the CHUNK_TOKENS floor called ok."""
    from sglang.srt.weg2 import front

    floor, _ = cc.route_floor()
    log = _write(tmp_path, _limit_lines([5119] * 3, size=5688))
    assert cc.census(log, expected_ranks=3, floor=front.CHUNK_TOKENS).verdict == "ok"
    assert cc.census(log, expected_ranks=3, floor=floor).verdict == "below_floor"


# ------------------------------------------------- the population it reads from
def test_the_mamba_pool_warning_cannot_enter_the_census(tmp_path):
    """THE DEFECT ITSELF.  A log carrying both pool warnings and the KV
    carrier's own limit line yields the carrier's number, not min(19, 30518)."""
    lines = [KV_WARNING, MAMBA_WARNING] + _limit_lines([27466] * 3)
    c = cc.census(_write(tmp_path, lines), expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466
    assert 19 not in c.per_rank.values()
    assert int(0.9 * 19) != c.bound


def test_the_cutover_rebind_site_is_not_counted(tmp_path):
    """``log_prefetch_limit`` also runs after every cutover rebind.  The census
    is taken at launch, so a later rebind line must not join the sample.  The
    site string is the REAL one (hicache_phase_binding.py:971), so this models
    the line it names."""
    lines = _limit_lines([27466] * 3) + _limit_lines([9] * 3, site="rebind_for_cutover")
    c = cc.census(_write(tmp_path, lines), expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466 and len(c.lines) == 3


def test_both_launch_time_cache_init_sites_are_counted(tmp_path):
    """``log_prefetch_limit`` has THREE call sites, and TWO of them are cache
    constructors: unified_radix_cache.py:1049 (init_hicache) and
    hiradix_cache.py:216 (hiradix_init).  A group D built on HiRadixCache emits
    only the second; admitting one site alone would W45-refuse a healthy
    carrier."""
    assert set(cc.CENSUS_SITES) == {"init_hicache", "hiradix_init"}
    c = cc.census(_write(tmp_path, _limit_lines([27466] * 3, site="hiradix_init")),
                  expected_ranks=3, floor=4096)
    assert c.verdict == "ok" and c.bound == 27466


def test_the_three_call_sites_of_the_emitter_are_the_ones_named():
    """The site set is a DENOMINATOR claim about the emitter.  If a fourth call
    site appears, or one is renamed, the census's population silently changes --
    so the claim is checked against the source, not remembered."""
    import inspect
    import re as _re

    from sglang.srt.mem_cache import (
        hicache_phase_binding,
        hiradix_cache,
        unified_radix_cache,
    )

    # one call site nests another call, so scan forward from the name rather
    # than to the first ')'
    pat = _re.compile(r"log_prefetch_limit\(.{0,200}?site=\"([a-z_]+)\"", _re.S)
    sites = set()
    for mod in (unified_radix_cache, hiradix_cache, hicache_phase_binding):
        sites |= set(pat.findall(inspect.getsource(mod)))
    assert sites == {"init_hicache", "hiradix_init", "rebind_for_cutover"}, sites
    assert sites - set(cc.CENSUS_SITES) == {"rebind_for_cutover"}


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
    # LAST wins, like the server's own argparse: the launcher writes --tp-size 3
    # into group D's argv and then appends the operator's --extra-d tail, so an
    # override there is the rank count the group actually runs.
    (["python", "--tp-size", "3", "--tp-size", "6"], 6),
    (["python", "--tp-size", "3", "--tp-size=6"], 6),
])
def test_tp_size_comes_from_the_group_argv(argv, expected):
    assert cc.tp_size_of(argv) == expected


def test_a_census_that_measured_nothing_says_so_instead_of_printing_zero(tmp_path):
    """``fraction=0.0 host_size=0`` on a log line reads as a MEASURED zero.
    The launcher's CARRIER BOUND line prints ``terms()``, which says which of
    the two it is (indicator law)."""
    c = cc.census(str(tmp_path / "absent.log"), expected_ranks=3, floor=4096)
    assert c.verdict == "missing"
    assert "not measured" in c.terms()
    assert "fraction=0.0" not in c.terms()
    good = cc.census(_write(tmp_path, _limit_lines([27466] * 3)), expected_ranks=3, floor=4096)
    assert good.terms() == "role=staging fraction=0.9 host_size=30518"


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


def test_the_launcher_has_no_flag_that_ships_a_bound_of_zero():
    """FIX 2.  Fix 1's --no-carrier-route shipped --carrier-max-tokens 0 while
    its help, its log line and the W45 remedy sentence all said it switched the
    round trip OFF.  Both front guards are `carrier_max_tokens > 0`, so 0
    switches the BYPASS off and leaves the round trip UNBOUNDED -- the W16 shape
    of boot weg2ls4b2.  The flag is gone; what replaces it is an operator bound
    that clears the same floor."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    # the flag itself, not the retraction comment that records why it went
    assert 'add_argument("--no-carrier-route"' not in src, (
        "the off switch that did the opposite of what it said is back")
    assert "no_carrier_route" not in src, (
        "the launcher still reads the removed flag off the namespace")
    assert '"--carrier-max-tokens", type=int, default=None' in src
    assert "operator_below_floor" in src
    # no arm of the region may assign a bound of 0
    assert "carrier_max_tokens = 0" not in src


def test_zero_is_documented_the_same_way_in_the_front_and_in_the_launcher():
    """The two files must not say opposite things about the same number, and
    what they say must be what the guards do."""
    import inspect

    from sglang.srt.weg2 import front, launcher

    fsrc = inspect.getsource(front.main)
    assert "0 = no CARRIER-EXCEEDS route" in fsrc
    assert "NOT an off switch" in fsrc
    assert "0 is NOT an off switch" in inspect.getsource(launcher)

    # the guards those sentences describe are still the guards
    assert ("if self.carrier_max_tokens > 0 and carrier_est > self.carrier_max_tokens:"
            in inspect.getsource(front.Front.handle_generate))
    assert ("if self.carrier_max_tokens > 0 and pt > self.carrier_max_tokens"
            in inspect.getsource(front.Front.leg1))


def test_the_mamba_pool_carries_the_label_it_registers_under():
    """One identity, three readers.  MambaPoolHost does not call
    HostKVCache.__init__, so it had no ``budget_label``: the warning printed the
    class name, the pinned-host post registered under a literal, and the
    teardown at pool_host/base.py:215 unregistered under ``type(self).__name__``
    -- a post that was never registered, so the #550 joint budget kept charging
    for a freed buffer."""
    import inspect

    from sglang.srt.mem_cache import memory_pool_host

    src = inspect.getsource(memory_pool_host.MambaPoolHost.__init__)
    assert 'self.budget_label = "HiCache Mamba anchor host pool"' in src
    assert 'name="HiCache Mamba anchor host pool"' not in src
    assert "name=self.budget_label," in src


# ------------------------------------------- the floor against the REAL router
class _Req:
    """The two attributes ``Front.handle_generate`` touches on a request."""

    def __init__(self, path, payload):
        self.path = path
        self._payload = payload

    async def json(self):
        return self._payload


def _route_of(bound: int, n_chars: int) -> str:
    """Which branch the REAL ``Front.handle_generate`` takes for a prompt of
    ``n_chars`` characters under ``carrier_max_tokens=bound``, with group D in
    its serving steady state (awake=D, admit_d, serving) -- the best case for
    SHORT and therefore the honest case for a floor.

    BATCH is detected by the branch it takes, not by a return value: it queues
    the request and awaits leg 1, which never completes here.
    """
    import asyncio

    from sglang.srt.weg2.front import Front

    async def go():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=bound)
        f.state = "serving"
        f.admit_d = True

        async def fake_leg2(request, rid, payload, text, stream, pending=None,
                            single_prefill=False):
            return "served"

        f.leg2 = fake_leg2
        task = asyncio.ensure_future(
            f.handle_generate(_Req("/generate", {"text": "x" * n_chars})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass  # queued on the leg-1 future: BATCH, i.e. the round trip
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        taken = [k for k in ("route_carrier_exceeds", "route_short", "route_batch")
                 if f.counters[k]]
        assert len(taken) == 1, (taken, dict(f.counters))
        return taken[0]

    return asyncio.run(go())


def test_at_the_floor_the_real_front_round_trips_nothing_and_one_above_it_does():
    """THE FLOOR, MEASURED AGAINST THE ROUTER IT DESCRIBES.  At `floor` no
    prompt length reaches BATCH; at `floor + 1` the shortest non-SHORT prompt
    does.  This is the coupling the CHUNK_TOKENS floor lacked: 4096 passed the
    census while the front round-tripped nothing."""
    from sglang.srt.weg2 import front

    floor, _ = cc.route_floor()
    shortest_non_short = int(front.CHUNK_TOKENS * front.CHARS_PER_TOKEN)  # 12288 chars

    assert _route_of(floor + 1, shortest_non_short) == "route_batch"
    assert _route_of(floor, shortest_non_short) == "route_carrier_exceeds"
    # and nothing at all round-trips at the floor, across the length axis
    for n in (10, 12287, shortest_non_short, 40000, 300000):
        assert _route_of(floor, n) in ("route_short", "route_carrier_exceeds"), n


def test_a_bound_the_old_floor_accepted_round_trips_nothing():
    """5119 (a 5688-token host pool x 0.9) cleared the CHUNK_TOKENS floor and is
    an off switch: measured against the real router, over the whole axis."""
    from sglang.srt.weg2 import front

    assert 5119 > front.CHUNK_TOKENS
    for n in (10, 12287, 12288, 12300, 40000, 300000):
        assert _route_of(5119, n) in ("route_short", "route_carrier_exceeds"), n


# ------------------------------------------ the regex against the LIVE emitter
def test_the_regex_matches_the_live_emitter_not_only_a_recorded_line():
    """Both REPLAY tests read RECORDED logs, which by construction cannot notice
    a change to the emitter's format string -- a drift would turn every boot
    into a W45 `missing`.  Drive the real emitter and match its own output."""
    import logging
    from types import SimpleNamespace

    from sglang.srt.mem_cache import prefetch_budget

    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    ctrl = SimpleNamespace(
        prefetch_capacity_limit=27466,
        prefetch_capacity_fraction=0.9,
        mem_pool_host=SimpleNamespace(size=30518, anchor_entry=None),
        host_role="staging",
    )
    lg = prefetch_budget.logger
    h = _Cap()
    lg.addHandler(h)
    lvl = lg.level
    lg.setLevel(logging.INFO)
    try:
        prefetch_budget.log_prefetch_limit(ctrl, site=cc.CENSUS_SITES[0])
    finally:
        lg.removeHandler(h)
        lg.setLevel(lvl)

    assert len(records) == 1, records
    assert "could not be formed" not in records[0], records[0]
    # the server writes the '[<ts> TP<k>]' prefix the rank is read from
    m = cc.LIMIT_RE.search(f"[2026-09-08 05:08:00 TP0] {records[0]}")
    assert m, f"LIMIT_RE no longer matches its own emitter: {records[0]!r}"
    assert m.group("rank") == "0" and m.group("now") == "27466"
    assert m.group("fraction") == "0.9" and m.group("host_size") == "30518"
    assert m.group("role") == "staging" and m.group("site") == cc.CENSUS_SITES[0]
