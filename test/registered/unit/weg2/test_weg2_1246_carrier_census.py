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


#: The SHORT bound these floor cases are derived against. It used to be
#: ``SHORT_BOUND``; slice A replaced that constant with the per-boot
#: ``--tp-prefill-max-tokens`` (X), so the bound is STATED here and handed to
#: ``route_floor`` rather than read off a module that no longer owns it. Every
#: number below is the same number it always was, at the same bound.
SHORT_BOUND = 4096

# ------------------------------------------------------------------ the floor
def test_route_floor_is_in_carrier_est_tokens_not_chunk_tokens():
    """FIX 2, the unit defect.  ``carrier_max_tokens`` is compared against
    ``carrier_est`` (CARRIER_CHARS_PER_TOKEN=2.4), not against the SHORT
    remainder (CHARS_PER_TOKEN=3.0), so the floor is 1.25x CHUNK_TOKENS.  Taking
    CHUNK_TOKENS itself passed every bound in (4096, 5120] as usable while NO
    prompt of ANY length could round-trip under one."""
    from sglang.srt.weg2 import front

    floor, why = cc.route_floor(SHORT_BOUND)
    assert floor > SHORT_BOUND
    assert floor == int(SHORT_BOUND * front.CHARS_PER_TOKEN
                        / front.CARRIER_CHARS_PER_TOKEN)
    assert floor == 5120  # at today's constants; the line above is the rule


def test_route_floor_carries_its_derivation_and_resolves_its_citations():
    """The floor is never printed bare, and the front line numbers in the
    provenance are RESOLVED, not transcribed (instrument-text law)."""
    import inspect

    from sglang.srt.weg2 import front

    floor, why = cc.route_floor(SHORT_BOUND)
    assert str(floor) in why
    assert "CHARS_PER_TOKEN" in why and "CARRIER_CHARS_PER_TOKEN" in why
    assert "CARRIER-EXCEEDS" in why and "SHORT" in why
    # the scope qualifier: SHORT is four conjuncts, so the floor is a
    # conservative refusal in D's serving steady state, not a licence
    assert "four conjuncts" in why and "conservative refusal" in why

    lines = inspect.getsourcelines(front)[0]
    # #1290 unpoisoned these two anchors: both bounds moved into
    # `serviceable_route`, which asks them TOGETHER. The citations must name
    # the lines that decide today, not the lines that used to.
    for anchor in ("fits_carrier = carrier_max <= 0 or carrier_est <= carrier_max",
                   "fits_d_prefill = x_tokens <= 0 or uncached <= x_tokens"):
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
    floor, _ = cc.route_floor(SHORT_BOUND)
    c = cc.census(_write(tmp_path, _limit_lines([floor] * 3)), expected_ranks=3, floor=floor)
    assert c.verdict == "below_floor"
    c2 = cc.census(_write(tmp_path, _limit_lines([floor + 1] * 3), name="d2.log"),
                   expected_ranks=3, floor=floor)
    assert c2.verdict == "ok" and c2.bound == floor + 1


def test_the_old_chunk_tokens_floor_would_have_passed_an_off_switch(tmp_path):
    """The defect FIX 2 closes, as a value: a group D whose KV host pool is
    5688 tokens yields limit 5119, which the CHUNK_TOKENS floor called ok."""
    from sglang.srt.weg2 import front

    floor, _ = cc.route_floor(SHORT_BOUND)
    log = _write(tmp_path, _limit_lines([5119] * 3, size=5688))
    assert cc.census(log, expected_ranks=3, floor=SHORT_BOUND).verdict == "ok"
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
    # FIX 3: the refusal text is built where the decision is taken, so the
    # launcher no longer carries a second copy of it.
    assert "decide_bound" in src
    assert "W45 Weg2CarrierCensusRefused" in inspect.getsource(cc)


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
    assert "operator_below_floor" in inspect.getsource(cc)
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
    # #1290: the carrier guard lives in `serviceable_route` now -- one
    # verdict on both bounds. The sentence it describes is unchanged.
    assert ("fits_carrier = carrier_max <= 0 or carrier_est <= carrier_max"
            in inspect.getsource(front.serviceable_route))
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
        # X IS THE SHORT BOUND NOW (slice A), so the router must be built at
        # the same bound the floor was derived against, or the two halves of
        # this test would price different fronts.
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=bound, tp_prefill_max_tokens=SHORT_BOUND)
        f.state = "serving"
        f.admit_d = True

        async def fake_leg2(request, rid, payload, text, stream, pending=None,
                            single_prefill=False, seat=None):
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

    floor, _ = cc.route_floor(SHORT_BOUND)
    shortest_non_short = int(SHORT_BOUND * front.CHARS_PER_TOKEN)  # 12288 chars

    # #1317d RETIRED THE OFF-SWITCH HAZARD THIS TEST WAS WRITTEN FOR, and the
    # law is restated rather than the assertion flipped.
    #
    # WHAT #1246 GUARDED: a carrier bound below the floor silently DISABLED the
    # round trip -- every non-short prompt fell to `route_carrier_exceeds` and
    # P was never used, which is why the floor had to be coupled to the real
    # router instead of to CHUNK_TOKENS.
    #
    # WHY IT NO LONGER APPLIES: since #1317d the carrier bound does not gate
    # round-tripping at all. D's host staging pool is TRANSIT for the window
    # loop, not a cap on prompt length, so a prompt above the carrier takes the
    # P route (`long` -> `route_batch`) instead of a single prefill or a 413.
    # A too-low bound therefore no longer switches the round trip OFF -- it
    # forces it ON for more of the axis, which is the safe direction and is the
    # whole point of the change.
    #
    # THE BOUND'S NEW FAILURE MODE, named so it is not rediscovered: a
    # mis-set (too low) carrier bound now routes prompts to P that D could
    # have single-prefilled cheaply. That costs a leg, never a refusal.
    assert _route_of(floor + 1, shortest_non_short) == "route_batch"
    assert _route_of(floor, shortest_non_short) == "route_batch", (
        "since #1317d a below-floor carrier bound must NOT disable the round "
        "trip; it routes the same prompt through P instead of refusing it"
    )
    # and the round trip is now available across the whole length axis at the
    # floor -- the axis that used to read `route_carrier_exceeds` everywhere.
    for n in (10, 12287, shortest_non_short, 40000, 300000):
        assert _route_of(floor, n) in ("route_short", "route_batch"), n


def test_a_bound_the_old_floor_accepted_is_no_longer_an_off_switch():
    """5119 (a 5688-token host pool x 0.9) cleared the CHUNK_TOKENS floor and
    WAS an off switch: over the whole length axis nothing round-tripped.

    #1317d: it is not an off switch any more, and this is the pair to the
    restated test above. The same bound, the same axis, the same router -- but
    a prompt above the carrier now takes the P route instead of falling to a
    single prefill, so `route_carrier_exceeds` is no longer the answer
    everywhere. The #1246 FLOOR still has a job (it is the bound below which
    the carrier stops describing anything real, and the census still refuses
    it), but the HAZARD it was sized against -- a silent loss of the round
    trip -- cannot occur through this branch any more."""
    from sglang.srt.weg2 import front

    assert 5119 > SHORT_BOUND
    seen = set()
    for n in (10, 12287, 12288, 12300, 40000, 300000):
        r = _route_of(5119, n)
        assert r in ("route_short", "route_batch"), n
        seen.add(r)
    assert "route_batch" in seen, (
        "at least one length must round-trip through P at this bound; if none "
        "does, the off-switch hazard #1246 was written for is back"
    )


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


# ======================================================================= FIX 3
# THE OVERRIDE MAY ONLY LOWER A MEASURED BOUND.
#
# Fix 2 replaced --no-carrier-route with --carrier-max-tokens N and checked N
# against the FLOOR alone.  That left the escape hatch open on the side the
# bound exists to guard: N above what group D's carrier reports it will enforce
# is a bound the store cannot honour, so every prompt priced between the
# measured bound and N takes leg 1 on P and then a leg-2 read the store must
# refuse -- '#915 PREFETCH REFUSED' / W16, boot weg2ls4b2.  Both round-2
# refuters reached it independently, at 262144 (this rig's --max-model-len).
#
# The interval is floor < N <= measured, and a census that measured NOTHING has
# no interval at all.

FLOOR = 5120  # cc.route_floor(SHORT_BOUND) at today's constants; asserted as a rule above


def _census_of(tmp_path, values, *, name, ranks=3, floor=FLOOR, **kw):
    """A census over a written log, at the REAL floor unless told otherwise."""
    return cc.census(_write(tmp_path, _limit_lines(values, **kw), name=name),
                     expected_ranks=ranks, floor=floor)


def _ok_census(tmp_path, name="ok.log", now=27466):
    return _census_of(tmp_path, [now] * 3, name=name)


def test_an_override_above_the_measured_bound_is_refused(tmp_path):
    """THE BLOCKING DEFECT OF FIX 2.  27466 is what the carrier enforces; 262144
    is the number an operator reaches for (this rig's --max-model-len, and the
    value the front-side help's wording invites).  Shipping it does not enlarge
    the carrier -- it only stops the front bypassing what the carrier cannot
    read back."""
    cen = _ok_census(tmp_path)
    assert cen.measured == 27466
    for n in (27467, 100000, 262144, 10 ** 9):
        d = cc.decide_bound(cen, n, floor_why="w")
        assert d.refused and d.reason == "operator_above_measured", (n, d.reason)
        assert d.bound == 0
        assert str(n) in d.detail and "27466" in d.detail
        assert "W16" in d.detail and "weg2ls4b2" in d.detail
        assert f"{FLOOR} < N <= 27466" in d.detail


def test_an_override_inside_the_interval_ships_and_names_measured_floor_and_n(tmp_path):
    """The one accepted shape, and the log line the operator is owed: it names
    all three numbers, so a reader of the boot log can check the decision
    without re-deriving it."""
    cen = _ok_census(tmp_path)
    for n in (FLOOR + 1, 9000, 27466):
        d = cc.decide_bound(cen, n, floor_why="w")
        assert not d.refused and d.bound == n and d.reason == ""
        assert d.source == "operator --carrier-max-tokens"
        assert f"N={n}" in d.note
        assert "measured=27466" in d.note
        assert f"floor={FLOOR}" in d.note
        assert "never raise one" in d.note


@pytest.mark.parametrize("n", [0, 1, 17, 4096, FLOOR])
def test_an_override_at_or_below_the_floor_is_still_refused(tmp_path, n):
    """Fix 2's arm, kept: it is the OTHER end of the same interval.  0 is in it
    and is not an off switch."""
    d = cc.decide_bound(_ok_census(tmp_path), n, floor_why="w")
    assert d.refused and d.reason == "operator_below_floor" and d.bound == 0
    assert "weg2rg5" in d.detail
    assert f"{FLOOR} < N <= 27466" in d.detail


@pytest.mark.parametrize("verdict, build", [
    ("missing", lambda t: cc.census(str(t / "absent.log"), expected_ranks=3, floor=FLOOR)),
    ("missing", lambda t: _census_of(t, [27466, 27466], name="partial.log")),
    ("disagree", lambda t: cc.census(
        _write(t, _limit_lines([27466, 27466]) + [LINE_TPL.format(
            rank=2, now=24000, frac="0.9", size=30518, role="staging", pool=9,
            site=cc.CENSUS_SITES[0])], name="disagree.log"),
        expected_ranks=3, floor=FLOOR)),
])
def test_an_override_is_refused_when_the_census_measured_nothing(tmp_path, verdict, build):
    """The re-armed failure the refusal exists for: with no measured number
    there is nothing to lower and nothing to check N against, so the flag cannot
    stand in for a census.  Fix the census, not the number."""
    cen = build(tmp_path)
    assert cen.verdict == verdict and cen.measured is None
    for n in (9000, 27466, 262144):
        d = cc.decide_bound(cen, n, floor_why="w")
        assert d.refused and d.reason == "operator_without_measured_bound", (n, d.reason)
        assert d.bound == 0
        assert "measured no carrier bound" in d.detail
        assert "Fix the census, not the number" in d.detail


def test_a_measured_bound_below_the_floor_leaves_no_interval_at_all(tmp_path):
    """boot weg2rg5's own shape: the carrier measured 17.  `measured` exists, so
    the refusal is not `without_measured_bound` -- but floor < N <= 17 is empty,
    so EVERY override is refused by one of the two ends."""
    cen = _census_of(tmp_path, [17] * 3, name="rg5.log", size=19)
    assert cen.verdict == "below_floor" and cen.measured == 17
    seen = set()
    for n in (0, 17, FLOOR, FLOOR + 1, 27466, 262144):
        d = cc.decide_bound(cen, n, floor_why="w")
        assert d.refused, n
        seen.add(d.reason)
    assert seen == {"operator_below_floor", "operator_above_measured"}


def test_the_census_path_ships_only_ok_and_is_untouched_by_fix_3(tmp_path):
    """The healthy no-override path: unchanged."""
    d = cc.decide_bound(_ok_census(tmp_path), None, floor_why="w")
    assert not d.refused and d.bound == 27466 and d.source == "census" and d.note == ""

    for cen in (_census_of(tmp_path, [17] * 3, name="low.log", size=19),
                cc.census(str(tmp_path / "absent.log"), expected_ranks=3, floor=FLOOR)):
        r = cc.decide_bound(cen, None, floor_why="w")
        assert r.refused and r.reason == cen.verdict and r.bound == 0


def test_the_remedy_sentence_describes_the_refusal_the_operator_would_actually_get(tmp_path):
    """INSTRUMENT-TEXT LAW, KLASSE A, and the exact form round 1 and round 2 both
    blocked on: the W45 remedy is the ONE sentence an operator is ever told to
    follow, so what it promises is checked against what the flag then does.  Fix
    2's said the flag 'is checked against the SAME floor, so there is no way to
    ship a bound that cannot carry the route' -- and 10^9 shipped."""
    for cen in (cc.census(str(tmp_path / "absent.log"), expected_ranks=3, floor=FLOOR),
                _census_of(tmp_path, [17] * 3, name="low2.log", size=19)):
        detail = cc.decide_bound(cen, None, floor_why="w").detail
        assert "fix the MEASUREMENT, not the number" in detail
        assert "may only LOWER" in detail
        assert "floor < N <= measured" in detail
        # and the promise is true: the flag really is refused on this census
        for n in (9000, 262144):
            assert cc.decide_bound(cen, n, floor_why="w").refused, (cen.verdict, n)
        # the sentence no longer claims the flag rescues the boot
        assert "pass --carrier-max-tokens N with a bound you measured yourself" not in detail


def test_the_launcher_delegates_the_whole_decision_and_keeps_no_second_check(tmp_path):
    """ONE MECHANISM.  The launcher must not hold its own comparison beside
    decide_bound's -- that split is how the floor check ended up alone."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    assert "_cc.decide_bound(_cen, ns.carrier_max_tokens" in src
    assert "raise Weg2LaunchRefused(_dec.detail)" in src
    assert "_override" not in src, "the launcher still compares the operator's number itself"
    assert "if not _cen.ok" not in src


def test_the_flag_help_states_the_interval_and_every_refusal_it_can_produce():
    """The help is the other operator-facing surface; it must name what the code
    does, including the arm fix 2 had no text for because it had no check."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    start = src.index('"--carrier-max-tokens", type=int, default=None')
    help_text = src[start:src.index("ap.add_argument(", start + 10)]
    assert "floor < N <= measured" in help_text
    for code in ("operator_above_measured", "operator_below_floor",
                 "operator_without_measured_bound"):
        assert code in help_text, code
        assert code in inspect.getsource(cc), code
    assert "0 is NOT an off switch" in help_text
    assert "fixing the census, not by typing a number" in help_text


def test_terms_are_the_ones_measured_even_on_a_refusal_that_parsed_lines(tmp_path):
    """NONBLOCKING ITEM 1.  The `expected_ranks <= 0` branch returned before the
    first row was taken, so a log whose three lines parsed perfectly printed
    `role=? fraction=0.0 host_size=0` -- a MEASURED zero where an observation
    existed."""
    c = _census_of(tmp_path, [27466] * 3, name="notp.log", ranks=0)
    assert c.verdict == "missing" and len(c.lines) == 3
    assert c.terms() == "role=staging fraction=0.9 host_size=30518"
    assert "fraction=0.0" not in c.terms() and "role=?" not in c.terms()
    # and a census that really measured nothing still says so
    empty = cc.census(str(tmp_path / "absent.log"), expected_ranks=0, floor=FLOOR)
    assert "not measured" in empty.terms()


def test_measured_is_none_exactly_when_there_is_nothing_to_lower(tmp_path):
    """`bound` holds 0 on missing/disagree because a frozen field must hold
    something; `measured` is the property that says whether it is an
    observation."""
    assert _ok_census(tmp_path).measured == 27466
    assert _census_of(tmp_path, [17] * 3, name="bf.log", size=19).measured == 17
    assert cc.census(str(tmp_path / "absent.log"), expected_ranks=3, floor=FLOOR).measured is None
    assert _census_of(tmp_path, [27466, 27466], name="part.log").measured is None


def test_the_source_docstring_names_the_condition_of_its_emitter():
    """NONBLOCKING ITEM 2, instrument-text law applied to this module's own
    prose: it convicted the old source of hiding a condition while calling the
    new one 'unconditional', and one of the two emitting sites sits inside
    `if storage_backend is not None:`."""
    import inspect

    from sglang.srt.mem_cache import hiradix_cache, unified_radix_cache

    doc = cc.__doc__
    assert "once per rank, unconditionally" not in doc  # fix 1's claim
    assert "NOT unconditional" in doc
    assert "if storage_backend is not None:" in doc
    assert "storage backend" in doc

    # the condition the docstring names is the condition in the source
    u = inspect.getsource(unified_radix_cache)
    i = u.index('log_prefetch_limit(self.cache_controller, site="init_hicache")')
    assert "if storage_backend is not None:" in u[:i]
    h = inspect.getsource(hiradix_cache.HiRadixCache.__init__)
    assert 'site="hiradix_init"' in h


def test_printed_citations_lead_with_symbols_that_exist():
    """NONBLOCKING ITEM 3.  A line number in a printed string cannot see its own
    drift; a symbol can be pinned.  The citations name real attributes of the
    real module, and the resolved line still holds the anchor."""
    from sglang.srt.weg2 import front

    assert callable(front.Front.handle_generate) and callable(front.Front.leg1)
    assert isinstance(SHORT_BOUND, int)

    _floor, why = cc.route_floor(SHORT_BOUND)
    assert "front.Front.handle_generate, front.py:" in why
    
    detail = cc.decide_bound(
        cc.census("/nonexistent", expected_ranks=3, floor=FLOOR), None, floor_why="w").detail
    assert "front.py:279" not in detail  # the citation fix 1 printed for SHORT
    import inspect
    anchor = "and pt > self.carrier_max_tokens"
    ref = cc._front_ref("front.Front.leg1", anchor, -1)
    assert ref.startswith("front.Front.leg1, front.py:")
    cited = int(ref.rsplit(":", 1)[1])
    assert anchor in inspect.getsourcelines(front)[0][cited - 1], (
        f"front.py:{cited} does not hold {anchor!r} -- the citation drifted")


@pytest.mark.skipif(not (os.path.exists(RG3_LOG) and os.path.exists(RG5_LOG)),
                    reason="evidence logs absent")
def test_replay_both_real_logs_still_decide_27466_with_no_flag():
    """The healthy path through the WHOLE decision, on both real D logs: the
    number the front gets is unchanged by fix 3."""
    for path in (RG3_LOG, RG5_LOG):
        cen = cc.census(path, expected_ranks=3, floor=cc.route_floor(SHORT_BOUND)[0])
        d = cc.decide_bound(cen, None, log_path=path, floor_why="w")
        assert not d.refused and d.bound == 27466 and d.source == "census"
        # and the same logs refuse the number that re-arms W16
        assert cc.decide_bound(cen, 262144, log_path=path).reason == "operator_above_measured"


# ------------------------------- the launcher's OWN region, executed verbatim
def _decision_region() -> str:
    """The launcher's carrier-bound decision, lifted out of ``launcher.py`` by
    text so the test drives THE CODE THAT SHIPS rather than a paraphrase of it.

    The region's f-strings and its raise only ever run inside a boot, which is
    why fix 1 and fix 2 could each ship an operator-facing sentence that did not
    match the arm beneath it: nothing at desk level executed the arm.  The
    anchors are the two lines that bracket the decision in every version of this
    region -- the source-line loop above it and the state write below it.
    """
    import inspect
    import textwrap

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    a = src.index("for _sl in _cen.lines:")
    b = src.index("state.carrier_max_tokens = carrier_max_tokens", a)
    body = src[a:b].split("\n")[2:]  # drop the loop itself, keep what follows
    return textwrap.dedent("\n".join(body))


def _launcher_decision(cen, override, *, floor_why="the floor, derived",
                       x_measured=True):
    """Run that region.  Returns ``(shipped_bound, log_lines)`` or raises
    ``Weg2LaunchRefused`` exactly as the launcher would.

    ``x_measured`` is the region's #1299 input: the floor is 1.25x X, so the
    region has to know whether that X was measured on this rig or came from
    the recorded fallback pair.  Default True = the pre-#1299 behaviour, so
    every expectation below is unchanged.
    """
    from types import SimpleNamespace

    from sglang.srt.weg2 import carrier_census as _cc_mod
    from sglang.srt.weg2.launcher import Weg2LaunchRefused, XSeed

    logged = []
    g = {
        "_cc": _cc_mod,
        "_cen": cen,
        "ns": SimpleNamespace(carrier_max_tokens=override),
        "log": logged.append,
        "Weg2LaunchRefused": Weg2LaunchRefused,
        "_floor": cen.floor,
        "_floor_why": floor_why,
        "spec_d": SimpleNamespace(log="<group D log>"),
        "x_seed": XSeed(8742, "X=8742 source=<test>", x_measured),
    }
    exec(compile(_decision_region(), "<launcher decision region>", "exec"), g)
    return g["carrier_max_tokens"], logged


def test_the_launcher_region_refuses_an_override_above_the_measured_bound(tmp_path):
    """THE BLOCKING DEFECT, AT THE SEAM THAT SHIPS IT.  On this census the
    launcher holds 27466 -- it prints it on the line above -- and fix 2 shipped
    262144, 10^9 and 2^62 beside it, logging only that they cleared the floor."""
    from sglang.srt.weg2.launcher import Weg2LaunchRefused

    cen = _ok_census(tmp_path)
    for n in (27467, 262144, 10 ** 9, 2 ** 62):
        with pytest.raises(Weg2LaunchRefused) as e:
            _launcher_decision(cen, n)
        assert "operator_above_measured" in str(e.value), n
        assert "27466" in str(e.value)


def test_the_launcher_region_does_not_refuse_a_below_floor_on_an_unmeasured_x(tmp_path):
    """#1299, AT THE SEAM THAT SHIPS IT: the region must pass x_seed.measured
    into decide_bound, not merely have it in scope.

    The measured shape of boots dec2b (f929987a9c) and shadow B (edbf7007c8):
    a real carrier bound of 27,466 against a floor of 28,195 = 1.25 x an X of
    22,556 that came from the recorded fallback pair, not from this rig. Both
    were refused pre-READY on that comparison.
    """
    cen = _census_of(tmp_path, [27466] * 3, name="belowfloor.log", floor=28195)
    assert cen.verdict == "below_floor"

    bound, logged = _launcher_decision(cen, None, x_measured=False)
    assert bound == 27466, "the MEASURED bound still ships"
    assert any("UNGRADED" in ln for ln in logged)
    assert any("did NOT measure" in ln for ln in logged)


def test_the_launcher_region_still_refuses_a_below_floor_on_a_measured_x(tmp_path):
    """The exemption must not become an off switch: with both terms measured
    the W45 is a real finding and the region must still raise."""
    from sglang.srt.weg2.launcher import Weg2LaunchRefused

    cen = _census_of(tmp_path, [27466] * 3, name="belowfloor2.log", floor=28195)
    with pytest.raises(Weg2LaunchRefused) as e:
        _launcher_decision(cen, None, x_measured=True)
    assert "below_floor" in str(e.value)


def test_the_launcher_region_ships_the_interval_and_logs_all_three_numbers(tmp_path):
    """What an accepted override must leave in the boot log: N, the measured
    bound it lowered, and the floor -- so the decision can be checked from the
    log alone."""
    cen = _ok_census(tmp_path)
    bound, logged = _launcher_decision(cen, 9000)
    assert bound == 9000
    override_lines = [ln for ln in logged if "OPERATOR OVERRIDE" in ln]
    assert len(override_lines) == 1
    line = override_lines[0]
    assert "N=9000" in line and "measured=27466" in line and f"floor={FLOOR}" in line
    assert any("--carrier-max-tokens 9000" in ln for ln in logged)


def test_the_launcher_region_refuses_an_override_with_no_measured_bound(tmp_path):
    """A census that measured nothing is fixed by fixing the census.  This is
    the state in which the W45 remedy sentence is actually printed, so it is the
    state in which fix 2's advice ('pass a bound you measured yourself') would
    have been followed -- with no measured ceiling to aim at."""
    from sglang.srt.weg2.launcher import Weg2LaunchRefused

    for cen in (cc.census(str(tmp_path / "absent.log"), expected_ranks=3, floor=FLOOR),
                _census_of(tmp_path, [27466, 27466], name="part2.log")):
        for n in (9000, 262144):
            with pytest.raises(Weg2LaunchRefused) as e:
                _launcher_decision(cen, n)
            assert "operator_without_measured_bound" in str(e.value)


def test_the_launcher_region_is_unchanged_on_the_healthy_path(tmp_path):
    """No override, healthy census: the front gets the measured bound, no
    OPERATOR OVERRIDE line, and a refusing census still refuses."""
    from sglang.srt.weg2.launcher import Weg2LaunchRefused

    bound, logged = _launcher_decision(_ok_census(tmp_path), None)
    assert bound == 27466
    assert not [ln for ln in logged if "OPERATOR OVERRIDE" in ln]
    assert any("front --carrier-max-tokens 27466" in ln for ln in logged)

    with pytest.raises(Weg2LaunchRefused) as e:
        _launcher_decision(_census_of(tmp_path, [17] * 3, name="rg5b.log", size=19), None)
    assert "W45 Weg2CarrierCensusRefused (below_floor)" in str(e.value)
