"""Weg 2 (#1233): two process groups per card, one awake, the store as carrier.

Modules:

* ``host_ledger`` -- the #721 host ledger for the six-process shape at both
  moments (launch, run), the S/M arm ladder, the tmpfs store size, W20.
* ``launcher`` -- the sequenced group launcher of spec section 1.5: group P
  (PP=3 prefill, :30031) up, asleep with its measured dormant residue, group D
  (TP=3+NEXTN decode, :30032) up and awake, the front up, deadmen armed.
* ``front`` -- the phase router on :30030: request ledger, the sequential
  two-leg route, drain-and-flip on the #1011 work-exhaustion clocks plus the
  V1 fairness bound, the named refusals of spec section 5.

Nothing here duplicates an upstream mechanism: sleep/wake is the upstream
memory saver (slice S1), the carrier is the canonical page store (slice S5),
and the groups are two stock ``sglang.launch_server`` launches.
"""

#: THE TWO SCHEDULING BATCH SIZES, EACH WRITTEN EXACTLY ONCE.
#:
#: THE ORDER IN FORCE, user 2026-09-09, verbatim: "der decode bs6 soll mit bs6
#: (nicht mehr bs4) der standard werden."  D's default is therefore 6.
#:
#: SUPERSEDED, same day, and kept because it is the source of P's number and of
#: the re-measurability clause that still governs both -- user 2026-09-09,
#: verbatim: "nimm jetzt vorerst bs4 fuer decode und bs2 fuer prefill. wenn
#: alles fertig ist kann man das immernoch nachmessen wo da das optimum fuer
#: meinen anwendungsfall liegt."  P stays 2 from that order; D's 4 was replaced
#: by the order above before it ever reached metal.
#:
#: So these two numbers remain a PROVISIONAL operating point, explicitly
#: re-measurable once the strand is finished -- not a derived or proven
#: optimum, and nothing may treat them as one.  The supersession inside a
#: single day is the cheapest possible evidence that the pair moves: it moved
#: once already, and the only reason that cost one edit instead of twenty is
#: the single-site rule below.
#:
#: They live here, in the package root, because BOTH ends need them and neither
#: may own the other's copy: ``launcher`` owns ``--p-bs`` / ``--d-bs`` and every
#: pricing helper that reads a bs off the argv it builds, and ``front`` owns
#: ``--p-concurrency`` / ``--d-bs`` for the standalone case (the launcher always
#: writes both explicitly, R-6/R-12, so the front's own defaults bind only when
#: it is run by hand).  A second copy in either module would drift the day one
#: of the two numbers is re-measured -- exactly the class ``_max_running_requests``
#: and ``_p_page_size`` already avoid by reading the flag off the argv instead of
#: restating it.  Re-measuring the operating point must be an edit to THESE TWO
#: LINES and nowhere else.
#:
#: They stay INDEPENDENT (law 2 / C1-R-12): P's prefill concurrency and D's
#: decode seat count are two knobs, and their being unequal here is the point.

#: K1: group P's ``--max-running-requests`` AND the front's leg-1 concurrency.
DEFAULT_P_BS = 2

#: K2: group D's ``--max-running-requests`` AND the number of front D seats.
#: 6 by the order of 2026-09-09 quoted above ("nicht mehr bs4").
DEFAULT_D_BS = 6

#: K3 -- THE P CUT'S POOL FLOOR, moved here from ``launcher.py`` by the serve-next5
#: train (2026-09-09): the three shipped operating-point constants of the two
#: user orders of that day live in ONE place, and ``launcher`` imports this one
#: exactly as it imports the bs pair above (upstream-minimal: one home, not two).
#: Its only consumer is still ``launcher.resolve_pool_floor``.
#: THE SHIPPED FLOOR UNDER ``--pp-solve-objective``, in WORLD KV TOKENS, and
#: WRITTEN EXACTLY ONCE (the pin in
#: ``test/registered/unit/weg2/test_weg2_p_cut_default_0909.py`` keeps it that
#: way, the same way ``DEFAULT_P_BS`` is pinned on its own branch).
#:
#: THE ORDER IN FORCE, user 2026-09-09, verbatim: "39,13,12 mit bs2 im pp
#: layout soll standard werden vorerst."
#:
#: THE CUT IS STILL SOLVED, NOT PINNED, and that distinction is the whole
#: reason this is a floor and not a ``--pp-stage-ratio``.  A hand pin would
#: make ``39,13,12`` survive a card change, a checkpoint change or a re-measure
#: that made it wrong; the solver stays the authority and this number only says
#: HOW MUCH POOL the boot refuses to go below.  ``--pp-solve-objective``
#: (makespan, itself a user default since 2026-09-08) then ranks over what is
#: left, so the shipped rule reads "the FASTEST cut that still holds
#: 448,027 tokens" -- and on this rig's frontier that is 39,13,12.
#:
#: WHY THIS NUMBER -- the rule, so the next re-solve can redo it rather than
#: copy it.  Read off the ``PP-CUT FRONTIER`` line of #1286b (SECTION 1am-b,
#: ``/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md``), the two frontier
#: points that bracket the order are
#:
#:     40,12,12 / 10,3,3   total_ms=417.6   pool=414654   <- next FASTER
#:     39,13,12 /  9,4,3   total_ms=420.1   pool=481400   <- THE ORDER
#:
#: Under "fastest above F" the order is selected for any F in the half-open
#: interval (414654, 481400].  The floor is the MIDPOINT of that interval,
#: (414654 + 481400) // 2 = 448027: it sits 33,373 tokens from each boundary,
#: which is the largest equal headroom the interval allows.
#:
#: A floor survives a uniform re-pricing of the frontier by factor s only while
#: it stays inside (414654*s, 481400*s], so every floor has TWO tolerances --
#: how far pools may GROW before 40,12,12 clears it too and wins on speed, and
#: how far they may SHRINK before 39,13,12 stops clearing it -- and the SMALLER
#: of the two is the one that fails first.  The midpoint tolerates +8.05 %%
#: growth and -6.93 %% shrink, so 6.93 %% binds.
#:
#: THE REJECTED RULE, named because it is the obvious one: "39,13,12's priced
#: pool minus the priced-vs-realised tolerance" gives 481400 * 0.999 = 480918.
#: It tolerates +15.98 %% growth -- better than the midpoint -- but only
#: -0.10 %% shrink, and that is not a coincidence: subtracting a 0.10 %%
#: tolerance BUILDS a 0.10 %% shrink margin, so the rule's binding margin is
#: always exactly the measurement error it was derived from (0.10 %% MEASURED,
#: weg2sb5f priced 304,946 against 304,655 realised).  A margin the size of
#: one's own error is no margin: the first re-price that shaved a tenth of a
#: percent off the pool would silently ship 38,13,13 (453.4 ms) or refuse.
#: Ranked on distance from the LOWER boundary alone the rejected rule looks
#: like the better one, which is exactly why that is not the comparison -- the
#: test asserts it on the binding margin, 6.93 %% against 0.10 %%, 69x.
#:
#: THE TRADE THE ORDER BOUGHT, so nobody re-derives it as a regression: against
#: the unfloored makespan winner 44,10,10 (359.0 ms / 304,946 tokens) this is
#: +17.0 %% ms/chunk for +57.9 %% pool.  The user chose the knee.
#:
#: OFF IS STILL REACHABLE AND STILL EXACT: ``--pp-solve-pool-floor 0`` restores
#: the unfloored makespan byte for byte.  The floor is PROVISIONAL in the same
#: sense the bs pair is -- "vorerst" is in the order -- so re-measuring it must
#: stay an edit to THIS ONE LINE.
DEFAULT_PP_SOLVE_POOL_FLOOR = 448_027
