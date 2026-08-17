# #602 FILL SIDE — per-card attribution, and why the sizer cannot claim it

Desk analysis of the open half. **Verdict: the gap is DELIBERATE plus
STRUCTURAL, not dead reservation. This is the labeled accounting table, not a
forced fix** — and the claim path, where one exists, is the CUT, not the sizer.

Attributed from the instrument chain built for #704/#707 (budget posts, the
holdback line, the sizing line), all read from the instrumented boot rather than
modelled.

---

## 1 — Per-rank accounting (MiB)

| rank | budget | weights+runtime | mamba | rest | holdback | allowed | used | **slack** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PP0 | 31,800 | 16,064.5 | 916.5 | 14,819.5 | 6,690.1 | 8,129.5 | 5,964.7 | **2,164.8** |
| PP1 | 18,800 | 10,061.8 | 654.3 | 8,083.4 | 3,562.7 | 4,520.7 | 4,260.5 | **260.2** |
| PP2 | 19,800 | 10,699.8 | 523.3 | 8,576.3 | 5,167.9 | 3,408.4 | 3,408.4 | **0.0** |

`allowed` is the post-holdback KV budget; `used` is the world pool
(436,278 tokens) priced at that rank's cell. **Total slack: 2,425 MiB.**

## 2 — The two terms, and which is claimable

**HOLDBACK — deliberate, not slack.** Established in #707: the pool is capped so
the resting free column still holds the arming floor
(`allowed = id_space + (free_at_measure - arming_floor - margin)/cell`). This is
not memory sitting idle; it is the level a flip must find free to arm at all,
and sizing below it produces the boot that holds the corridor and never flips
(#656 boots E/G). **Not claimable.**

**SLACK — structural, and still not claimable by the sizer.** The 2,425 MiB is
almost entirely on PP0 (2,164.8), and **the binder PP2 has exactly zero**. That
is the whole finding:

> The pool is ONE global token count. A non-binding rank cannot spend its extra
> allowance without the binder moving, because every rank must hold the same
> tokens. PP0's 2,164.8 MiB is not booked-and-untouched — it is unreachable at
> this cut.

A sizer that "claimed" it would hand the pool tokens PP2 cannot back, and PP2 is
the rank that OOMs. That is precisely the #593-family direction the brief warns
about, so no red-first test was written for a fix that must not exist: the
falsifier here would assert a wrong answer.

## 3 — Where the capacity actually is

The slack is a property of the CUT, not of the sizer, so the claim path is the
cut solve — which is already answered on another strand. #702 rev5 found the
incumbent [28,20,16] is **not pool-optimal**: the binding rank switches, and
moving layers OFF the binder raises the pool. [30,18,16] gives **+20 %**
(462,231 vs 384,209 model rows) *and* 1.11x pipelined prefill.

So the fill-side capacity is real, and it is claimed by rebalancing the cut, not
by the sizer relaxing a reserve. Those are different tickets and the second one
would be unsafe.

## 4 — What this note does NOT claim

The ticket's measured shape is **2.0 / 5.7 / 3.7 GiB free per card**. Those
figures are from a different operating point than the instrumented boot
attributed above, and the instruments that would attribute them per-term
(budget posts, holdback, sizing line) **only exist on boots carrying
2a6305dd3b / 5f3e61f0db / f55c1a8adf**. Mapping this table onto those three
numbers would be exactly the kind of cross-boot arithmetic this strand has
already had to retract twice.

What can be said without that: against the measured arming floors
(1,728 / 1,825 / 2,467 MiB) the per-card free sits **above** the ~1,024 MiB
corridor target *by construction*, because the arming floor — not the corridor —
is the binding level on a flip-enabled boot. Reading the gap against 1,024
overstates it on every rank.

**Ask for the next boot:** capture the three instrument lines at the operating
point that produced 2.0 / 5.7 / 3.7, and this table can be re-run against it
directly. That is a window item, not desk work.

## 5 — Corridor law note

Per the 2026-08-16 softening, ~1,024 MiB free/card is a TARGET, not a hard
bound, and free is NVML-FREE. Nothing here proposes to hold *more* than the
target; it explains why the observed free exceeds it without that being waste:
`free >= arming_floor` is the real constraint, and the floors exceed the target
on all three cards.
