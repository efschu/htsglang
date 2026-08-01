# REVIEW #111 — the KvLink seam and NcclLink group formation

Fresh-context review, requested by the author. Read + verdict, no rewrites.

**Subject**: `feat/pd-kv-nccl-111` (`da22478c77`, merged at `5390b4d10a`) —
`disaggregation/nccl/link.py`, `disaggregation/nccl/contract.py`,
`docs/dev/TASK_111_PD_KV_NCCL.md`, with attention on the part the author
flagged as *"reasoned about but could not exercise"*.

**Overall: GO for the wire slice, with three preconditions** (§7). The seam is
well cut and the honesty about what has not run is exemplary — `transfer()`
raising `NotImplementedError` with a pointer to the ticket is the right shape,
and it means none of the findings below can corrupt anything today. Two of
them, however, are things the wire slice would inherit silently if it is
written against the current contract as-is.

---

## Axis 1 — Group formation: **RISK (unenforced fixed universe)**

*Evidence*: `link.py:238-243` states the rule correctly — *"Both sides learn
the peer set from the bootstrap exchange [...] No rank asks the group who is
present -- that would be a membership decision taken by a collective, which is
the #94 family"*. `setup()` (`link.py:272-281`) refuses to proceed without a
`group_factory` and says why.

The rule is right and the refusal is right. The problem is that **the rule is
asserted in prose and cannot be checked at the seam.** `setup()` calls

```python
self._group = self._group_factory(session_id=..., is_sender=..., peer=...)
```

and then trusts whatever comes back. Nothing constrains the factory to build
the bootstrap-agreed universe; a factory that internally discovers members, or
one that silently builds a smaller world because one peer was slow to arrive,
returns an object that passes every existing test. The seam has **no way to
notice that the world it was handed is not the world bootstrap agreed on** —
which is precisely the property the docstring claims.

*Why this matters more than a style point*: the #94/#259 family is not "don't
call a discovery API", it is "membership must not be re-derived at runtime".
An unconstrained factory relocates the re-derivation one frame outward, where
the seam's own tests cannot see it.

*Cheap fix, for the wire slice to adopt*: have the bootstrap exchange yield an
expected member count / rank list, pass it into `setup()`, and assert
`torch.distributed.get_world_size(self._group) == expected` immediately after
construction — a named `LinkError` on mismatch. Two lines, and it converts a
prose invariant into a checked one. The ticket's "not done" item 3 already
notes the rendezvous is unwritten; this says what its acceptance test is.

---

## Axis 2 — Bounded waits: **DEFECT (formation is not bounded)**

*Evidence*: `link.py:44-47` declares the rule — *"Default bound for any link
wait [...] Never unbounded: #259 found a HiCache collective sitting on a
7200 s gloo default"* — and `DEFAULT_LINK_TIMEOUT_S = 120.0`. `link.py:244-247`
claims *"Every wait is bounded through `bounded_collective` /
`bounded_barrier`"*.

But `setup()` does **not** pass `self._timeout_s` to the factory, and does not
wrap the call:

```python
self._group = self._group_factory(
    session_id=session_id, is_sender=is_sender, peer=peer
)   # no timeout, no bound
```

Group formation is a wait — a TCPStore rendezvous blocks until the peer
arrives — and it is the *first* wait in the sequence. Who wins today: **torch's
own default**, which for `init_process_group`/TCPStore is on the order of tens
of minutes, not 120 s. So the one claim the contract makes universally
("every wait 120 s bounded") is false for the very wait most likely to hang,
because a peer that never boots is the ordinary failure of a cross-instance
rendezvous.

This is the #259 defect in its original shape — a long library default
inherited because nobody passed a bound — one layer above where #259 found it.

*Fix*: pass `timeout_s` into the factory contract and require the factory to
apply it to the store/init call; assert it in the conformance suite by handing
in a factory that sleeps past the bound and requiring `LinkTimeoutError`.

---

## Axis 3 — Failure semantics: **RISK (mid-formation undistinguished)**

*Evidence*: the error taxonomy is good — `LinkError`, `LinkRegistrationError`,
`LinkTimeoutError` (`link.py:50-64`), each with a stated reason, and
`LinkTimeoutError` documented to name the peer and the elapsed time. The GPU
ticket's item 5 covers *"Peer death ends the wait. Kill the prefill instance
mid-transfer"*.

Two gaps:

1. **Mid-formation is not on the ticket and not bounded** (follows directly
   from Axis 2). A peer that dies — or never arrives — *before* the group
   exists produces a long block, not a named error. That is the #312 spin in
   the one window where it is most likely.
2. **The two deaths are not distinguished by type.** Mid-formation and
   mid-transfer are operationally different failures — the first means "the
   pairing never happened, retry or re-place", the second means "a session is
   half-moved and its KV state is now ambiguous" — and an operator reading a
   log needs to tell them apart without inspecting a stack. Today both would
   surface as `LinkTimeoutError` at best.

*Fix*: a `LinkFormationTimeout` subclass, plus a ticket item mirroring item 5
for the formation window (kill the peer *before* the group forms).

---

## Axis 4 — What the LoopbackLink gate does and does not constrain: **SOUND, with named residuals**

*Evidence*: `link.py:24-30` is honest and accurate — *"`LoopbackLink` is fully
exercised: it is what makes the byte-identity gate a real test rather than a
mock assertion. `NcclLink` is written but has NEVER run on a card [...] Its
contract is pinned by the same tests through the shared `KvLink` conformance
suite, which is a statement about its INTERFACE, not about its wire
behaviour."*

That is exactly the right claim, neither over- nor under-stated. The
conformance suite genuinely constrains: the block-plan builder, the region
validation and its all-or-nothing rule (#221), the handshake refusal path, the
route policy, and the error taxonomy's shape.

**Residual risks the loopback gate cannot catch — the wire slice's checklist:**

| # | residual | why loopback misses it |
| --- | --- | --- |
| R1 | the group ever forming at all, and forming with the right members | loopback has no peer and no group |
| R2 | the formation wait's bound (Axis 2) | no wait exists to time out |
| R3 | real registration semantics | NCCL registration is a documented no-op; loopback validates the same list, so both agree by construction and neither exercises pinning |
| R4 | transfer ORDERING and completion | loopback is synchronous and local; NCCL completion is a separate event, and "the call returned" is not "the peer has the bytes" |
| R5 | multi-rank fan-out | the seam is written per (session, role) pair; a decode instance with several ranks pulling from one prefill is untested shape |
| R6 | byte identity **over the wire** as opposed to in-process | the gate proves the plan is right, not that NCCL moved it unchanged — endianness, padding and stride assumptions all live past the seam |
| R7 | partial transfer on peer death | loopback cannot half-fail |
| R8 | DCP row ownership (Axis 5) | loopback has one owner by construction |

R6 and R8 are the two that produce *wrong bytes* rather than *no bytes*, so
they are the ones the GPU ticket should front-load.

---

## Axis 5 — Identity handshake: **DEFECT (the dcp_size non-comparison is not safe)**

*Evidence*: `contract.py:176-181` justifies the exclusion — *"`tp_size` /
`pp_size` / `dcp_size` are deliberately NOT compared: PD already supports
differing prefill/decode TP (KVArgs carries `state_dim_per_tensor` and
`state_dim_offsets` precisely so the sender can re-slice), so demanding
equality would refuse working configurations."* The ticket repeats it at
line 56.

**The justification is sound for `tp_size` and `pp_size`, and does not
transfer to `dcp_size`.** `state_dim_per_tensor` / `state_dim_offsets`
describe the **state/head dimension** — they let a sender re-slice *within a
row*. DCP shards the **token axis**: under the fork's owner rule a global slot
`L` lives on rank `L % S` (even-modulo) or in the weighted prefix range, at
compact row `L // S`. That changes **which rows a rank owns**, not how a row
is sliced, and no offset table re-derives it.

The file says so itself, four lines above the exclusion —
`contract.py:160-161`: *"Fork geometry: token-sharded KV changes which rows a
rank owns."* The field's own docstring states the reason it must be compared,
and then it is excluded on a justification that covers a different axis.

Concretely: a prefill instance at `dcp_size=1` and a decode instance at
`dcp_size=3` have entirely different row→token mappings. The handshake — whose
stated purpose is that a mismatch is *"checked before the first transfer"*
because otherwise it is *"wrong tokens, not an error"* — would wave that
through.

*Why I am calling it a DEFECT and not a risk*: it cannot bite today only
because `transfer()` raises. The wire slice is exactly the change that makes
it bite, and it is **not on the ticket** — the ticket's item 6 tests handshake
refusal on *"different"* configurations, but the excluded fields are precisely
the ones it will not catch.

*Fix, either is acceptable*: add `dcp_size` to `COMPARED`; or keep it
excluded and require equality of a derived **row-ownership descriptor**
(`(S, lo, hi)` from the owner rule) which is the quantity that actually has to
match. The second is better if differing-DCP PD is ever wanted; the first is
correct today and is one line.

---

## 6. What I checked and found nothing wrong with

Stated so the review's silence is not mistaken for absence of inspection:

* The seam's narrowness (`setup`/`register`/`transfer`/`close`) and the
  argument for it (`link.py:13-23`) — a later fastpath is a new member, not a
  second stack. Sound, and the reason is the right one.
* Registration all-or-nothing with a first-bad-region error, carrying the
  #221 lesson explicitly rather than by comment.
* `IncompatiblePeer` spelling out *which* field differs alongside the hash —
  the "two-minute fix vs a bisect" reasoning is correct and rare.
* `state_types` **is** compared, which preempts the #212 hybrid failure. Good
  catch by the author; it is the one geometry field that most needed it.
* The opt-in posture and the construction-time warning while unvalidated.

---

## 7. Go / no-go

**GO for the wire slice**, conditional on three preconditions, all small and
all inside the slice's own scope:

1. **Bound the formation wait** (Axis 2 — the only DEFECT that is a hang) and
   add the conformance test that a slow factory raises.
2. **Verify the handed-in world** against the bootstrap-agreed member count in
   `setup()` (Axis 1) — this is what turns the fixed-universe rule from prose
   into a check.
3. **Close the DCP hole** (Axis 5), by comparison or by a row-ownership
   descriptor, before `transfer()` moves its first byte.

Add to the GPU ticket: a mid-**formation** peer-death arm (mirroring item 5),
and front-load R6 (byte identity over the wire) and R8 among the residuals,
since those two produce wrong bytes rather than no bytes.

Nothing here argues against the design. The seam is the right shape, the
honesty about the unexercised half is the right posture, and the three
preconditions are all "convert an asserted invariant into a checked one" —
which is the cheapest kind of finding to act on and the most expensive kind to
discover on hardware.
