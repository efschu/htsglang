"""broadcast over the BAR1 direct path -- the arithmetic, the table, the gate.

Context. Before this, ``BarlinkBar1Transport.BARLINK_OPS`` covered all_reduce,
all_to_all and all_gather, and the standard run died while capturing the
DRAFT graph::

    RuntimeError: barlink: 'broadcast' with 128 bytes during a
    CUDA graph capture, but bar1 reports handles('broadcast', 128)
    -> False; covered there are all_gather, all_reduce, all_to_all,
    all_to_all_single.

128 bytes. Not the bandwidth was missing, the coverage was -- and under
capture a missing coverage is not a slower path but the end of the run, since
the fallback (host-staged gloo) executes once at capture time and never again
on replay.

broadcast now rides the same a2a kernel as all_gather, with a table in which
exactly one rank fills a row. A payload larger than one slot runs in
``ceil(n/slot)`` rounds -- on EVERY rank, including the ones that send
nothing, because the round count is what the barrier is counted in.

Everything here is CPU-only. No card, no window, no extension: ``bc_plan`` is
a pure function, ``barlink_broadcast`` is driven against a stub that records the
tables it would have handed to the kernel, and those tables are then checked
against a reference broadcast AND against each other. What a GPU still has to
prove is benchmark/bar1_graph_check.py, cases ``broadcast`` and
``broadcast-two-graphs``.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    bc_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The size from the handover -- the exact payload that stopped the run.
ABNAHME_BYTES = 128


def _stub(rank: int = 0, welt: int = 3, **kw):
    """A transport instance without __init__ -- no card, no window, no ext.

    Only the fields the broadcast path reads. Anything the test does not set
    stays absent on purpose: a new condition reading a new field then fails
    loudly here instead of being silently skipped.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.bc_an = True
    t.a2a_an = True
    t._a2a_beleg = True
    t._bc_beleg = True
    # The SHIPPED defaults, not round numbers: that the stub really
    # mirrors them is pinned down against the source by
    # TestTheShippedFloor. A stub with a more convenient floor would have
    # missed the 12-byte abort just as much as the first proof did.
    t.bc_min_bytes = 1
    t.bc_max_runden = 16
    t._fenster_minimum = 96 << 20
    t._geo = {
        "off_a2a": 4096,
        "a2a_schlitz": 8 << 20,
        "region_bytes": 90 << 20,
    }
    t._auf = True
    t._ext = object()
    t._belege_stehen = True
    t.welt = welt
    t.rank = rank
    for k, v in kw.items():
        if k in ("off_a2a", "a2a_schlitz", "region_bytes"):
            t._geo[k] = v
        else:
            setattr(t, k, v)
    return t


def _fahre(t, tensor, src):
    """Run ``barlink_broadcast`` on a stub and collect the tables it builds.

    The kernel call is replaced by a recorder that also does what the kernel
    would do LOCALLY: copy the own block. Everything that would cross the
    aperture is filled in by :func:`_zustellen` afterwards, so a table that
    addresses the wrong peer shows up as wrong bytes.
    """
    aufrufe = []

    def _a2a(_self, comm, output, inp, s_len, e_len, s_off, e_off,
             kern_last=None):
        aufrufe.append(
            {
                "s_len": list(s_len), "e_len": list(e_len),
                "s_off": list(s_off), "e_off": list(e_off),
                "kern_last": kern_last, "out": output, "in": inp,
            }
        )
        n = int(s_len[t.rank])
        if n:
            a, b = int(s_off[t.rank]), int(e_off[t.rank])
            output[b:b + n] = inp[a:a + n]
        return output

    with mock.patch.object(type(t), "barlink_all_to_all_single", _a2a):
        ergebnis = t.barlink_broadcast(None, tensor, src)
    return ergebnis, aufrufe


class TestOpRegistration(CustomTestCase):
    """What the transport claims, and what it deliberately still does not."""

    def test_broadcast_is_covered(self):
        self.assertIn("broadcast", BarlinkBar1Transport.BARLINK_OPS)

    def test_broadcast_method_is_not_the_placeholder(self):
        """The F811 trap, nailed down.

        ``barlink_broadcast = _no_collective`` used to stand in the class
        body, BELOW where the real method now is. A plain assignment in a
        class body wins against a ``def`` of the same name further up --
        silently. That is exactly how all_gather would have raised
        NotImplementedError after handles() had promised it, and the bar
        would have looked like a transport fault.
        """
        self.assertIsNot(
            BarlinkBar1Transport.barlink_broadcast,
            BarlinkBar1Transport._no_collective,
        )
        self.assertEqual(
            BarlinkBar1Transport.barlink_broadcast.__name__, "barlink_broadcast"
        )

    def test_reduce_scatter_is_still_the_placeholder(self):
        """The other half of the same check: nothing was removed by accident."""
        self.assertNotIn("reduce_scatter", BarlinkBar1Transport.BARLINK_OPS)
        self.assertIs(
            BarlinkBar1Transport.barlink_reduce_scatter,
            BarlinkBar1Transport._no_collective,
        )

    def test_ops_and_methods_agree(self):
        for op in BarlinkBar1Transport.BARLINK_OPS:
            name = f"barlink_{op}"
            if op == "all_to_all":
                name = "barlink_all_to_all_single"
            self.assertTrue(
                hasattr(BarlinkBar1Transport, name),
                msg=f"BARLINK_OPS claims {op!r} but there is no {name}()",
            )

    def test_the_placeholder_message_names_broadcast_as_covered(self):
        """It reads BARLINK_OPS, so adding an op cannot leave it stale."""
        t = _stub()
        with self.assertRaises(NotImplementedError) as ctx:
            BarlinkBar1Transport._no_collective(t, None, None, 0)
        text = str(ctx.exception)
        self.assertIn("broadcast", text)
        self.assertIn("reduce_scatter", text)


class TestBcPlanArithmetic(CustomTestCase):
    """The round decomposition. One sender, but every rank counts along."""

    def _check_coverage(self, nbytes, slot):
        plan = bc_plan(nbytes, slot)
        seen = []
        for versatz, length in plan:
            self.assertGreater(length, 0)
            self.assertLessEqual(length, slot, msg="a round must fit one slot")
            self.assertLessEqual(versatz + length, nbytes)
            seen.extend(range(versatz, versatz + length))
        self.assertEqual(
            seen, list(range(nbytes)), msg=f"{nbytes} bytes not covered once"
        )
        return plan

    def test_the_handover_payload_is_one_round(self):
        self.assertEqual(self._check_coverage(ABNAHME_BYTES, 8384512),
                         [(0, 128)])

    def test_single_round(self):
        self.assertEqual(len(self._check_coverage(1024, 4096)), 1)

    def test_many_rounds(self):
        self.assertEqual(self._check_coverage(10, 4), [(0, 4), (4, 4), (8, 2)])

    def test_exact_multiple_does_not_add_an_empty_round(self):
        self.assertEqual(len(self._check_coverage(8, 4)), 2)

    def test_a_payload_just_over_the_slot(self):
        """The case byte_proof_broadcast drives: slot + 16."""
        plan = self._check_coverage(4096 + 16, 4096)
        self.assertEqual(plan, [(0, 4096), (4096, 16)])

    def test_round_count_is_rank_uniform(self):
        """The hang, not the error.

        Only the source moves bytes -- but the round count falls out of
        ``nbytes`` and the slot alone, never out of "how much do I send".
        A rank that shortened its count would leave the source standing in
        the barrier of a round nobody else runs.
        """
        nbytes, slot = 10600448, 8384512
        counts = {len(bc_plan(nbytes, slot)) for _ in range(8)}
        self.assertEqual(counts, {2})

    def test_send_offsets_are_aligned_when_the_slot_is(self):
        for nbytes in (4096 * 3 + 5, 100, 16, 1 << 20):
            for versatz, _laenge in bc_plan(nbytes, 4096):
                self.assertEqual(versatz % 16, 0, msg=f"{nbytes}")

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            bc_plan(16, 0)
        with self.assertRaises(ValueError):
            bc_plan(-1, 16)
        self.assertEqual(bc_plan(0, 16), [])


def _tabellen(welt, nbytes, src, slot=8 << 20):
    """The tables of EVERY rank for the same call.

    Module level on purpose: the size ladder needs the same construction
    and reaching into another TestCase for it would tie the two together
    by inheritance rather than by the thing they share.
    """
    aus = {}
    for r in range(welt):
        t = _stub(rank=r, welt=welt, a2a_schlitz=slot)
        tensor = torch.full((nbytes,), r + 1, dtype=torch.uint8)
        _, aufrufe = _fahre(t, tensor, src)
        aus[r] = aufrufe
    return aus


def _lauf(welt, nbytes, src, slot):
    daten = {
        r: bytes(((r * 251 + k * 17) % 256) for k in range(nbytes))
        for r in range(welt)
    }
    transporte = {r: _stub(rank=r, welt=welt, a2a_schlitz=slot)
                  for r in range(welt)}
    puffer = {r: torch.frombuffer(bytearray(daten[r]), dtype=torch.uint8)
              for r in range(welt)}
    tabellen, ergebnisse = {}, {}
    for r in range(welt):
        ergebnisse[r], tabellen[r] = _fahre(
            transporte[r], puffer[r].clone(), src
        )
    # Whatever crossed the aperture did not happen in the recorder: it is
    # replayed here, read from the SENDER's table and written into the
    # RECEIVER's -- so a swapped row shows up.
    for k in range(len(tabellen[0])):
        for empf in range(welt):
            for send in range(welt):
                if send == empf:
                    continue
                n = tabellen[empf][k]["e_len"][send]
                if not n:
                    continue
                a = tabellen[send][k]["s_off"][empf]
                b = tabellen[empf][k]["e_off"][send]
                ergebnisse[empf][b:b + n] = torch.frombuffer(
                    bytearray(daten[send][a:a + n]), dtype=torch.uint8
                )
    return daten, ergebnisse


def _pruefe_alle_halten_die_quelle(fall, welt, nbytes, src, slot):
    """Every rank must end up holding the source's bytes. Byte for byte.

    Takes the test case rather than being a method, because both the
    protocol class and the size ladder need it and neither owns it.
    """
    daten, ergebnisse = _lauf(welt, nbytes, src, slot)
    for r in range(welt):
        fall.assertEqual(
            bytes(ergebnisse[r].numpy().tobytes()),
            daten[src],
            msg=f"rank {r} holds the wrong bytes "
                f"(welt={welt} src={src} n={nbytes} slot={slot})",
        )


class TestTheTableItBuilds(CustomTestCase):
    """The a2a table for a broadcast, checked where it is really decided.

    ``barlink_broadcast`` is driven against a stub that records what it would
    hand to the kernel. That is the level a wrong table shows up at: the
    extension checks what it can see locally (bounds, slot, own block) but
    never that ``empfangs_bytes[i]`` here equals ``sende_bytes[rank]`` on
    rank ``i`` -- for that it would have to run a collective, which is the
    host sync this path exists to avoid.
    """


    def test_only_the_source_sends(self):
        tabellen = _tabellen(3, 4096, src=1)
        for r, aufrufe in tabellen.items():
            (ruf,) = aufrufe
            if r == 1:
                self.assertEqual(ruf["s_len"], [4096] * 3)
            else:
                self.assertEqual(ruf["s_len"], [0] * 3)

    def test_exactly_one_receive_block_is_nonzero(self):
        for r, aufrufe in _tabellen(3, 4096, src=2).items():
            (ruf,) = aufrufe
            self.assertEqual(ruf["e_len"], [0, 0, 4096])

    def test_the_send_offset_is_constant_across_destinations(self):
        """Every destination gets the SAME slice -- that is the broadcast."""
        for aufrufe in _tabellen(4, 3 * (8 << 20), src=0).values():
            for ruf in aufrufe:
                self.assertEqual(len(set(ruf["s_off"])), 1)

    def test_the_pairwise_contract_holds(self):
        """``e_len[i]`` here == ``s_len[rank]`` on rank ``i``, per round.

        The one condition the extension cannot check and a hang would be
        the symptom of.
        """
        for welt in (2, 3, 8):
            for src in range(welt):
                tabellen = _tabellen(welt, 5000, src)
                runden = len(tabellen[0])
                for k in range(runden):
                    for r in range(welt):
                        for i in range(welt):
                            self.assertEqual(
                                tabellen[r][k]["e_len"][i],
                                tabellen[i][k]["s_len"][r],
                                msg=f"welt={welt} src={src} round={k} "
                                    f"r={r} i={i}",
                            )

    def test_the_own_block_matches_in_both_directions(self):
        """``send_len[r] == recv_len[r]`` -- the extension asserts this."""
        for welt in (2, 3):
            for src in range(welt):
                for r, aufrufe in _tabellen(welt, 5000, src).items():
                    for ruf in aufrufe:
                        self.assertEqual(ruf["s_len"][r], ruf["e_len"][r])

    def test_the_round_count_is_the_same_on_every_rank(self):
        # 3 full slots and a remainder of 7 bytes -- four rounds, and the
        # remainder is one of them, not an extra.
        tabellen = _tabellen(3, 3 * (8 << 20) + 7, src=2)
        self.assertEqual({len(a) for a in tabellen.values()}, {4})

    def test_the_kernel_variant_is_decided_group_uniformly(self):
        """``kern_last`` is handed in, not computed from the own row.

        Computed locally it would be ``(R-1)*n`` on the source and 0
        everywhere else -- the same collective would run as a cooperative
        launch on one rank and as a single block on the others.
        """
        tabellen = _tabellen(3, 4096, src=1)
        lasten = {ruf["kern_last"] for a in tabellen.values() for ruf in a}
        self.assertEqual(lasten, {4096 * 2})

    def test_input_and_output_are_never_the_same_buffer(self):
        """The extension rejects ``in is out``, and it is right to.

        The kernel's send and receive phases sit around ONE barrier; a
        buffer that is both would already be overwritten in the second.
        """
        for aufrufe in _tabellen(3, 4096, src=0).values():
            for ruf in aufrufe:
                self.assertIsNot(ruf["in"], ruf["out"])
                self.assertNotEqual(
                    ruf["in"].data_ptr(), ruf["out"].data_ptr()
                )


class TestAgainstAReference(CustomTestCase):
    """Replay the whole slot protocol in Python and compare byte for byte.

    Every rank writes its (possibly empty) block into every peer's slot, the
    barrier passes, every rank copies each peer's slot into its own output.
    The reference is: everybody holds the source's bytes.
    """

    def test_single_round_every_source(self):
        for src in range(3):
            _pruefe_alle_halten_die_quelle(self, 3, 64, src, 256)

    def test_multi_round_every_source(self):
        for src in range(3):
            _pruefe_alle_halten_die_quelle(self, 3, 300, src, 64)

    def test_ragged_tail(self):
        _pruefe_alle_halten_die_quelle(self, 3, 301, 1, 64)

    def test_world_two_and_eight(self):
        _pruefe_alle_halten_die_quelle(self, 2, 200, 1, 64)
        _pruefe_alle_halten_die_quelle(self, 8, 129, 7, 16)

    def test_the_handover_size(self):
        _pruefe_alle_halten_die_quelle(self, 3, ABNAHME_BYTES, 0, 8384512)


class TestInPlaceContract(CustomTestCase):
    """The seam promises the SAME tensor back, filled."""

    def test_the_tensor_object_comes_back(self):
        t = _stub(rank=1, welt=3)
        tensor = torch.zeros(64, dtype=torch.uint8)
        ergebnis, _ = _fahre(t, tensor, 1)
        self.assertIs(ergebnis, tensor)

    def test_a_non_contiguous_tensor_is_still_filled_in_place(self):
        """``reshape(-1)`` copies there, so the result has to be copied back."""
        t = _stub(rank=0, welt=3)
        gross = torch.zeros((8, 4), dtype=torch.uint8)
        sicht = gross[:, ::2]                     # not contiguous
        sicht.copy_(torch.arange(16, dtype=torch.uint8).reshape(8, 2))
        ergebnis, _ = _fahre(t, sicht, 0)
        self.assertIs(ergebnis, sicht)
        self.assertTrue(
            torch.equal(
                sicht, torch.arange(16, dtype=torch.uint8).reshape(8, 2)
            )
        )

    def test_shape_and_dtype_survive(self):
        t = _stub(rank=0, welt=2)
        tensor = torch.arange(32, dtype=torch.int64).reshape(4, 8)
        ergebnis, aufrufe = _fahre(t, tensor, 0)
        self.assertEqual(tuple(ergebnis.shape), (4, 8))
        self.assertEqual(ergebnis.dtype, torch.int64)
        # 32 elements x 8 bytes -- the kernel counts in BYTES, not elements.
        self.assertEqual(aufrufe[0]["s_len"][0], 256)

    def test_an_empty_tensor_is_a_no_op(self):
        t = _stub(rank=0, welt=2)
        tensor = torch.zeros(0, dtype=torch.uint8)
        ergebnis, aufrufe = _fahre(t, tensor, 0)
        self.assertIs(ergebnis, tensor)
        self.assertEqual(aufrufe, [])

    def test_a_source_outside_the_group_is_refused(self):
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            Bar1Unavailable,
        )

        t = _stub(rank=0, welt=3)
        for src in (-1, 3, 99):
            with self.assertRaises(Bar1Unavailable):
                _fahre(t, torch.zeros(64, dtype=torch.uint8), src)


class TestHandlesGate(CustomTestCase):
    """When the transport says yes -- and when it says no, and why."""

    def test_yes_for_the_handover_payload(self):
        t = _stub(a2a_schlitz=8384512)
        self.assertTrue(t._handles_broadcast(ABNAHME_BYTES))
        self.assertTrue(t.handles("broadcast", ABNAHME_BYTES))

    def test_over_the_slot_is_not_rejected_but_split(self):
        """The whole point. Rejecting would abort a capture, not slow it."""
        t = _stub(a2a_schlitz=4096)
        self.assertTrue(t._handles_broadcast(4096 * 4))
        self.assertEqual(t.bc_rounds(4096 * 4), 4)

    def test_unaligned_payload_is_accepted(self):
        """Unlike all_reduce: the a2a kernel has a byte path for the tail."""
        self.assertTrue(_stub()._handles_broadcast(1023))

    def test_off_switch(self):
        self.assertFalse(_stub(bc_an=False)._handles_broadcast(65536))

    def test_needs_the_a2a_area_and_both_proofs(self):
        self.assertFalse(_stub(a2a_an=False)._handles_broadcast(65536))
        self.assertFalse(_stub(_a2a_beleg=False)._handles_broadcast(65536))
        self.assertFalse(_stub(_bc_beleg=False)._handles_broadcast(65536))
        self.assertFalse(_stub(off_a2a=-1)._handles_broadcast(65536))
        self.assertFalse(_stub(a2a_schlitz=0)._handles_broadcast(65536))

    def test_needs_the_window(self):
        self.assertFalse(_stub(_fenster_minimum=1 << 20)._handles_broadcast(65536))

    def test_only_the_empty_payload_is_below_the_floor(self):
        """There is no floor beyond "non-empty" -- and that is the fix.

        16 used to stand here ("one packet", copied from a2a). The standard
        run sends a 12-byte broadcast; it was refused while the 128-byte
        case from the first crash went through, so the coverage looked
        complete and the run still died.
        """
        self.assertFalse(_stub()._handles_broadcast(0))
        self.assertFalse(_stub()._handles_broadcast(-1))
        for n in (1, 4, 8, 12, 15, 16):
            self.assertTrue(_stub()._handles_broadcast(n), msg=f"{n} B")

    def test_round_cap(self):
        t = _stub(a2a_schlitz=1024, bc_max_runden=4)
        self.assertTrue(t._handles_broadcast(1024 * 4))
        self.assertFalse(t._handles_broadcast(1024 * 4 + 1))

    def test_the_answer_does_not_depend_on_the_rank(self):
        """Two ranks answering differently is a hang, not an error."""
        antworten = {
            _stub(rank=r, welt=4).handles("broadcast", ABNAHME_BYTES)
            for r in range(4)
        }
        self.assertEqual(antworten, {True})

    def test_the_other_ops_are_unaffected(self):
        t = _stub(ag_an=True, ag_min_bytes=16, ag_max_runden=16)
        self.assertFalse(t.handles("reduce_scatter", 65536))
        self.assertFalse(t.handles("nonsense", 65536))
        self.assertTrue(t.handles("all_gather", 65536))


#: The ladder against which every new condition of this path is measured --
#: from one byte to one round past the slot. ``slot`` is chosen small so
#: the byte simulation stays affordable per size; the ratios (below one
#: packet, above one packet, exactly one slot, one over) are the same as
#: at 8 MiB.
_SLOT = 1024
_LEITER = (1, 4, 12, 128, 1024 - 1, 1024, 1024 + 1, 4096)


class TestTheShippedFloor(CustomTestCase):
    """The default in the SOURCE, not the one the stub is convenient with.

    The first attempt had both: a stub that said 16 and a shipped default of
    16, agreeing with each other and with nothing else. What broke was a
    12-byte broadcast from the standard run, and no test could see it
    because every test asked the same wrong number.
    """

    def _default(self, name: str) -> str:
        import re
        from pathlib import Path

        import sglang.srt.distributed.device_communicators.barlink_bar1 as mod

        text = Path(mod.__file__).read_text(encoding="utf-8")
        treffer = re.search(
            r"""os\.environ\.get\(\s*["']%s["']\s*,\s*["'](\d+)["']""" % name,
            text,
        )
        self.assertIsNotNone(treffer, msg=f"{name} not in source")
        return treffer.group(1)

    def test_broadcast_floor_is_one_byte(self):
        self.assertEqual(self._default("SGLANG_BARLINK_BAR1_BC_MIN_BYTES"), "1")

    def test_all_gather_floor_is_one_byte(self):
        """The twin of the same bug, and it was live.

        all_gather had the identical 16 from the identical copy, the same
        ragged-tail path in the kernel and the same "no fallback under
        capture" situation. A 12-byte shard would have aborted a run the
        same way.
        """
        self.assertEqual(self._default("SGLANG_BARLINK_BAR1_AG_MIN_BYTES"), "1")

    def test_the_stub_mirrors_the_shipped_floor(self):
        self.assertEqual(
            str(_stub().bc_min_bytes),
            self._default("SGLANG_BARLINK_BAR1_BC_MIN_BYTES"),
        )


class TestTheSizeLadder(CustomTestCase):
    """One byte to one round past the slot -- gate, plan and bytes each.

    Three questions per size, because they fail in different places: does
    the transport SAY yes, does the round decomposition cover the payload,
    and do the bytes actually arrive. The 12-byte case had a yes for two of
    them and a no for the first, which is why the run died with a coverage
    message rather than wrong data.
    """

    def _stub(self, rank=0, welt=3):
        return _stub(rank=rank, welt=welt, a2a_schlitz=_SLOT)

    def test_the_gate_says_yes_to_every_rung(self):
        for n in _LEITER:
            self.assertTrue(
                self._stub().handles("broadcast", n), msg=f"{n} B rejected"
            )

    def test_the_plan_covers_every_rung_exactly_once(self):
        for n in _LEITER:
            plan = bc_plan(n, _SLOT)
            gesehen = []
            for versatz, length in plan:
                self.assertGreater(length, 0, msg=f"{n} B: empty round")
                self.assertLessEqual(length, _SLOT, msg=f"{n} B")
                gesehen.extend(range(versatz, versatz + length))
            self.assertEqual(gesehen, list(range(n)), msg=f"{n} B")

    def test_the_round_count_matches_the_gate(self):
        for n in _LEITER:
            self.assertEqual(
                len(bc_plan(n, _SLOT)), self._stub().bc_rounds(n), msg=f"{n} B"
            )

    def test_the_rung_below_one_packet_is_a_single_ragged_round(self):
        """12 bytes: one incomplete packet, and nothing else.

        This is the rung the kernel treats specially (assemble in a
        register, read back byte by byte) and the one no other size on the
        ladder exercises.
        """
        self.assertEqual(bc_plan(12, _SLOT), [(0, 12)])
        self.assertEqual(self._stub().bc_rounds(12), 1)

    def test_every_rung_delivers_the_right_bytes(self):
        for n in _LEITER:
            for src in range(3):
                _pruefe_alle_halten_die_quelle(self, 3, n, src, _SLOT)

    def test_every_rung_keeps_the_pairwise_contract(self):
        """Per round: ``e_len[i]`` here == ``s_len[rank]`` on rank ``i``."""
        for n in _LEITER:
            tabellen = _tabellen(3, n, src=1, slot=_SLOT)
            for k in range(len(tabellen[0])):
                for r in range(3):
                    for i in range(3):
                        self.assertEqual(
                            tabellen[r][k]["e_len"][i],
                            tabellen[i][k]["s_len"][r],
                            msg=f"n={n} round={k} r={r} i={i}",
                        )


class TestLoudBar(CustomTestCase):
    """The bar in barlink._select, before and after this change."""

    def _comm(self, transport):
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = transport
        c._path_dispatcher = None
        return c

    def test_the_handover_case_no_longer_raises(self):
        from sglang.srt.distributed.device_communicators import barlink as mod

        t = _stub(a2a_schlitz=8384512)
        c = self._comm(t)
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            self.assertIs(
                mod.BarlinkCommunicator._select(c, "broadcast", ABNAHME_BYTES),
                t,
            )

    def test_reduce_scatter_still_raises_and_names_the_new_coverage(self):
        from sglang.srt.distributed.device_communicators import barlink as mod

        c = self._comm(_stub())
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            with self.assertRaises(RuntimeError) as ctx:
                mod.BarlinkCommunicator._select(c, "reduce_scatter", 4096)
        text = str(ctx.exception)
        self.assertIn("reduce_scatter", text)
        # Derived from BARLINK_OPS, never a literal list in the message -- so it
        # cannot claim broadcast is missing now that it is not.
        self.assertIn("broadcast", text)
        self.assertIn("all_gather", text)

    def test_a_broadcast_past_the_round_cap_still_raises(self):
        """The bar is not disarmed by the coverage, only narrowed.

        This used to read "below the floor" and used 8 bytes -- and that is
        precisely what was wrong: the floor refused sizes the stack really
        sends. What is left above the bar is a NAMED limit (more rounds than
        `bc_max_runden` would make the transport a loop), not a threshold
        somebody copied.
        """
        from sglang.srt.distributed.device_communicators import barlink as mod

        t = _stub(a2a_schlitz=1024, bc_max_runden=4)
        c = self._comm(t)
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            with self.assertRaises(RuntimeError) as ctx:
                mod.BarlinkCommunicator._select(c, "broadcast", 1024 * 4 + 1)
        self.assertIn("broadcast", str(ctx.exception))

    def test_the_twelve_byte_case_from_the_standard_run_passes(self):
        """The whole point of the follow-up.

        The first attempt put broadcast in the coverage list, and the bar
        still fired -- with a message that named broadcast as covered. That
        reads like a contradiction and was one: coverage is not the op, it
        is the op AND the size.
        """
        from sglang.srt.distributed.device_communicators import barlink as mod

        t = _stub()
        c = self._comm(t)
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            for n in (1, 4, 12, 128):
                self.assertIs(
                    mod.BarlinkCommunicator._select(c, "broadcast", n), t,
                    msg=f"{n} B",
                )


if __name__ == "__main__":
    unittest.main()
