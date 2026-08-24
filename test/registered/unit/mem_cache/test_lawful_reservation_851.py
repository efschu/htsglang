"""#851 F2: the boot VA reservation must cover every backing the dial may reach.

W22, on the rank holding the group floor::

    GROUP FLOOR CANNOT FUND THE LIVE SET: this rank holds the group floor at
    126976 rows and the group's live set needs 126977 ... the pool's IMMUTABLE
    VA reservation is 125052 rows and the target is 126977: this rank's dial
    has already moved its size past its own boot reservation, so no grow can
    ever be accepted here.  -> exit=RESERVATION-CAPPED

The #839-METAL actuator built for exactly this gap ran 49 times and could not
pay, because the reservation is fixed at construction from the size at that
instant (`reserved_num_tokens=self.size`) and never reassigned, while `size`
stays mutable and the #330 dial writes it.

THE LAW. The rung's floor is `max_live + 1 + margin + admission_reserve`, and
`max_live` may legitimately sit at the pool's own cap. So the largest backing
the dial may lawfully be asked to reach is `size + 1 + margin + reserve`.
Reserving `size` under-reserves by exactly the floor's headroom -- which is why
the wall lands precisely on the rank that holds the floor.

THE DIRECTION IS NOT SYMMETRIC, which is the argument for rounding up: a VA
reservation is ADDRESS SPACE, not committed pages (the point of
`swappable_backing`). Over-reserving costs address space; under-reserving is a
permanent wall no runtime actuator can lift. #848's merged commits made this
failure NAMED; this makes it not happen.

Hermetic: a pure function, no pool, no driver, no CUDA.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import lawful_reservation_rows


class TestTheReservationCoversTheFloor(unittest.TestCase):
    def test_the_w22_pool_would_have_been_reservable(self):
        """The specimen. 126976 backed, 4096 admission reserve."""
        self.assertEqual(lawful_reservation_rows(126976, 4096, 0), 131073)

    def test_it_covers_the_floor_the_rung_can_demand(self):
        """THE PROPERTY, not the number: reservation >= any lawful floor.

        The floor is max_live + 1 + margin + reserve and max_live <= size, so
        the reservation must dominate it for every max_live in range. Asserted
        across the range rather than at one point, so a formula that happens to
        match 131073 by coincidence still fails.
        """
        for size in (1024, 126976, 212992):
            for reserve in (0, 4096, 8192):
                for margin in (0, 256):
                    got = lawful_reservation_rows(size, reserve, margin)
                    worst_floor = size + 1 + margin + reserve
                    self.assertGreaterEqual(got, worst_floor, f"{size}/{reserve}/{margin}")

    def test_it_never_reserves_less_than_the_pool(self):
        """CAN-FAIL TWIN. The pre-#851 value is the FLOOR of the new one."""
        for size in (0, 1, 1024, 126976):
            self.assertGreaterEqual(lawful_reservation_rows(size, 0, 0), size)

    def test_an_empty_pool_reserves_nothing(self):
        """A pool with no rows must not acquire a reservation out of nowhere."""
        self.assertEqual(lawful_reservation_rows(0, 4096, 256), 0)
        self.assertEqual(lawful_reservation_rows(-5, 4096, 256), 0)

    def test_negative_knobs_cannot_shrink_the_reservation(self):
        """A misconfigured knob must not produce an under-reservation."""
        self.assertGreaterEqual(lawful_reservation_rows(1024, -9999, -9999), 1024)


if __name__ == "__main__":
    unittest.main()
