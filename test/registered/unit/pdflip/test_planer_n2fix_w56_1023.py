"""planer-n2fix 1023 (B7): the W56 line of the corridor pass said "every user reserve is 0" on the 27B duo boot of 23.10. (05:46Z) while the boot
ran --user-reserve-mib 1800,1400: the reserve is CONTAINED in each card's booked awake rest (VRAM-GRUNDGESETZ 29.09., pdflip/budget_rest.py), so the
corridor pass sees 0.  The line now says per card which of the two it is and names the launcher's own flag (--user-reserve-mib).

planer-n2fix2 1023 (review A1): "contained in the booked awake rest" read as if the external headroom were kept; it is not -- on a card with a
booked rest the reserve is NOT CHARGED, the measured rest replaces it (laws 19.09. keine-korridor-reserve-nie / 29.09. VRAM-GRUNDGESETZ) and no
external headroom stays on that card.  The old advice "pass --user-reserve-mib ... to make it actuate" used the reserve as a run lever, which
reserve-semantik forbids: it is gone from the line (and from corridor_budget's REFUSED-UNMEASURED-FLOOR verdict)."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L  # noqa: E402

U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U3080 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=499),
        L.Card(nvml_index=1, uuid=U3080, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _w56(lines):
    got = [ln for ln in lines if "PDFLIP-BUDGET corridor-constrained" in ln and "W56" in ln]
    assert len(got) == 1, lines
    return got[0]


def test_the_duo_boot_line_names_the_reserve_inside_the_booked_rest():
    lines = []
    out = L.budgets_from_dc(_cards(), {U5090: 894, U3080: 794}, lines.append, "D", corridor_constrain=True,
                            user_reserve_by_card={U5090: 1800, U3080: 1400}, booked_rest_mib=[3191, 2079],
                            booked_rest_provenance="RECORD test", charge_driver_carve=True)
    # the boot's own figures (boot_weg2_dkr27bnvfp4duobar1duo10230546 front.log): 28016 / 17176 -- unchanged by the text fix
    assert out == [28016, 17176]
    w = _w56(lines)
    assert "every user reserve is 0" not in w
    assert "nvml0 1800 MiB NOT charged: the measured awake rest 3191 MiB replaces it (laws 19.09./29.09.), no external headroom is kept on this card" in w
    assert "nvml1 1400 MiB NOT charged: the measured awake rest 2079 MiB replaces it" in w
    assert "contained in" not in w
    # no reserve-as-lever advice (reserve-semantik): neither flag spelling, no "actuate" hint
    assert "--user-reserve-mib" not in w and "--rank-user-reserve-mib" not in w and "actuate" not in w


def test_a_card_without_reserve_says_zero():
    lines = []
    L.budgets_from_dc(_cards(), {U5090: 894, U3080: 794}, lines.append, "D", corridor_constrain=True,
                      user_reserve_by_card={}, booked_rest_mib=[3191, 2079], booked_rest_provenance="RECORD test")
    w = _w56(lines)
    assert "nvml0 0 MiB;" in w and "contained" not in w and "NOT charged" not in w

