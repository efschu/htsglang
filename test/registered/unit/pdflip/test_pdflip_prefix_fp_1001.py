"""L15-18: unit tests for the front's prefix-fingerprint ledger (pure, CPU).

Red-green for AP L15-18 (L3-RETURN stage 2): the front marks each leg-1
prompt with a fingerprint of its leading prefix and says whether that
fingerprint was already in the ledger BEFORE this boot started, so
monitor M2's L3-RETURN check stops counting never-seen prompts as MISS.
"""

import hashlib
import struct

from flliper.srt.pdflip.prefix_fp import PrefixSeen, fingerprint


def test_fingerprint_stable_and_prefix_only():
    ids_a = list(range(5000)) + [11] * 900
    ids_b = list(range(5000)) + [22] * 900
    # same first 4096 ids -> same fingerprint; a different tail is ignored
    assert fingerprint(ids_a) == fingerprint(ids_b)
    assert fingerprint(ids_a[:4096]) == fingerprint(ids_a)
    # a difference inside the window changes it
    other = list(range(4095)) + [999] + [11] * 900
    assert fingerprint(other) != fingerprint(ids_a)
    assert len(fingerprint(ids_a)) == 16
    # encoding is int64 little-endian bytes of the first n ids
    expect = hashlib.sha1(struct.pack("<qqq", 7, 8, 9)).hexdigest()[:16]
    assert fingerprint([7, 8, 9]) == expect


def test_fingerprint_text_path():
    window = 4096 * 4  # first n*4 chars are hashed
    t1 = "x" * window + "AAA"
    t2 = "x" * window + "BBB"
    assert fingerprint(t1) == fingerprint(t2)  # tail beyond the window ignored
    assert fingerprint("A" * (window + 1)) != fingerprint("B" + "A" * window)
    assert fingerprint("short text") == hashlib.sha1(
        "short text".encode("utf-8")
    ).hexdigest()[:16]


def test_prefix_seen_two_boots(tmp_path):
    path = str(tmp_path / "seen.txt")
    boot1 = PrefixSeen(path)
    old = fingerprint("old prompt")
    assert boot1.seen_before(old) is False
    boot1.add(old)
    boot1.save()

    boot2 = PrefixSeen(path)
    assert boot2.seen_before(old) is True  # loaded at boot start
    new = fingerprint("new prompt")
    boot2.add(new)
    assert boot2.seen_before(new) is False  # this boot's additions never count
    # an old prompt re-added during boot 2 stays "seen before"
    boot2.add(old)
    assert boot2.seen_before(old) is True
    boot2.save()

    boot3 = PrefixSeen(path)
    assert boot3.seen_before(new) is True


def test_prefix_seen_missing_and_corrupt_file(tmp_path):
    missing = str(tmp_path / "nope.txt")
    ps = PrefixSeen(missing)  # missing file -> empty set, never raises
    assert ps.seen_before("deadbeefdeadbeef") is False

    corrupt = tmp_path / "seen.txt"
    corrupt.write_bytes(b"\x00\x01not a ledger \xff\xfe\nzz\n")
    ps2 = PrefixSeen(str(corrupt))  # corrupt file -> empty, never raises
    assert ps2.seen_before("deadbeefdeadbeef") is False
    ps2.add("deadbeefdeadbeef")
    ps2.save()
    assert PrefixSeen(str(corrupt)).seen_before("deadbeefdeadbeef") is True


def test_prefix_seen_cap_keeps_newest(tmp_path):
    path = str(tmp_path / "seen.txt")
    ps = PrefixSeen(path)
    for i in range(8300):
        ps.add("%016x" % i)
    ps.save()
    again = PrefixSeen(path)
    assert again.seen_before("%016x" % 8299) is True  # newest kept
    assert again.seen_before("%016x" % 0) is False  # oldest dropped


def test_front_leg1_block_wired():
    import flliper.srt.pdflip.front as front

    with open(front.__file__, encoding="utf-8") as f:
        src = f.read()
    # the logger call, not the docstring that merely quotes the format
    i = src.find('logger.info("PDFLIP-SERVED group=P leg=1 rid=%s')
    assert i > 0
    block = src[i:i + 2500]
    assert "FLLIPER_PDFLIP_PREFIX_FP_PATH" in block
    assert "prefix_fp" in block
    assert "PDFLIP-PREFIX-FP" in block
