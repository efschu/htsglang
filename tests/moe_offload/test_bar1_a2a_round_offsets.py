"""bar1 all_to_all round loop (fn7j 19.09.): a block that finished in an
earlier round has length 0 -- its offset must stay inside its own block
instead of walking on by k*slot past the end of the tensor."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed.device_communicators import barlink_bar1 as bb


def _rounds_seen(send_bytes, recv_bytes, slot):
    t = bb.BarlinkBar1Transport.__new__(bb.BarlinkBar1Transport)
    t._geo = {"a2a_slot": slot}
    seen = []

    def one_round(comm, output, inp, s_len, e_len, s_off, e_off, kernel_bytes, op_label="x"):
        seen.append((list(s_len), list(e_len),
                     None if s_off is None else list(s_off),
                     None if e_off is None else list(e_off)))

    t._a2a_one_round = one_round
    rounds = bb.a2a_rounds(max(send_bytes + recv_bytes), slot)
    t.barlink_all_to_all_single(None, None, None, send_bytes, recv_bytes, rounds=rounds)
    return seen


def test_finished_blocks_keep_their_offset_inside_the_tensor():
    # heads 15/5/4 x 4 MB rows, slot 8 MB: the 4-head block finishes in round 3
    send = [60, 20, 16]
    recv = [60, 60, 60]
    seen = _rounds_seen(send, recv, slot=8)
    assert len(seen) == 8  # ceil(60 / 8)
    total = sum(send)
    for s_len, e_len, s_off, e_off in seen:
        base = 0
        for z, length in enumerate(send):
            assert base <= s_off[z] and s_off[z] + s_len[z] <= base + length
            assert s_off[z] + s_len[z] <= total
            base += length
    # every byte of every block is moved exactly once
    for z, length in enumerate(send):
        assert sum(r[0][z] for r in seen) == length
    for z, length in enumerate(recv):
        assert sum(r[1][z] for r in seen) == length


def test_single_round_is_untouched():
    seen = _rounds_seen([4, 4, 4], [4, 4, 4], slot=8)
    # one round: the offsets stay None, _a2a_one_round derives the prefix sums
    assert seen == [([4, 4, 4], [4, 4, 4], None, None)]
