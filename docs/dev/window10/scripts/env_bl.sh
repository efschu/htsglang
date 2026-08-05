# barlink, the standing default transport (user law 2026-08-03/05). With
# SGLANG_BARLINK_TRANSPORT unset, barlink.py:67 resolves to "device" -- which
# is what production runs. The record's bar1 sub-transport is NOT available in
# this window: it needs /dev/dmabuf_holder, absent on this host since the
# reboot (stock NVIDIA open 595.58.03, no smallbar holder module loaded).
export SGLANG_BARLINK=1
unset SGLANG_BARLINK_TRANSPORT
unset SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY
unset SGLANG_BARLINK_PEER_WATCHDOG
