# barlink device with the #517 fast path DISABLED: no peer watchdog thread
# means barlink_abort_gate.should_poll_status() is False, so the transport
# keeps the pre-#517 in-line blocking read at every check.
export SGLANG_BARLINK=1
unset SGLANG_BARLINK_TRANSPORT
unset SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY
export SGLANG_BARLINK_PEER_WATCHDOG=0
